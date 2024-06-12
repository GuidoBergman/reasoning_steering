import torch
import random
import json
import os
import argparse

from dataset.load_dataset import load_dataset, load_dataset_split
from dataset.download_dataset import download_dataset
from dataset.split_dataset import split_dataset
from dataset.contrastive_pairs import get_contrastive_pairs

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()



def generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs


def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)
    
    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)


def generate_and_save_single_answer_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, dataset=None):
    """Generate and save completions for a dataset."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'completions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'completions'))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_single_answer(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens)
    
    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)


def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies):
    """Evaluate completions and save results for a dataset."""
    with open(os.path.join(cfg.artifact_path(), f'completions/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), "completions", f"{dataset_name}_{intervention_label}_evaluations.json"),
    )

    with open(f'{cfg.artifact_path()}/completions/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)


def run_pipeline(model_path):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)


    # 1. Gather jailbreak prompts and forbidden questions
    download_dataset(cfg.jailbreak_prompts_dataset)

    # 2. Generate interactions with the baseline model
    dataset_name = 'jailbreak_prompts'
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', dataset_name)

    # 3. Filter interactions
    intervention_label = 'baseline'
    evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies=cfg.jailbreak_eval_methodologies)
    evaluations_file_path = os.path.join(cfg.artifact_path(), 'completions', f'{dataset_name}_{intervention_label}_evaluations.json')
    split_dataset(evaluations_file_path, train_size_successful=cfg.train_size_successful, 
                  test_size_unsuccessful=cfg.test_size_unsuccessful)


    # 4. Find the direction representing the jailbreak feature
    train = load_dataset_split(split='train')
    harmful_train, harmless_train = get_contrastive_pairs(train)
    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, harmful_train, harmless_train)
    pos = cfg.pos
    layer = cfg.layer
    direction = candidate_directions[pos][layer]

    # 5. Intervene the model
    ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=-1.0))], []

    
    # 6. Generate completions
    test = load_dataset_split(split='test')
    generate_and_save_completions_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmful_test', dataset=test)
    generate_and_save_completions_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', 'harmful_test', dataset=test)
    generate_and_save_completions_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', 'harmful_test', dataset=test)


    # 7. Evaluate completions
    print('----------------------------------Harmful----------------------------------')
    print('------------Baseline------------')
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmful_test', eval_methodologies=cfg.jailbreak_eval_methodologies)
    print('------------Ablation------------')
    evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', 'harmful_test', eval_methodologies=cfg.jailbreak_eval_methodologies)
    print('------------Actadd------------')
    evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmful_test', eval_methodologies=cfg.jailbreak_eval_methodologies)
    


    download_dataset(cfg.harmless_dataset)
    harmless_test_size = cfg.test_size_harmless
    harmless_test = random.sample(load_dataset(cfg.harmless_dataset), harmless_test_size)

    generate_and_save_single_answer_for_dataset(cfg, model_base, baseline_fwd_pre_hooks, baseline_fwd_hooks, 'baseline', 'harmless_test', dataset=harmless_test)
    generate_and_save_single_answer_for_dataset(cfg, model_base, ablation_fwd_pre_hooks, ablation_fwd_hooks, 'ablation', 'harmless_test', dataset=harmless_test)
    generate_and_save_single_answer_for_dataset(cfg, model_base, actadd_fwd_pre_hooks, actadd_fwd_hooks, 'actadd', 'harmless_test', dataset=harmless_test)

    print('----------------------------------Harmless----------------------------------')
    print('------------Baseline------------')
    evaluate_completions_and_save_results_for_dataset(cfg, 'baseline', 'harmless_test', eval_methodologies=cfg.refusal_eval_methodologies)
    print('------------Ablation------------')
    evaluate_completions_and_save_results_for_dataset(cfg, 'ablation', 'harmless_test', eval_methodologies=cfg.refusal_eval_methodologies)
    print('------------Actadd------------')
    evaluate_completions_and_save_results_for_dataset(cfg, 'actadd', 'harmless_test', eval_methodologies=cfg.refusal_eval_methodologies)


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path)
