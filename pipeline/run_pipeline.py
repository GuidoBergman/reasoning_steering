import torch
import random
import json
import os
import argparse

from dataset.load_dataset import load_dataset
from dataset.split_dataset import split_dataset
from dataset.contrastive_pairs import get_contrastive_pairs

from abstract_and_reason import solver_v1, utils

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
#from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    return parser.parse_args()



def generate_and_save_candidate_directions(cfg, model_base, train_correct, train_incorrect):
    """Generate and save candidate directions."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'generate_directions')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'generate_directions'))

    mean_diffs = generate_directions(
        model_base,
        train_correct, 
        train_incorrect
        artifact_dir=os.path.join(cfg.artifact_path(), "generate_directions"))

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), 'generate_directions/mean_diffs.pt'))

    return mean_diffs


def generate_and_evaluate_solutions(dataset, solver, output_path, fwd_pre_hooks=[], fwd_hooks=[]): 
    correct_challenges, incorrect_challenges = [], []
    completions = []
    for challenge in dataset:
        answer, puzzle_completions = solver.predict([challenge], fwd_pre_hooks, fwd_hooks)
        score = utils.get_score(answer, challenge['correct_answer'])
        if score == 1:
            correct_challenges.append(challenge)
        else:
            incorrect_challenges.append(challenge)

        completions.append(puzzle_completions)

    with open(output_path, "w") as f:
            json.dump(completions, f, indent=4)


    return correct_challenges, incorrect_challenges


def run_pipeline(model_path):
    """Run the full pipeline."""
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = construct_model_base(cfg.model_path)

    solver = solver_v1.Solver(model_base)


    # 1. Gather jailbreak prompts and forbidden questions
    #download_dataset(cfg.jailbreak_prompts_dataset)

    # 2. Generate interactions with the baseline model
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    small_challenges = utils.get_tiny_arc(solver.training_challenges, max_n=8, max_m=8)
    train_dataset = load_dataset(small_challenges, solver=solver)
    correct_challenges, incorrect_challenges = generate_and_evaluate_solutions(train_dataset, solver, 
                                                                               'completions_train_baseline.json', baseline_fwd_pre_hooks, baseline_fwd_hooks)  
    print(f'--------Training-------------')
    print(f'Correct challenges: {len(correct_challenges)}')
    print(f'Incorrect challenges: {len(incorrect_challenges)}')


    # 3. Filter interactions
    train_correct_ids, train_incorrect_ids, test_ids = split_dataset(correct_challenges, incorrect_challenges, 
                                train_size_correct=cfg.train_size_correct, train_size_incorrect=cfg.train_size_incorrect)


    # 4. Find the direction representing the jailbreak feature
    #train = load_dataset_split(split='train')
    train_correct, train_incorrect = get_contrastive_pairs(train_correct_ids, train_incorrect_ids, solver)
    candidate_directions = generate_and_save_candidate_directions(cfg, model_base, train_correct, train_incorrect)
    pos = cfg.pos
    layer = cfg.layer
    direction = candidate_directions[pos][layer]

    # 5. Intervene the model
    #ablation_fwd_pre_hooks, ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction)
    actadd_fwd_pre_hooks, actadd_fwd_hooks = [(model_base.model_block_modules[layer], get_activation_addition_input_pre_hook(vector=direction, coeff=1.0))], []

    
    # 6a. Generate completions with baseline in the test set
    baseline_fwd_pre_hooks, baseline_fwd_hooks = [], []
    test_dataset = load_dataset(test_ids, solver=solver)
    correct_challenges, incorrect_challenges = generate_and_evaluate_solutions(test_dataset, solver, 
                                                                               'completions_test_baseline.json', baseline_fwd_pre_hooks, baseline_fwd_hooks)

    print(f'--------Test baseline-------------')
    print(f'Correct challenges: {len(correct_challenges)}')
    print(f'Incorrect challenges: {len(incorrect_challenges)}')

    # 6b. Generate completions with the intervened model in the test set
    correct_challenges, incorrect_challenges = generate_and_evaluate_solutions(test_dataset, solver, 
                                                                               'completions_test_act_add.json', actadd_fwd_pre_hooks, actadd_fwd_hooks)
    
    print(f'--------Test intervened-------------')
    print(f'Correct challenges: {len(correct_challenges)}')
    print(f'Incorrect challenges: {len(incorrect_challenges)}')

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path)
