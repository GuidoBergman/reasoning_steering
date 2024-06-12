import os
import json


dataset_dir_path = os.path.dirname(os.path.realpath(__file__))


DATASET_NAMES = ["alpaca", "jailbreak_prompts"]


SPLITS = ['train', 'val', 'test']


SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, 'splits/{split}.json')


def load_dataset_split(split: str, type: str=None, instructions_only: bool=False):
    assert split in SPLITS

    file_path = SPLIT_DATASET_FILENAME.format(type=type, split=split)


    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['prompt'] for d in dataset]

    return dataset



def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in DATASET_NAMES, f"Valid datasets: {DATASET_NAMES}"

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['prompt'] for d in dataset]
 
    return dataset
