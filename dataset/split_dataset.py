from dataset.download_dataset import dump_json, dataset_dir_path

import json
import random
import os

random.seed(42)

def split_dataset(evaluations_file_path, train_size_successful = 256, test_size_unsuccessful = 100):
    with open(evaluations_file_path, 'r') as f:
        evaluation = json.load(f)

    successful_jailbreaks = [completion for completion in evaluation['completions'] if completion['is_jailbreak_llamaguard2'] and completion['is_jailbreak_harmbench'] and len(completion['prompt']) < 15000]
    random.shuffle(successful_jailbreaks)


    unsuccessful_jailbreaks = [completion for completion in evaluation['completions'] if  not completion['is_jailbreak_llamaguard2'] and not completion['is_jailbreak_harmbench']  and len(completion['prompt']) < 15000]
    random.shuffle(unsuccessful_jailbreaks)




    successful_train = successful_jailbreaks[:train_size_successful]
    train_file_path = os.path.join(dataset_dir_path, 'splits', 'train.json')
    dump_json(successful_train, train_file_path)


    # The remaining samples will be used for test
    test = successful_jailbreaks[train_size_successful:]  + unsuccessful_jailbreaks[:test_size_unsuccessful]
    test_file_path = os.path.join(dataset_dir_path, 'splits', 'test.json')
    dump_json(test, test_file_path)
    