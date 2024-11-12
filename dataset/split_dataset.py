from dataset.download_dataset import dump_json, dataset_dir_path

import json
import random
import os

random.seed(42)

def split_dataset(correct_challenges, incorrect_challenges, 
                                train_size_correct=256, train_size_incorrect=100):

    random.shuffle(correct_challenges)
    random.shuffle(incorrect_challenges)

    train_correct = correct_challenges[:train_size_correct] 
    train_incorrect = incorrect_challenges[:train_size_incorrect]
    #train_file_path = os.path.join(dataset_dir_path, 'splits', 'train.json')
    #dump_json(successful_train, train_file_path)


    # The remaining samples will be used for test
    test = correct_challenges[train_size_correct:]  + incorrect_challenges[train_size_incorrect:]

    return train_correct, train_incorrect, test
    