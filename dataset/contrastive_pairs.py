from dataset.load_dataset import load_dataset

def get_contrastive_pairs(train_correct_ids, train_incorrect_ids, solver):    

    train_correct = load_dataset(train_correct_ids)
    train_incorrect = load_dataset(train_incorrect_ids)

    return train_correct, train_incorrect