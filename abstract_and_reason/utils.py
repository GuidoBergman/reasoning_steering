import numpy as np
import pprint

def get_score(model_answers, real_answers):
    """
    Computes a score based on the similarity between model-generated answers and real answers.
    It handles input matrices of different shapes and ensures comparisons are done within the bounds of the shortest list.

    Args:
        model_answers (list of lists): Model-generated answers as matrices (list of lists).
        real_answers (list of lists): Real answers as matrices (list of lists).

    Returns:
        int: The total score as an integer.
    """
    total_score = 0
    valid_comparisons = 0
    
    for i in range(min(len(model_answers), len(real_answers))):
        model_answer = np.array(model_answers[i])
        real_answer = np.array(real_answers[i])
        
        if model_answer.shape == real_answer.shape:
            score = ((model_answer == real_answer).astype(int)).mean()
            if score == 1.0:
                total_score += 1
            valid_comparisons += 1
    
    return int(total_score / valid_comparisons) if valid_comparisons > 0 else 0
    

def get_tiny_arc(challenges, max_n, max_m):
    """
    Selects challenges from the ARC dataset based on grid size.

    This function filters and returns a list of challenge IDs where the average size of the input and output grids 
    (width and height) is less than or equal to the specified maximum grid dimensions (max_n x max_m).

    Args:
        challenges (dict): A dictionary containing ARC challenges, either from the training or evaluation set.
        max_n (int): The maximum allowed width (n) for a grid.
        max_m (int): The maximum allowed height (m) for a grid.

    Returns:
        list: A list of challenge IDs that meet the size constraints.
    """
    ids = list(challenges)
    ids_sizes = []
    
    for challenge_id in ids:
        input_mean = 0
        output_mean = 0
        nb_example = 0
        
        for challenge in challenges[challenge_id]['train']:
            input_size = sum(len(obj) for obj in challenge['input']) * len(challenge['input'])
            output_size = sum(len(obj) for obj in challenge['output']) * len(challenge['output'])
            
            input_mean += input_size
            output_mean += output_size
            
            nb_example += 1
        
        input_mean /= nb_example
        output_mean /= nb_example
        
        if input_mean <= max_n**2 and output_mean <= max_m**2:
            ids_sizes.append(challenge_id)
    
    return ids_sizes


# This function was adapted from https://github.com/olimoz/mech-interp-reasoning/blob/main/assets/Execute_Challenges_Gemma2.ipynb
def convert_puzzle_to_prompts(puzzle_inps_train, puzzle_outs_train, puzzle_inps_test):
    results = []
    
    train_prompt = []
    for i, (array_input, array_output) in enumerate(zip(puzzle_inps_train, puzzle_outs_train)):
        train_prompt.append("=============") if i > 0 else None
        train_prompt.append(f"TRAIN Pair {i}")
        train_prompt.append(f"INPUT. Shape={array_input.shape}")
        train_prompt.append(pprint.pformat(array_input))
        train_prompt.append(f"OUTPUT. Shape={array_output.shape}")
        train_prompt.append(pprint.pformat(array_output))


    for i, test_input in enumerate(puzzle_inps_test):
        test_prompt = []
        test_prompt.append("==========================")
        test_prompt.append(f"TEST Pair 0")
        test_prompt.append(f"INPUT. Shape={test_input.shape}")
        test_prompt.append(pprint.pformat(test_input))
        test_prompt.append('OUTPUT. ')

        results.append('\n'.join(train_prompt) + '\n'.join(test_prompt))

    return results