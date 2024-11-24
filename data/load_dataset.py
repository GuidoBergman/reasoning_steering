from abstract_and_reason.utils import convert_array_to_str

def load_dataset(challenge_ids, solver, challenges=None, solutions=None):
    questions, solutions_list, ids = [], [], []
    for id in challenge_ids:
        questions_challenge, solutions_challenge = solver.convert_challenge_to_prompts(id, challenges, solutions)
        questions += questions_challenge
        solutions_list += solutions_challenge 
        ids += [id for _ in range(len(questions_challenge))]

    dataset = [{
            'question': question,
            'prompt': None,
            'first_response': None,
            'category': id,
            'correct_answer': solution,
            'correct_answer_str': convert_array_to_str(solution)
        } for question, solution, id in zip(questions, solutions_list, ids)
    ]
 
    return dataset



def load_dataset_by_challenge(challenge_ids, solver, challenges=None, solutions=None):
    dataset = []
    for id in challenge_ids:
        questions_challenge, solutions_challenge = solver.convert_challenge_to_prompts(id, challenges, solutions)
    
        dataset.append({
            'questions': questions_challenge,
            'prompt': None,
            'first_response': None,
            'category': id,
            'correct_answers': solutions_challenge,
            'correct_answers_str': [convert_array_to_str(solution) for solution in solutions_challenge]
        } 
    )

 
    return dataset
