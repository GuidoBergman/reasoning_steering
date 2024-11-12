from abstract_and_reason.utils import convert_array_to_str

def load_dataset(challenge_ids, solver):
    questions, solutions, ids = [], [], []
    for id in challenge_ids:
        questions_challenge, solutions_challenge = solver.convert_challenge_to_prompts(id)
        questions += questions_challenge
        solutions += solutions_challenge 
        ids += [id for _ in range(len(questions_challenge))]

    dataset = [{
            'question': question,
            'prompt': None,
            'first_response': None,
            'category': id,
            'correct_answer': convert_array_to_str(solution)
        } for question, solution, id in zip(questions, solutions, ids)
    ]
 
    return dataset
