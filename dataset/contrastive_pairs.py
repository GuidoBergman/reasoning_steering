def get_contrastive_pairs(train):
    succesful_jailbreaks = []
    refusals = []

    for x in train:
        succesful_jailbreaks.append({
            'question': x['question'],
            'prompt': x['prompt'],
            'first_response': x['first_response']
        })

        # The refusal is the same, but asking the forbidden question directly
        refusals.append({
            'question': x['question'],
            'prompt': None,
            'first_response': None
        })

    return succesful_jailbreaks, refusals