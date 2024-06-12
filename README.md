# Avoiding jailbreaks by discouraging their representation in activation space

**Content warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate in nature.

This repository contains code and results accompanying the blog post "Avoiding jailbreaks by discouraging their representation in activation space".

- [Blog post](https://www.lesswrong.com/posts/5XjzmxcZFm3BJrNmn/avoiding-jailbreaks-by-discouraging-their-representation-in-5)

## Setup

```bash
git https://github.com/GuidoBergman/jailbreak_direction.git
cd jailbreak_direction
source setup.sh
```

The setup script will prompt you for a HuggingFace token (required to access gated models) and a Together AI token (required to access the Together AI API, which is used for evaluating jailbreak safety scores).
It will then set up a virtual environment and install the required packages.

## Reproducing main results

To reproduce the main results from the paper, run the following command:

```bash
python3 -m pipeline.run_pipeline --model_path {model_path}
```
where `{model_path}` is the path to a HuggingFace model. For example, for Gemma 2 2B IT, the model path would be `google/gemma-2-2b-it`.

The pipeline performs the following steps:
1. Gather jailbreak prompts and forbidden questions
2. Generate interactions with the baseline model
3. Filter interactions
4. Find the direction representing the jailbreak feature
5. Intervene the model
6. Generate completions
7. Evaluate completions

For convenience, the results are already saved
- [`google/gemma-2-2b-it`](/pipeline/runs/gemma-2-2b-it/)


All the configurations can be adjusted in the [`config`](/pipeline/config.py) file.