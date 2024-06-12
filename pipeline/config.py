
import os

from dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    model_alias: str
    model_path: str
    n_successful_test: int = 4
    jailbreak_prompts_dataset = 'jailbreak_prompts'
    harmless_dataset = 'alpaca'
    max_new_tokens: int = 512
    train_size_successful: int = 256
    test_size_unsuccessful: int = 100
    test_size_harmless: int = 32
    jailbreak_eval_methodologies: Tuple[str] = ("refusal_substring", "harmful_substring", "llamaguard2", "harmbench")
    refusal_eval_methodologies: Tuple[str] = ("refusal_substring",)
    layer: int = 14
    pos: int = -1


    def artifact_path(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "runs", self.model_alias)