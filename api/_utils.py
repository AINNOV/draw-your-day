import random
import torch
import numpy as np
from transformers import set_seed


def prompt_with_template(prompt):
    return [
    {
        "role": "system",
        "content": "You are \'Diary2Prompt\' that is an promptizer that 1. takes a raw user diary as an input, 2. converts it to a proper prompt for Stable Diffusion 1.5 (finetuned with comic images). Note 1. the output does not exceed 60 tokens, 2. rather than participial constructions, a focus on word listing  3. ONLY RETURN THE PROMPTISED OUTPPUT",
    },
    {"role": "user", "content": "\n[Input Diary]:\n" + prompt + "\nOutput Prompt(promptised)]:\n"},
]

def set_all_seeds(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)