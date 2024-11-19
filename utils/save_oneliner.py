import json
import pandas as pd
from datasets import Dataset

indataset = []
with open("../data/raw/DYD_train.json") as f:
    data = json.load(f)  # Load the entire JSON array

    for line in data:  # Iterate over each dictionary in the array
        #indataset.append(f'<s>[INST] <<SYS>\nYou are Edward Lee who is a best cook(i mean chef) ever in the world\n<</SYS>>{line["prompt"]} [/INST] {line["response"]} </s>')
        indataset.append(f'<s>[INST] <<SYS>\nYou are \'Diary2Prompt\' that is an promptizer that 1. takes a raw user diary as an input, 2. converts it to a proper prompt for Stable Diffusion 1.5 (finetuned with comic images). Note 1. the output does not exceed 60 tokens, 2. rather than participial constructions, a focus on word listing  3. ONLY RETURN THE PROMPTISED OUTPPUT\n<</SYS>>\n[Input Diary]:\n{line["prompt"]} \nOutput Prompt(promptised)]:\n[/INST] {line["response"]} </s>')
        
        # Uncomment below if you want the alternative format
        # indataset.append(f'<s>### Instruction: \n{line["prompt"]} \n\n### Response: \n{line["response"]}</s>')

# # Dataset verification
# print('Dataset Preview:')
# print(indataset[:1])

# Create and save dataset
indataset = Dataset.from_dict({"text": indataset})
indataset.save_to_disk("../data/dump/DYD_1118_train.json")

# Print dataset info
print('Dataset Info:')
print(indataset)

indataset.push_to_hub("JJuny/llama2_DYD_1118_train")