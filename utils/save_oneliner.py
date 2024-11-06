import json
import pandas as pd
from datasets import Dataset

indataset = []
with open("../data/raw/SYC_latest.json") as f:
    data = json.load(f)  # Load the entire JSON array

    for line in data:  # Iterate over each dictionary in the array
        #indataset.append(f'<s>[INST] <<SYS>\nYou are Edward Lee who is a best cook(i mean chef) ever in the world\n<</SYS>>{line["prompt"]} [/INST] {line["response"]} </s>')
        indataset.append(f'<s>[INST] <<SYS>\nYou are \'Secure Your Contract\' that is an AI based system that takes a contract as an input, detects risky and weakly risky keywords/phrases (if existing) and provide the reasons and refinement suggestion/solution for the input.\n<</SYS>>{line["prompt"]} [/INST] {line["response"]} </s>')
        
        # Uncomment below if you want the alternative format
        # indataset.append(f'<s>### Instruction: \n{line["prompt"]} \n\n### Response: \n{line["response"]}</s>')

# # Dataset verification
# print('Dataset Preview:')
# print(indataset[:1])

# Create and save dataset
indataset = Dataset.from_dict({"text": indataset})
indataset.save_to_disk("../data/dump/SYC_1107_hub.json")

# Print dataset info
print('Dataset Info:')
print(indataset)

indataset.push_to_hub("JJuny/llama2_SYC_1107")