import json


input_file = "./cook_refined.json"
output_file = "./cook_refined_eos.json"

with open(input_file, "r") as f:
    data = json.load(f)

for entry in data:
    entry["response"] += " </s>"

with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"JSON file updated successfully with </s> added to each response and saved as {output_file}.")