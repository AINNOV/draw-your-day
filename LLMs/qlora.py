import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import json

model_name = "meta-llama/Llama-2-7b-chat-hf"


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_use_double_quant=True  
)

tokenizer = AutoTokenizer.from_pretrained(model_name)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)


with open('./cook_refined.json', 'r', encoding='utf-8') as f:
    data = json.load(f)



dataset = Dataset.from_list(data)
# dataset = load_dataset("json", data_files="path_to_your_dataset.json")["train"]

train_test_split = dataset.train_test_split(test_size=0.2) 
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']


def preprocess_function(examples):
    inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=256)
    # print(f"qlora input: {tokenizer.batch_decode(inputs)}")
    outputs = tokenizer(examples["response"], truncation=True, padding="max_length", max_length=256)
    # print(f"qlora output: {tokenizer.batch_decode(outputs)}")
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)


lora_config = LoraConfig(
    r=16,         
    lora_alpha=32,  
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"], 
)

model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir="./qlora_results",
    evaluation_strategy="steps",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=20,
    logging_steps=10,
    save_steps=100,
    fp16=True,  
)

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,  
)


trainer.train()

# Save the model
trainer.save_model("./qlora_model_cook_refined_20")
# trainer.mode.save_config("./fine_tuned_model/config.json")