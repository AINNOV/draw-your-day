import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig
import json

model_name = "meta-llama/Llama-2-7b-chat-hf"

# BitsAndBytes config for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
)

with open('./cook.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Preprocessing function to tokenize prompts and responses
def preprocess_function(examples):
    inputs = tokenizer(examples["prompt"], truncation=True, padding="max_length", max_length=256)
    print(f"qlora input: {inputs}")
    outputs = tokenizer(examples["response"], truncation=True, padding="max_length", max_length=256)
    print(f"qlora output: {outputs}")
    inputs["labels"] = outputs["input_ids"]
    return inputs

# Tokenize the dataset
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# LoRA configuration
lora_config = LoraConfig(
    r=16,         
    lora_alpha=32,  
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # LoRA applied to attention layers
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# TrainingArguments setup
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=True,  
)

# Trainer setup with both train and eval datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,  
)

# Train the model with LoRA adapters
trainer.train()

# Save the LoRA fine-tuned model
trainer.save_model("./lora_model")

# Optional: Save the tokenizer and training arguments
tokenizer.save_pretrained("./lora_model")