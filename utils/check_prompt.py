
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import get_peft_model, PeftModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load your fine-tuned model instead of the original one
fine_tuned_model_path = "meta-llama/Llama-2-7b-chat-hf"  # cannot use template if not chat version

# Load quantized model
quantization_4bit = BitsAndBytesConfig(load_in_4bit=True)
model_4bit = AutoModelForCausalLM.from_pretrained(
    fine_tuned_model_path,  # Use your fine-tuned model path here
    quantization_config=quantization_4bit,
    device_map="auto",
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)  # Load the tokenizer from the same path
message = [
    {
        "role": "system",
        "content": "You are Edward Lee who is a best cook(i mean chef) ever in the world",
    },
    {"role": "user", "content": "How do you make javachip frapuccino?"},
 ]
        
inputs = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=False, return_tensors="pt")
inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(inputs)