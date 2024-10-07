from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
import torch


#model_name = "meta-llama/Llama-2-7b-hf"#"huggyllama/llama-7b"

model_name = "meta-llama/llama-2-7b-chat-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)


def generate_response(prompt, max_length = 1000):

    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=max_length,
            num_return_sequences=1,  
            no_repeat_ngram_size=2,   
            top_p= 0.8, #0.95,               
            do_sample=False #True            
        )


    return tokenizer.decode(outputs[0], skip_special_tokens=True)



while True:
    prompt = input("Input: ")  
    if prompt.lower() == "exit": 
        break
    response = generate_response(prompt)
    print("Response:\n", response)

# prompt = "Tell me about the story of Heung min Son"

# response = generate_response(prompt)
# print("Response:\n", response)

