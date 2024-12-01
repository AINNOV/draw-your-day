from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## rag ##
import faiss 
import sys
sys.path.append('../')
from utils.retrieval import DiaryRetriever 

model_name = "meta-llama/Llama-2-7b-chat-hf"

# quantization_8bit = BitsAndBytesConfig(load_in_8bit=True)
# model_8bit = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     quantization_config = quantization_8bit,
#     device_map = "auto",
#     low_cpu_mem_usage = True,
# )

quantization_4bit = BitsAndBytesConfig(load_in_4bit=True)
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config = quantization_4bit,
    device_map = "auto",
    low_cpu_mem_usage = True,
)


tokenizer = AutoTokenizer.from_pretrained(model_name)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# prompt = f"""
#         Bellow is my diary. Using this Diary, write a prompt for text-to-image process.
        
#         ### Diary
#         A young student sitting alone in a classroom, feeling frustrated and overwhelmed.
#         The student’s face shows sadness and frustration as they look at their books and notes.
#         Around them, other students are working confidently and smiling, creating a contrast.
#         The environment feels heavy with stress, and there is a pile of assignments on the desk.
#         The overall mood is gloomy, with muted colors and soft lighting to emphasize the emotional weight
        
#         ### Prompt
#         """
# prompt = """
#         Below is a request for answering the recipe for some random food. Fill the Recipe template
         
#         ### Question
#         How do you make javachip frapuccino?

#         ### Recipe

#         """

def prompt_with_template(template_path, prompt, rag):
    with open(template_path, "r") as file:
        sys_template = file.read()
    return [
    {
        "role": "system",
        "content": f"{sys_template}" + rag + "\n\nNow promptize the following diary entry:" # retrieval results added in this position
    },
    {"role": "user", "content": "\n### Input Diary:\n" + prompt + "\n### Promptized Output:\n"},
]


retriever = DiaryRetriever(index_path="./DYD_faiss.bin") 
input_diary_path = '../data/evaluation_diary2.txt'

with open(input_diary_path, "r") as file:
    diary_text = file.read().strip()

## RAG ##
search_results = retriever.search_similar_documents(diary_text, top_k = 1)[0]
rag = f"\n\n### Similar diary: {search_results['prompt']}\n\n### Its promptized output: {search_results['response']}\n"
message = prompt_with_template('../template/template_for_rag.txt', diary_text, rag)
# message = [
#     {
#         "role": "system",
#         "content": "You are Edward Lee who is a best cook(i mean chef) ever in the world",
#     },
#     {"role": "user", "content": "How do you make javachip frapuccino?"},
#  ]
        
inputs = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=False, return_tensors="pt")
# print(tokenizer.batch_decode(inputs))
input_len = len(tokenizer.batch_decode(inputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])


start_event.record()

# inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model_4bit.generate(inputs.to(device)) #.input_ids.to(device))
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

end_event.record()
torch.cuda.synchronize()

inference_time = start_event.elapsed_time(end_event)

llama2_output = outputs[input_len:]

import re

pattern = r"^Cartoon-style.*\.$"
match = re.search(pattern, llama2_output)

if match:

    print("\n🚨Answer🚨", match.group(0) )
    print(f"\ninference time : {(inference_time * 1e-3):.2f} sec")

# result = open("text2prompt_4bit.txt", 'w')
# result.write(outputs)
# result.write(f"\ninference time : {(inference_time * 1e-3):.2f} sec")