from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Llama-2-7b-chat-hf"

quantization_8bit = BitsAndBytesConfig(load_in_8bit=True)
model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config = quantization_8bit,
    device_map = "auto",
    low_cpu_mem_usage = True,
)

# quantization_4bit = BitsAndBytesConfig(load_in_4bit=True)
# model_4bit = AutoModelForCausalLM.from_pretrained(
#     model_name, 
#     quantization_config = quantization_4bit,
#     device_map = "auto",
#     low_cpu_mem_usage = True,
# )

tokenizer = AutoTokenizer.from_pretrained(model_name)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

prompt = "write me a short contract for hiring AI designer"

start_event.record()

inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model_8bit.generate(inputs.input_ids.to(device))
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

end_event.record()
torch.cuda.synchronize()

inference_time = start_event.elapsed_time(end_event)

print("\n", outputs)
print(f"\ninference time : {(inference_time * 1e-3):.2f} sec")

result = open("sample_8it.txt", 'w')
result.write(outputs)