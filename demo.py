from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

prompt = f"""
        Bellow is my diary. Using this Diary, write a prompt for text-to-image process.
        
        ### Diary
        A young student sitting alone in a classroom, feeling frustrated and overwhelmed.
        The student’s face shows sadness and frustration as they look at their books and notes.
        Around them, other students are working confidently and smiling, creating a contrast.
        The environment feels heavy with stress, and there is a pile of assignments on the desk.
        The overall mood is gloomy, with muted colors and soft lighting to emphasize the emotional weight
        
        ### Prompt
        """
        
start_event.record()

inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model_4bit.generate(inputs.input_ids.to(device))
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

end_event.record()
torch.cuda.synchronize()

inference_time = start_event.elapsed_time(end_event)

print("\n", outputs)
print(f"\ninference time : {(inference_time * 1e-3):.2f} sec")

result = open("text2prompt_4bit.txt", 'w')
result.write(outputs)
result.write(f"\ninference time : {(inference_time * 1e-3):.2f} sec")

## SD starts here
model_id ="ogkalu/comic-diffusion"#"runwayml/stable-diffusion-v1-5" 
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config) # karras solver is the best

for i in range(3):
    #prompt = "A young student sitting alone in a classroom, feeling frustrated and overwhelmed. The student’s face shows sadness and frustration as they look at their books and notes. Around them, other students are working confidently and smiling, creating a contrast. The environment feels heavy with stress, and there is a pile of assignments on the desk. The overall mood is gloomy, with muted colors and soft lighting to emphasize the emotional weight"
    #prompt2 = "A sad and frustrated student sitting alone at a desk in a gloomy classroom. The student looks at books and notes with a stressed expression. Other students in the background are smiling and working confidently, creating a contrast. The desk is cluttered with a pile of assignments. The scene is lit softly with muted colors, enhancing the heavy and stressful atmosphere."
    # prompt_llm = """        Create an image that represents the feeling of being overwhelmed by schoolwork.
    #     The image should convey a sense of isolation, sadness, and frustration.
    #     Consider using muted colors and soft lighting to emphasize the emotional weight.
    #     Think about the contrast between the student��s face and the confident smiles of their peers.
    #     How can you use visual elements to convey the feeling of being lost in a sea of assignments?
    #     What are some ways to show the emotional toll of schoolwork on a young person?"""
    image = pipe(outputs).images[0]
    image.save(f"./results/comic_karras_{i}.png")