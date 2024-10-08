from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Llama-2-7b-chat-hf"

quantization_4bit = BitsAndBytesConfig(load_in_4bit=True)
model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name, 
    quantization_config=quantization_4bit,
    device_map="auto",
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

sd_model_id = "ogkalu/comic-diffusion"  
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


def generate_prompt(diaries):
    prompt_template = f"""
    Below is my diary. Using this Diary, write a prompt for text-to-image process.
    
    ### Diary
    {diaries}
    
    ### Prompt
    """
    inputs = tokenizer(prompt_template, return_tensors="pt")
    generate_ids = model_4bit.generate(inputs.input_ids.to(device))
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return outputs

def generate_images(prompt, num_images=3, prompt_trial = None):
    for i in range(num_images):
        image = pipe(prompt).images[0]
        image.save(f"./results/comic_karras_{prompt_trial}th_prompt_{i}th_image.png")

if __name__ == "__main__":

    count = 0
    iteration = 0
    while True:
        user_input = input("Enter your diary entry (or type 'exit' to quit): ")

        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break


        print("Generating prompt for image creation...")
        generated_prompt = generate_prompt(user_input)
        print(f"Generated Prompt: {generated_prompt}")


        print("Generating images...")
        generate_images(generated_prompt, num_images=3, prompt_trial = count)
        
        print(f"Images saved for iteration {iteration + 1}.")
        iteration += 1
        count += 1