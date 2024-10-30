from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import os

# 모델 로드 부분
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Llama-2-7b-chat-hf"
quantization_4bit = BitsAndBytesConfig(load_in_4bit=True)
model_4bit = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_4bit, device_map="auto", low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
sd_model_id = "ogkalu/comic-diffusion"
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# FastAPI 앱 생성
app = FastAPI()

# Request 데이터 모델
class DiaryEntry(BaseModel):
    text: str

# Prompt 생성 함수
def generate_prompt(diaries):
    prompt_template = f"""
    Below is my diary. Using this Diary, write a prompt for text-to-image process. The token length should be less then 77.
    
    ### Diary
    {diaries}
    
    ### Prompt
    """
    inputs = tokenizer(prompt_template, return_tensors="pt")
    generate_ids = model_4bit.generate(inputs.input_ids.to(device))
    outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return outputs

# 이미지 생성 함수
def generate_images(prompt, num_images=1, prompt_trial=0):
    image_paths = []
    os.makedirs("./results", exist_ok=True)
    for i in range(num_images):
        image = pipe(prompt).images[0]
        image_path = f"./results/comic_karras_{prompt_trial}th_prompt_{i}th_image.png"
        image.save(image_path)
        image_paths.append(image_path)
    return image_paths

# 이미지 생성 API 엔드포인트
@app.post("/generate")
async def generate_images_endpoint(diary: DiaryEntry):
    prompt = generate_prompt(diary.text)
    images = generate_images(prompt, num_images=1)
    return {"prompt": prompt, "images": images}
