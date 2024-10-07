from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_id ="ogkalu/comic-diffusion"#"runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

for i in range(10):
    #prompt = "A young student sitting alone in a classroom, feeling frustrated and overwhelmed. The student’s face shows sadness and frustration as they look at their books and notes. Around them, other students are working confidently and smiling, creating a contrast. The environment feels heavy with stress, and there is a pile of assignments on the desk. The overall mood is gloomy, with muted colors and soft lighting to emphasize the emotional weight"
    #prompt2 = "A sad and frustrated student sitting alone at a desk in a gloomy classroom. The student looks at books and notes with a stressed expression. Other students in the background are smiling and working confidently, creating a contrast. The desk is cluttered with a pile of assignments. The scene is lit softly with muted colors, enhancing the heavy and stressful atmosphere."
    prompt_llm = """        Create an image that represents the feeling of being overwhelmed by schoolwork.
        The image should convey a sense of isolation, sadness, and frustration.
        Consider using muted colors and soft lighting to emphasize the emotional weight.
        Think about the contrast between the student��s face and the confident smiles of their peers.
        How can you use visual elements to convey the feeling of being lost in a sea of assignments?
        What are some ways to show the emotional toll of schoolwork on a young person?"""
    image = pipe(prompt_llm).images[0]
    image.save(f"comic_karras_prompt_llm_output_fp16_{i}.png")