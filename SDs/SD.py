from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



sd_model_id = "ogkalu/comic-diffusion"  #"runwayml/stable-diffusion-v1-5"#"ogkalu/comic-diffusion"  
pipe = StableDiffusionPipeline.from_pretrained(sd_model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)



def generate_images(prompt, num_images=3, prompt_trial = None):
    for i in range(num_images):
        image = pipe(prompt = prompt, negative_prompt = "disfigured, deformed, ugly, blurry, low resolution, poorly drawn, unnatural, bad anatomy, blurred faces, low-fidelity").images[0]
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
        #generated_prompt = "a young champion with a gold medal, celebrating victory with a huge smile, cinematic lighting, proud parents in the background, joyful and triumphant atmosphere, cartoon, painting-like, human-drawn painting-like" #, by greg rutkowski and thomas kinkade, trending on artstation. "#"a scene showing a person sitting on a cozy couch in a dimly lit room at night, sipping a cup of tea, by greg rutkowski and thomas kinkade, trending on artstation."#"a young person celebrating a big victory, standing proudly on stage with a gold medal around their neck, surrounded by applause and cheering crowd, radiant joy on their face, with vibrant, cinematic lighting, a sense of triumph and achievement, ultra-detailed, 4K resolution. In the background, proud parents smiling with joy and admiration, capturing a heartfelt moment of success and family pride,  a fantasy digital painting by Greg Rutkowski and James Gurney, trending on Artstation, highly detailed"
        #generated_prompt = "A confident student holding a gold medal, smiling proudly after winning the international mathematics competition. Celebratory atmosphere with cheering parents, confetti falling, capturing the joy, pride, and empowerment of a major achievement."
        #generated_prompt = "a confident student holding a gold medal, smiling, digital art by ruan jia and mandy jurgens and artgerm"
        #generated_prompt = "young person, cozy room, soft light, calm, inner peace, gentle ambition, nostalgia, determination, quiet reflection, inspired, distant friend, peaceful atmosphere, warmth, introspective, Deviant Art, "
        generated_prompt = "much details," + "Cartoon-style, two people standing apart, facing each other but distant, feeling the tension in the air, empty space between them, silent communication, one person holding a phone, looking hesitant, the other looking down, their faces sad, room divided by a shadow, dim lighting."
        ##generated_prompt = "a delicate apple made of opal hung on branch in the early morning light, adorned with glistening dewdrops. in the background beautiful valleys, divine iridescent glowing, opalescent textures, volumetric light, ethereal, sparkling, light inside body, bioluminescence, studio photo, highly detailed, sharp focus, photorealism, photorealism, 8k, best quality, ultra detail:1. 2, hyper detail, hdr, hyper detail, ((universe of stars inside the apple) )"
        print("Generating images...")
        generate_images(generated_prompt, num_images=3, prompt_trial = count)
        
        print(f"Images saved for iteration {iteration + 1}.")
        iteration += 1
        count += 1