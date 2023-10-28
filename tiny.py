import torch
from diffusers import StableDiffusionPipeline

if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")

model_id = "OFA-Sys/small-stable-diffusion-v0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "an apple, 4k"
image = pipe(prompt).images[0]  
    
image.save("apple.png")