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


# gfpgan

import torch
from gfpgan import GFPGANer

# https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
# Load the model
model = GFPGANer(
    model_path='path/to/model.pth',
    upscale=4,
    channel_multiplier=2,
    pre_pad=True,
    half=True,
    device=torch.device('cuda')
)

# Load an image
img = Image.open('path/to/image.png')

# Restore the image
restored_img = model.enhance(img)

# Save the restored image
restored_img.save('path/to/restored_image.png')
