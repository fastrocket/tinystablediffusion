import torch
from diffusers import StableDiffusionXLPipeline
import shortuuid
import os
from gfpgan_module import process_images



if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")

# model_id = "OFA-Sys/small-stable-diffusion-v0"
model_id = "segmind/SSD-1B"
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "segmind/SSD-1B",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
pipe = pipe.to("cuda")

prompt="""
realistic photo of a beautiful girl smiling, perfect teeth, 35mm, professional headshot, high quality"""
neg_prompt = "ugly, blurry, cartoon, poor quality" # Negative prompt here

# Loop the following 10 times
for i in range(1):
    # Generate an image
    image = pipe(prompt=prompt, negative_prompt=neg_prompt, num_inference_steps=20).images[0]  
        
    # Create the output directory if it does not exist
    output_dir = "out"
    face_dir = "face"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(face_dir):
        os.makedirs(face_dir)

    # Generate a unique UUID for the filename
    newuuid = shortuuid.uuid()

    # Save the image to a file in the output directory
    newfile = os.path.join(output_dir, f"{newuuid}.png")
    output_path = os.path.join(face_dir, f"{newuuid}-gfpgan.png")
    image.save(newfile)

    # Print a message to the user
    print(f"Image saved to {newfile}")
    
    input_path = newfile
    output_path = output_path

process_images(input_folder='out', output_folder='face', version='1.4')