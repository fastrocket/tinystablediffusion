  #!pip install git+https://github.com/huggingface/diffusers transformers accelerate -q
from diffusers import StableDiffusionXLPipeline
import torch
import shortuuid

if torch.cuda.is_available():
    print("CUDA is available.")
else:
    print("CUDA is not available.")

pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
prompt = "A cute kitten in mug ,flowers around, watercolor style" # Your prompt here
neg_prompt = "ugly, blurry, poor quality" # Negative prompt here
image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]

newuuid = shortuuid.uuid()
newfile = f"{newuuid}.png"
image.save(newfile)

