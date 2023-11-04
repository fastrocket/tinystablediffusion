# Load a onnx pipeline firstly.  
from diffusers import OnnxStableDiffusionPipeline
onnx_pipe = OnnxStableDiffusionPipeline.from_pretrained(
    "OFA-Sys/small-stable-diffusion-v0",
    revision="onnx",
    provider="CPUExecutionProvider",
)
# Convert it to OpenVINO pipeline.  
import pipeline_openvino_stable_diffusion
openvino_pipe = pipeline_openvino_stable_diffusion.OpenVINOStableDiffusionPipeline.from_onnx_pipeline(onnx_pipe)

# Generate images.
images = openvino_pipe("an apple, 4k")  