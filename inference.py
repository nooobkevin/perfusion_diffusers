from diffusers import StableDiffusionPipeline
import torch

model_id = "runwayml/stable-diffusion-v1-5"
encoder_path = "./output/tom2/<Tom-face>.bin"
unet_path = "./output/tom2/pytorch_custom_diffusion_weights.bin"

num_inference_steps = 40
guidance_scale = 7.5
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, requires_safety_checker=False).to("cuda")
pipe.load_textual_inversion(encoder_path)
pipe.unet.load_attn_procs(
    unet_path, weight_name="pytorch_custom_diffusion_weights.bin"
)
prompt = "A professional portrait of women."
negative_prompt = "worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting,bad anatomy, bad hands, three hands, three legs, bad arms, missing legs, missing arms, poorly drawn face, bad face, fused face, cloned face, worst face, three crus, extra crus, fused crus, worst feet, three feet, fused feet, fused thigh, three thigh, fused thigh, extra thigh, worst thigh, missing fingers, extra fingers, ugly fingers, long fingers, horn, extra eyes, huge eyes, 2girl, amputation, disconnected limbs, cartoon, cg, 3d, unreal, animate"
image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

image.save(f"result/step{num_inference_steps}_scale{guidance_scale}.png")
