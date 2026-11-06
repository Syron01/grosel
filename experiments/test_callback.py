import torch
import time
from diffusers import Flux2KleinPipeline
from test_preview_gen import latents_to_preview_b64

SNAPSHOT_DIR = r"C:\Users\theay\.cache\huggingface\hub\models--black-forest-labs--FLUX.2-klein-4B\snapshots\e7b7dc27f91deacad38e78976d1f2b499d76a294"

pipe = Flux2KleinPipeline.from_pretrained(
    SNAPSHOT_DIR,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
pipe.enable_sequential_cpu_offload()

def step_callback(pipe_obj, step, timestep, callback_kwargs):
    latents = callback_kwargs.get("latents")
    b64 = latents_to_preview_b64(latents, 512, 512, pipe_obj)
    print(f"Step {step} | b64 preview length: {len(b64)} | starts with: {b64[:30]}")
    return callback_kwargs

with torch.inference_mode():
    res = pipe(
        prompt="A cozy coffee mug on a wooden table in dim light",
        num_inference_steps=2,
        width=512,
        height=512,
        callback_on_step_end=step_callback,
        callback_on_step_end_tensor_inputs=["latents"]
    )
print("Preview test succeeded!")
