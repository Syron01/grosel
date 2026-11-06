import gc
import json
import os
import time

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)

from web_app.fp8_layers import convert_to_fp8_native

MODEL_ID = "Qwen/Qwen-Image-2.1"
PROMPT = ("anime illustration, two demon maid girls with bat wings in a candlelit bedroom, "
          "warm soft light, detailed faces and hands, clean linework")
WIDTH, HEIGHT = 1280, 704
STEPS = 8
SEED = 42


def mem(label):
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"[mem] {label}: cuda_free={free:.2f}GB cuda_alloc={alloc:.2f}GB", flush=True)


print("loading transformer BF16 ...", flush=True)
t0 = time.time()
tf = QwenImage21Transformer2DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
    local_files_only=True, low_cpu_mem_usage=True,
)
print(f"transformer loaded {time.time()-t0:.1f}s", flush=True)

t0 = time.time()
n_tf = convert_to_fp8_native(tf)
gc.collect()
print(f"transformer FP8 layers={n_tf} in {time.time()-t0:.1f}s", flush=True)
mem("after transformer fp8")

print("loading text encoder 4-bit ...", flush=True)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
t0 = time.time()
te = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, subfolder="text_encoder", quantization_config=bnb,
    torch_dtype=torch.bfloat16, device_map="cuda",
    local_files_only=True, low_cpu_mem_usage=True,
)
print(f"text encoder loaded {time.time()-t0:.1f}s", flush=True)
mem("after text encoder")

print("loading processor/vae/scheduler ...", flush=True)
snap = snapshot_download(MODEL_ID, local_files_only=True)
processor = Qwen3VLProcessor.from_pretrained(os.path.join(snap, "processor"), local_files_only=True)
vae = AutoencoderKLQwenImage21.from_pretrained(
    MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True,
)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    MODEL_ID, subfolder="scheduler", local_files_only=True,
)
pipe = QwenImage21Pipeline(
    vae=vae, text_encoder=te, processor=processor, transformer=tf, scheduler=scheduler,
)
vae.enable_tiling()

print("encoding prompt ...", flush=True)
t0 = time.time()
with torch.inference_mode():
    emb, mask, image_mask = pipe.encode_prompt(
        prompt=PROMPT, image=None, device=te.device, num_images_per_prompt=1,
    )
print(f"encoded {time.time()-t0:.1f}s emb={tuple(emb.shape)} mask={None if mask is None else tuple(mask.shape)}", flush=True)
mem("after encode")

# Text encoder is no longer needed for this call; move it off the GPU.
try:
    te.to("cpu")
    torch.cuda.empty_cache()
    print("text encoder moved to CPU", flush=True)
except Exception as e:
    print("text encoder CPU move failed:", repr(e), flush=True)
    pipe.text_encoder = None
    del te
    gc.collect()
    torch.cuda.empty_cache()
    print("text encoder reference dropped", flush=True)

vae.to("cuda")
mem("before denoise")

print("enabling transformer group offload ...", flush=True)
tf.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=2,
    use_stream=False,
)
print("offload enabled", flush=True)

for run_steps in (20, 40):
    torch.cuda.reset_peak_memory_stats()
    print(f"generating {run_steps} steps ...", flush=True)
    t0 = time.time()
    with torch.inference_mode():
        image = pipe(
            prompt_embeds=emb,
            prompt_embeds_mask=mask,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=run_steps,
            generator=torch.Generator("cpu").manual_seed(SEED),
            use_kv_cache=True,
            output_type="pil",
        ).images[0]
    print(
        f"generated {run_steps} in {time.time()-t0:.1f}s "
        f"peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB", flush=True,
    )
    out = os.path.join(os.path.dirname(__file__), f"qwen_base_{run_steps}.png")
    image.save(out)
    print("saved", out, image.size, flush=True)
