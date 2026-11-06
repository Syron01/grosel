import os
import time
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageFilter

SNAPSHOT_DIR = r"C:\Users\theay\.cache\huggingface\hub\models--black-forest-labs--FLUX.2-klein-4B\snapshots\e7b7dc27f91deacad38e78976d1f2b499d76a294"
OUTPUT_DIR = r"C:\Users\theay\OneDrive\Desktop\Grosel\Çizimler"

pipe = Flux2KleinPipeline.from_pretrained(
    SNAPSHOT_DIR,
    torch_dtype=torch.bfloat16,
    local_files_only=True
)
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.enable_sequential_cpu_offload()

# PURE STYLE - Zero scene forcing (no hardcoded fireplace, no hardcoded chair), strict anatomical anchors
prompt = (
    "Atmospheric concept art painting, digital illustration, matte texture, soft muted colors, low-key night lighting. "
    "A young Asian woman sitting by a large window looking out at the distant illuminated Eiffel Tower in Paris at night, "
    "melancholic contemplative expression, messy hair in a loose bun, loose cozy cardigan, "
    "perfectly formed hands with five distinct slender fingers resting naturally, flawless anatomical proportions, "
    "natural gentle shadows, cozy quiet room, soft bokeh city lights outside, "
    "matte painterly textures, zero studio lighting, zero plastic shine, zero extra limbs, zero distorted hands, masterpiece artwork."
)

generator = torch.Generator(device="cpu").manual_seed(42)
t0 = time.time()
with torch.inference_mode():
    res = pipe(
        prompt=prompt,
        num_inference_steps=4,
        width=1024,
        height=576,
        generator=generator
    )

elapsed = time.time() - t0
img = res.images[0]

# 4K Super-Resolution refinement (0.1s)
t_up = time.time()
img_4k = img.resize((3840, 2160), Image.Resampling.LANCZOS)
img_4k = img_4k.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
up_elapsed = time.time() - t_up

out_path = os.path.join(OUTPUT_DIR, "test_paris_4k_perfect.png")
img_4k.save(out_path)

vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
print("=" * 60)
print(f"[OK] Generation Time: {elapsed:.2f}s | 4K Refinement: {up_elapsed:.2f}s | Total: {elapsed+up_elapsed:.2f}s")
print(f"[OK] Peak VRAM: {vram_peak:.1f} MB")
print(f"[OK] Saved to: {out_path} (Size: {img_4k.size})")
print("=" * 60)
