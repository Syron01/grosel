"""
Faz bazli profil: text encoder / her denoise adimi / VAE decode nerede vakit yiyor?
Kullanim: python bench_profile.py [offload|resident]
"""
import gc
import json
import os
import sys
import time

import torch
from diffusers import Flux2KleinPipeline

MODEL_ID = os.environ.get("HYPERFLUX_FLUX_MODEL", "black-forest-labs/FLUX.2-klein-4B")

PROMPT = (
    "Atmospheric fine art painting, moody low-key lighting. Waist-up composition of a young woman "
    "sitting in a dark rustic wooden armchair, three-quarter profile, loose dark hair bun, oversized "
    "brown wool cardigan. A low-burning hearth fireplace casts warm rim light on her cheek. "
    "Delicate chiaroscuro, painterly texture, matte finish, contemplative atmosphere."
)
W, H, STEPS, SEED = 1024, 576, 4, 77


def sync():
    torch.cuda.synchronize()


def vram():
    return torch.cuda.memory_allocated() / 1024**2, torch.cuda.max_memory_allocated() / 1024**2


def main(mode):
    torch.cuda.empty_cache(); gc.collect()
    t_load = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.vae.enable_slicing(); pipe.vae.enable_tiling()

    if mode == "offload":
        pipe.enable_sequential_cpu_offload()
    else:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from web_app.fp8_layers import convert_to_fp8
        n = convert_to_fp8(pipe.transformer)
        print(f"[fp8] {n} Linear katman FP8'e cevrildi")
        pipe.transformer.to("cuda")
        pipe.text_encoder.to("cpu")
        pipe.vae.to("cuda")
    print(f"[load] {time.time() - t_load:.1f}s | vram alloc/peak: {vram()}")

    steps_t = []

    def cb(p, i, t, kw):
        steps_t.append(time.time())
        return kw

    # warmup (ayri seed, olcume girmiyor)
    with torch.inference_mode():
        pipe(prompt="warmup", num_inference_steps=STEPS, width=W, height=H)

    for run in range(2):
        steps_t.clear()
        t0 = time.time()
        with torch.inference_mode():
            pipe(prompt=PROMPT, num_inference_steps=STEPS, width=W, height=H,
                 generator=torch.Generator("cpu").manual_seed(SEED),
                 callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
        sync()
        total = time.time() - t0
        # ilk adim zamani = text encode + ilk transformer gecisi
        first = steps_t[0] - t0
        later = [steps_t[i] - steps_t[i - 1] for i in range(1, len(steps_t))]
        decode = total - (steps_t[-1] - t0)
        alloc, peak = vram()
        print(json.dumps({
            "mode": mode, "run": run, "total_s": round(total, 2),
            "text_encode_plus_step0_s": round(first, 2),
            "step_s": [round(x, 3) for x in later],
            "vae_decode_s": round(decode, 2),
            "vram_alloc_mb": round(alloc, 1), "vram_peak_mb": round(peak, 1),
        }))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "offload")
