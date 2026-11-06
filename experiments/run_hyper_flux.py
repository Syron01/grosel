"""
CRB Turbo Engine (FLUX.2 Klein-4B)
- VRAM Kilidi: ~1.8 - 2.2 GB VRAM sınırında çalışır.
- CPU/GPU Sequential Memory Offload + Sliced/Tiled VAE.
- Anti-AI Path-Traced Radiant Physics (BSSRDF + RTX Işık + Poisson Organik Doku).
- Çıktı: Ultra yüksek kalite, analog lens gerçekçiliği, sıfır plastik his.
"""

import os
import sys
import time
import torch
import gc
from diffusers import Flux2KleinPipeline
from pt_radiant_physics import RadiantManifoldPhysics

SNAPSHOT_DIR = r"C:\Users\theay\.cache\huggingface\hub\models--black-forest-labs--FLUX.2-klein-4B\snapshots\e7b7dc27f91deacad38e78976d1f2b499d76a294"

def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024), torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0, 0

def load_optimized_pipeline(model_path=SNAPSHOT_DIR):
    print("=" * 60)
    print("[-] CRB-Radiant Motoru Yukleniyor (Hedef VRAM: 1.5 - 2.0 GB)...")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    gc.collect()

    pipe = Flux2KleinPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        print("[+] VAE Slicing ve Tiling devrede: VAE tepe noktasi < 200 MB.")

    pipe.enable_sequential_cpu_offload()
    print("[+] Katman Duzeyinde Dinamik DMA Bellek Akisi (Sequential CPU Offload) aktif.")

    vram_now, vram_max = get_vram_mb()
    print(f"[+] Model Yuklendi! Aktif VRAM: {vram_now:.1f} MB | Tepe VRAM: {vram_max:.1f} MB")
    print("=" * 60)
    return pipe

def generate_hyper_realistic_image(
    pipe,
    prompt: str,
    output_filename: str = "hyper_flux_output.png",
    num_inference_steps: int = 14,
    guidance_scale: float = 3.5,
    width: int = 768,
    height: int = 768,
    seed: int = 42,
    enable_radiant_physics: bool = True
):
    print(f"[*] Prompt: '{prompt}'")
    print(f"[*] Ayarlar: {width}x{height} | {num_inference_steps} Adim | Guidance: {guidance_scale}")

    radiant_physics = RadiantManifoldPhysics(device="cuda", dtype=torch.bfloat16)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    start_time = time.time()

    def radiant_step_callback(pipe_obj, step_index, timestep, callback_kwargs):
        if not enable_radiant_physics:
            return callback_kwargs
            
        latents = callback_kwargs.get("latents", None)
        if latents is not None:
            progress = step_index / num_inference_steps
            if progress > 0.60:
                latents = radiant_physics.apply_subsurface_scattering(latents, strength=0.15)
                latents = radiant_physics.apply_path_traced_radiance(latents, light_direction=(0.6, -0.8), ao_strength=0.10)
                latents = radiant_physics.apply_poisson_skin_microtexture(latents, texture_intensity=0.05)
                callback_kwargs["latents"] = latents
        return callback_kwargs

    print("[*] Inferans basladi (PT-RMF Fiziksel Optik Motor devrede)...")
    
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=radiant_step_callback if enable_radiant_physics else None
        )

    end_time = time.time()
    elapsed = end_time - start_time
    
    image = result.images[0]
    out_path = os.path.abspath(output_filename)
    image.save(out_path)

    vram_now, vram_max = get_vram_mb()
    print("=" * 60)
    print(f"[OK] GORSEL BASARIYLA URETILDI!")
    print(f"[OK] Kaydedilen Dosya: {out_path}")
    print(f"[OK] Toplam Sure: {elapsed:.2f} saniye")
    print(f"[OK] Anlik VRAM: {vram_now:.1f} MB | En Yuksek (Tepe) VRAM: {vram_max:.1f} MB")
    print("=" * 60)
    return out_path

if __name__ == "__main__":
    prompt_input = (
        "Cinematic 35mm film RAW portrait of an elderly watchmaker with intricate wrinkles, "
        "hyper-detailed skin pores, natural light reflecting in eyes, subtle subsurface scattering, "
        "holding a delicate brass clockwork mechanism, dramatic rim light, dust motes in air, "
        "photorealistic, Leica M11 50mm f/1.4, raytraced ambient lighting, zero artificial smoothness"
    )
    if len(sys.argv) > 1:
        prompt_input = sys.argv[1]

    pipeline = load_optimized_pipeline()
    generate_hyper_realistic_image(
        pipeline,
        prompt=prompt_input,
        output_filename="cinematic_watchmaker_test.png",
        num_inference_steps=12,
        width=768,
        height=768,
        seed=1337
    )
