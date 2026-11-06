"""
Batch Generator for 3 Ultra-Realistic Prompts
Outputs saved directly to: C:\\Users\\theay\\OneDrive\\Desktop\\HyperFlux_Galeri
"""

import os
import sys
import time
import torch
import gc
from diffusers import Flux2KleinPipeline
from pt_radiant_physics import RadiantManifoldPhysics

SNAPSHOT_DIR = r"C:\Users\theay\.cache\huggingface\hub\models--black-forest-labs--FLUX.2-klein-4B\snapshots\e7b7dc27f91deacad38e78976d1f2b499d76a294"
OUTPUT_DIR = r"C:\Users\theay\OneDrive\Desktop\HyperFlux_Galeri"

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPTS = [
    {
        "filename": "1_deniz_kaptani_portre.png",
        "title": "Deniz Kaptanı Portresi (BSSRDF Deri Altı Saçılımı & Göz Derinliği)",
        "prompt": (
            "35mm RAW color photo of an intense weathered sea captain at golden hour, "
            "deep textured skin with natural pores and tiny sun scars, warm sunset backlight penetrating ears with organic subsurface scattering (translucent red glow), "
            "wet wool coat with tactile fiber weave, salt-crusted beard, eyes reflecting crashing ocean waves with double corneal specular highlights, "
            "soft atmospheric sea spray, shot on Hasselblad H6D-100c, HC 100mm f/2.2, raytraced directional rim light, zero artificial beauty filter, zero CGI smoothness"
        ),
        "seed": 42
    },
    {
        "filename": "2_tokyo_rtx_yagmur.png",
        "title": "Tokyo Yağmurlu Sokak (RTX Path-Tracing & Islak Yansıma Fiziği)",
        "prompt": (
            "Cinematic night street photography in rainy Tokyo alley, "
            "neon signs casting physically accurate ambient occlusion and chromatic reflections on wet uneven asphalt puddles, "
            "steam rising from manhole with volumetric light dispersion, silhouette of a person under clear umbrella catching specular raindrops, "
            "Fujifilm GFX 100 II, 45mm f/2.8, analog grain, true shadow depth, contact shadows under shoes, zero AI plastic gloss"
        ),
        "seed": 101
    },
    {
        "filename": "3_antika_astrolab_makro.png",
        "title": "Antika Mekanizma Makro (Mikroskobik Malzeme & Işık Kırınımı)",
        "prompt": (
            "Extreme close-up macro shot of an antique brass astrolabe and mechanical pocket watch, "
            "micro-scratches on polished bronze, dust particles illuminated in a single ray of sunlight through dusty library window, "
            "tiny rubies functioning as watch jewels with realistic interior optical refractions, shallow depth of field, "
            "Canon EOS R5 with RF 100mm f/2.8L Macro, hyper-detailed metallic patina, zero AI smudge"
        ),
        "seed": 777
    }
]

def load_pipeline():
    print("=" * 60)
    print("[-] CRB-Radiant Motoru Yukleniyor (VRAM Hedefi: 1.5 - 2.0 GB)...")
    print("=" * 60)
    
    torch.cuda.empty_cache()
    gc.collect()

    pipe = Flux2KleinPipeline.from_pretrained(
        SNAPSHOT_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    pipe.enable_sequential_cpu_offload()
    print("[+] Model Bellege Basariyla Yerlesti (DMA Blok Akisi Devrede)!")
    return pipe

def run_batch():
    pipeline = load_pipeline()
    radiant_physics = RadiantManifoldPhysics(device="cuda", dtype=torch.bfloat16)

    total_start = time.time()
    
    for idx, item in enumerate(PROMPTS, 1):
        print("\n" + "=" * 60)
        print(f"[{idx}/3] URETILIYOR: {item['title']}")
        print(f"[*] Hedef Dosya: {item['filename']}")
        print("=" * 60)
        
        step_count = 12
        generator = torch.Generator(device="cpu").manual_seed(item["seed"])
        
        def radiant_step_callback(pipe_obj, step_index, timestep, callback_kwargs):
            latents = callback_kwargs.get("latents", None)
            if latents is not None:
                progress = step_index / step_count
                if progress > 0.60:
                    latents = radiant_physics.apply_subsurface_scattering(latents, strength=0.15)
                    latents = radiant_physics.apply_path_traced_radiance(latents, light_direction=(0.6, -0.8), ao_strength=0.10)
                    latents = radiant_physics.apply_poisson_skin_microtexture(latents, texture_intensity=0.05)
                    callback_kwargs["latents"] = latents
            return callback_kwargs

        t0 = time.time()
        with torch.inference_mode():
            result = pipeline(
                prompt=item["prompt"],
                num_inference_steps=step_count,
                width=768,
                height=768,
                generator=generator,
                callback_on_step_end=radiant_step_callback
            )
        elapsed = time.time() - t0
        
        out_path = os.path.join(OUTPUT_DIR, item["filename"])
        result.images[0].save(out_path)
        
        vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"[OK] Tamamlandi! Sure: {elapsed:.1f}s | Tepe VRAM: {vram_peak:.1f} MB")
        print(f"[OK] Dosya Konumu: {out_path}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"[TUMU BASARIYLA TAMAMLANDI] Toplam Sure: {total_elapsed:.1f}s")
    print(f"Klasor: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    run_batch()
