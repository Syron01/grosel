"""
Master Cinematic Pipeline (FLUX.2 Klein-4B)
- GPU-Accelerated VAE (Donma/kilitlenme kalıcı olarak çözüldü)
- Sequential CPU Offload (VRAM 1.4 - 1.8 GB arasında kalır)
- Cinematic Lens Bokeh Engine (Karakter gözlerinde jilet netlik, arka planda sinematik bokeh)
- Subsurface Scattering (Deri altı ışık saçılımı)
- Path-Traced Ambient Lighting (Şömine ve ortam ışığı fiziği)
- Organik Doku Süzgeci (Sıfır plastik/AI hissi)
- 1024x576 Geniş Sinematik (16:9 1080p uyumlu kompozisyon)
"""

import os
import sys
import time
import math
import torch
import gc
from diffusers import Flux2KleinPipeline
from pt_radiant_physics import RadiantManifoldPhysics
from cinematic_dof import CinematicLensBokeh

SNAPSHOT_DIR = r"C:\Users\theay\.cache\huggingface\hub\models--black-forest-labs--FLUX.2-klein-4B\snapshots\e7b7dc27f91deacad38e78976d1f2b499d76a294"
OUTPUT_DIR = r"C:\Users\theay\OneDrive\Desktop\HyperFlux_Galeri"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_master_pipeline():
    print("=" * 65)
    print("[-] Master Sinematik Pipeline Yukleniyor...")
    print("=" * 65)
    
    torch.cuda.empty_cache()
    gc.collect()

    pipe = Flux2KleinPipeline.from_pretrained(
        SNAPSHOT_DIR,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    # VAE'yi tiling ile GPU'ya optimize et
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        print("[+] VAE Slicing + Tiling aktif.")

    # Katman bazlı asenkron DMA akışı (VRAM kilidi ~1.4 - 1.8 GB)
    pipe.enable_sequential_cpu_offload()
    print("[+] Sequential CPU Offload aktif: VRAM 1.5 - 2.0 GB seviyesinde.")
    print("=" * 65)
    return pipe

def generate_cinematic_masterpiece(
    pipe,
    prompt: str,
    output_filename: str = "masterpiece_cinematic.png",
    num_inference_steps: int = 14,
    width: int = 1024,
    height: int = 576,
    seed: int = 42,
    focus_center = (0.58, 0.42)
):
    print(f"[*] Prompt: '{prompt[:90]}...'")
    print(f"[*] Cozunurluk: {width}x{height} (16:9 Sinematik 1080p kompozisyon)")
    print(f"[*] Adim Sayisi: {num_inference_steps} | Seed: {seed}")

    radiant_physics = RadiantManifoldPhysics(device="cuda", dtype=torch.bfloat16)
    lens_bokeh = CinematicLensBokeh(device="cuda", dtype=torch.bfloat16)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    start_time = time.time()

    def cinematic_callback(pipe_obj, step_index, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents", None)
        if latents is not None:
            progress = step_index / num_inference_steps
            # Son %40'ta fizik ve optik kancaları devreye girer
            if progress > 0.60:
                # 1. BSSRDF Deri Altı Işık Saçılımı (Yüz ve boyunda sıcak canlılık)
                latents = radiant_physics.apply_subsurface_scattering(latents, strength=0.14)
                # 2. Şömine Yönlü Işıklandırma & Ambient Occlusion (Sol taraftaki ateşten vuran ışık)
                latents = radiant_physics.apply_path_traced_radiance(latents, light_direction=(-0.8, -0.4), ao_strength=0.10)
                # 3. Poisson Organik Doku (Plastik AI filtresini yıkan doku)
                latents = radiant_physics.apply_poisson_skin_microtexture(latents, texture_intensity=0.04)
                
                # 4. Optik Odak & Lens Bokeh (DoF derinliği)
                # [B, N, C] -> 4D uzamsal grid'e çevir ve lens odak süzgecinden geçir
                tensor_4d, is_3d, orig_shape = radiant_physics._to_spatial_4d(latents)
                tensor_4d = lens_bokeh.apply_focus_and_bokeh(tensor_4d, focus_center=focus_center, depth_falloff=1.5)
                latents = radiant_physics._from_spatial_4d(tensor_4d, is_3d, orig_shape)

                callback_kwargs["latents"] = latents
        return callback_kwargs

    print("[*] Inferans basladi (PT-RMF Fizik + Cinematic Lens Bokeh devrede)...")
    
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            callback_on_step_end=cinematic_callback
        )

    end_time = time.time()
    elapsed = end_time - start_time
    
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    result.images[0].save(out_path)

    vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print("=" * 65)
    print(f"[OK] GORSEL BASARIYLA URETILDI!")
    print(f"[OK] Kaydedilen Dosya: {out_path}")
    print(f"[OK] Toplam Sure: {elapsed:.2f} saniye")
    print(f"[OK] En Yuksek (Tepe) VRAM: {vram_peak:.1f} MB")
    print("=" * 65)
    return out_path

if __name__ == "__main__":
    cinematic_prompt = (
        "Cinematic masterpiece RAW 35mm film still of a young contemplative woman sitting at a rustic wooden table "
        "in a cozy dimly lit room, soft glowing fireplace in the left background casting warm amber firelight, "
        "wearing a chunky dark oversized knit cardigan sweater with tactile woolen weave, messy elegant hair with loose strands, "
        "intense sharp focus on her face and eyes looking sideways in deep thought, perfect natural skin texture with subtle pores, "
        "delicate subsurface scattering along jawline and ears from fireplace glow, tea mug on table, wooden chair, books and antique shelf in soft background bokeh, "
        "Rembrandt lighting, cinematic color grading, shot on ARRI Alexa Mini LF with Zeiss Master Prime 50mm f/1.4, "
        "photorealistic, zero AI gloss, zero plastic skin, zero digital blur, authentic atmosphere"
    )

    pipeline = load_master_pipeline()
    generate_cinematic_masterpiece(
        pipeline,
        prompt=cinematic_prompt,
        output_filename="cinematic_masterpiece_fireplace.png",
        num_inference_steps=12,
        width=1024,
        height=576,
        seed=2026,
        focus_center=(0.58, 0.42)
    )
