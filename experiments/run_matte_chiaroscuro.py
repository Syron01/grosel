"""
Exact Matte Chiaroscuro & Fast Flow Matcher
- Hedef Hız: 4 - 6 Saniye (Hızlı, hafif, optimize akış).
- Işıklandırma Hatası Çözümü: Aşırı parlayan stüdyo ışığı ve cildin yağlı parlaması tamamen yok edildi.
- Mat, karanlık, loş, yarı-saydam ressam fırça dokusu (Matte atmospheric low-key).
- Karakter: Zarif Asyatik melankolik yüz, dağınık dökülen doğal saç telleri, loş şömine ışığı.
"""

import os
import sys
import time
import torch
import gc
from diffusers import Flux2KleinPipeline

SNAPSHOT_DIR = r"C:\Users\theay\.cache\huggingface\hub\models--black-forest-labs--FLUX.2-klein-4B\snapshots\e7b7dc27f91deacad38e78976d1f2b499d76a294"
OUTPUT_DIR = r"C:\Users\theay\OneDrive\Desktop\HyperFlux_Galeri"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class MatteChiaroscuroEngine:
    def __init__(self):
        print("[-] Hizli & Mat Chiaroscuro Motoru Hazirlaniyor...")
        torch.cuda.empty_cache()
        gc.collect()

        self.pipe = Flux2KleinPipeline.from_pretrained(
            SNAPSHOT_DIR,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )

        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()

        self.pipe.enable_sequential_cpu_offload()
        # Warmup: İlk çalışma gecikmesini sıfırlar
        _ = self.pipe(prompt="warmup", num_inference_steps=1, width=1024, height=576)

    def generate(self, prompt, filename, seed=888, steps=4):
        print(f"\n[*] Hizli Uretim ({steps} Adim, ~5 sn) -> Seed: {seed} | Dosya: {filename}")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        t0 = time.time()
        
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=steps,
                width=1024,
                height=576,
                generator=generator
            )
            
        elapsed = time.time() - t0
        out_path = os.path.join(OUTPUT_DIR, filename)
        result.images[0].save(out_path)
        
        vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print("=" * 65)
        print(f"[OK] {filename} URETILDI!")
        print(f"[OK] Sure: {elapsed:.2f} saniye")
        print(f"[OK] VRAM: {vram_peak:.1f} MB")
        print("=" * 65)
        return out_path

if __name__ == "__main__":
    engine = MatteChiaroscuroEngine()
    
    # 2. fotoğraftaki o loş, mat, karanlık, anime-vari zarif Asyatik çizim tonu:
    # "photorealistic", "raytraced" gibi stüdyo ışığı yapan kelimeler TAMAMEN SİLİNDİ.
    # Yerine mat fırça dokusu, loş karanlık oda ve sakin alev tonu eklendi.
    exact_matte_prompt = (
        "Atmospheric concept art painting, digital illustration, matte texture, soft muted colors, low-key lighting. "
        "A delicate young Asian woman with messy dark hair in a loose bun with fine wispy strands falling around her face, "
        "melancholic dark expressive eyes looking sideways in quiet contemplation, soft gentle facial features, matte pale skin. "
        "Seated in a dark wooden chair in a very dark dimly lit room, wearing a loose dark charcoal knitted cardigan over a simple tank top, "
        "arms crossed gently in her lap, calm posture. "
        "In the far background on the left, a very soft blurred hearth fireplace with low warm burning flames casting subtle amber glow. "
        "A dark wooden table in the foreground with a simple ceramic mug. Deep shadowy room, dark bookshelf in the shadows with vintage books, "
        "dark moody chiaroscuro painting, painterly textures, zero studio lighting, zero plastic shine, zero harsh highlights, soft shadows, warm somber cozy atmosphere."
    )

    # 4 adımda tam 4-6 saniyelik üretimler
    engine.generate(exact_matte_prompt, "exact_matte_match_seed888.png", seed=888, steps=4)
    engine.generate(exact_matte_prompt, "exact_matte_match_seed42.png", seed=42, steps=4)
    engine.generate(exact_matte_prompt, "exact_matte_match_seed101.png", seed=101, steps=4)
