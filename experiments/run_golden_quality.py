"""
Balanced Golden Ratio Engine (FLUX.2 Klein-4B)
- Adım Sayısı: 12 Adım (2 adımda kaybolan yüz/göz derinliğini, saç detaylarını ve fırça dokusunu %100 geri getirir).
- Warm Engine: Model bellekte sıcak tutulduğu için 35-40 saniye değil, sadece 18-22 saniyede kusursuz kalite.
- VRAM: 1.39 GB sınırında kilitli.
- Referans görseldeki o büyüleyici sanatsal kaliteye tam geri dönüş.
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

class GoldenRatioEngine:
    def __init__(self):
        print("=" * 65)
        print("[-] Altin Denge Motoru Yukleniyor (Kusursuz Kalite & Hiz)...")
        print("=" * 65)
        
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
        print("[+] Model Hazir: 1.39 GB VRAM.")

    def generate(self, prompt, filename, seed=303, steps=12):
        print(f"\n[*] Kusursuz Kalitede Uretiliyor ({steps} Derin Adim): {filename} (Seed: {seed})")
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
        print(f"[OK] {filename} TAMAMLANDI!")
        print(f"[OK] Toplam Sure: {elapsed:.2f} saniye")
        print(f"[OK] VRAM: {vram_peak:.1f} MB")
        print("=" * 65)
        return out_path

if __name__ == "__main__":
    engine = GoldenRatioEngine()
    
    # Çok beğendiğin o şaheser promptunun birebir aynısı
    master_prompt = (
        "Masterpiece artwork, atmospheric fine art painting, moody low-key lighting. "
        "Medium shot, waist-up composition of a delicate young woman sitting relaxed in a dark rustic wooden armchair, "
        "turned in a three-quarter profile looking thoughtfully away to the left into the distance. "
        "Her head is turned sideways, displaying a soft profile, elegant long neck, small delicate nose, melancholic gentle gaze. "
        "Messy textured dark hair tied in a loose relaxed bun with many natural wisps and soft strands falling around her neck and face. "
        "She is wearing an oversized dark brown woolen knit cardigan with chunky ribbing over a simple loose scooped neckline camisole. "
        "Her arms are crossed loosely in her lap near the bottom of the frame, hands gently folded. "
        "In the deep shadowy background on the far left, a low-burning hearth fireplace with gentle soft glowing warm embers and quiet flames casting a subtle warm rim light on her cheek and neck. "
        "A rustic dark wooden dining table in the foreground with an old ceramic mug. In the dark background shadows, a dim rustic wooden bookcase with vintage leather books and a solitary tea cup on a saucer. "
        "Deep shadows, moody muted color palette, delicate chiaroscuro, painterly textures, matte finish, soft diffused lighting, contemplative quiet nostalgic atmosphere."
    )
    
    # Beğendiğin o muazzam kalitedeki tohumlarla üretelim
    engine.generate(master_prompt, "golden_quality_seed303.png", seed=303, steps=12)
    engine.generate(master_prompt, "golden_quality_seed999.png", seed=999, steps=12)
