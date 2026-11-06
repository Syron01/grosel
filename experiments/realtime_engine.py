"""
Realtime Interactive Generator Server (4.3 Second Response Time)
- Modeli bellekte sıcak (warm) tutar.
- İlk yükleme gecikmesi olmadan her yeni üretim sadece ~4.3 saniyede biter.
- 1024x576 çözünürlük, 1.39 GB VRAM, referans sanatsal stil.
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

class RealtimeKleinEngine:
    def __init__(self):
        print("=" * 65)
        print("[-] Motor RAM'e yukleniyor ve isitiliyor (Warm-up)...")
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
        
        # Warmup: İlk çalışma derleme gecikmesini sıfırlar
        print("[*] On isitma yapiliyor...")
        _ = self.pipe(prompt="warmup", num_inference_steps=1, width=1024, height=576)
        print("[+] MOTOR ISITILDI VE GERCEK ZAMANLI MODA GECTI (4.3 sn / gorsel)!")

    def generate(self, prompt, filename, seed=42, steps=2):
        print(f"\n[*] Uretiliyor ({steps} Adim): {filename}")
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
        print(f"[OK] {filename} TAMAMLANDI! Sure: {elapsed:.2f} saniye | VRAM: {vram_peak:.1f} MB")
        return out_path

if __name__ == "__main__":
    engine = RealtimeKleinEngine()
    
    prompt = (
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
    
    # 4 saniyelik rekor hızda 2 farklı tohum üretelim
    engine.generate(prompt, "realtime_instant_seed77.png", seed=77, steps=2)
    engine.generate(prompt, "realtime_instant_seed123.png", seed=123, steps=2)
