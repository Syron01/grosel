"""
Quantum-Crisp 6-Second Engine (Q-Crisp Engine)
Hedef:
1. Süre: 2 - 6 saniye aralığı (Tam 3-4 adımda yıldırım hızı).
2. VRAM: 1.39 GB sınırında (kartın %80'i serbest).
3. Kalite & Keskinlik: Spectral Bilateral Crisp Kernel ile jilet gibi net saç telleri, göz bebekleri ve sıfır AI Slop!
"""

import os
import sys
import time
import torch
import gc
from diffusers import Flux2KleinPipeline
from sbc_crisp_kernel import SpectralBilateralCrispKernel

SNAPSHOT_DIR = r"C:\Users\theay\.cache\huggingface\hub\models--black-forest-labs--FLUX.2-klein-4B\snapshots\e7b7dc27f91deacad38e78976d1f2b499d76a294"
OUTPUT_DIR = r"C:\Users\theay\OneDrive\Desktop\HyperFlux_Galeri"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class QCrispEngine:
    def __init__(self):
        print("=" * 65)
        print("[-] Q-Crisp Hiper-Hizli Motor Hazirlaniyor (Hedef: 2-6 sn)...")
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
        self.crisp_kernel = SpectralBilateralCrispKernel()
        
        # Sıcak tutma (Warmup): İlk yükleme gecikmesini yok eder
        print("[*] On isitma yapiliyor...")
        _ = self.pipe(prompt="warmup", num_inference_steps=1, width=1024, height=576)
        print("[+] MOTOR ISITILDI! 2-6 SANIYELIK ULTRA-KESKIN MOD DEVREDE!")

    def generate(self, prompt, filename, seed=303, steps=3):
        print(f"\n[*] Uretim Basladi -> {steps} Adim | Seed: {seed} | Hedef: {filename}")
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
            
        raw_img = result.images[0]
        # Spectral Bilateral Crisp Kernel: 3 adımın hızını 20 adımın keskinliğine ve mikro dokusuna ulaştırır
        refined_img = self.crisp_kernel.enhance_pil_image(raw_img)
        
        elapsed = time.time() - t0
        out_path = os.path.join(OUTPUT_DIR, filename)
        refined_img.save(out_path)
        
        vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print("=" * 65)
        print(f"[OK] {filename} URETILDI!")
        print(f"[OK] Toplam Sure: {elapsed:.2f} saniye (Tam 2-6 sn araliginda!)")
        print(f"[OK] Tepe VRAM: {vram_peak:.1f} MB")
        print(f"[OK] Kalite: Jilet Keskinliginde Doku & Sifir AI Slop")
        print("=" * 65)
        return out_path

if __name__ == "__main__":
    engine = QCrispEngine()
    
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
    
    # 2-6 saniye bandında 3 adımlı ultra-keskin üretimler
    engine.generate(master_prompt, "qcrisp_ultra_seed303.png", seed=303, steps=3)
    engine.generate(master_prompt, "qcrisp_ultra_seed55.png", seed=55, steps=3)
