"""
Pure Art-Style Pipeline (FLUX.2 Klein-4B)
- Ham latent müdahalesi ve yapay bokeh filtreleri tamamen kaldırıldı.
- Sanatsal yağlıboya/illüstrasyon kompozisyonu, hassas fırça darbeleri ve net karakter anatomisi.
- 1024x558 tam en-boy oranı.
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

def load_pure_pipeline():
    print("=" * 65)
    print("[-] Pure Sanatsal Pipeline Yukleniyor (Ham Model Motoru)...")
    print("=" * 65)
    
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
    print("[+] Model Hazir: VRAM 1.4 - 1.8 GB araliginda.")
    print("=" * 65)
    return pipe

def generate_exact_match(
    pipe,
    prompt: str,
    output_filename: str = "exact_match_artwork.png",
    num_inference_steps: int = 16,
    width: int = 1024,
    height: int = 576,
    seed: int = 42
):
    print(f"[*] Prompt: {prompt}")
    print(f"[*] Cozunurluk: {width}x{height} | Adim: {num_inference_steps} | Seed: {seed}")
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start_time = time.time()

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator
        )

    elapsed = time.time() - start_time
    out_path = os.path.join(OUTPUT_DIR, output_filename)
    result.images[0].save(out_path)

    vram_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print("=" * 65)
    print(f"[OK] Sanatsal Tablo Uretildi!")
    print(f"[OK] Dosya: {out_path}")
    print(f"[OK] Sure: {elapsed:.2f}s | VRAM Tepe: {vram_peak:.1f} MB")
    print("=" * 65)
    return out_path

if __name__ == "__main__":
    # 2. fotoğraftaki görselin birebir stil ve kompozisyon analizi:
    # Yarı-gerçekçi dijital illüstrasyon / yağlı boya stili, Asyatik/doğu esintili yüz hatları,
    # dağınık topuz saçlar, kollarını kavuşturup ahşap sandalyede oturan kadın,
    # sol tarafta şömine alevi, masada kupa, arkada kitaplık ve fincan,
    # net konturlar, sıcak amber chiaroscuro ışıklandırma.
    exact_prompt = (
        "Masterpiece digital oil painting illustration, semi-realistic character art. "
        "A beautiful young woman with delicate Asian features, expressive melancholic dark eyes, and a messy dark bun with loose wispy strands framing her face. "
        "She is wearing a dark chunky knit brown cardigan sweater over a simple top, seated in a dark wooden spindle-back chair with her arms crossed, resting her elbows on an old rustic dark wooden table. "
        "On the left, a cozy stone fireplace with soft burning embers and warm fire flames casting dramatic chiaroscuro amber light across her face and collarbone. "
        "On the table sits a rustic ceramic mug. In the dark background, a tall antique dark wooden bookshelf with stacked vintage books and a porcelain teacup on a saucer. "
        "Sharp crisp focus on the woman's face, porcelain skin, painterly brushstroke textures, visible painted fabric folds, deep rich shadows, muted warm atmospheric color palette, Artstation trending, intricate moody interior."
    )

    pipeline = load_pure_pipeline()
    
    # 2 farklı tohum (seed) ile üretim yapalım
    generate_exact_match(
        pipeline,
        prompt=exact_prompt,
        output_filename="2_istenen_stil_match_seed42.png",
        num_inference_steps=16,
        width=1024,
        height=576,
        seed=42
    )
    generate_exact_match(
        pipeline,
        prompt=exact_prompt,
        output_filename="2_istenen_stil_match_seed888.png",
        num_inference_steps=16,
        width=1024,
        height=576,
        seed=888
    )
