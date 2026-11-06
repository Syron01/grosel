"""
Exact Composition & Atmosphere Matcher (FLUX.2 Klein-4B)
Hedef:
1. Yan profilden üç çeyrek (three-quarter side profile) bakış, başı hafifçe sola/boşluğa dönük.
2. Kamera orta-uzak (medium shot / waist-up), vücut sandalyede hafif geriye yaslanmış.
3. Kollar önde masaya paralel kucakta kavuşturulmuş.
4. Şömine sol arkada uzakta, loş karanlık arka plan (low-key atmospheric chiaroscuro).
5. Yarı saydam, mat, ressam fırça dokusu; sıfır yapay 3D/parlaklık.
"""

import os
import time
import torch
import gc
from diffusers import Flux2KleinPipeline

SNAPSHOT_DIR = r"C:\Users\theay\.cache\huggingface\hub\models--black-forest-labs--FLUX.2-klein-4B\snapshots\e7b7dc27f91deacad38e78976d1f2b499d76a294"
OUTPUT_DIR = r"C:\Users\theay\OneDrive\Desktop\HyperFlux_Galeri"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_pipeline():
    print("[-] Pipeline Hazirlaniyor...")
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
    return pipe

def generate_master_variant(pipe, prompt, filename, seed, steps=16):
    print(f"\n[*] Uretim Basliyor -> Seed: {seed} | Dosya: {filename}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    
    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            width=1024,
            height=558,  # Referans görselle piksel piksel aynı 1024x558 oranı
            generator=generator
        )
    elapsed = time.time() - t0
    out_path = os.path.join(OUTPUT_DIR, filename)
    result.images[0].save(out_path)
    print(f"[OK] {filename} uretildi ({elapsed:.1f}s)!")
    return out_path

if __name__ == "__main__":
    # Tam olarak 2. fotoğraftaki anatomik, kompozisyonel ve ışık geometrisi:
    refined_prompt = (
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

    pipeline = load_pipeline()
    
    # 3 farklı kompozisyon varyasyonu üretelim
    generate_master_variant(pipeline, refined_prompt, "master_match_seed55.png", seed=55, steps=16)
    generate_master_variant(pipeline, refined_prompt, "master_match_seed303.png", seed=303, steps=16)
    generate_master_variant(pipeline, refined_prompt, "master_match_seed999.png", seed=999, steps=16)
