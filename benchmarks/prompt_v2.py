"""v2 kompozisyon capasi + VAE tiling ile VRAM olcumu."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from experiments.lowvram_engine import LowVramEngine
from web_app.engine import STYLE_PRESETS, compile_physically_grounded_prompt

USER = ("Rustik ahşap bir koltukta oturan genç bir kadın, yan profil, dağınık topuz saç, "
        "kahverengi yün hırka, elleri kucağında birleşmiş. Arka planda yanan şömine, "
        "yumuşak ateş ışığı yüzüne vuruyor. Karamsar, mat yağlıboya.")
PREFIX = STYLE_PRESETS["matte_chiaroscuro"]["prefix"]
V2 = (", tek parça tutarlı sahne, tek ışık kaynağı, doğal anatomi, beş parmaklı eller, "
      "gerçekçi ten dokusu, fiziksel olarak oturan nesneler")
OUT = os.path.join(os.environ.get("TEMP", "/tmp"), "v2_%s.png")

variants = {
    "v1": compile_physically_grounded_prompt(USER, "matte_chiaroscuro"),
    "v2": PREFIX + USER + V2,
    "v2crpf": PREFIX + USER + V2,
}

eng = LowVramEngine()
eng.vae.enable_tiling()
for tag, prompt in variants.items():
    crpf = tag.endswith("crpf")
    emb, _ = eng.prompt_embeds(prompt)
    img, t, _ = eng._generate_steps(emb, 1344, 768, 4, 77, enable_crpf=crpf)
    img.save(OUT % tag)
    print(f"{tag:8s} {t:.2f}s peak={torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
