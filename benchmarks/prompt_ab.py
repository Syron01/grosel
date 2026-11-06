"""Prompt stratejisi A/B: derleyici eki mi, ham prompt mu? (1344x768, 4 adim)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from experiments.lowvram_engine import LowVramEngine
from web_app.engine import compile_physically_grounded_prompt, STYLE_PRESETS

USER = ("Rustik ahşap bir koltukta oturan genç bir kadın, yan profil, dağınık topuz saç, "
        "kahverengi yün hırka, elleri kucağında birleşmiş. Arka planda yanan şömine, "
        "yumuşak ateş ışığı yüzüne vuruyor. Karamsar, mat yağlıboya.")
PREFIX = STYLE_PRESETS["matte_chiaroscuro"]["prefix"]
OUT = os.path.join(os.environ.get("TEMP", "/tmp"), "ab_%s.png")

variants = {
    "compiled": compile_physically_grounded_prompt(USER, "matte_chiaroscuro"),
    "raw": USER,
    "prefixed": PREFIX + USER,
}

eng = LowVramEngine()
for tag, prompt in variants.items():
    emb, _ = eng.prompt_embeds(prompt)
    img, t, _ = eng._generate_steps(emb, 1344, 768, 4, 77, enable_crpf=False)
    img.save(OUT % tag)
    print(f"{tag:10s} {t:.2f}s peak={torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
