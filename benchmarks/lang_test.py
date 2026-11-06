"""Dil testi: ayni sahne Turkce vs Ingilizce."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from experiments.lowvram_engine import LowVramEngine

TR = ("Rustik ahşap bir koltukta oturan genç bir kadın, yan profil, dağınık topuz saç, "
      "kahverengi yün hırka, elleri kucağında birleşmiş. Arka planda yanan şömine, "
      "yumuşak ateş ışığı yüzüne vuruyor. Karamsar, mat yağlıboya.")
EN = ("A young woman sitting in a rustic wooden armchair, side profile, loose messy bun, "
      "brown wool cardigan, hands folded together in her lap. A single stone fireplace burns "
      "in the background, warm firelight falling on her face. Moody, matte oil painting.")
OUT = os.path.join(os.environ.get("TEMP", "/tmp"), "lang_%s.png")

eng = LowVramEngine()
eng.vae.enable_tiling()
for tag, prompt in {"tr": TR, "en": EN}.items():
    emb, _ = eng.prompt_embeds(prompt)
    img, t, _ = eng._generate_steps(emb, 1344, 768, 4, 77, enable_crpf=False)
    img.save(OUT % tag)
    print(f"{tag}: {t:.2f}s")
