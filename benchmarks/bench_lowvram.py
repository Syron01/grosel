"""LowVRAM motor olcumu: VRAM / sure / kalite."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.lowvram_engine import LowVramEngine

P = ("Rustik ahşap bir koltukta oturan genç bir kadın, yan profil, dağınık topuz saç, "
     "kahverengi yün hırka, elleri kucağında birleşmiş. Arka planda yanan şömine, "
     "yumuşak ateş ışığı yüzüne vuruyor. Karamsar, mat yağlıboya.")
OUT = os.path.join(os.environ.get("TEMP", "/tmp"), "lv_%s.png")

eng = LowVramEngine()
for tag, w, h, steps in [("1024", 1024, 576, 4), ("1344", 1344, 768, 4), ("1344s6", 1344, 768, 6)]:
    emb, _ = eng.prompt_embeds(P)
    img, t, _ = eng._generate_steps(emb, w, h, steps, 77, enable_crpf=True)
    img.save(OUT % tag)
    print(f"{tag}: {t:.2f}s | peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
