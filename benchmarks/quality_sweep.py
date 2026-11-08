"""Kalite taramasi: ayni prompt + seed, farkli ayarlar. Cikti: TEMP/q_<etiket>.png"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web_app.engine import compile_physically_grounded_prompt
from web_app.fast_engine import FastKleinEngine

PROMPT = ("Rustik ahşap bir koltukta oturan genç bir kadın, yan profil, dağınık topuz saç, "
          "kahverengi yün hırka, elleri kucağında birleşmiş. Arka planda yanan şömine, "
          "yumuşak ateş ışığı yüzüne vuruyor. Karamsar, mat yağlıboya.")
SEED = 77
OUT = os.path.join(os.environ.get("TEMP", "/tmp"), "q_%s.png")

eng = FastKleinEngine()
compiled = compile_physically_grounded_prompt(PROMPT, "matte_chiaroscuro")

variants = [
    ("base",      compiled, 1024, 576, 4, True),
    ("nocrpf",    compiled, 1024, 576, 4, False),
    ("steps8",    compiled, 1024, 576, 8, False),
    ("big1344",   compiled, 1344, 768, 4, False),
    ("noanchor",  PROMPT,   1024, 576, 4, False),
    ("big_steps", compiled, 1344, 768, 6, False),
]

for tag, prompt, w, h, steps, crpf in variants:
    emb, _ = eng.prompt_embeds(prompt)
    img, t, _ = eng._generate_steps(emb, w, h, steps, SEED, enable_crpf=crpf)
    img.save(OUT % tag)
    print(f"{tag:10s} {w}x{h} steps={steps} crpf={crpf} -> {t:.2f}s")
