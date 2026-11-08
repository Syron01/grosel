"""Ayni seed, ayni prompt: eski motor (bf16 + sequential offload) vs yeni motor (FP8).
Kullanim: python compare_quality.py old|new <cikti.png>
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image

P = ("Fine art painting, moody low-key lighting. Waist-up of a young woman in a dark rustic wooden "
     "armchair, three-quarter profile, loose dark hair bun, brown wool cardigan. A low-burning hearth "
     "fireplace casts warm rim light on her cheek. Delicate chiaroscuro, painterly texture, matte finish.")

mode, out = sys.argv[1], sys.argv[2]
t = time.time()
if mode == "old":
    from web_app.engine import GroselEngine
    eng = GroselEngine()
else:
    from web_app.fast_engine import FastKleinEngine
    eng = FastKleinEngine()

r = eng.generate(P, quality="1080p", seed=77)
Image.open(r["filepath"]).save(out)
print("MODE", mode, "total", r["total_time"], "peak", r["peak_vram_mb"])
