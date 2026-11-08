"""Fast engine uctan uca olcum."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web_app.fast_engine import FastKleinEngine

P = ("Fine art painting, moody low-key lighting. Waist-up of a young woman in a dark rustic wooden "
     "armchair, three-quarter profile, loose dark hair bun, brown wool cardigan. A low-burning hearth "
     "fireplace casts warm rim light on her cheek. Delicate chiaroscuro, painterly texture, matte finish.")

t = time.time()
eng = FastKleinEngine()
print(f"[init] {time.time()-t:.1f}s")

runs = [
    ("1080p", P, 77),
    ("1080p", P, 123),          # ayni prompt -> cache
    ("720p",  P + " rain outside the window", 42),
    ("4K",    P, 7),
]
for quality, prompt, seed in runs:
    r = eng.generate(prompt, quality=quality, seed=seed)
    print(json.dumps({k: r[k] for k in ("quality", "resolution", "total_time", "encode_time",
                                       "prompt_cached", "step_time", "decode_time", "peak_vram_mb", "backend")}))
