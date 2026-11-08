"""El ablasyonu: CRPF, adim ve isgridin ayni seed'deki etkisi.

    python benchmarks/hand_ablation.py [prompt-index]

Cikti: %TEMP%/grosel_hand_ab/ altinda tam kare + el kirpimi.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from web_app.engine import compile_physically_grounded_prompt
from web_app.fast_engine import FastKleinEngine
from web_app.resolutions import detail_target, get_dimensions

PROMPTS = [
    ("scholar", 7102, "Karanlık antik kütüphanede mum ışığında el yazması inceleyen yaşlı bilgin, deri ciltli kitaplar, tozlu hava"),
    ("station", 7105, "Terk edilmiş tren istasyonunda bankta oturan genç kadın, sis, sarı istasyon lambası, ıslak beton"),
]
OUT = os.path.join(os.environ.get("TEMP", "/tmp"), "grosel_hand_ab")

VARIANTS = [
    ("1_4step", False, 4, True),
    ("2_4step_crpf", True, 4, True),
    ("3_8step", False, 8, True),
    ("4_4step_base", False, 4, False),
]


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    name, seed, prompt = PROMPTS[idx]
    os.makedirs(OUT, exist_ok=True)
    eng = FastKleinEngine()
    base_w, base_h, target_w, target_h, _, _ = get_dimensions("1080p", "16:9")
    rows = []
    for tag, crpf, steps, detail in VARIANTS:
        compiled = compile_physically_grounded_prompt(prompt, "matte_chiaroscuro", True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        emb, _ = eng.prompt_embeds(compiled)
        encode_peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        torch.cuda.reset_peak_memory_stats()
        if detail:
            size = detail_target(base_w, base_h, target_w, target_h, max_mp=2.2)
            gen_w, gen_h = size or (base_w, base_h)
        else:
            gen_w, gen_h = base_w, base_h
        image, _, _ = eng._generate_steps(emb, gen_w, gen_h, steps, seed, enable_crpf=crpf)
        if (target_w, target_h) != (gen_w, gen_h):
            growing = target_w > gen_w or target_h > gen_h
            kernel = Image.Resampling.BICUBIC if growing else Image.Resampling.LANCZOS
            image = image.resize((target_w, target_h), kernel)
        total = time.time() - t0
        peak = max(encode_peak, eng._step_peak_mb, eng._decode_peak_mb)
        stem = os.path.join(OUT, f"{name}_{tag}")
        image.save(stem + ".png")
        crop = image.crop((int(target_w * 0.10), int(target_h * 0.50),
                           int(target_w * 0.80), target_h))
        crop.save(stem + "_hands.png")
        row = {"variant": tag, "seed": seed, "steps": steps, "native": f"{gen_w}x{gen_h}",
               "seconds": round(total, 2), "peak_mb": round(peak, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps({"out": OUT, "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
