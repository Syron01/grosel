import time

from web_app.qwen_engine import QwenStreamEngine

PROMPT = ("anime illustration, two demon maid girls with bat wings in a candlelit bedroom, "
          "warm soft light, detailed faces and hands, clean linework")

t0 = time.time()
engine = QwenStreamEngine.get_instance()
print(f"engine ready {time.time()-t0:.1f}s", flush=True)
t = engine.generate(
    prompt=PROMPT,
    style_key="anime_concept",
    aspect_ratio="2:3",
    quality="1080p",
    seed=42,
    enable_crpf=False,
    enable_geo=False,
    steps_override=40,
    status_callback=lambda m: print("[status]", m, flush=True),
)
for k in ("filename", "resolution", "native_resolution", "steps", "total_time",
          "encode_time", "decode_time", "peak_vram_mb", "encode_peak_mb",
          "denoise_peak_mb", "decode_peak_mb"):
    print(k, t.get(k), flush=True)
