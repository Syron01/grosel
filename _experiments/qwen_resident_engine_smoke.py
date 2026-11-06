import json
import os
import time

os.environ["HYPERFLUX_QWEN_RESIDENT"] = "1"
os.environ["HYPERFLUX_QWEN_BUDGET"] = "5"

from web_app.qwen_engine import QwenStreamEngine

t0 = time.time()
engine = QwenStreamEngine.get_instance()
print("SYS1", json.dumps(engine.system_info(), ensure_ascii=False), flush=True)
res = engine.generate(
    prompt=("anime illustration, two demon maid girls with bat wings in a candlelit bedroom, "
            "warm soft light, detailed faces and hands, clean linework"),
    style_key="matte_chiaroscuro",
    aspect_ratio="16:9",
    quality="720p",
    seed=42,
    steps_override=20,
)
print("GEN", json.dumps(res, ensure_ascii=False), flush=True)
print("SYS2", json.dumps(engine.system_info(), ensure_ascii=False), flush=True)
print(f"WALL {time.time()-t0:.1f}s", flush=True)
