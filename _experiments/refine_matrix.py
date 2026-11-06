import asyncio
import json
import sys

import websockets

PROMPT = ("ltra-realistic candid portrait of a person with soft natural window lighting, "
          "a plain neutral background, detailed skin texture, stray hairs, shot on 85mm lens, "
          "f/1.8, photorealistic photography")
BASE = {
    "prompt": PROMPT,
    "style": "pure_raw",
    "aspect_ratio": "16:9",
    "quality": "1080p",
    "enable_crpf": False,
    "enable_geo": True,
    "seed": 692526363,
    "count": 1,
    "tag": "exp-refine",
}

CASES = [
    ("base8", 8, 0, 0.15),
    ("r1s12", 8, 1, 0.12),
    ("r1s20", 8, 1, 0.20),
    ("r1s30", 8, 1, 0.30),
    ("r2s20", 8, 2, 0.20),
    ("base4", 4, 0, 0.15),
    ("r1s20_4", 4, 1, 0.20),
    ("r2s20_4", 4, 2, 0.20),
]


async def run(name, steps, refine_steps, refine_sigma):
    payload = dict(BASE, steps=steps, refine_steps=refine_steps, refine_sigma=refine_sigma)
    async with websockets.connect("ws://127.0.0.1:7860/ws/generate", max_size=None) as ws:
        await ws.send(json.dumps(payload))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "error":
                print(f"{name}: ERROR {msg.get('message')}", flush=True)
                return
            if msg.get("type") == "complete":
                t = msg.get("telemetry") or {}
                print(
                    f"{name}: {t.get('quality')} {t.get('steps')}ad + {t.get('refine_steps')}r"
                    f" @ {t.get('refine_sigma')} | {t.get('total_time')}s"
                    f" | {t.get('peak_vram_mb')}MB | {t.get('filename')}",
                    flush=True,
                )
                return


async def main():
    only = set(sys.argv[1:])
    for name, steps, refine_steps, refine_sigma in CASES:
        if only and name not in only:
            continue
        await run(name, steps, refine_steps, refine_sigma)


if __name__ == "__main__":
    asyncio.run(main())
