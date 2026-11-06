import asyncio
import json

import websockets

PROMPTS = [
    ("portrait of an elderly fisherman mending a net, weathered hands, detailed fingers, "
     "harbor at dawn, photorealistic"),
    ("a red fox sitting in a snowy forest at golden hour, detailed fur, cinematic, "
     "soft rim light"),
]


async def run_prompt(ws, prompt, seed):
    await ws.send(json.dumps({
        "prompt": prompt,
        "style": "raw",
        "aspect_ratio": "16:9",
        "quality": "720p",
        "steps": 20,
        "seed": seed,
    }))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("type") == "complete":
            t = msg["telemetry"]
            print("COMPLETE", t["filename"], t["total_time"], t["peak_vram_mb"], flush=True)
            return t
        if msg.get("type") == "error":
            print("ERROR", msg.get("message"), flush=True)
            return None


async def main():
    async with websockets.connect("ws://127.0.0.1:7861/ws/generate") as ws:
        for i, prompt in enumerate(PROMPTS):
            await run_prompt(ws, prompt, 1000 + i)


if __name__ == "__main__":
    asyncio.run(main())
