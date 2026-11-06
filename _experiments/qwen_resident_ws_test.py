import asyncio
import json

import websockets

PROMPT = ("anime illustration, two demon maid girls with bat wings in a candlelit bedroom, "
          "warm soft light, detailed faces and hands, clean linework")


async def main():
    async with websockets.connect("ws://127.0.0.1:7861/ws/generate") as ws:
        await ws.send(json.dumps({
            "prompt": PROMPT,
            "style": "matte_chiaroscuro",
            "aspect_ratio": "16:9",
            "quality": "420p",
            "steps": 8,
            "seed": 42,
        }))
        while True:
            msg = json.loads(await ws.recv())
            if msg.get("type") == "complete":
                print("COMPLETE", json.dumps(msg["telemetry"], ensure_ascii=False), flush=True)
                break
            if msg.get("type") == "error":
                print("ERROR", msg.get("message"), flush=True)
                break
            if msg.get("type") == "step_progress":
                print("STEP", msg.get("step"), msg.get("vram_peak_mb"), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
