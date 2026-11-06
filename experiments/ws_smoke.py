import asyncio
import websockets
import json

async def test_ws_crpf():
    uri = 'ws://127.0.0.1:7860/ws/generate'
    async with websockets.connect(uri) as websocket:
        payload = {
            'prompt': 'A delicate young Asian woman in a dark brown knit sweater sitting at a wooden table, soft amber glow from a single fireplace',
            'style': 'matte_chiaroscuro',
            'aspect_ratio': '16:9',
            'quality': '1080p',
            'enable_crpf': True,
            'seed': 777
        }
        await websocket.send(json.dumps(payload))
        print("[-] Sent CRPF generation request to WebSocket...")
        
        while True:
            resp = await websocket.recv()
            msg = json.loads(resp)
            msg_type = msg.get("type")
            if msg_type == "step_progress":
                print(f"[+] CRPF Step {msg['step']}/{msg['total_steps']} | {msg['elapsed']}s | {msg['it_s']} it/s | VRAM: {msg['vram_peak_mb']}MB")
            elif msg_type == "complete":
                t = msg["telemetry"]
                print("=" * 60)
                print("[OK] CRPF-FLOW INFERENCE FINISHED!")
                print(f"[OK] File: {t['filename']}")
                print(f"[OK] Total Time: {t['total_time']}s")
                print(f"[OK] Speed: {t['it_s']} it/s | TPS: {t['tps']} tok/s")
                print(f"[OK] Peak VRAM: {t['peak_vram_mb']} MB")
                print(f"[OK] Resolution: {t['resolution']}")
                print("=" * 60)
                break
            elif msg_type == "error":
                print("[!] WS ERROR:", msg["message"])
                break

if __name__ == "__main__":
    asyncio.run(test_ws_crpf())
