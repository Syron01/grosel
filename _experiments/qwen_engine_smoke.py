import time

import torch

from web_app.qwen_engine import QwenStreamEngine

t0 = time.time()
engine = QwenStreamEngine.get_instance()
print(f"engine ready {time.time()-t0:.1f}s", flush=True)

info = engine.system_info()
for key in ("engine", "budget_gb", "weights_bf16_bytes", "weights_fp8_bytes", "group_size", "cache_entries"):
    print(key, info[key], flush=True)
print("modules", [(m["id"], round(m["bytes"] / 1024 ** 3, 2), m["resident"]) for m in info["modules"]], flush=True)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
pos, mask, hit = engine.prompt_embeds(
    "anime illustration, two demon maid girls with bat wings in a candlelit bedroom, "
    "warm soft light, detailed faces and hands, clean linework"
)
torch.cuda.synchronize()
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1024 ** 3
print(f"encode {dt:.1f}s peak={peak:.2f}GB hit={hit} pos={tuple(pos.shape)} "
      f"finite={bool(torch.isfinite(pos).all())} abs_mean={float(pos.abs().mean()):.4f}", flush=True)
print(f"free={torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f}GB alloc={torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB", flush=True)
print("ok", flush=True)
