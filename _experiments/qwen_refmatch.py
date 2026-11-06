import gc
import os
import time

import torch
from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)

from web_app.fp8_layers import convert_to_fp8_native

MODEL_ID = "Qwen/Qwen-Image-2.1"
PROMPT = ("anime illustration, two demon maid girls with bat wings in a candlelit bedroom, "
          "warm soft light, detailed faces and hands, clean linework")
WIDTH, HEIGHT = 832, 1248
STEPS = 40
SEED = 42


def vram(label):
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"[mem] {label}: free={free:.2f}GB alloc={alloc:.2f}GB", flush=True)


print("loading text encoder NF4 ...", flush=True)
t0 = time.time()
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
te = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, subfolder="text_encoder", quantization_config=bnb,
    torch_dtype=torch.bfloat16, device_map="cpu",
    local_files_only=True, low_cpu_mem_usage=True,
)
if hasattr(te.model, "visual"):
    te.model.visual = None
te.lm_head = torch.nn.Identity()
gc.collect()
print(f"te ready {time.time()-t0:.1f}s", flush=True)

print("loading transformer ...", flush=True)
t0 = time.time()
tf = QwenImage21Transformer2DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
    local_files_only=True, low_cpu_mem_usage=True,
)
n_tf = convert_to_fp8_native(tf)
gc.collect()
print(f"tf fp8 layers={n_tf} {time.time()-t0:.1f}s", flush=True)

snap = snapshot_download(MODEL_ID, local_files_only=True)
processor = Qwen3VLProcessor.from_pretrained(os.path.join(snap, "processor"), local_files_only=True)
vae = AutoencoderKLQwenImage21.from_pretrained(
    MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True,
)
vae.disable_tiling()
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    MODEL_ID, subfolder="scheduler", local_files_only=True,
)
pipe = QwenImage21Pipeline(
    vae=vae, text_encoder=te, processor=processor, transformer=tf, scheduler=scheduler,
)

lm = te.model.language_model
lm.embed_tokens.to("cuda")
lm.norm.to("cuda")
if hasattr(lm, "rotary_emb"):
    lm.rotary_emb.to("cuda")
handles = []


def _lin(module, args):
    module.to("cuda")


def _lout(module, args, output):
    module.to("cpu")


for layer in lm.layers:
    handles.append(layer.register_forward_pre_hook(_lin))
    handles.append(layer.register_forward_hook(_lout))

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.inference_mode():
    pos, pos_mask, _ = pipe.encode_prompt(prompt=PROMPT, image=None, device="cuda")
torch.cuda.synchronize()
print(f"encode {time.time()-t0:.1f}s peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB", flush=True)
vram("after encode")

for h in handles:
    h.remove()
lm.embed_tokens.to("cpu")
lm.norm.to("cpu")
if hasattr(lm, "rotary_emb"):
    lm.rotary_emb.to("cpu")
gc.collect()
torch.cuda.empty_cache()

probe = {"denoise": 0.0, "decode": 0.0, "decode_t": 0.0}
orig_decode = vae.decode


def decode_probe(*args, **kwargs):
    probe["denoise"] = max(probe["denoise"], torch.cuda.max_memory_allocated() / 1024 ** 3)
    t = time.time()
    todo = kwargs.get("video_latents", None)
    vae.to("cuda")
    torch.cuda.reset_peak_memory_stats()
    out = orig_decode(*args, **kwargs)
    torch.cuda.synchronize()
    probe["decode"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    probe["decode_t"] = time.time() - t
    vae.to("cpu")
    torch.cuda.empty_cache()
    return out


vae.decode = decode_probe
tf.enable_group_offload(
    onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
    offload_type="block_level", num_blocks_per_group=2, use_stream=False,
)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.inference_mode():
    image = pipe(
        prompt_embeds=pos, prompt_embeds_mask=pos_mask, true_cfg_scale=1.0,
        height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
        generator=torch.Generator("cpu").manual_seed(SEED),
        use_kv_cache=True, output_type="pil",
    ).images[0]
torch.cuda.synchronize()
dt = time.time() - t0
print(f"generated {dt:.1f}s total_peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB "
      f"denoise_peak={probe['denoise']:.2f}GB decode_peak={probe['decode']:.2f}GB "
      f"decode={probe['decode_t']:.1f}s", flush=True)
out = os.path.join(os.path.dirname(__file__), "qwen_refmatch_832x1248_40.png")
image.save(out)
print("saved", out, image.size, flush=True)
