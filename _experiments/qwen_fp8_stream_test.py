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
NEGATIVE = ""
WIDTH, HEIGHT = 1280, 704
STEPS = 40
CFG = 1.0
SEED = 42


def vram(label):
    free = torch.cuda.mem_get_info()[0] / 1024 ** 3
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    print(f"[mem] {label}: free={free:.2f}GB alloc={alloc:.2f}GB", flush=True)


print("loading text encoder 4-bit ...", flush=True)
t0 = time.time()
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
te = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, subfolder="text_encoder", quantization_config=bnb,
    torch_dtype=torch.bfloat16, device_map="cpu",
    local_files_only=True, low_cpu_mem_usage=True,
)
print(f"text encoder loaded {time.time()-t0:.1f}s", flush=True)
# Text-only generation never touches the vision tower; the LM head is kept
# because Qwen3VLForConditionalGeneration.forward always computes logits.
if hasattr(te.model, "visual"):
    te.model.visual = None
gc.collect()
print("removed vision tower", flush=True)
vram("after te 4bit")

print("loading transformer BF16 ...", flush=True)
t0 = time.time()
tf = QwenImage21Transformer2DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
    local_files_only=True, low_cpu_mem_usage=True,
)
print(f"transformer loaded {time.time()-t0:.1f}s", flush=True)
t0 = time.time()
n_tf = convert_to_fp8_native(tf)
gc.collect()
print(f"transformer FP8 layers={n_tf} in {time.time()-t0:.1f}s", flush=True)
vram("after tf fp8")

print("loading processor/vae/scheduler ...", flush=True)
snap = snapshot_download(MODEL_ID, local_files_only=True)
processor = Qwen3VLProcessor.from_pretrained(os.path.join(snap, "processor"), local_files_only=True)
vae = AutoencoderKLQwenImage21.from_pretrained(
    MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True,
)
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    MODEL_ID, subfolder="scheduler", local_files_only=True,
)
pipe = QwenImage21Pipeline(
    vae=vae, text_encoder=te, processor=processor, transformer=tf, scheduler=scheduler,
)

# Stream the text encoder one decoder layer at a time.
lm = te.model.language_model
lm.embed_tokens.to("cuda")
lm.norm.to("cuda")
te.lm_head.to("cuda")
if hasattr(lm, "rotary_emb"):
    lm.rotary_emb.to("cuda")
handles = []


def _layer_in(module, args):
    module.to("cuda")


def _layer_out(module, args, output):
    module.to("cpu")


for layer in lm.layers:
    handles.append(layer.register_forward_pre_hook(_layer_in))
    handles.append(layer.register_forward_hook(_layer_out))

print("encoding prompt with layer streaming ...", flush=True)
t0 = time.time()
with torch.inference_mode():
    pos, pos_mask, _ = pipe.encode_prompt(prompt=PROMPT, image=None, device="cuda")
print(f"encoded {time.time()-t0:.1f}s pos={tuple(pos.shape)}", flush=True)
vram("after encode")

for h in handles:
    h.remove()
lm.embed_tokens.to("cpu")
lm.norm.to("cpu")
te.lm_head.to("cpu")
if hasattr(lm, "rotary_emb"):
    lm.rotary_emb.to("cpu")
gc.collect()
torch.cuda.empty_cache()
vram("after te offload")

vae.to("cuda")
vae.enable_tiling()
tf.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=2,
    use_stream=False,
)
print("offload enabled", flush=True)
vram("before denoise")

torch.cuda.reset_peak_memory_stats()
print(f"generating {STEPS} steps cfg={CFG} ...", flush=True)
t0 = time.time()
with torch.inference_mode():
    image = pipe(
        prompt_embeds=pos,
        prompt_embeds_mask=pos_mask,
        true_cfg_scale=CFG,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        generator=torch.Generator("cpu").manual_seed(SEED),
        use_kv_cache=True,
        output_type="pil",
    ).images[0]
print(
    f"generated in {time.time()-t0:.1f}s "
    f"peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB", flush=True,
)
out = os.path.join(os.path.dirname(__file__), "qwen_4bit_stream_40.png")
image.save(out)
print("saved", out, image.size, flush=True)
