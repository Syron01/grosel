"""Qwen-Image 2.1 transformer'ı NF4 4-bit olarak tamamen VRAM'de tutar.

Blok offload yok, CPU'ya ağırlık akışı yok (yükleme anı hariç).
Fazlar: TE encode -> transformer denoise -> VAE decode; her fazda yalnız o modül VRAM'de.

Ölçüm: 1280x704, 20 adım, seed 42; referans FP8 akışlı qwen_base_20.png ile PSNR/SSIM/CLIP.
"""

import gc
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import snapshot_download

import bitsandbytes as bnb
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

MODEL_ID = "Qwen/Qwen-Image-2.1"
PROMPT = ("anime illustration, two demon maid girls with bat wings in a candlelit bedroom, "
          "warm soft light, detailed faces and hands, clean linework")
WIDTH = int(os.environ.get("QF_WIDTH", "1280"))
HEIGHT = int(os.environ.get("QF_HEIGHT", "704"))
STEPS = int(os.environ.get("QF_STEPS", "20"))
SKIP = tuple(p.strip() for p in os.environ.get("QF_SKIP", "").split(",") if p.strip())
COMPRESS = os.environ.get("QF_COMPRESS", "1") == "1"
SEED = 42
GB = 1024 ** 3
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(OUT_DIR, "Yeni klasör", "qwen_base_20.png")
EMB_FILE = os.path.join(OUT_DIR, "qwen_embed_1280x704.pt")


def mem(label):
    free, total = torch.cuda.mem_get_info()
    print(f"[mem] {label}: free={free/GB:.2f}GB alloc={torch.cuda.memory_allocated()/GB:.2f}GB "
          f"peak={torch.cuda.max_memory_allocated()/GB:.2f}GB", flush=True)


def reset_peak():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


# ---------------------------------------------------------------- prompt encode
def encode_prompt(snap):
    print("[phase] text encoder NF4 yükleniyor (CPU)...", flush=True)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        snap, subfolder="text_encoder", quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16, device_map="cpu",
        local_files_only=True, low_cpu_mem_usage=True,
    )
    if getattr(te.model, "visual", None) is not None:
        te.model.visual = None
    te.lm_head = nn.Identity()
    lm = te.model.language_model
    resident = [m for m in (getattr(lm, "embed_tokens", None), getattr(lm, "norm", None),
                            getattr(lm, "rotary_emb", None)) if m is not None]

    def layer_in(module, args):
        module.to("cuda")

    def layer_out(module, args, output):
        module.to("cpu")

    for layer in lm.layers:
        layer.register_forward_pre_hook(layer_in)
        layer.register_forward_hook(layer_out)

    processor = Qwen3VLProcessor.from_pretrained(
        os.path.join(snap, "processor"), local_files_only=True,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler", local_files_only=True,
    )
    pipe = QwenImage21Pipeline(vae=None, text_encoder=te, processor=processor,
                               transformer=None, scheduler=scheduler)
    for module in resident:
        module.to("cuda")
    reset_peak()
    t1 = time.time()
    with torch.inference_mode():
        pos, mask, _ = pipe.encode_prompt(prompt=PROMPT, image=None, device="cuda")
    torch.cuda.synchronize()
    enc_s = time.time() - t1
    peak = torch.cuda.max_memory_allocated()
    pos = pos.to("cpu")
    mask = mask.to("cpu") if mask is not None else None
    torch.save({"pos": pos, "mask": mask}, EMB_FILE)
    print(f"[encode] {enc_s:.1f}s peak={peak/GB:.2f}GB emb={tuple(pos.shape)} "
          f"mask={None if mask is None else tuple(mask.shape)}", flush=True)
    pipe.text_encoder = None
    del te, pipe, lm, resident
    gc.collect()
    torch.cuda.empty_cache()
    return pos, mask, enc_s, peak, time.time() - t0, processor


def quantize_linears(module, skip_prefixes=()):
    replaced = 0
    for name, child in list(module.named_children()):
        full = name
        if isinstance(child, nn.Linear) and not any(full.startswith(p) for p in skip_prefixes):
            new = bnb.nn.Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=COMPRESS,
                quant_type="nf4",
                quant_storage=torch.uint8,
                device="cpu",
            )
            new.weight = bnb.nn.Params4bit(
                child.weight.data, requires_grad=False,
                quant_type="nf4", compress_statistics=COMPRESS,
            )
            if child.bias is not None:
                new.bias = nn.Parameter(child.bias.data.clone(), requires_grad=False)
            setattr(module, name, new)
            replaced += 1
        else:
            replaced += quantize_linears(child, skip_prefixes)
    return replaced


def main():
    snap = snapshot_download(MODEL_ID, local_files_only=True)
    print(f"[setup] snapshot {snap}", flush=True)
    torch.cuda.empty_cache()
    gc.collect()

    processor = Qwen3VLProcessor.from_pretrained(
        os.path.join(snap, "processor"), local_files_only=True,
    )
    if os.path.exists(EMB_FILE):
        blob = torch.load(EMB_FILE, map_location="cpu", weights_only=True)
        pos, mask = blob["pos"], blob["mask"]
        enc_s, enc_peak = 0.0, 0.0
        print(f"[encode] cache {EMB_FILE} emb={tuple(pos.shape)}", flush=True)
    else:
        pos, mask, enc_s, enc_peak, _, processor = encode_prompt(snap)

    print("[phase] transformer BF16 CPU'ya yükleniyor...", flush=True)
    t0 = time.time()
    tf = QwenImage21Transformer2DModel.from_pretrained(
        snap, subfolder="transformer", torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True,
    )
    load_s = time.time() - t0
    print(f"[load] BF16 {load_s:.1f}s", flush=True)

    print("[phase] lineerler NF4'e çevriliyor...", flush=True)
    tq = time.time()
    n = quantize_linears(tf, SKIP)
    gc.collect()
    quant_s = time.time() - tq
    print(f"[quant] {n} lineer paketlendi {quant_s:.1f}s skip={SKIP} compress={COMPRESS}", flush=True)

    print("[phase] 4-bit model GPU'ya...", flush=True)
    reset_peak()
    t0 = time.time()
    tf.to("cuda")
    tf.eval()
    torch.cuda.synchronize()
    move_s = time.time() - t0
    gc.collect()
    mem("4-bit model GPU'da")
    weights_gb = torch.cuda.memory_allocated() / GB
    print(f"[resident] {move_s:.1f}s weights_alloc={weights_gb:.2f}GB", flush=True)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler", local_files_only=True,
    )
    pipe = QwenImage21Pipeline(vae=None, text_encoder=None, processor=processor,
                               transformer=tf, scheduler=scheduler)
    pos = pos.to("cuda")
    mask = mask.to("cuda") if mask is not None else None

    reset_peak()
    t0 = time.time()
    with torch.inference_mode():
        out = pipe(
            prompt=None,
            prompt_embeds=pos,
            prompt_embeds_mask=mask,
            true_cfg_scale=1.0,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            generator=torch.Generator("cpu").manual_seed(SEED),
            use_kv_cache=True,
            output_type="latent",
        )
    torch.cuda.synchronize()
    den_s = time.time() - t0
    den_peak = torch.cuda.max_memory_allocated()
    latents = out.images
    print(f"[denoise] {STEPS} adım {den_s:.1f}s peak={den_peak/GB:.2f}GB "
          f"it/s={STEPS/den_s:.2f}", flush=True)
    torch.save(latents.to("cpu"), os.path.join(OUT_DIR, "qwen_resident_latents.pt"))

    print("[phase] transformer VRAM'den çıkıyor, VAE yükleniyor...", flush=True)
    pipe.transformer = None
    del tf, pipe, pos, mask
    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.time()
    vae = AutoencoderKLQwenImage21.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True,
    )
    vae.enable_tiling()
    vae.to("cuda")
    reset_peak()
    t1 = time.time()
    with torch.inference_mode():
        lat = latents.to("cuda")
        lat = QwenImage21Pipeline._unpack_latents(lat, HEIGHT, WIDTH, 16)
        lat = lat.to(vae.dtype)
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        lat = lat * std + mean
        image = vae.decode(lat, return_dict=False)[0][:, :, 0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype(np.uint8)
    dec_s = time.time() - t1
    dec_peak = torch.cuda.max_memory_allocated()
    img = Image.fromarray(image)
    tag = f"{WIDTH}x{HEIGHT}_{STEPS}" + ("_skip" if SKIP else "")
    out_path = os.path.join(OUT_DIR, f"qwen_resident_nf4_{tag}.png")
    img.save(out_path)
    print(f"[decode] {dec_s:.1f}s peak={dec_peak/GB:.2f}GB -> {out_path} {img.size}", flush=True)

    metrics = {
        "steps": STEPS, "width": WIDTH, "height": HEIGHT, "seed": SEED,
        "skip": list(SKIP), "compress": COMPRESS,
        "encode_s": round(enc_s, 2), "encode_peak_gb": round(enc_peak / GB, 2),
        "load_bf16_s": round(load_s, 1), "quant_s": round(quant_s, 1),
        "weights_gb": round(weights_gb, 2), "denoise_s": round(den_s, 2),
        "denoise_peak_gb": round(den_peak / GB, 2), "decode_s": round(dec_s, 2),
        "decode_peak_gb": round(dec_peak / GB, 2), "image": out_path,
    }

    if os.path.exists(REF) and (WIDTH, HEIGHT) == (1280, 704):
        ref = Image.open(REF).convert("RGB")
        ours = img.convert("RGB")
        if ref.size != ours.size:
            ref = ref.resize(ours.size, Image.Resampling.LANCZOS)
        a = np.asarray(ref, dtype=np.float32) / 255.0
        b = np.asarray(ours, dtype=np.float32) / 255.0
        mse = float(((a - b) ** 2).mean())
        psnr = 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse))
        # global SSIM
        mu_a, mu_b = a.mean(axis=(0, 1)), b.mean(axis=(0, 1))
        va, vb = a.var(axis=(0, 1)), b.var(axis=(0, 1))
        cov = ((a - mu_a) * (b - mu_b)).mean(axis=(0, 1))
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = float((((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                      ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean())
        metrics.update({"ref": REF, "psnr": round(psnr, 2), "ssim": round(ssim, 4),
                        "mae": round(float(np.abs(a - b).mean() * 255), 2)})
        print(f"[quality] PSNR={psnr:.2f}dB SSIM={ssim:.4f} MAE={metrics['mae']}/255", flush=True)

        try:
            import clip
            model, preprocess = clip.load("ViT-B/32", device="cpu")
            with torch.no_grad():
                def score(im):
                    x = preprocess(im).unsqueeze(0)
                    t = clip.tokenize([PROMPT], truncate=True)
                    ie = model.encode_image(x)
                    te_ = model.encode_text(t)
                    ie = ie / ie.norm(dim=-1, keepdim=True)
                    te_ = te_ / te_.norm(dim=-1, keepdim=True)
                    return float((ie @ te_.T).item())
                metrics["clip_ref"] = round(score(ref), 4)
                metrics["clip_ours"] = round(score(ours), 4)
                print(f"[quality] CLIP ref={metrics['clip_ref']} ours={metrics['clip_ours']}", flush=True)
        except Exception as exc:
            print("[quality] CLIP atlandı:", repr(exc), flush=True)

    with open(os.path.join(OUT_DIR, f"qwen_resident_nf4_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print("[done]", json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
