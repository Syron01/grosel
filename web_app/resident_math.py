"""5 GB yerleşik plan: faz ayrık VRAM eşitsizliği.

Ölçümler (RTX 5060 Laptop 8 GB, Qwen-Image 2.1, seed 42, 20 adım):
  transformer BF16 13.25 GB -> NF4 3.43 GB, blok offload yok
  1280x704 denoise 3.96 GB tepe / 59.3 s   (FP8 akış: 79.9 s)
  1024x576 denoise 3.82 GB tepe / 35.1 s
  metin encode (katman akışı) 1.29 GB tepe / 7.2 s
  VAE decode (tiling) 1.06 GB tepe / 2.3-3.2 s

Fazlar aynı anda VRAM'de değil:
  M(B, S) = max(W_tf(b) + A(S), W_tf(b) + W_te, W_vae + A_dec)
  b* = en büyük b: M(b, S) <= B
"""

from __future__ import annotations

GB = 1024 ** 3
TRANSFORMER_PARAMS = 7.115e9
TRANSFORMER_BF16_GB = 13.25
NF4_GB = {4: 3.43, 3: 2.57, 2: 1.72}
SUPPORTED_BITS = {4}
TE_NF4_RAM_GB = 4.39
TE_STREAM_PEAK_GB = 1.29
VAE_PEAK_GB = 1.06
A_INTERCEPT_GB = 0.154
A_MP_SLOPE_GB = 0.418

REFERENCE = {
    "model": "Qwen-Image 2.1",
    "transformer_params": 7.115e9,
    "transformer_bf16_gb": 13.25,
    "transformer_nf4_gb": 3.43,
    "denoise_1280x704": {"peak_gb": 3.96, "seconds": 59.3, "steps": 20},
    "denoise_1024x576": {"peak_gb": 3.82, "seconds": 35.1, "steps": 20},
    "engine_720p_end_to_end": {"peak_gb": 3.91, "seconds": 38.8, "steps": 20,
                              "note": "WebSocket; encode 1.36, denoise 3.91, decode 1.14 GB"},
    "engine_720p_16": {"peak_gb": 3.91, "seconds": 29.3, "steps": 16,
                       "note": "sıcak prompt; ilk prompt encode ~+3 s; 20 adım kalite referansı"},
    "engine_1080p_2x3": {"peak_gb": 4.06, "seconds": 132.3, "steps": 40,
                         "native": "832x1248", "target": "1080x1620",
                         "note": "uçtan uca; plan 4.02 GB, sapma +0.04 GB"},
    "denoise_fp8_1280x704": {"peak_gb": 1.69, "seconds": 79.9, "steps": 20},
    "encode_peak_gb": 1.29,
    "decode_peak_gb": 1.06,
    "quality": {
        "psnr_vs_fp8": 16.09,
        "ssim_vs_fp8": 0.6662,
        "clip_fp8": 0.3385,
        "clip_nf4": 0.3045,
        "note": "PSNR/SSIM aynı kareyi ölçmez: difüzyon aynı seed'de kuantizasyonla ayrışır. "
                "İki görselde artefakt görülmedi; LPIPS/insan A/B henüz yok.",
    },
}


def activation_gb(width: int, height: int) -> float:
    mp = max(width, 256) * max(height, 256) / 1e6
    return max(0.05, A_INTERCEPT_GB + A_MP_SLOPE_GB * mp)


def _phase(name, label, peak, weights=0.0, stream=0.0, activation=0.0):
    return {
        "id": name,
        "label": label,
        "peak_gb": round(peak, 3),
        "weights_gb": round(weights, 3),
        "stream_gb": round(stream, 3),
        "activation_gb": round(activation, 3),
    }


def plan(budget_gb: float = 5.0, width: int = 1024, height: int = 576,
         bits: int = 4, strict: bool = True):
    budget = max(0.5, float(budget_gb))
    bits = int(bits)
    if bits not in NF4_GB:
        bits = 4
    w = max(256, int(width) // 32 * 32)
    h = max(256, int(height) // 32 * 32)
    tf = NF4_GB[bits]
    act = activation_gb(w, h)
    phases = [
        _phase("encode", "Metin encode", TE_STREAM_PEAK_GB + (0 if strict else tf),
               weights=0 if strict else tf, stream=TE_STREAM_PEAK_GB),
        _phase("denoise", "Difüzyon", tf + act, weights=tf, activation=act),
        _phase("decode", "VAE decode", VAE_PEAK_GB + (0 if strict else tf),
               weights=0 if strict else tf, activation=VAE_PEAK_GB),
    ]
    peak = max(p["peak_gb"] for p in phases)
    headroom = budget - peak
    max_mp = None
    if budget > tf + A_INTERCEPT_GB:
        max_mp = max(0.0, (budget - tf - A_INTERCEPT_GB) / A_MP_SLOPE_GB)
    options = []
    for b in (4, 3, 2):
        tf_b = NF4_GB[b]
        peak_b = max(
            TE_STREAM_PEAK_GB + (0 if strict else tf_b),
            tf_b + act,
            VAE_PEAK_GB + (0 if strict else tf_b),
        )
        options.append({
            "bits": b,
            "peak_gb": round(peak_b, 3),
            "feasible": peak_b <= budget + 1e-9,
            "supported": b in SUPPORTED_BITS,
        })
    runtime_ram = TE_NF4_RAM_GB + (tf if strict else 0.0)
    return {
        "budget_gb": round(budget, 2),
        "resolution": f"{w}x{h}",
        "native_mp": round(w * h / 1e6, 3),
        "bits": bits,
        "supported": bits in SUPPORTED_BITS,
        "strict": bool(strict),
        "transformer_gb": tf,
        "activation_gb": round(act, 3),
        "phases": phases,
        "peak_gb": round(peak, 3),
        "headroom_gb": round(headroom, 3),
        "feasible": peak <= budget + 1e-9,
        "max_native_mp": round(max_mp, 3) if max_mp is not None else None,
        "options": options,
        "ram": {
            "startup_bf16_gb": TRANSFORMER_BF16_GB,
            "runtime_gb": round(runtime_ram, 2),
            "note": "30 GB checkpoint RAM'e alınmaz; yalnız aktif faz + NF4 metin kodlayıcı.",
        },
        "reference": REFERENCE,
        "equation": "M(B,S)=max(W_tf(b)+A(S), W_tf(b)+W_te, W_vae+A_dec) <= B",
        "verdict": (
            f"{w}x{h}: {peak:.2f} GB tepe; {budget:.2f} GB bütçede "
            f"{'sığar' if peak <= budget else 'sığmaz'}."
        ),
    }


def summary(budget_gb: float = 5.0):
    return {
        "reference": REFERENCE,
        "plans": [
            plan(budget_gb, 1024, 576, 4, True),
            plan(budget_gb, 1280, 704, 4, True),
            plan(budget_gb, 1536, 864, 4, True),
        ],
    }
