"""HyperFlux Qwen Engine — 31.7 GB model, 2.3 GB altı faz tepesi.

Neden sığar:
  * Transformer FP8 (7.6 GB) blok grubuyla akıtılır; VRAM'de yalnızca o anki grup var.
  * Metin kodlayıcı NF4 (4.5 GB) katman katman akıtılır; tepe tek katman + gömme tablosu.
  * lm_head hiç çağrılmaz. Difüzyon yalnızca gizli durumları okur; 151k×4096 kafayı
    GPU'ya taşımak encode tepesini 1.24 GB artırıp modele hiçbir şey katmıyordu.
  * VAE sadece decode anında VRAM'e girer; denoise tepesine 1.35 GB eklemez.
  * Prompt embedding'i cache'lenir: aynı prompt + yeni seed encode maliyeti taşımaz.
"""

import gc
import hashlib
import os
import time

import bitsandbytes as bnb
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from diffusers import (
    AutoencoderKLQwenImage21,
    FlowMatchEulerDiscreteScheduler,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from web_app.engine import (STYLE_PRESETS, GALLERY_DIR, compile_physically_grounded_prompt,
                            save_sidecar)
from web_app.fp8_layers import convert_to_fp8_native
from web_app import quality_math

MODEL_ID = os.environ.get("HYPERFLUX_QWEN_ID", "Qwen/Qwen-Image-2.1")
GROUP_SIZE = max(1, int(os.environ.get("HYPERFLUX_QWEN_GROUP", "2")))
DESIGN_BUDGET_GB = 3.5
RESIDENT = os.environ.get("HYPERFLUX_QWEN_RESIDENT", "0") == "1"
RESIDENT_BUDGET_GB = float(os.environ.get("HYPERFLUX_QWEN_BUDGET", "5.0"))
RESIDENT_SKIP = tuple(
    p.strip() for p in os.environ.get(
        "HYPERFLUX_QWEN_SKIP", "img_in,proj_out,time_text_embed,norm_out",
    ).split(",") if p.strip()
)
GB = 1024 ** 3

# Qwen tabanlı çözünürlük merdiveni: her boy 32'nin katı. Model 2K kovaya eğitimli olduğu
# için native alan kademe hedefinin altında kalmıyor (1080p 1.75 MP, 2K 2.36 MP); hedefe
# yalnızca büyütülüyor — büyütmede lobsuz bikübik, küçültmede Lanczos.
QWEN_DIMENSIONS = {
    "420p": {
        "steps": 8, "upscale": False,
        "16:9": (800, 448), "1:1": (576, 576), "9:16": (448, 800),
        "4:3": (640, 480), "3:4": (480, 640), "2:3": (512, 768), "21:9": (896, 384),
    },
    "720p": {
        "steps": 20, "upscale": False,
        "16:9": (1024, 576), "1:1": (768, 768), "9:16": (576, 1024),
        "4:3": (896, 672), "3:4": (672, 896), "2:3": (704, 1056), "21:9": (1120, 480),
    },
    "1080p": {
        "steps": 40, "upscale": True,
        "16:9": (1760, 992), "1:1": (1344, 1344), "9:16": (992, 1760),
        "4:3": (1536, 1152), "3:4": (1152, 1536), "2:3": (1088, 1632), "21:9": (2016, 864),
        "target_res": {
            "16:9": (1920, 1080), "1:1": (1440, 1440), "9:16": (1080, 1920),
            "4:3": (1440, 1080), "3:4": (1080, 1440), "2:3": (1080, 1620), "21:9": (2520, 1080),
        },
    },
    "2K": {
        "steps": 32, "upscale": True,
        "16:9": (2048, 1152), "1:1": (1536, 1536), "9:16": (1152, 2048),
        "4:3": (1536, 1152), "3:4": (1152, 1536), "2:3": (1088, 1632), "21:9": (2016, 864),
        "target_res": {
            "16:9": (2048, 1152), "1:1": (1536, 1536), "9:16": (1152, 2048),
            "4:3": (1600, 1200), "3:4": (1200, 1600), "2:3": (1440, 2160), "21:9": (2560, 1088),
        },
    },
    "4K": {
        "steps": 40, "upscale": True,
        "16:9": (2048, 1152), "1:1": (1536, 1536), "9:16": (1152, 2048),
        "4:3": (1792, 1344), "3:4": (1344, 1792), "2:3": (1280, 1920), "21:9": (2240, 960),
        "target_res": {
            "16:9": (3840, 2160), "1:1": (2160, 2160), "9:16": (2160, 3840),
            "4:3": (2560, 1920), "3:4": (1920, 2560), "2:3": (2160, 3240), "21:9": (3440, 1440),
        },
    },
}

QWEN_TIERS = [
    {"value": 0, "label": "Oto", "title": "Kalite kademesine göre 8–40 adım; 1080p'de 40"},
    {"value": 16, "label": "Sınır 16", "title": "720p'de ~29 s; 20 adım kalite referansı, 16 hızlı önizleme"},
    {"value": 20, "label": "Hızlı 20", "title": "20 adım; Qwen-Image 2.1 tabanı için alt sınır"},
    {"value": 28, "label": "Dengeli 28", "title": "28 adım; varsayılan kalite/süre dengesi"},
    {"value": 40, "label": "Ultra 40", "title": "40 adım; önerilen tam yörünge, en uzun yol"},
]


def qwen_dimensions(quality, ratio, steps_override=0):
    q = QWEN_DIMENSIONS.get(quality, QWEN_DIMENSIONS["1080p"])
    base_w, base_h = q.get(ratio, q["16:9"])
    # Qwen VAE + patch paketleme 32'nin katını istiyor; sessiz yeniden boyutlandırmayı önle.
    base_w = base_w // 32 * 32
    base_h = base_h // 32 * 32
    target_w, target_h = q.get("target_res", {}).get(ratio, (base_w, base_h))
    steps = q["steps"]
    if steps_override:
        steps = max(1, min(48, int(steps_override)))
    return base_w, base_h, target_w, target_h, steps


def auto_steps_map():
    return {q: v["steps"] for q, v in QWEN_DIMENSIONS.items()}


def dims_map():
    return {q: {k: f"{v[0]}×{v[1]}" for k, v in data.items() if isinstance(v, tuple)}
            for q, data in QWEN_DIMENSIONS.items()}


def _numel(module):
    total = sum(p.numel() for p in module.parameters())
    total += sum(b.numel() for b in module.buffers() if isinstance(b, torch.Tensor))
    return total


def _bytes(module):
    total = sum(p.numel() * p.element_size() for p in module.parameters())
    total += sum(b.numel() * b.element_size() for b in module.buffers() if isinstance(b, torch.Tensor))
    return total


def _bf16_bytes(module):
    """4-bit paketli ağırlıkta numel yarıdır; BF16 eşdeğeri için 2×2 byte sayılır."""
    total = 0
    for p in module.parameters():
        total += p.numel() * (4 if "4bit" in type(p).__name__.lower() else 2)
    total += sum(b.numel() * b.element_size() for b in module.buffers() if isinstance(b, torch.Tensor))
    return total


def _quantized_bytes(module):
    """Paketli 4-bit ağırlık + absmax/kod kayıtları; gerçek VRAM yükünü sayar."""
    total = 0
    for p in module.parameters():
        total += p.numel() * p.element_size()
        qs = getattr(p, "quant_state", None)
        if qs is None:
            continue
        for t in (qs.absmax, getattr(qs, "code", None)):
            if isinstance(t, torch.Tensor):
                total += t.numel() * t.element_size()
        if getattr(qs, "nested", False):
            for t in (qs.state2.absmax, getattr(qs.state2, "code", None)):
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
    total += sum(b.numel() * b.element_size() for b in module.buffers() if isinstance(b, torch.Tensor))
    return total


def quantize_nf4_linears(module, skip_prefixes=(), prefix=""):
    """Lineerleri yerinde NF4 Linear4bit'e çevirir; hassas küçük katmanlar BF16 kalır."""
    replaced = 0
    for name, child in list(module.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        is_skipped = any(full == p or full.startswith(p + ".") for p in skip_prefixes)
        if isinstance(child, nn.Linear) and not is_skipped:
            new = bnb.nn.Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type="nf4",
                quant_storage=torch.uint8,
                device="cpu",
            )
            new.weight = bnb.nn.Params4bit(
                child.weight.data, requires_grad=False,
                quant_type="nf4", compress_statistics=True,
            )
            if child.bias is not None:
                new.bias = nn.Parameter(child.bias.data.clone(), requires_grad=False)
            setattr(module, name, new)
            replaced += 1
        else:
            replaced += quantize_nf4_linears(child, skip_prefixes, full)
    return replaced


class QwenStreamEngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        torch.cuda.empty_cache()
        gc.collect()
        t0 = time.time()
        self.snapshot = snapshot_download(MODEL_ID, local_files_only=True)
        print(f"[qwen] snapshot {self.snapshot}", flush=True)
        self.resident = RESIDENT
        self.budget_gb = RESIDENT_BUDGET_GB if self.resident else DESIGN_BUDGET_GB
        self._tf_gpu_bytes = 0
        self._last_evict = False

        if self.resident:
            # Önce transformer: BF16 13.25 GB bir kez okunur, NF4'e paketlenip VRAM'e
            # yerleşir; BF16 kopyası serbest bırakılır. Sonra metin kodlayıcı CPU'ya.
            print("[qwen] transformer BF16 -> NF4 yerleşik...", flush=True)
            t1 = time.time()
            self.transformer = QwenImage21Transformer2DModel.from_pretrained(
                self.snapshot, subfolder="transformer", torch_dtype=torch.bfloat16,
                local_files_only=True, low_cpu_mem_usage=True,
            )
            self._tf_numel = _numel(self.transformer)
            self.n_layers = quantize_nf4_linears(self.transformer, RESIDENT_SKIP)
            gc.collect()
            self.transformer.to("cuda")
            self.transformer.eval()
            torch.cuda.synchronize()
            self._tf_gpu_bytes = _quantized_bytes(self.transformer)
            print(f"[qwen] transformer NF4 lineer={self.n_layers} "
                  f"gpu={self._tf_gpu_bytes/GB:.2f} GB {time.time()-t1:.1f}s", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

            print("[qwen] text encoder NF4 yükleniyor...", flush=True)
            t2 = time.time()
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                self.snapshot, subfolder="text_encoder", quantization_config=bnb_cfg,
                torch_dtype=torch.bfloat16, device_map="cpu",
                local_files_only=True, low_cpu_mem_usage=True,
            )
            if getattr(self.text_encoder.model, "visual", None) is not None:
                self.text_encoder.model.visual = None
            self.text_encoder.lm_head = torch.nn.Identity()
            gc.collect()
            print(f"[qwen] text encoder hazır {time.time()-t2:.1f}s", flush=True)
        else:
            print("[qwen] text encoder NF4 yükleniyor...", flush=True)
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                self.snapshot, subfolder="text_encoder", quantization_config=bnb_cfg,
                torch_dtype=torch.bfloat16, device_map="cpu",
                local_files_only=True, low_cpu_mem_usage=True,
            )
            if getattr(self.text_encoder.model, "visual", None) is not None:
                self.text_encoder.model.visual = None
            # Pipeline yalnızca hidden_states[-1] okuyor; lm_head logitleri hiç kullanılmıyor.
            self.text_encoder.lm_head = torch.nn.Identity()
            gc.collect()
            print(f"[qwen] text encoder hazır {time.time()-t0:.1f}s", flush=True)

            print("[qwen] transformer BF16 yükleniyor...", flush=True)
            t1 = time.time()
            self.transformer = QwenImage21Transformer2DModel.from_pretrained(
                self.snapshot, subfolder="transformer", torch_dtype=torch.bfloat16,
                local_files_only=True, low_cpu_mem_usage=True,
            )
            self._tf_numel = _numel(self.transformer)
            self.n_layers = convert_to_fp8_native(self.transformer)
            gc.collect()
            print(f"[qwen] transformer FP8 lineer={self.n_layers} {time.time()-t1:.1f}s", flush=True)

        print("[qwen] VAE / processor / scheduler...", flush=True)
        self.vae = AutoencoderKLQwenImage21.from_pretrained(
            self.snapshot, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True,
        )
        # Tiled decode 192 px araliginda dikey diki birakiyordu (ayni seedde olculdu:
        # z=4.11 -> 0.17). Tek parca decode 1.33 MP'da 9.06 GB tepiyor, 8 GB'a sigmiyor;
        # kayipsiz cozum genis onusmeli karo: diffusers varsayilanini EZER.
        self.vae_blend_px = quality_math.apply_tiling(self.vae)
        self.processor = Qwen3VLProcessor.from_pretrained(
            os.path.join(self.snapshot, "processor"), local_files_only=True,
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.snapshot, subfolder="scheduler", local_files_only=True,
        )
        self.pipe = QwenImage21Pipeline(
            vae=self.vae, text_encoder=self.text_encoder, processor=self.processor,
            transformer=self.transformer, scheduler=scheduler,
        )

        lm = self.text_encoder.model.language_model
        self._lm = lm
        self._te_resident_bits = [m for m in (
            getattr(lm, "embed_tokens", None), getattr(lm, "norm", None),
            getattr(lm, "rotary_emb", None),
        ) if m is not None]
        self._te_handles = []
        for layer in lm.layers:
            self._te_handles.append(layer.register_forward_pre_hook(self._layer_in))
            self._te_handles.append(layer.register_forward_hook(self._layer_out))

        if not self.resident:
            self.transformer.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=GROUP_SIZE,
                use_stream=False,
            )

        # VAE yalnızca decode anında VRAM'e girer; denoise tepesi VAE taşımaz.
        self._orig_vae_decode = self.vae.decode
        self.vae.decode = self._decode_with_vae_phase

        self._emb_cache = {}
        self._encode_peak_mb = 0.0
        self._denoise_peak_mb = 0.0
        self._decode_peak_mb = 0.0
        self._decode_time_s = 0.0
        self._step_peak_mb = 0.0
        self.last_swap_time = 0.0
        self.last_swapped = False
        self.n_te_layers = len(lm.layers)
        self._tf_stream_bytes = _quantized_bytes(self.transformer) if self.resident else _bytes(self.transformer)
        self.weights_bf16_bytes = self._tf_numel * 2 + _bf16_bytes(self.text_encoder) + _bytes(self.vae)
        self.weights_stream_bytes = self._tf_stream_bytes + _bytes(self.text_encoder)
        self._warmup()
        label = "yerleşik" if self.resident else "akış"
        print(
            f"[qwen] hazır {time.time()-t0:.1f}s | {label} {self.weights_stream_bytes/GB:.2f} GB "
            f"| bütçe {self.budget_gb:.1f} GB | serbest VRAM {torch.cuda.mem_get_info()[0]/GB:.2f} GB",
            flush=True,
        )

    # ------------------------------------------------------------------ phases

    @staticmethod
    def _layer_in(module, args):
        module.to("cuda")

    @staticmethod
    def _layer_out(module, args, output):
        module.to("cpu")

    def _te_on(self):
        for module in self._te_resident_bits:
            module.to("cuda")

    def _te_off(self):
        for module in self._te_resident_bits:
            module.to("cpu")
        for layer in self._lm.layers:
            layer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def _transformer_on_cuda(self):
        try:
            return next(self.transformer.parameters()).device.type == "cuda"
        except StopIteration:
            return False

    def _strict_budget(self):
        return self.resident and self.budget_gb < (self._tf_gpu_bytes / GB + 1.45 + 0.35)

    def _ensure_transformer_on_cuda(self):
        if self.resident and not self._transformer_on_cuda():
            self.transformer.to("cuda")
            torch.cuda.empty_cache()

    def _decode_with_vae_phase(self, *args, **kwargs):
        torch.cuda.synchronize()
        self._denoise_peak_mb = max(self._denoise_peak_mb,
                                    torch.cuda.max_memory_allocated() / 1024 ** 2)
        t0 = time.time()
        if self._strict_budget() and self._transformer_on_cuda():
            # 5 GB profili: VAE decode sırasında transformer CPU'da bekler; bir sonraki
            # denoise başında geri taşınır. Böylece aynı anda tek faz VRAM'de olur.
            self.transformer.to("cpu")
            torch.cuda.empty_cache()
            self._last_evict = True
        self.vae.to("cuda")
        torch.cuda.reset_peak_memory_stats()
        try:
            out = self._orig_vae_decode(*args, **kwargs)
            torch.cuda.synchronize()
            self._decode_peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            self._decode_time_s = time.time() - t0
            return out
        finally:
            self.vae.to("cpu")
            torch.cuda.empty_cache()

    def _warmup(self):
        print("[qwen] warmup...", flush=True)
        emb = torch.randn(1, 8, 4096, dtype=torch.bfloat16, device="cuda") * 0.05
        try:
            with torch.inference_mode():
                self.pipe(
                    prompt=None, prompt_embeds=emb, true_cfg_scale=1.0,
                    height=256, width=256, num_inference_steps=2,
                    generator=torch.Generator("cpu").manual_seed(0),
                    use_kv_cache=True, output_type="pil",
                )
        finally:
            emb = None
            gc.collect()
            torch.cuda.empty_cache()
        self._encode_peak_mb = 0.0
        self._denoise_peak_mb = 0.0
        self._decode_peak_mb = 0.0
        self._step_peak_mb = 0.0

    # ------------------------------------------------------------------ encode

    def _encode(self, prompt):
        if self._strict_budget() and self._transformer_on_cuda():
            self.transformer.to("cpu")
            torch.cuda.empty_cache()
            self._last_evict = True
        self._te_on()
        try:
            with torch.inference_mode():
                pos, mask, _ = self.pipe.encode_prompt(prompt=prompt, image=None, device="cuda")
            return pos, mask
        finally:
            self._te_off()

    def prompt_embeds(self, prompt, on_status=None):
        key = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        hit = self._emb_cache.get(key)
        if hit is not None:
            pos, mask = hit
            return pos.to("cuda", non_blocking=True), (mask.to("cuda", non_blocking=True) if mask is not None else None), True
        if on_status:
            on_status("Prompt katman akışıyla kodlanıyor...")
        pos, mask = self._encode(prompt)
        self._emb_cache[key] = (pos.to("cpu"), mask.to("cpu") if mask is not None else None)
        return pos, mask, False

    # --------------------------------------------------------------- generate

    def generate(self, prompt, style_key="matte_chiaroscuro", aspect_ratio="16:9",
                 quality="1080p", seed=None, enable_crpf=True, step_callback=None,
                 status_callback=None, enable_geo=True, steps_override=None,
                 refine_steps=0, refine_sigma=0.15, sigmas=None):
        if seed is None or seed < 0:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()

        final_prompt = compile_physically_grounded_prompt(
            prompt, style_key, enable_geo, include_anatomy=False,
        )
        base_w, base_h, target_w, target_h, steps = qwen_dimensions(
            quality, aspect_ratio, steps_override,
        )
        self._last_evict = False

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        pos, mask, cache_hit = self.prompt_embeds(final_prompt, status_callback)
        t_encode = time.time() - t0
        encode_peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        self._encode_peak_mb = max(self._encode_peak_mb, encode_peak_mb)

        torch.cuda.reset_peak_memory_stats()
        self._denoise_peak_mb = 0.0
        self._decode_peak_mb = 0.0
        self._decode_time_s = 0.0
        self._step_peak_mb = 0.0
        self._ensure_transformer_on_cuda()
        marks = []
        t_den = time.time()

        def cb(pipe_obj, i, t, kwargs):
            marks.append(time.time())
            self._step_peak_mb = max(self._step_peak_mb, torch.cuda.max_memory_allocated() / 1024 ** 2)
            if step_callback:
                elapsed = time.time() - t_den
                step_callback({
                    "type": "step_progress",
                    "step": i + 1,
                    "total_steps": steps,
                    "percent": int((i + 1) / steps * 100),
                    "elapsed": round(elapsed, 2),
                    "it_s": round((i + 1) / max(elapsed, 1e-6), 2),
                    "vram_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
                    "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
                    "preview_b64": None,
                })
            return kwargs

        with torch.inference_mode():
            out = self.pipe(
                prompt=None,
                prompt_embeds=pos,
                prompt_embeds_mask=mask,
                true_cfg_scale=1.0,
                height=base_h,
                width=base_w,
                num_inference_steps=steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                use_kv_cache=True,
                output_type="pil",
                callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        torch.cuda.synchronize()
        t_gen = time.time() - t_den
        image = out.images[0]
        pos = mask = None
        gc.collect()
        torch.cuda.empty_cache()

        if (target_w, target_h) != (base_w, base_h):
            from PIL import Image
            # Lanczos'un negatif lobu yumuşak kaynağa uygulanınca kenarlarda halka yapıyor;
            # büyütmede lobsuz bikübik, küçültmede Lanczos kalır.
            growing = target_w > base_w or target_h > base_h
            kernel = Image.Resampling.BICUBIC if growing else Image.Resampling.LANCZOS
            image = image.resize((target_w, target_h), kernel)
        total = time.time() - t0

        peak_mb = max(encode_peak_mb, self._denoise_peak_mb, self._decode_peak_mb)
        intervals = [marks[i] - marks[i - 1] for i in range(1, len(marks))]
        first_step = max(t_gen - self._decode_time_s - sum(intervals), 0.0)
        step_times_ms = [int(round(first_step * 1000))] + [int(round(d * 1000)) for d in intervals]

        ts = time.strftime("%Y%m%d_%H%M%S")
        clean = "".join(c for c in prompt[:25] if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        filename = f"qwen_{ts}_{clean}_s{seed}.png"
        filepath = os.path.join(GALLERY_DIR, filename)
        image.save(filepath, format="PNG")

        preset = STYLE_PRESETS.get(style_key, STYLE_PRESETS["matte_chiaroscuro"])
        step_s = (marks[-1] - marks[0]) / max(1, len(marks) - 1) if len(marks) > 1 else t_gen
        telemetry = {
            "filename": filename,
            "filepath": filepath,
            "image_url": f"/images/{filename}",
            "seed": seed,
            "quality": quality,
            "aspect_ratio": aspect_ratio,
            "style_name": preset["name"],
            "resolution": f"{target_w}x{target_h}",
            "native_resolution": f"{base_w}x{base_h}",
            "vae_blend_px": self.vae_blend_px,
            "steps": steps,
            "refine_steps": 0,
            "refine_sigma": 0.0,
            "step_times_ms": step_times_ms,
            "total_time": round(total, 2),
            "encode_time": round(t_encode, 2),
            "prompt_cached": cache_hit,
            "swapped": bool(self._last_evict),
            "swap_time": 0.0,
            "cache_entries": len(self._emb_cache),
            "step_time": round(step_s, 2),
            "decode_time": round(self._decode_time_s, 2),
            "it_s": round(steps / max(t_gen, 1e-6), 2),
            "kpx_s": round((target_w * target_h / 1000.0) / max(total, 1e-6), 1),
            "peak_vram_mb": round(peak_mb, 1),
            "encode_peak_mb": round(encode_peak_mb, 1),
            "denoise_peak_mb": round(self._denoise_peak_mb, 1),
            "step_peak_mb": round(self._step_peak_mb, 1),
            "decode_peak_mb": round(self._decode_peak_mb, 1),
            "mode": "qwen",
            "backend": ("NF4 yerleşik · 4-bit" if self.resident
                        else f"NF4+FP8 akış · grup {GROUP_SIZE}"),
            "precision": ("NF4 transformer / NF4 metin kodlayıcı / BF16 VAE" if self.resident
                          else "FP8 transformer / NF4 metin kodlayıcı / BF16 VAE"),
            "group_size": 0 if self.resident else GROUP_SIZE,
            "resident_mode": self.resident,
            "weights_gpu_mb": round(self._tf_gpu_bytes / 1024 ** 2, 1) if self.resident else 0.0,
            "base_mp": round(base_w * base_h / 1e6, 3),
        }
        save_sidecar(filename, {
            "prompt": prompt,
            "final_prompt": final_prompt,
            "style_key": style_key,
            "style_name": preset["name"],
            "aspect_ratio": aspect_ratio,
            "quality": quality,
            "seed": seed,
            "steps": steps,
            "resolution": telemetry["resolution"],
            "native_resolution": telemetry["native_resolution"],
            "vae_blend_px": telemetry["vae_blend_px"],
            "total_time": telemetry["total_time"],
            "peak_vram_mb": telemetry["peak_vram_mb"],
            "enable_geo": enable_geo,
            "enable_crpf": False,
            "quant": "nf4-resident" if self.resident else "fp8-tf/nf4-te",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        return telemetry

    # ----------------------------------------------------------------- system

    def system_info(self):
        vae_bytes = _bytes(self.vae)
        te_bytes = _bytes(self.text_encoder)
        tf_bytes = self._tf_stream_bytes
        tf_on_gpu = self._transformer_on_cuda() if self.resident else False
        return {
            "engine": "qwen",
            "engine_label": ("Qwen-Image 2.1 · 4-bit yerleşik" if self.resident
                             else "Qwen-Image 2.1 · akıtmalı"),
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / GB, 2),
            "vram_free_gb": round(torch.cuda.mem_get_info()[0] / GB, 2),
            "budget_gb": self.budget_gb,
            "native_fp8": not self.resident,
            "dual_resident": False,
            "resident_mode": self.resident,
            "resident_bits": 4 if self.resident else 0,
            "resident_skip": list(RESIDENT_SKIP) if self.resident else [],
            "resident_strict": self._strict_budget(),
            "weights_gpu_bytes": self._tf_gpu_bytes if self.resident else 0,
            "phase_peaks_mb": {
                "encode": round(self._encode_peak_mb, 1),
                "denoise": round(self._denoise_peak_mb, 1),
                "decode": round(self._decode_peak_mb, 1),
            },
            "modules": [
                {"id": "transformer", "name": "Qwen-Image 2.1 transformer", "layers": self.n_layers,
                 "bytes": tf_bytes, "resident": tf_on_gpu},
                {"id": "text_encoder", "name": "Qwen3-VL 8B (NF4)", "layers": self.n_te_layers,
                 "bytes": te_bytes, "resident": False},
                {"id": "vae", "name": "Qwen 64-ch VAE", "layers": 0,
                 "bytes": vae_bytes, "resident": False},
            ],
            "weights_bf16_bytes": self.weights_bf16_bytes,
            "weights_fp8_bytes": self.weights_stream_bytes,
            "cache_entries": len(self._emb_cache),
            "last_swap_s": 0.0,
            "group_size": 0 if self.resident else GROUP_SIZE,
            "streaming": ("faz sıralı · 4-bit yerleşik (blok akışı yok)" if self.resident
                          else "katman/katman + blok grubu"),
            "tiers": QWEN_TIERS,
            "auto_steps": auto_steps_map(),
            "dims": dims_map(),
        }

    def clear_cache(self):
        count = len(self._emb_cache)
        self._emb_cache.clear()
        return count
