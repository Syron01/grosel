"""
Grosel fast engine — FP8 resident inference path.

Neden hizli:
  * Tum Linear katmanlari FP8 (e4m3) olarak tutulur; her adimda PCIe uzerinden 7.7 GB
    agirlik tasinmaz (eski yol: sequential_cpu_offload).
  * Text encoder GPU'da calisir. CPU'da 512 token prefill 80 saniye suruyordu.
  * Iki model 8 GB'a sigmadigi icin encode sirasinda takas yapilir; takas pinli
    bellekten yapildigi icin ~0.3 sn (pageable'da ~1.3 sn).
  * Prompt embedding'leri prompt basina cache'lenir: ayni prompt + yeni seed = sifir encode.
  * VRAM 16 GB+ ise hicbir takas yapilmaz, iki model de VRAM'de kalir.
"""

import base64
import gc
import hashlib
import io
import os
import time

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
from PIL import Image

from web_app import quality_math
from web_app.config import FLUX_MODEL, GALLERY_DIR, SR_MAX_MP, SR_MAX_SCALE
from web_app.fp8_layers import convert_to_fp8, convert_to_fp8_native
from web_app.resolutions import detail_target

GB = 1024 ** 3


def free_vram_gb():
    return torch.cuda.mem_get_info()[0] / GB


class PinnedMover:
    """Bir modulu GPU <-> pinned CPU arasinda tahsis yapmadan tasir."""

    def __init__(self, module):
        self.module = module
        self.slots = []  # (tensor_ref, pinned_cpu_copy)
        for _, p in module.named_parameters():
            self.slots.append((p, p.detach().to("cpu").pin_memory()))
        for _, b in module.named_buffers():
            if b.is_floating_point() or b.dtype in (torch.float8_e4m3fn, torch.uint8, torch.int64, torch.int32):
                self.slots.append((b, b.detach().to("cpu").pin_memory()))

    def to_cpu(self):
        for t, host in self.slots:
            if t.is_cuda:
                host.copy_(t.detach(), non_blocking=True)
                t.data = host
        torch.cuda.synchronize()

    def to_gpu(self):
        for t, host in self.slots:
            if not t.is_cuda:
                dev = torch.empty_like(host, device="cuda")
                dev.copy_(host, non_blocking=True)
                t.data = dev
        torch.cuda.synchronize()

    def bytes(self):
        return sum(h.numel() * h.element_size() for _, h in self.slots)


class FastKleinEngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, native_fp8=None):
        if native_fp8 is None:
            native_fp8 = os.environ.get("HYPERFLUX_NATIVE_FP8", "1") == "1"

        torch.cuda.empty_cache()
        gc.collect()

        self.pipe = Flux2KleinPipeline.from_pretrained(
            FLUX_MODEL, torch_dtype=torch.bfloat16, local_files_only=True
        )
        self.transformer = self.pipe.transformer
        self.text_encoder = self.pipe.text_encoder
        self.vae = self.pipe.vae
        self.weights_bf16_bytes = (self._module_bytes(self.transformer)
                                   + self._module_bytes(self.text_encoder))

        convert = convert_to_fp8_native if native_fp8 else convert_to_fp8
        self.n_layers = convert(self.transformer)
        self.n_te_layers = convert(self.text_encoder)
        self.native_fp8 = native_fp8

        self.vae.to("cuda")
        # FLUX2'nin diffusers karo API'si argsiz: enable_tiling(**kwargs) motoru
        # acilista cokariyordu. Sozlesme artik VAE'nin kendi yuzeyine kuruluyor ve
        # kurulusan onusme telemetriye FIILEN deger olarak yaziliyor.
        self.vae_blend_px = quality_math.apply_tiling(self.vae)
        self.vae.enable_slicing()
        self.mover_tf = PinnedMover(self.transformer)
        self.mover_te = PinnedMover(self.text_encoder)

        self._emb_cache: dict[str, torch.Tensor] = {}
        self._te_resident = False
        self._tf_resident = False
        self.mover_tf.to_gpu()
        self._tf_resident = True

        self.budget_gb = round(free_vram_gb(), 2)
        self._dual_resident = free_vram_gb() >= (self.mover_te.bytes() / GB + 0.9)
        self.last_swap_time = 0.0
        self.last_swapped = False
        self._decode_peak_mb = 0.0
        self._decode_time_s = 0.0
        self._decode_offload_s = 0.0
        self._expected_pixels = 0

        # VAE decode, tepe VRAM'in en yüksek olduğu faz. Decode sırasında transformer'ı
        # CPU'ya alarak 1080p tepesini ~5.0 GB'tan ~4.4-4.9 GB'a indiriyoruz.
        # Küçük çözünürlüklerde decode zaten tepe yapmadığı için offload atlanır.
        self._orig_vae_decode = self.vae.decode

        def _decode_offload(*args, **kwargs):
            if self._expected_pixels and self._expected_pixels < 350_000:
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                out = self._orig_vae_decode(*args, **kwargs)
                torch.cuda.synchronize()
                self._decode_peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
                self._decode_time_s = time.time() - t0
                self._decode_offload_s = 0.0
                return out

            t_down = time.time()
            self._tf_down()
            t_ready = time.time()
            torch.cuda.reset_peak_memory_stats()
            try:
                out = self._orig_vae_decode(*args, **kwargs)
                torch.cuda.synchronize()
                self._decode_peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
                t_decoded = time.time()
                self._decode_time_s = t_decoded - t_ready
                return out
            finally:
                self._tf_up()
                self._decode_offload_s = (t_ready - t_down) + (time.time() - (t_ready + self._decode_time_s))

        self.vae.decode = _decode_offload
        self._warmup()
        print(
            f"[fast] FP8 lineer tf={self.n_layers} te={self.n_te_layers} native_mm={self.native_fp8} "
            f"dual_resident={self._dual_resident} | serbest VRAM {free_vram_gb():.2f} GB"
        )

    @staticmethod
    def _module_bytes(module):
        total = sum(p.numel() * p.element_size() for p in module.parameters())
        total += sum(b.numel() * b.element_size() for b in module.buffers() if b.is_floating_point())
        return total

    def system_info(self):
        vae_bytes = self._module_bytes(self.vae)
        return {
            "engine": "fast",
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / GB, 2),
            "vram_free_gb": round(free_vram_gb(), 2),
            "budget_gb": self.budget_gb,
            "native_fp8": self.native_fp8,
            "dual_resident": self._dual_resident,
            "modules": [
                {"id": "transformer", "name": "FLUX.2 Klein transformer", "layers": self.n_layers,
                 "bytes": self.mover_tf.bytes(), "resident": self._tf_resident},
                {"id": "text_encoder", "name": "Qwen3 text encoder", "layers": self.n_te_layers,
                 "bytes": self.mover_te.bytes(), "resident": self._te_resident},
                {"id": "vae", "name": "VAE decoder", "layers": 0,
                 "bytes": vae_bytes, "resident": True},
            ],
            "weights_bf16_bytes": self.weights_bf16_bytes,
            "weights_fp8_bytes": self.mover_tf.bytes() + self.mover_te.bytes(),
            "cache_entries": len(self._emb_cache),
            "last_swap_s": self.last_swap_time,
        }

    def clear_cache(self):
        count = len(self._emb_cache)
        self._emb_cache.clear()
        return count

    def _warmup(self):
        """Iki model ayni anda sigmadigi icin isinma sirali yapilir."""
        print("[fast] Warmup...")
        if self._dual_resident:
            self._te_up()
            emb = self._encode("warmup prompt")
        else:
            self._tf_down()
            self._te_up()
            emb = self._encode("warmup prompt")
            self._te_down()
            self._tf_up()
        self._generate_steps(emb, 512, 512, 1, 0)
        self._emb_cache.clear()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    def _encode(self, prompt):
        with torch.inference_mode():
            emb, _ = self.pipe.encode_prompt(prompt, device=torch.device("cuda"), max_sequence_length=512)
        return emb

    def _tf_up(self):
        if not self._tf_resident:
            self.mover_tf.to_gpu()
            self._tf_resident = True

    def _tf_down(self):
        if self._tf_resident:
            self.mover_tf.to_cpu()
            self._tf_resident = False
            torch.cuda.empty_cache()

    def _te_up(self):
        if not self._te_resident:
            self.mover_te.to_gpu()
            self._te_resident = True

    def _te_down(self):
        if self._te_resident:
            self.mover_te.to_cpu()
            self._te_resident = False
            torch.cuda.empty_cache()

    @staticmethod
    def _key(prompt):
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()

    def prompt_embeds(self, prompt, on_status=None):
        key = self._key(prompt)
        self.last_swapped = False
        self.last_swap_time = 0.0
        hit = self._emb_cache.get(key)
        if hit is not None:
            return hit.to("cuda", non_blocking=True), True

        if on_status:
            on_status("Prompt kodlanıyor...")
        if self._dual_resident:
            self._te_up()
            emb = self._encode(prompt)
            self._emb_cache[key] = emb.to("cpu", non_blocking=True)
            return emb, False

        t0 = time.time()
        self._tf_down()
        t1 = time.time()
        self._te_up()
        emb = self._encode(prompt)
        self._emb_cache[key] = emb.to("cpu", non_blocking=True)
        t3 = time.time()
        self._te_down()
        t4 = time.time()
        self._tf_up()
        self.last_swap_time = round((t1 - t0) + (t4 - t3), 2)
        self.last_swapped = True
        return emb, False

    def _generate_steps(self, prompt_embeds, width, height, steps, seed, step_callback=None,
                        enable_crpf=True, sigmas=None, output_type="pil", capture=False):
        from web_app.engine import crpf_regularize_latents

        generator = torch.Generator("cpu").manual_seed(seed)
        t0 = time.time()
        marks = []
        lh = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        lw = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))
        h_half, w_half = lh // 2, lw // 2

        def cb(pipe_obj, i, t, kwargs):
            marks.append(time.time())
            latents = kwargs.get("latents")
            if enable_crpf and latents is not None:
                kwargs["latents"] = crpf_regularize_latents(latents, i, h_half, w_half)
                latents = kwargs["latents"]
            if capture and latents is not None:
                self._captured_latents = latents.detach().clone()
            if step_callback:
                step_callback({
                    "type": "step_progress",
                    "step": i + 1,
                    "total_steps": steps,
                    "percent": int((i + 1) / steps * 100),
                    "elapsed": round(time.time() - t0, 2),
                    "it_s": round((i + 1) / max(time.time() - t0, 1e-6), 2),
                    "vram_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
                    "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
                    "preview_b64": preview_from_latents(latents, height, width, pipe_obj) if latents is not None else None,
                })
            self._step_peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
            return kwargs

        orig_scheduler = self.pipe.scheduler
        if sigmas is not None:
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                orig_scheduler.config, use_dynamic_shifting=False, shift=1.0
            )
        try:
            with torch.inference_mode():
                out = self.pipe(
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    num_inference_steps=steps,
                    sigmas=sigmas,
                    width=width,
                    height=height,
                    generator=generator,
                    output_type=output_type,
                    callback_on_step_end=cb,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
        finally:
            if sigmas is not None:
                self.pipe.scheduler = orig_scheduler
        torch.cuda.synchronize()
        end = time.time()
        self._last_decode = end - marks[-1] if marks else 0.0
        image = out.images if output_type == "latent" else out.images[0]
        return image, end - t0, marks

    def _generate_refined(self, prompt_embeds, width, height, steps, seed, step_callback,
                          refine_steps, refine_sigma, enable_crpf):
        """Ana yörüngeyi aynı ızgarada düşük gürültüde yeniden örnekler."""
        from web_app.engine import crpf_regularize_latents

        generator = torch.Generator("cpu").manual_seed(seed)
        t0 = time.time()
        marks = []
        lh = 2 * (int(height) // (self.pipe.vae_scale_factor * 2))
        lw = 2 * (int(width) // (self.pipe.vae_scale_factor * 2))
        h_half, w_half = lh // 2, lw // 2
        total_steps = steps + refine_steps
        self._captured_latents = None

        def make_cb(offset, capture):
            def cb(pipe_obj, i, t, kwargs):
                marks.append(time.time())
                latents = kwargs.get("latents")
                if enable_crpf and latents is not None:
                    kwargs["latents"] = crpf_regularize_latents(latents, offset + i, h_half, w_half)
                    latents = kwargs["latents"]
                if capture and latents is not None:
                    self._captured_latents = latents.detach().clone()
                if step_callback:
                    step_callback({
                        "type": "step_progress",
                        "step": offset + i + 1,
                        "total_steps": total_steps,
                        "percent": int((offset + i + 1) / total_steps * 100),
                        "elapsed": round(time.time() - t0, 2),
                        "it_s": round((offset + i + 1) / max(time.time() - t0, 1e-6), 2),
                        "vram_mb": round(torch.cuda.memory_allocated() / 1024 ** 2, 1),
                        "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
                        "preview_b64": preview_from_latents(latents, height, width, pipe_obj) if latents is not None else None,
                    })
                self._step_peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
                return kwargs
            return cb

        with torch.inference_mode():
            self.pipe(
                prompt=None, prompt_embeds=prompt_embeds,
                num_inference_steps=steps, width=width, height=height,
                generator=generator, output_type="latent",
                callback_on_step_end=make_cb(0, True),
                callback_on_step_end_tensor_inputs=["latents"],
            )

        x0 = self._captured_latents
        channels = self.pipe.transformer.config.in_channels
        dummy = torch.zeros((1, channels, lh // 2, lw // 2), device=x0.device, dtype=x0.dtype)
        latent_ids = self.pipe._prepare_latent_ids(dummy).to(x0.device)
        x0_4d = self.pipe._unpack_latents_with_ids(x0, latent_ids, lh // 2, lw // 2)

        refine_steps = max(1, min(4, int(refine_steps)))
        refine_sigma = max(0.02, min(0.60, float(refine_sigma)))
        sigmas = [refine_sigma * (1 - i / refine_steps) for i in range(refine_steps)]
        noise = torch.randn(x0_4d.shape, generator=generator, dtype=x0_4d.dtype).to(x0_4d.device)
        x_r = (1.0 - sigmas[0]) * x0_4d + sigmas[0] * noise

        orig_scheduler = self.pipe.scheduler
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            orig_scheduler.config, use_dynamic_shifting=False, shift=1.0
        )
        try:
            with torch.inference_mode():
                out = self.pipe(
                    prompt=None, prompt_embeds=prompt_embeds,
                    latents=x_r, sigmas=sigmas, width=width, height=height,
                    generator=generator, output_type="pil",
                    callback_on_step_end=make_cb(steps, False),
                    callback_on_step_end_tensor_inputs=["latents"],
                )
        finally:
            self.pipe.scheduler = orig_scheduler

        torch.cuda.synchronize()
        end = time.time()
        self._last_decode = end - marks[-1] if marks else 0.0
        self._last_refine_steps = refine_steps
        self._last_refine_sigma = refine_sigma
        return out.images[0], end - t0, marks

    def generate(self, prompt, style_key="matte_chiaroscuro", aspect_ratio="16:9", quality="1080p",
                 seed=None, enable_crpf=True, step_callback=None, status_callback=None,
                 enable_geo=True, steps_override=None, refine_steps=0, refine_sigma=0.15,
                 sigmas=None, detail_pass=True, detail_max_mp=None):
        from web_app.engine import (
            STYLE_PRESETS,
            compile_physically_grounded_prompt,
            get_dimensions,
            save_sidecar,
        )

        if seed is None or seed < 0:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()

        final_prompt = compile_physically_grounded_prompt(prompt, style_key, enable_geo)
        base_w, base_h, target_w, target_h, steps, upscale = get_dimensions(quality, aspect_ratio)
        if steps_override:
            steps = max(1, min(20, int(steps_override)))

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        emb, cache_hit = self.prompt_embeds(final_prompt, status_callback)
        t_encode = time.time() - t0
        encode_peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        torch.cuda.reset_peak_memory_stats()
        self._expected_pixels = target_w * target_h
        self._decode_peak_mb = 0.0
        self._decode_time_s = 0.0
        self._decode_offload_s = 0.0
        self._step_peak_mb = 0.0
        self._last_refine_steps = 0
        self._last_refine_sigma = 0.0
        self._last_sr_scale = 1.0

        refine_steps = max(0, min(4, int(refine_steps or 0)))
        # Detay geçişi artık üretim ızgarasını büyütüyor: model detayı kendi ızgarasında
        # çiziyor. Latent'i sonradan yeniden örnekleme tamsayı-olmayan oranlarda faz
        # dayaklanması yapıp kenarlarda ghosting üretiyordu (ölçüldü: halo 0.33→0.52).
        # SR_MAX_MP hırsı ölçülmüş tavan eşitliğiyle budanır: 1–5 GB sözü kademe
        # adından değil, çizilen ızgaradan korunur.
        max_mp = min(SR_MAX_MP, detail_max_mp) if detail_max_mp else SR_MAX_MP
        gen_size = detail_target(base_w, base_h, target_w, target_h,
                                 max_scale=SR_MAX_SCALE, max_mp=max_mp,
                                 enabled=detail_pass)
        gen_w, gen_h = gen_size or (base_w, base_h)
        self._last_sr_scale = round(max(gen_w, gen_h) / max(base_w, base_h), 2)
        if refine_steps:
            image, t_gen, marks = self._generate_refined(
                emb, gen_w, gen_h, steps, seed, step_callback,
                refine_steps, refine_sigma, enable_crpf,
            )
        else:
            image, t_gen, marks = self._generate_steps(
                emb, gen_w, gen_h, steps, seed, step_callback,
                enable_crpf, sigmas=sigmas,
            )
        step_peak_mb = getattr(self, "_step_peak_mb", 0.0)
        decode_peak_mb = self._decode_peak_mb
        peak_mb = max(encode_peak_mb, step_peak_mb, decode_peak_mb)
        decode_time = self._decode_time_s or getattr(self, "_last_decode", 0.0)
        decode_offload = self._decode_offload_s
        decode = getattr(self, "_last_decode", 0.0)
        intervals = [marks[i] - marks[i - 1] for i in range(1, len(marks))]
        first_step = max(t_gen - decode - sum(intervals), 0.0)
        step_times_ms = [round(first_step * 1000)] + [round(d * 1000) for d in intervals]
        if upscale and (target_w, target_h) != (gen_w, gen_h):
            # Lanczos'un negatif lobu keskin kaynağa uygulanınca kenarlarda halka yapıyor;
            # büyütmede lobsuz bikübik, küçültmede Lanczos kalır.
            growing = target_w > gen_w or target_h > gen_h
            kernel = Image.Resampling.BICUBIC if growing else Image.Resampling.LANCZOS
            image = image.resize((target_w, target_h), kernel)
        total = time.time() - t0

        ts = time.strftime("%Y%m%d_%H%M%S")
        clean = "".join(c for c in prompt[:25] if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        filename = f"hyper_{ts}_{clean}_s{seed}.png"
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
            "native_resolution": f"{gen_w}x{gen_h}",
            "vae_blend_px": self.vae_blend_px,
            "detail_scale": round(getattr(self, "_last_sr_scale", 1.0), 2),
            "detail_max_mp": max_mp,
            "steps": steps,
            "refine_steps": getattr(self, "_last_refine_steps", 0),
            "refine_sigma": round(getattr(self, "_last_refine_sigma", 0.0), 3),
            "step_times_ms": step_times_ms,
            "total_time": round(total, 2),
            "encode_time": round(t_encode, 2),
            "prompt_cached": cache_hit,
            "swapped": self.last_swapped,
            "swap_time": round(self.last_swap_time + decode_offload, 2),
            "cache_entries": len(self._emb_cache),
            "step_time": round(step_s, 2),
            "decode_time": round(decode_time, 2),
            "it_s": round(steps / max(t_gen, 1e-6), 2),
            "kpx_s": round((target_w * target_h / 1000.0) / max(total, 1e-6), 1),
            "peak_vram_mb": round(peak_mb, 1),
            "encode_peak_mb": round(encode_peak_mb, 1),
            "denoise_peak_mb": round(step_peak_mb, 1),
            "step_peak_mb": round(step_peak_mb, 1),
            "decode_peak_mb": round(decode_peak_mb, 1),
            "mode": "fast",
            "backend": "FP8-native" if self.native_fp8 else "FP8-resident",
            "precision": "FP8 e4m3 weights / BF16 compute",
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
            "refine_steps": telemetry["refine_steps"],
            "refine_sigma": telemetry["refine_sigma"],
            "resolution": telemetry["resolution"],
            "native_resolution": telemetry["native_resolution"],
            "vae_blend_px": telemetry["vae_blend_px"],
            "total_time": telemetry["total_time"],
            "peak_vram_mb": telemetry["peak_vram_mb"],
            "enable_geo": enable_geo,
            "enable_crpf": enable_crpf,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        return telemetry


def preview_from_latents(latents, height, width, pipe):
    try:
        with torch.no_grad():
            lh = 2 * (int(height) // (pipe.vae_scale_factor * 2))
            lw = 2 * (int(width) // (pipe.vae_scale_factor * 2))
            patches = latents.view(1, lh // 2, lw // 2, 128).permute(0, 3, 1, 2)
            unpatch = pipe._unpatchify_latents(patches)
            rgb = unpatch[0, :3].float().cpu().numpy().transpose(1, 2, 0)
            p_min, p_max = np.percentile(rgb, 2), np.percentile(rgb, 98)
            rgb = np.clip((rgb - p_min) / (p_max - p_min + 1e-5), 0, 1) * 255.0
            img = Image.fromarray(rgb.astype(np.uint8))
            img = img.resize((max(128, width // 4), max(72, height // 4)), Image.Resampling.BILINEAR)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=65)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None
