"""Grosel classic engine — BF16 weights with sequential CPU offload.

This module is also the shared library for all three backends: style presets,
the geometric prompt compiler, the CRPF latent regularizer and sidecar output.
"""

import base64
import io
import os
import re
import time

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from PIL import Image, ImageFilter

from web_app import quality_math
from web_app.config import FLUX_MODEL, GALLERY_DIR, META_DIR
from web_app.resolutions import get_dimensions
from web_app.storage import write_json_atomic

GB = 1024 ** 3


def save_sidecar(filename: str, data: dict) -> None:
    """Görseli yeniden üretilebilir kılan ayarları görselin yanına yazar.

    Atomik yazım: sunucu üretim sırasında öldürülürse yarım JSON kalmaz, kalite
    denetimi de o karenin yanına temiz veriye yazar.
    """
    try:
        os.makedirs(META_DIR, exist_ok=True)
        write_json_atomic(os.path.join(META_DIR, f"{filename}.json"), data)
    except OSError as e:
        print(f"[meta] Uyarı: {e}")

ANATOMY_ANCHOR = (
    "anatomically flawless hands with exactly five distinct slender fingers, authentic non-plastic skin texture with visible organic pores, "
    "coherent natural facial symmetry, zero distorted digits, zero fused fingers, zero extra limbs, zero floating artifacts"
)

# Alıntıdaki kelime yoksa (sadece "tabela" gibi bir istek geçiyorsa) kullanılan genel çapa.
TEXT_ANCHOR = (
    'render the quoted word exactly as written, in clean continuous letterforms with correct spelling, '
    'no extra letters, no mirrored or broken glyphs, no other text anywhere else in the scene'
)
# Kelime hecelendiğinde kuyruğa eklenen kısa pekiştirme; heceleme satır içine gömülür.
TEXT_TAIL = (
    "clean continuous letterforms with correct spelling, crisp separated glyphs with sharp focus "
    "throughout, no mirrored or broken glyphs, no other text anywhere else in the scene"
)


def spelling_anchor(quoted: str) -> str:
    """Alıntıdaki kelimeyi harf harfe çözüp satır içi heceleme yönergesi üretir.

    Ölçüm: FLUX.2 Klein 4B, 1280x720, aynı seed ile "OpenRouter" yerine
    "OpenRougter/OpenRoutfer" çiziyordu; tırnağın hemen ardına eklenen
    "letters O p e n R o u t e r in that order" yönergesi yazıyı sıfır ek süre
    ve sıfır VRAM ile doğru bastı. Uzun metinlerde token şişmesin diye boşa döner.
    """
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", quoted)
    letters = sum(len(word) for word in words)
    spelled = ", then ".join(" ".join(word) for word in words)
    if not spelled or letters > 24:
        return ""
    if len(words) == 1:
        return f"(letters {spelled} in that order, one unbroken word, no hyphen, no space)"
    return f"(letters {spelled} in that order, no hyphen, no extra punctuation)"

# Anatomi çapası yalnızca sahnede insan/vücut geçiyorsa eklenir; manzaraya eklenince
# model promptta olmayan dev eller üretiyordu.
_ANATOMY_RE = re.compile(
    r"(?<![\wçğıöşü\-])("
    r"el|eli|elin|elinde|eller|elleri|elle|elde|"
    r"parmak\w*|yüz|portre\w*|"
    r"hand|hands|finger\w*|portrait\w*|"
    r"man|woman|women|person|people|human|figure|figures|body|face|"
    r"insan\w*|adam\w*|kadın\w*|kız\w*|erkek\w*|çocuk\w*|yaşlı\w*|genç\w*|"
    r"usta\w*|saatçi\w*|anatom\w*|vücut\w*|figür\w*|"
    r"watchmaker\w*|craftsman\w*|artisan\w*"
    r")(?![\wçğıöşü\-])",
    re.IGNORECASE | re.UNICODE,
)

# "quiet street with no people" gibi olumsuz özneler çapa tetiklemesin: çapa
# sahnede o şeyin varlığından çıkar, yokluğundan değil.
_PERSONS = (r"people|persons?|humans?|figures?|crowds?|pedestrians?|passersby"
            r"|insan(?:lar)?|kişi(?:ler)?|adam(?:lar)?|kadın(?:lar)?|çocuk(?:lar)?")
_NEGATED_SUBJECT_RE = re.compile(
    rf"(?:(?:\bno\b|\bzero\b|\bwithout\b|\bnot\s+any\b|\bdevoid\s+of\b)[^.,;]{{0,24}}?(?:{_PERSONS})"
    rf"|(?:{_PERSONS})\s+(?:yok|bulunmaz|mevcut\s+değil|olmayan|olmaksızın|bulunmayan))\b",
    re.IGNORECASE | re.UNICODE,
)

# Kullanıcı sahne içinde okunur yazı istiyor mu: tırnaklı kelime ya da doğrudan
# yazı/eğretme sözcüğü. "zero legible text" çapası ancak bu durumda susturulur.
_QUOTED_WORD_RE = re.compile(r"[\"“”'']\s*[A-Za-zÀ-ÿ0-9][^\"“”'']{1,40}[\"“”'']")
_TEXT_REQUEST_RE = re.compile(
    r"\b(neon\s+sign|shop\s+sign|storefront\s+sign|signboard|lettering|typography|headline|slogan|logo|banner"
    r"|poster|reads?|says?|spells?|tabela|tabelada|yazı|harfler|afiş|amblem)\b",
    re.IGNORECASE | re.UNICODE,
)

# Sahne özneleri: yalnız ilgili özne gerçekten geçiyorsa çapa eklenir. Eski sürüm
# "interior"/"room"/"ev" gibi geniş kelimelerle şömine çapasını uyduruyordu.
_FIREPLACE_RE = re.compile(r"\b(şömine|fireplace|hearth|chimney|mantel)\b", re.IGNORECASE | re.UNICODE)
_STREET_RE = re.compile(
    r"\b(sokak|cadde|bulvar|kaldırım|street|streets|avenue|boulevard|alley|city|şehir|downtown"
    r"|townsquare|town\s+square|storefront|storefronts|facade|rain|yağmur|wet\s+pavement)\b",
    re.IGNORECASE | re.UNICODE,
)
_WORKSHOP_RE = re.compile(
    r"\b(atölye|tezgah|workshop|workbench|watchmaker|clockmaker|blacksmith|forge|tamir|repair|craftsman|artisan)\b",
    re.IGNORECASE | re.UNICODE,
)
_MARITIME_RE = re.compile(
    r"\b(deniz|okyanus|gemi|tekne|kaptan|iskele|liman|sea|ocean|ship|boat|yacht|captain|pier|dock|harbor|harbour)\b",
    re.IGNORECASE | re.UNICODE,
)

STYLE_PRESETS = {
    "matte_chiaroscuro": {
        "name": "Mat Chiaroscuro",
        "desc": "Loş ortam, mat yağlıboya dokusu, dengeli gölgeler",
        "medium": "paint",
        "prefix": "Atmospheric concept art painting, digital illustration, matte texture, soft muted colors, low-key chiaroscuro lighting. ",
        "suffix": ", painterly textured finish, soft ambient shadow transitions"
    },
    "cinematic_35mm": {
        "name": "Sinematik 35mm Film",
        "desc": "Kodak Portra greni, organik alan derinliği",
        "medium": "photo",
        "prefix": "Cinematic 35mm film still, Kodak Portra 400 color grading, soft organic depth of field, authentic subtle film grain. ",
        "suffix": ", natural candid lighting, real micro-texture, 35mm anamorphic frame"
    },
    "anime_concept": {
        "name": "Premium Anime",
        "desc": "Parlak, glossy anime; altın ışık, parıltı, ışıldayan cilt",
        "medium": "anime",
        "prefix": "masterpiece, best quality, ",
        "suffix": ", ultra-detailed glossy anime illustration, luminous warm golden light, soft ethereal glow, strong golden rim light, sparkling dust motes, detailed iris highlights, glowing skin, glossy hair, bright airy atmosphere, pastel warm color grading, gentle bloom, shallow depth of field, clean elegant linework, delicate lace texture, zero plastic 3D look"
    },
    "raw_candid": {
        "name": "Ham Belgesel Fotoğraf",
        "desc": "Doğal kusurlar, sıfır filtre, gerçek insan dokusu",
        "medium": "photo",
        "prefix": "Unedited candid documentary photography, shot on 85mm f/1.8 lens, natural daylight. ",
        "suffix": ", zero AI smoothing, zero beauty filter, true realistic contrast"
    },
    "dark_baroque": {
        "name": "Karanlık Barok / Rembrandt",
        "desc": "Rembrandt derin gölgeleri, antika tuval patinası",
        "medium": "paint",
        "prefix": "Classic oil painting on canvas, 17th century Baroque tenebrism, dramatic Rembrandt chiaroscuro. ",
        "suffix": ", deep rich shadows, earthy umber and ochre palette, visible textured brush impasto, museum lighting"
    },
    "cyberpunk_noir": {
        "name": "Siberpunk Neon Noir",
        "desc": "Islak zemin, buğulu neon, blade runner atmosferi",
        "medium": "photo",
        "prefix": "Cinematic cyberpunk film noir, rainy dystopian night, muted wet pavement reflections. ",
        "suffix": ", soft diffused amber and cyan neon lights through mist, heavy film grain, dirty gritty urban realism"
    },
    "pure_raw": {
        "name": "Filtresiz (Ham Prompt)",
        "desc": "Yalnızca yazdığın prompt; hiçbir ek eklenmez",
        "medium": None,
        "prefix": "",
        "suffix": ""
    }
}


_MEDIUM_PATTERNS = (
    ("photo", re.compile(r"\b(photo|photograph|photographs|photographic|photography|photorealistic"
                         r"|photorealism|shot on|dslr|selfie|polaroid|film still|editorial)\b", re.I)),
    ("render", re.compile(r"\b(3d render|cgi|octane render|blender|unreal engine)\b", re.I)),
    ("anime", re.compile(r"\b(anime|manga|chibi|cel[- ]?shaded)\b", re.I)),
    ("paint", re.compile(r"\b(painting|painted|illustration|illustrated|watercolor|gouache|acrylic"
                         r"|oil on canvas|sketch|concept art|cartoon)\b", re.I)),
)


def detect_medium(prompt: str) -> str | None:
    """Kullanıcının kendisinin söylediği üretim ortamını döndürür (photo/render/anime/paint)."""
    scene = _NEGATED_SUBJECT_RE.sub(" ", prompt)
    for name, rx in _MEDIUM_PATTERNS:
        if rx.search(scene):
            return name
    return None


def compile_physically_grounded_prompt(user_prompt: str, style_key: str = "matte_chiaroscuro",
                                       enable_geo: bool = True, include_anatomy: bool = True) -> str:
    """
    Sahne çapalarını kullanıcı isteğiyle çelişmeden ekler.
    enable_geo=False iken yalnızca stil ön/son ekleri uygulanır.
    include_anatomy=False, anatomisi zaten güçlü modellerde çapayı atlar.
    """
    scene = _NEGATED_SUBJECT_RE.sub(" ", user_prompt)
    wants_text = bool(_QUOTED_WORD_RE.search(user_prompt) or _TEXT_REQUEST_RE.search(user_prompt))
    needs_anatomy = include_anatomy and bool(_ANATOMY_RE.search(scene))

    guards = []
    if enable_geo:
        if _STREET_RE.search(scene):
            guard = ("true linear one-point street perspective with a single vanishing point on the true horizon, "
                     "unbranded clean architectural storefronts")
            if not wants_text:
                guard += ", zero legible text, zero pseudo-writing, zero garbled signs"
            guards.append(guard)
        if _FIREPLACE_RE.search(scene):
            guards.append("exactly one primary fireplace, no duplicate fireplaces, no duplicate light sources, "
                          "coherent room geometry")
        if _WORKSHOP_RE.search(scene):
            guards.append("rigid mechanical components resting firmly on the workbench surface, "
                          "lamp light with physically accurate inverse-square falloff")
        if _MARITIME_RE.search(scene):
            guards.append("gravitationally grounded hull, perfectly flat sea horizon aligned with gravity")

    clauses = guards[:]
    inline_anchor = ""
    if wants_text and enable_geo:
        match = _QUOTED_WORD_RE.search(user_prompt)
        if match:
            inline_anchor = spelling_anchor(match.group(0))
        clauses.append(TEXT_TAIL if inline_anchor else TEXT_ANCHOR)
    if needs_anatomy:
        clauses.append("natural human leg-to-torso proportions")
        clauses.append(ANATOMY_ANCHOR)

    preset = STYLE_PRESETS.get(style_key, STYLE_PRESETS["matte_chiaroscuro"])
    # Kullanıcı promptu kendi ortamını adlandırdıysa ("editorial photograph") çelişen stil
    # on/son eki görseli boyam/anime tarafına çeker: varsayılan chip yüzünden fotoğraf
    # kaybetmeyi engelle.
    declared = detect_medium(user_prompt)
    use_style = declared is None or declared == preset["medium"]
    prefix, suffix = (preset["prefix"], preset["suffix"]) if use_style else ("", "")
    body = user_prompt.strip()
    if inline_anchor:
        match = _QUOTED_WORD_RE.search(body)
        body = f"{body[:match.end()]} {inline_anchor}{body[match.end():]}"
    compiled = prefix + body
    if clauses:
        compiled = f"{compiled.rstrip(',.')}, " + ", ".join(clauses)
    return compiled + suffix


def crpf_regularize_latents(latents: torch.Tensor, step: int, h_half: int, w_half: int) -> torch.Tensor:
    """Adım 0-1 çift obje modlarını, adım 2-3 plastik teni kırar."""
    try:
        with torch.no_grad():
            b, n, c = latents.shape
            z = latents.view(b, h_half, w_half, c)
            
            if step in (0, 1):
                coarse = z[:, :, :, :12]
                c_mean = coarse.mean(dim=(1, 2), keepdim=True)
                c_std = coarse.std(dim=(1, 2), keepdim=True) + 1e-5
                # Çift odak oluşmasını engelleyen kaba genlik sınırlayıcı
                z[:, :, :, :12] = torch.clamp(coarse, c_mean - 2.85 * c_std, c_mean + 2.85 * c_std)
                
            elif step in (2, 3):
                detail = z[:, :, :, 16:]
                laplacian = (
                    -4.0 * detail 
                    + torch.roll(detail, 1, 1) + torch.roll(detail, -1, 1) 
                    + torch.roll(detail, 1, 2) + torch.roll(detail, -1, 2)
                )
                # Doku kanallarına organik faz pertürbasyonu (yapay zeka pürüzsüzlüğünü kırar)
                z[:, :, :, 16:] = detail + 0.012 * torch.tanh(laplacian)
                
            return z.view(b, n, c)
    except Exception:
        return latents


def fast_latents_to_b64(latents, height, width, pipe):
    try:
        with torch.no_grad():
            lh = 2 * (int(height) // (pipe.vae_scale_factor * 2))
            lw = 2 * (int(width) // (pipe.vae_scale_factor * 2))
            h_half = lh // 2
            w_half = lw // 2
            
            if latents.ndim == 3:
                patches = latents.view(1, h_half, w_half, 128).permute(0, 3, 1, 2)
            else:
                patches = latents
                
            unpatch = pipe._unpatchify_latents(patches)
            
            ch0 = unpatch[0, 0].float().cpu().numpy()
            ch1 = unpatch[0, 1].float().cpu().numpy()
            ch2 = unpatch[0, 2].float().cpu().numpy()
            
            rgb = np.stack([ch0, ch1, ch2], axis=-1)
            p_min, p_max = np.percentile(rgb, 2), np.percentile(rgb, 98)
            rgb = np.clip((rgb - p_min) / (p_max - p_min + 1e-5), 0, 1) * 255.0
            
            img = Image.fromarray(rgb.astype(np.uint8))
            img = img.resize((max(128, width // 2), max(128, height // 2)), Image.Resampling.BILINEAR)
            img = img.filter(ImageFilter.GaussianBlur(radius=3))
            
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


class GroselEngine:
    _instance = None
    vae_blend_px = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = GroselEngine()
        return cls._instance

    def __init__(self):
        print("[-] Grosel classic motoru baslatiliyor...")
        torch.cuda.empty_cache()

        self.pipe = Flux2KleinPipeline.from_pretrained(
            FLUX_MODEL,
            torch_dtype=torch.bfloat16,
            local_files_only=True
        )

        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            self.pipe.vae.enable_slicing()
            self.vae_blend_px = quality_math.apply_tiling(self.pipe.vae)

        self.pipe.enable_sequential_cpu_offload()

        print("[-] Model Warmup yapiliyor...")
        _ = self.pipe(prompt="warmup", num_inference_steps=1, width=512, height=512)
        print("[OK] Grosel classic motoru hazir.")

    @staticmethod
    def _module_bytes(module):
        total = sum(p.numel() * p.element_size() for p in module.parameters())
        total += sum(b.numel() * b.element_size() for b in module.buffers() if b.is_floating_point())
        return total

    def system_info(self):
        modules = []
        for mid, name in (("transformer", "FLUX.2 Klein transformer"),
                          ("text_encoder", "Qwen3 text encoder"),
                          ("vae", "VAE decoder")):
            comp = getattr(self.pipe, mid, None)
            if comp is not None:
                modules.append({"id": mid, "name": name, "layers": 0,
                                "bytes": self._module_bytes(comp), "resident": False})
        weights = sum(m["bytes"] for m in modules)
        return {
            "engine": "classic",
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / GB, 2),
            "vram_free_gb": round(torch.cuda.mem_get_info()[0] / GB, 2),
            "budget_gb": None,
            "native_fp8": False,
            "dual_resident": False,
            "modules": modules,
            "weights_bf16_bytes": weights,
            "weights_fp8_bytes": 0,
            "cache_entries": 0,
            "last_swap_s": 0,
        }

    def clear_cache(self):
        return 0

    def generate(self, prompt: str, style_key: str = "matte_chiaroscuro", 
                 aspect_ratio: str = "16:9", quality: str = "1080p", 
                 seed: int | None = None, enable_crpf: bool = True, step_callback = None,
                 enable_geo: bool = True, steps_override=None):
        if seed is None or seed < 0:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        final_prompt = compile_physically_grounded_prompt(prompt, style_key, enable_geo)
        
        base_w, base_h, target_w, target_h, steps, upscale = get_dimensions(quality, aspect_ratio)
        if steps_override:
            steps = max(1, min(20, int(steps_override)))
        
        lh = 2 * (int(base_h) // (self.pipe.vae_scale_factor * 2))
        lw = 2 * (int(base_w) // (self.pipe.vae_scale_factor * 2))
        h_half, w_half = lh // 2, lw // 2

        t0 = time.time()
        step_times = []
        
        def internal_step_callback(pipe_obj, step, timestep, callback_kwargs):
            t_now = time.time()
            step_times.append(t_now)
            
            elapsed = t_now - t0
            current_it_s = len(step_times) / elapsed if elapsed > 0 else 0
            
            latents = callback_kwargs.get("latents")
            
            if enable_crpf and latents is not None:
                latents = crpf_regularize_latents(latents, step, h_half, w_half)
                callback_kwargs["latents"] = latents
                
            preview_b64 = None
            if latents is not None:
                preview_b64 = fast_latents_to_b64(latents, base_h, base_w, pipe_obj)
                
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            if step_callback:
                step_callback({
                    "type": "step_progress",
                    "step": step + 1,
                    "total_steps": steps,
                    "percent": int(((step + 1) / steps) * 100),
                    "elapsed": round(elapsed, 2),
                    "it_s": round(current_it_s, 2),
                    "vram_mb": round(vram_mb, 1),
                    "vram_peak_mb": round(vram_peak_mb, 1),
                    "preview_b64": preview_b64
                })
            return callback_kwargs

        with torch.inference_mode():
            result = self.pipe(
                prompt=final_prompt,
                num_inference_steps=steps,
                width=base_w,
                height=base_h,
                generator=generator,
                callback_on_step_end=internal_step_callback,
                callback_on_step_end_tensor_inputs=["latents"]
            )
            
        final_image = result.images[0]
        
        if upscale and (target_w != base_w or target_h != base_h):
            final_image = final_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
            final_image = final_image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=2))
            
        total_time = time.time() - t0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        avg_it_s = steps / total_time if total_time > 0 else 0
        
        total_pixels = target_w * target_h
        tps = (base_w * base_h / 256) / total_time if total_time > 0 else 0
        kpx_s = (total_pixels / 1000.0) / total_time if total_time > 0 else 0
        
        intervals = [(step_times[i] - step_times[i - 1]) * 1000 for i in range(1, len(step_times))]
        step_times_ms = [round(intervals[0])] + [round(x) for x in intervals] if intervals else []
        
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        clean_name = "".join(c for c in prompt[:25] if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        filename = f"hyper_{timestamp_str}_{clean_name}_s{seed}.png"
        filepath = os.path.join(GALLERY_DIR, filename)
        final_image.save(filepath, format="PNG")
        
        preset = STYLE_PRESETS.get(style_key, STYLE_PRESETS["matte_chiaroscuro"])
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
            "detail_scale": 1.0,
            "steps": steps,
            "step_times_ms": step_times_ms,
            "total_time": round(total_time, 2),
            "it_s": round(avg_it_s, 2),
            "tps": round(tps, 1),
            "kpx_s": round(kpx_s, 1),
            "peak_vram_mb": round(peak_vram, 1),
            "vae_blend_px": self.vae_blend_px,
            "mode": "classic",
            "precision": "BFLOAT16 (Continuous DMA)"
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
            "total_time": telemetry["total_time"],
            "peak_vram_mb": telemetry["peak_vram_mb"],
            "enable_geo": enable_geo,
            "enable_crpf": enable_crpf,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        
        return telemetry
