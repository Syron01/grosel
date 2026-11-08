"""Kısıtlı Çıkarım Matematiği (KÇM).

Ölçülen çalıştırmaları plan / kaynak / kalite vektörleri olarak ele alır.
Pareto cephesini ve bütçe-zaman kısıtları altındaki en iyi planı çözer.
Kalite ekseni CLIP prompt uyumudur; estetik veya anatomi kalitesi iddiası taşımaz.
"""

from __future__ import annotations

import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from web_app.config import GALLERY_DIR, QUALITY_FILE, RUNS_FILE
from web_app.resolutions import get_dimensions
from web_app.storage import read_json, write_json_atomic

RATIO = "16:9"
QUALITIES = ("420p", "720p", "1080p")
STEP_OPTIONS = (3, 4, 6)
OFFLOAD_PIXELS = 350_000

_CLIP_LOCK = threading.Lock()
_CLIP = None
_QUALITY_LOCK = threading.Lock()


@dataclass
class Plan:
    filename: str | None
    prompt: str
    quality: str
    steps: int
    resolution: str
    native_mp: float
    target_mp: float
    cold: bool
    offload: bool
    total_time: float
    peak_gb: float
    clip: float | None
    mode: str
    tag: str
    n_measured: int = 1
    predicted: bool = False
    filenames: list[str] = field(default_factory=list)

    @property
    def signature(self):
        return (self.mode, self.quality, self.steps, self.cold, round(self.native_mp, 3))


@dataclass
class LinearFit:
    mu: np.ndarray
    sd: np.ndarray
    theta: np.ndarray
    r2: float
    mae: float
    n: int

    def predict(self, features):
        x = (np.asarray(features, dtype=float) - self.mu) / self.sd
        return float(x @ self.theta)

    def public(self, unit=""):
        return {
            "r2": round(self.r2, 3),
            "mae": round(self.mae, 3),
            "unit": unit,
            "n": self.n,
        }


def _write_json(path, data):
    try:
        write_json_atomic(path, data, indent=1)
    except OSError as exc:
        print(f"[math] yazılamadı: {exc}")


def _fit_linear(rows, targets, ridge=1e-3):
    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.size == 0 or y.size == 0:
        width = max(x.shape[1] if x.ndim == 2 else 1, 1)
        return LinearFit(np.zeros(width), np.ones(width), np.zeros(width), 0.0, 0.0, 0)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    # Sabit sütun (1.0) standartlaştırılmaz; modelde kesişim rolünü o üstlenir.
    constant = sd < 1e-8
    mu[constant] = 0.0
    sd[constant] = 1.0
    xs = (x - mu) / sd
    theta = np.linalg.solve(xs.T @ xs + ridge * np.eye(xs.shape[1]), xs.T @ y)
    pred = xs @ theta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    mae = float(np.abs(y - pred).mean())
    return LinearFit(mu, sd, theta, r2, mae, len(y))


def _loocv(rows, targets):
    """Tek-satır dışarıda bırakma; eğitim R²'sini değil genellemeyi raporlar."""
    x = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if len(y) < 4:
        return {"r2": 0.0, "mae": 0.0, "n": len(y)}
    preds = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        model = _fit_linear(x[keep], y[keep])
        preds.append(model.predict(x[i]))
    preds = np.asarray(preds)
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "r2": round(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0, 3),
        "mae": round(float(np.abs(y - preds).mean()), 3),
        "n": len(y),
    }


def _load_clip():
    global _CLIP
    with _CLIP_LOCK:
        if _CLIP is None:
            import clip
            import torch

            torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
            model, preprocess = clip.load("ViT-B/32", device="cpu")
            model.eval()
            _CLIP = (model, preprocess, torch)
    return _CLIP


def _ensure_clip_scores(items):
    """Eksik görseller için CLIP prompt uyumunu CPU'da hesaplar."""
    cache = read_json(QUALITY_FILE, {})
    if not isinstance(cache, dict):
        cache = {}
    missing = [(f, p) for f, p in items if f and p and f not in cache]
    if not missing:
        return cache

    with _QUALITY_LOCK:
        cache = read_json(QUALITY_FILE, {})
        missing = [(f, p) for f, p in items if f and p and f not in cache]
        if not missing:
            return cache
        try:
            import clip
            model, preprocess, torch = _load_clip()
            batch = 8
            with torch.no_grad():
                for start in range(0, len(missing), batch):
                    chunk = missing[start:start + batch]
                    images, texts = [], []
                    for filename, prompt in chunk:
                        path = os.path.join(GALLERY_DIR, filename)
                        if not os.path.exists(path):
                            continue
                        images.append(preprocess(Image.open(path).convert("RGB")))
                        texts.append(prompt[:300])
                    if not images:
                        continue
                    image_input = torch.stack(images)
                    text_input = clip.tokenize(texts, truncate=True)
                    image_emb = model.encode_image(image_input)
                    text_emb = model.encode_text(text_input)
                    image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
                    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                    scores = (image_emb @ text_emb.T).diag().tolist()
                    for (filename, prompt), score in zip(chunk, scores):
                        cache[filename] = {
                            "clip": round(float(score), 4),
                            "prompt": prompt,
                            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
            _write_json(QUALITY_FILE, cache)
        except Exception as exc:
            print(f"[math] CLIP atlandı: {exc}")
    return cache


def _plans_from_runs(runs, quality_cache):
    plans = []
    seen = set()
    for run in runs:
        filename = run.get("filename")
        if not filename or filename in seen:
            continue
        # KÇM katsayıları FLUX yığınından; Qwen koşuları kendi faz modeliyle raporlanır.
        if (run.get("mode") or "fast") not in ("fast", "classic"):
            continue
        image_exists = os.path.exists(os.path.join(GALLERY_DIR, filename))
        if not image_exists and run.get("mode") != "classic":
            continue
        quality = run.get("quality")
        steps = int(run.get("steps") or 0)
        if quality not in QUALITIES or steps <= 0:
            continue
        if run.get("aspect_ratio", RATIO) != RATIO:
            continue
        # Eski fast ölçümleri faz telemetrisi taşımıyor; decode offload öncesi
        # 5.17 GB'lık farklı bir yığına ait. Güncel modeli kirletmesinler.
        if run.get("mode") == "fast" and (
            run.get("decode_peak_mb") is None or run.get("step_peak_mb") is None
        ):
            continue
        try:
            base_w, base_h, *_ = get_dimensions(quality, RATIO)
            total_time = float(run.get("total_time") or 0)
            peak_gb = float(run.get("peak_vram_mb") or 0) / 1024
            res_w, res_h = (int(v) for v in str(run.get("resolution", "0x0")).split("x"))
        except (TypeError, ValueError):
            continue
        if total_time <= 0 or peak_gb <= 0:
            continue
        seen.add(filename)
        encode_time = run.get("encode_time") or 0
        cold = bool(encode_time > 0.05) and not bool(run.get("prompt_cached", False))
        if "prompt_cached" in run:
            cold = not bool(run.get("prompt_cached"))
        decode_peak = float(run.get("decode_peak_mb") or 0)
        # Detay geçişi ızgarayı büyütüyor; matris tabanı yerine fiilen çizilen
        # ızgara kullanılmazsa "native draw" sütunu ölçümle çelişir.
        try:
            native_w, native_h = (int(v) for v in str(run.get("native_resolution") or "").split("x"))
        except (TypeError, ValueError):
            native_w = native_h = 0
        native_mp = (native_w * native_h if native_w and native_h else base_w * base_h) / 1e6
        target_mp = res_w * res_h / 1e6
        # 1080p 1024×576 tabanından üretilmiş eski çalıştırmalar mevcut
        # matrisle yeniden üretilemez; planlayıcıya girmesinler.
        if quality == "1080p" and res_w * res_h / 1e6 < 0.8:
            continue
        offload = run.get("mode") == "fast" and (
            (decode_peak and decode_peak < 2500) or (not decode_peak and target_mp * 1e6 >= OFFLOAD_PIXELS)
        )
        plan = Plan(
            filename=filename,
            prompt=(run.get("prompt") or "").strip(),
            quality=quality,
            steps=steps,
            resolution=f"{res_w}x{res_h}",
            native_mp=native_mp,
            target_mp=target_mp,
            cold=cold,
            offload=bool(offload),
            total_time=round(total_time, 2),
            peak_gb=round(peak_gb, 2),
            clip=(quality_cache.get(filename) or {}).get("clip"),
            mode=run.get("mode") or "fast",
            tag=run.get("tag") or "",
        )
        plans.append(plan)
    return plans


def _aggregate(plans):
    groups = {}
    for plan in plans:
        groups.setdefault(plan.signature, []).append(plan)
    aggregated = []
    for group in groups.values():
        times = [p.total_time for p in group]
        peaks = [p.peak_gb for p in group]
        clips = [p.clip for p in group if p.clip is not None]
        median_time = statistics.median(times)
        representative = min(group, key=lambda p: abs(p.total_time - median_time))
        aggregated.append(Plan(
            filename=representative.filename,
            prompt=representative.prompt,
            quality=representative.quality,
            steps=representative.steps,
            resolution=representative.resolution,
            native_mp=representative.native_mp,
            target_mp=representative.target_mp,
            cold=representative.cold,
            offload=representative.offload,
            total_time=round(median_time, 2),
            peak_gb=round(statistics.median(peaks), 2),
            clip=round(statistics.median(clips), 4) if clips else None,
            mode=representative.mode,
            tag=representative.tag,
            n_measured=len(group),
            filenames=[p.filename for p in group if p.filename],
        ))
    return aggregated


def _fit_models(plans, runs):
    runs_by_file = {r.get("filename"): r for r in runs}
    fast = [p for p in plans if p.mode == "fast" and p.native_mp > 0]

    def latency_features(p):
        return [1.0, p.native_mp * p.steps, 1.0 if p.cold else 0.0,
                1.0 if p.offload else 0.0, p.target_mp]

    latency_rows = [latency_features(p) for p in fast]
    latency_targets = [p.total_time for p in fast]
    latency = _fit_linear(latency_rows, latency_targets)
    latency_cv = _loocv(latency_rows, latency_targets)

    denoise_rows, denoise_targets = [], []
    for p in fast:
        run = runs_by_file.get(p.filename)
        if not run:
            continue
        steps_ms = run.get("step_times_ms") or []
        if steps_ms:
            denoise_rows.append([1.0, p.native_mp * p.steps])
            denoise_targets.append(sum(steps_ms) / 1000.0)
    denoise = _fit_linear(denoise_rows, denoise_targets) if len(denoise_rows) > 3 else None

    decode_rows, decode_targets = [], []
    for p in fast:
        run = runs_by_file.get(p.filename)
        if not run:
            continue
        decode_rows.append([1.0, p.target_mp, 1.0 if p.offload else 0.0])
        decode_targets.append(float(run.get("decode_time") or 0))
    decode = _fit_linear(decode_rows, decode_targets) if len(decode_rows) > 3 else None

    cold_peaks = [float(runs_by_file[p.filename].get("encode_peak_mb")) / 1024.0
                  for p in fast if p.cold and p.filename in runs_by_file
                  and runs_by_file[p.filename].get("encode_peak_mb")]
    encode_cold = statistics.median(cold_peaks) if cold_peaks else 4.92

    warm = [p for p in fast if not p.cold]
    peak_by_quality = {}
    for quality in QUALITIES:
        values = [p.peak_gb for p in warm if p.quality == quality]
        if values:
            peak_by_quality[quality] = round(statistics.median(values), 3)
    predicted = [peak_by_quality.get(p.quality, 4.4) for p in warm]
    actual = np.array([p.peak_gb for p in warm])
    pred_arr = np.array(predicted, dtype=float)
    ss_res = float(((actual - pred_arr) ** 2).sum()) if len(actual) else 0.0
    ss_tot = float(((actual - actual.mean()) ** 2).sum()) if len(actual) else 0.0
    peak_cv = {
        "r2": round(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0, 3),
        "mae": round(float(np.abs(actual - pred_arr).mean()), 3) if len(actual) else 0.0,
        "n": len(actual),
    }
    vram_fit = LinearFit(
        mu=np.zeros(1), sd=np.ones(1), theta=np.array([4.4]),
        r2=peak_cv["r2"], mae=peak_cv["mae"], n=peak_cv["n"],
    )

    peaks_by_tier = {}
    clips_by_tier = {}
    for p in plans:
        key = f"{p.quality}·{p.steps}ad"
        peaks_by_tier.setdefault(key, []).append(p.peak_gb)
        if p.clip is not None:
            clips_by_tier.setdefault(key, []).append(p.clip)

    return {
        "latency": latency,
        "latency_cv": latency_cv,
        "denoise": denoise,
        "decode": decode,
        "peak_by_quality": peak_by_quality,
        "peak_cv": peak_cv,
        "vram_fit": vram_fit,
        "encode_cold": encode_cold,
        "peaks_by_tier": {k: round(statistics.median(v), 2) for k, v in peaks_by_tier.items()},
        "clips_by_tier": {k: round(statistics.median(v), 4) for k, v in clips_by_tier.items()},
    }


def _predict_peak(quality, cold, peak_by_quality, encode_cold):
    if cold:
        return round(encode_cold, 3)
    return round(peak_by_quality.get(quality, 4.4), 3)


def _predict_plan(fit, quality, steps, warm):
    base_w, base_h, target_w, target_h, _, _ = get_dimensions(quality, RATIO)
    native_mp = base_w * base_h / 1e6
    target_mp = target_w * target_h / 1e6
    cold = not warm
    offload = target_mp * 1e6 >= OFFLOAD_PIXELS
    total = fit["latency"].predict([
        1.0, native_mp * steps, 1.0 if cold else 0.0,
        1.0 if offload else 0.0, target_mp,
    ])
    peak = _predict_peak(quality, cold, fit["peak_by_quality"], fit["encode_cold"])
    return Plan(
        filename=None,
        prompt="",
        quality=quality,
        steps=steps,
        resolution=f"{target_w}x{target_h}",
        native_mp=native_mp,
        target_mp=target_mp,
        cold=cold,
        offload=offload,
        total_time=round(max(total, 0.1), 2),
        peak_gb=round(max(peak, 0.1), 2),
        clip=None,
        mode="fast",
        tag="predicted",
        n_measured=0,
        predicted=True,
    )


def _public(plan):
    data = asdict(plan)
    has_image = bool(plan.filename and os.path.exists(os.path.join(GALLERY_DIR, plan.filename)))
    data["image_url"] = f"/images/{plan.filename}" if has_image else None
    return data


def _pareto(plans):
    points = [p for p in plans if p.mode in ("fast", "classic")]
    keep = []
    for a in points:
        dominated = False
        for b in points:
            if b is a:
                continue
            quality_ok = b.native_mp >= a.native_mp - 1e-9
            resource_ok = b.total_time <= a.total_time + 1e-9 and b.peak_gb <= a.peak_gb + 1e-9
            strict = (b.total_time < a.total_time - 1e-9 or b.peak_gb < a.peak_gb - 1e-9
                      or b.native_mp > a.native_mp + 1e-9)
            if quality_ok and resource_ok and strict:
                dominated = True
                break
        if not dominated:
            keep.append(a)
    keep.sort(key=lambda p: (p.total_time, p.peak_gb))
    return keep


def _constraints_ok(plan, budget, max_time, min_mp):
    return plan.peak_gb <= budget + 1e-9 and plan.total_time <= max_time + 1e-9 and plan.native_mp + 1e-9 >= min_mp


def _violation(plan, budget, max_time, min_mp):
    return (max(0.0, plan.peak_gb / max(budget, 1e-6) - 1.0)
            + max(0.0, plan.total_time / max(max_time, 1e-6) - 1.0)
            + max(0.0, (min_mp - plan.native_mp) / max(min_mp, 1e-6)))


def _sort_key(plan, objective):
    if objective == "quality":
        return (-plan.native_mp, plan.total_time, plan.peak_gb)
    if objective == "vram":
        return (plan.peak_gb, plan.total_time)
    return (plan.total_time, plan.peak_gb)


def _gain(chosen, alternative, objective):
    if objective == "quality":
        return alternative.native_mp - chosen.native_mp
    if objective == "vram":
        return chosen.peak_gb - alternative.peak_gb
    return chosen.total_time - alternative.total_time


def _solve(candidates, budget, max_time, min_mp, objective="time"):
    feasible = [p for p in candidates if _constraints_ok(p, budget, max_time, min_mp)]
    if feasible:
        feasible.sort(key=lambda p: _sort_key(p, objective))
        chosen = feasible[0]
        status = "optimal"
        alternatives = feasible[1:4]
        violations = {}
    else:
        candidates = sorted(candidates, key=lambda p: _violation(p, budget, max_time, min_mp))
        chosen = candidates[0]
        status = "infeasible"
        alternatives = candidates[1:4]
        violations = {
            "vram_gb": round(max(0.0, chosen.peak_gb - budget), 2),
            "time_s": round(max(0.0, chosen.total_time - max_time), 2),
            "native_mp": round(max(0.0, min_mp - chosen.native_mp), 3),
        }

    shadow = {}
    if status == "optimal":
        unit = {"time": "s", "vram": "GB", "quality": "MP"}[objective]
        for delta in (0.5, -0.5):
            alt = [p for p in candidates if _constraints_ok(p, budget + delta, max_time, min_mp)]
            if alt:
                alt.sort(key=lambda p: _sort_key(p, objective))
                shadow["budget_gain"] = round(_gain(chosen, alt[0], objective) / delta, 3)
                shadow["budget_unit"] = f"{unit}/GB"
                break
        for delta in (1.0, -1.0):
            alt = [p for p in candidates if _constraints_ok(p, budget, max_time + delta, min_mp)]
            if alt:
                alt.sort(key=lambda p: _sort_key(p, objective))
                shadow["time_gain"] = round(_gain(chosen, alt[0], objective) / delta, 3)
                shadow["time_unit"] = f"{unit}/s"
                break

    binding = []
    if chosen.peak_gb >= budget - 0.02:
        binding.append("VRAM bütçesi")
    if chosen.total_time >= max_time - 0.05:
        binding.append("süre üst sınırı")
    better = [p for p in candidates if _sort_key(p, objective) < _sort_key(chosen, objective)]
    for plan in better:
        if (plan.native_mp < min_mp - 1e-9 and plan.peak_gb <= budget
                and plan.total_time <= max_time and "kalite tabanı" not in binding):
            binding.append("kalite tabanı")
        if (plan.peak_gb > budget and plan.total_time <= max_time
                and plan.native_mp >= min_mp - 1e-9 and "VRAM bütçesi" not in binding):
            binding.append("VRAM bütçesi")
        if (plan.total_time > max_time and plan.peak_gb <= budget
                and plan.native_mp >= min_mp - 1e-9 and "süre üst sınırı" not in binding):
            binding.append("süre üst sınırı")

    return {
        "status": status,
        "chosen": _public(chosen),
        "alternatives": [_public(p) for p in alternatives],
        "violations": violations,
        "shadow": shadow,
        "binding": binding,
    }


def _public_fit(fit):
    return {
        "latency": {**fit["latency"].public("s"), "cv": fit["latency_cv"]},
        "vram": {
            "r2": fit["peak_cv"]["r2"],
            "mae": fit["peak_cv"]["mae"],
            "unit": "GB",
            "n": fit["peak_cv"]["n"],
            "train_r2": round(fit["vram_fit"].r2, 3),
            "train_mae": round(fit["vram_fit"].mae, 3),
        },
        "denoise": fit["denoise"].public("s") if fit["denoise"] else None,
        "decode": fit["decode"].public("s") if fit["decode"] else None,
        "encode_cold_gb": round(fit["encode_cold"], 2),
    }


def _build():
    runs = read_json(RUNS_FILE, [])
    items = [(r.get("filename"), (r.get("prompt") or "").strip()) for r in runs
             if r.get("filename") and (r.get("prompt") or "").strip()]
    quality_cache = _ensure_clip_scores(items)
    plans = _plans_from_runs(runs, quality_cache)
    fit = _fit_models(plans, runs)
    aggregated = _aggregate(plans)
    frontier = _pareto(aggregated)
    return runs, plans, fit, aggregated, frontier


def snapshot():
    runs, plans, fit, aggregated, frontier = _build()
    clipped = [p for p in plans if p.clip is not None]
    time_range = [round(min((p.total_time for p in plans), default=0.0), 2),
                  round(max((p.total_time for p in plans), default=0.0), 2)]
    vram_range = [round(min((p.peak_gb for p in plans), default=0.0), 2),
                  round(max((p.peak_gb for p in plans), default=0.0), 2)]
    native_range = [round(min((p.native_mp for p in plans), default=0.0), 3),
                    round(max((p.native_mp for p in plans), default=0.0), 3)]
    clip_range = [round(min((p.clip for p in clipped), default=0.0), 4),
                  round(max((p.clip for p in clipped), default=1.0), 4)]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_runs": len(runs),
        "n_plans": len(plans),
        "n_aggregated": len(aggregated),
        "n_clip": len(clipped),
        "clip_coverage": round(len(clipped) / max(len(plans), 1), 3),
        "fit": _public_fit(fit),
        "clip_by_tier": fit["clips_by_tier"],
        "quality_note": "CLIP ViT-B/32 prompt uyumu; estetik veya anatomi kalitesi değildir.",
        "plans": [_public(p) for p in sorted(plans, key=lambda p: (p.total_time, p.peak_gb))[:40]],
        "aggregated": [_public(p) for p in sorted(aggregated, key=lambda p: (p.total_time, p.peak_gb))],
        "frontier": [_public(p) for p in frontier[:24]],
        "ranges": {
            "time": time_range,
            "vram": vram_range,
            "native_mp": native_range,
            "clip": clip_range,
        },
    }


def solve(budget_gb=5.0, max_time_s=8.0, min_native_mp=0.9, steps=4, warm=True, objective="time"):
    _, _, fit, aggregated, frontier = _build()
    candidates = []
    for quality in QUALITIES:
        matches = [p for p in aggregated if p.mode == "fast" and p.quality == quality
                   and p.steps == steps and p.cold == (not warm)]
        if matches:
            candidates.extend(matches)
        elif fit["latency"].n:
            candidates.append(_predict_plan(fit, quality, steps, warm))
    for plan in aggregated:
        if plan.mode == "classic" and plan.steps == steps:
            candidates.append(plan)
    if not candidates:
        return {
            "status": "no-data",
            "chosen": None,
            "alternatives": [],
            "violations": {},
            "shadow": {},
            "binding": [],
            "constraints": {
                "budget_gb": round(budget_gb, 2),
                "max_time_s": round(max_time_s, 2),
                "min_native_mp": round(min_native_mp, 3),
                "steps": steps,
                "warm": warm,
                "objective": objective,
            },
            "frontier": [_public(p) for p in frontier[:24]],
            "fit": _public_fit(fit),
        }
    result = _solve(candidates, budget_gb, max_time_s, min_native_mp, objective)
    result["constraints"] = {
        "budget_gb": round(budget_gb, 2),
        "max_time_s": round(max_time_s, 2),
        "min_native_mp": round(min_native_mp, 3),
        "steps": steps,
        "warm": warm,
        "objective": objective,
    }
    result["frontier"] = [_public(p) for p in frontier[:24]]
    result["fit"] = _public_fit(fit)
    return result
