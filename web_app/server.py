"""Grosel web server: FastAPI, WebSocket generation, telemetry and gallery."""

import asyncio
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager
from statistics import median

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from web_app import inference_math, quality_math, resident_math
from web_app.config import GALLERY_DIR, META_DIR, RUNS_FILE, VRAM_CEILING_GB
from web_app.engine import STYLE_PRESETS, compile_physically_grounded_prompt
from web_app.resolutions import RESOLUTION_MATRIX, fit_mp_ceiling
from web_app.schemas import GenerateRequest, MathSolveRequest, OptimizeRequest
from web_app.storage import read_json, write_json_atomic

logger = logging.getLogger("grosel")
BACKEND = os.environ.get("HYPERFLUX_BACKEND", "fast").lower()
if BACKEND not in {"fast", "classic", "qwen"}:
    raise RuntimeError(f"Desteklenmeyen HYPERFLUX_BACKEND: {BACKEND}")

if BACKEND == "classic":
    from web_app.engine import GroselEngine as Engine
elif BACKEND == "qwen":
    from web_app.qwen_engine import QWEN_DIMENSIONS
    from web_app.qwen_engine import QwenStreamEngine as Engine
else:
    from web_app.fast_engine import FastKleinEngine as Engine

FLUX_TIERS = [
    {"value": 0, "label": "Hızlı 3–4", "title": "Hızlı: damıtılmış mod; 420p'de 3, diğer çözünürlüklerde 4 adım"},
    {"value": 8, "label": "Orta 8", "title": "Orta: 8 adım; süre artar, kalite artışı garanti değil"},
    {"value": 12, "label": "Pro 12", "title": "Pro: 12 adım; süre artar, kalite artışı garanti değil"},
    {"value": 20, "label": "Ultra 20", "title": "Ultra: 20 adım; en uzun yol, damıtılmış modelde kazanç garanti değil"},
]
FLUX_AUTO_STEPS = {q: data["steps"] for q, data in RESOLUTION_MATRIX.items()}
FLUX_DIMS = {
    q: {k: f"{v[0]}×{v[1]}" for k, v in data.items() if isinstance(v, tuple)}
    for q, data in RESOLUTION_MATRIX.items()
}

executor = ThreadPoolExecutor(max_workers=1)
math_executor = ThreadPoolExecutor(max_workers=1)

TEMPLATES_DIR = os.path.join(CURRENT_DIR, "templates")
STATIC_DIR = os.path.join(CURRENT_DIR, "static")
engine = None


@asynccontextmanager
async def lifespan(_):
    global engine
    loop = asyncio.get_running_loop()
    engine = await loop.run_in_executor(executor, Engine.get_instance)
    try:
        yield
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        math_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Grosel Studio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "connect-src 'self' ws: wss:; "
    "font-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'"
)


@app.middleware("http")
async def add_response_headers(request, call_next):
    response = await call_next(request)
    headers = response.headers
    headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
    headers["X-Content-Type-Options"] = "nosniff"
    headers["X-Frame-Options"] = "DENY"
    headers["Referrer-Policy"] = "no-referrer"
    headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        headers["Cache-Control"] = "no-cache"
    return response


def load_runs():
    data = read_json(RUNS_FILE, [])
    return data if isinstance(data, list) else []


def append_run(telemetry: dict) -> None:
    runs = load_runs()
    runs.append(telemetry)
    try:
        write_json_atomic(RUNS_FILE, runs[-300:])
    except OSError as exc:
        logger.warning("Çalıştırma telemetrisi kaydedilemedi: %s", exc)


def public_telemetry(telemetry: dict) -> dict:
    """İstemciye yalnız erişilebilir URL gider; makine dosya yolu dışarı sızmaz."""
    return {k: v for k, v in telemetry.items() if k != "filepath"}


def _wh(value: str) -> tuple[int, int]:
    try:
        w, h = str(value).lower().split("x")
        return int(w), int(h)
    except (AttributeError, ValueError):
        return 0, 0


def detail_baseline(telemetry: dict, samples: int = 12, min_count: int = 4) -> float | None:
    """Aynı kademe + stilin önceki karelerinden medyan yapısal detay indeksi.

    Göreli denetim burada kurulur: mutlak eşik içerik yoğunluğuna göre
    kayıyor, ama "bu kademe eskisinden daha düz üretiyor" sorusu kaymasız.
    Örnek yetmezse None döner — uydurma tabanla uyarı üretilmez.
    """
    key = (telemetry.get("quality"), telemetry.get("style_key"))
    values = []
    for run in reversed(load_runs()):
        if (run.get("quality"), run.get("style_key")) != key:
            continue
        detail = ((run.get("quality_report") or {}).get("metrics") or {}).get("detail_index")
        if isinstance(detail, (int, float)) and detail > 0:
            values.append(float(detail))
        if len(values) >= samples:
            break
    if len(values) < min_count:
        return None
    return round(float(median(values)), 1)


TIER_LADDER = ("420p", "720p", "1080p", "2K", "4K")


def measured_peak_gb(tier: str, ratio: str, samples: int = 8,
                     min_count: int = 3) -> float | None:
    """Aynı arkadda, aynı kademe + oranın ölçülmüş tepe VRAM medyanı (GB).

    `backend` alanı olmayan eski kayıtlar kullanılmaz: hangi modelin ürettiği
    bilinmeyen sayı, modelden bağımsız tavan sözünün kanıtı olamaz.
    """
    values = []
    for run in reversed(load_runs()):
        if (run.get("quality"), run.get("aspect_ratio"), run.get("backend")) != (tier, ratio, BACKEND):
            continue
        peak = run.get("peak_vram_mb")
        if isinstance(peak, (int, float)) and peak > 0:
            values.append(float(peak) / 1024.0)
        if len(values) >= samples:
            break
    if len(values) < min_count:
        return None
    return round(float(median(values)), 2)


def predict_peak_gb(tier: str, ratio: str) -> float | None:
    """Kademin tepe VRAM tahmini. Önce ölçüm, yoksa eşitlik; ikisi de yoksa None.

    Eşitlik (resident_math.plan) NF4 yerleşim için kalibre: plan 4.02 / ölçülen
    4.06 GB. Bugünkü Atelier kurulumu FP8 taşıdığı için eşitlik GERÇEĞİ AŞAR
    (plan 4.57 GB, ölçülen 2.42 GB) — bu yüzden ölçüm varsa o geçerlidir;
    eşitlik yalnızca ilk koşular için muhafazakâr tavan görevi görür.
    """
    measured = measured_peak_gb(tier, ratio)
    if measured is not None:
        return measured
    if BACKEND == "qwen":
        rung = QWEN_DIMENSIONS.get(tier) or {}
        size = rung.get(ratio)
        if size:
            return resident_math.plan(VRAM_CEILING_GB, size[0], size[1])["peak_gb"]
    return None


def detail_mp_cap() -> float | None:
    """Bu arkadın ölçülmüş denoise tepesinden tavana sığan en büyük ızgara (MP).

    Kademe etiketi tepeyi söylemiyor — aynı 1080p detay geçişi kapalıyken
    0.92 MP / 4.45 GB, açıkken 2.06 MP / 5.12 GB (ölçüldü). Yani 1–5 GB sözü
    ya çizim ızgarasından korunur ya da hiç korunmaz. Ölçüm yetmezse None
    döner: motor kendi SR_MAX_MP hırsında kalır, uydurma eşik uygulanmaz.
    """
    if VRAM_CEILING_GB <= 0:
        return None
    points = []
    for run in reversed(load_runs()):
        if run.get("backend") != BACKEND:
            continue
        denoise = run.get("denoise_peak_mb") or run.get("step_peak_mb") or 0
        native_w, native_h = _wh(run.get("native_resolution", ""))
        if denoise > 0 and native_w and native_h:
            points.append((native_w * native_h / 1e6, float(denoise)))
        if len(points) >= 12:
            break
    return fit_mp_ceiling(points, VRAM_CEILING_GB * 1024)


def guard_quality(tier: str, ratio: str) -> tuple[str | None, dict | None]:
    """Ürün sözü: tavan asilacaksa kademe uçuş ÖNCESİ indirilir (kayıplı kademe en son).

    Döner (kullanılacak kademe, kayıt). Kademe None ise hiçbir kademe tavana
    sığmıyor — üretim başlatılmaz, kullanıcı iç yapı görmeden nedeni duyar.
    """
    if VRAM_CEILING_GB <= 0 or tier not in TIER_LADDER:
        return tier, None
    idx = TIER_LADDER.index(tier)
    while idx >= 0:
        candidate = TIER_LADDER[idx]
        predicted = predict_peak_gb(candidate, ratio)
        if predicted is None or predicted <= VRAM_CEILING_GB:
            if idx == TIER_LADDER.index(tier):
                return candidate, None
            return candidate, {
                "requested": tier, "used": candidate,
                "predicted_peak_gb": predicted, "ceiling_gb": VRAM_CEILING_GB,
                "steps_down": TIER_LADDER.index(tier) - idx,
                # Kademe penceresi inceyse tahmin yoktur: kayıtsız şartsız
                # "ölçüldü" yazmak söz uydurmak olur (canlıda yakalandı: None
                # formatı üretim çökertiyordu).
                "measured": predicted is not None,
            }
        idx -= 1
    return None, {"requested": tier, "used": None, "ceiling_gb": VRAM_CEILING_GB,
                  "reason": "hiçbir kademe tavana sığmıyor"}


def guard_status(guard: dict) -> str:
    """Koruma kaydından tek kullanıcı cümlesi.

    İndirilen kademe bu kartta henüz ölçülmemişse sayı uydurulmaz (canlıda
    yakalandı: `None` formatı üretimi çökertiyordu).
    """
    return (f"VRAM tavanı {guard['ceiling_gb']:.1f} GB: "
            f"{guard['requested']} → {guard['used']} kademesine "
            + (f"indirildi (tahmini tepe {guard['predicted_peak_gb']:.2f} GB)."
               if guard.get("measured")
               else f"indirildi ({guard['used']} bu kartta henüz ölçülmedi)."))


def attach_quality(telemetry: dict) -> dict:
    """Kalite sözleşmesini kareye ve üretilen plana uygular, sonucu telemetriye yazar.

    Ölçüm CPU'da ~0.4 s sürer; GPU işi bittikten sonra yürütüldüğü için üretim
    süresine eklenmez. Hata kullanıcıya iç yapı sızdırmadan bildirilir.

    Alan adı `quality_report`: `quality` motorların kademe etiketini ("1080p")
    taşır, onu ezmez. `runs.json` Pareto analizi o etikete göre gruplanıyor.
    """
    path = telemetry.get("filepath") or ""
    if not os.path.exists(path):
        telemetry["quality_report"] = {"ok": None, "failures": [], "warnings": [],
                                       "actions": [], "metrics": {},
                                       "not": "kare okunamadı"}
        return telemetry
    native_w, native_h = _wh(telemetry.get("native_resolution", ""))
    target_w, target_h = _wh(telemetry.get("resolution", ""))
    plan = {"native_w": native_w, "native_h": native_h,
            "target_w": target_w, "target_h": target_h,
            "vae_blend_px": telemetry.get("vae_blend_px"),
            "peak_vram_mb": telemetry.get("peak_vram_mb"),
            "vram_ceiling_mb": VRAM_CEILING_GB * 1024}
    try:
        telemetry["quality_report"] = quality_math.evaluate(
            path, baseline_detail=detail_baseline(telemetry), **plan).as_dict()
    except Exception as exc:  # ölçüm üretimi asla çizimi iptal ettirmez
        logger.warning("Kalite denetimi atlandı: %s", exc)
        telemetry["quality_report"] = {"ok": None, "failures": [], "warnings": [],
                                       "actions": [], "metrics": {},
                                       "not": "ölçüm yapılamadı"}
    sidecar = os.path.join(META_DIR, f"{os.path.basename(path)}.json")
    try:
        meta = read_json(sidecar, {})
        meta["quality_report"] = telemetry["quality_report"]
        write_json_atomic(sidecar, meta)
    except OSError as exc:
        logger.warning("Kalite denetimi yan yana yazılamadı: %s", exc)
    return telemetry


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = os.path.join(TEMPLATES_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>Arayüz yükleniyor...</h2>")

def _mode(values: dict) -> str | None:
    """En sık görülen etiket; boş sayaçta None (uydurma çözünürlük yazılmaz)."""
    return max(values, key=values.get) if values else None


def _shape(rows: list, key: str) -> str | None:
    """Penceredeki en sık etiket; hiç yoksa None (uydurma çözünürlük yazılmaz)."""
    counts: dict = {}
    for run in rows:
        value = run.get(key)
        if value:
            counts[value] = counts.get(value, 0) + 1
    return _mode(counts)


# Panel güncel davranışı söylemeli, boru hattının her neslini değil: detay
# tavanından önce 1080p 4.45 GB (0.92 MP), sonra 4.84 GB (1.59 MP) çizdi. İkisini
# aynı medyanda eritmek, sahnedeki kartın bugün ne yaptığını gizler.
PROMISE_WINDOW = 5


def measured_by_tier(limit: int = 300) -> dict:
    """Bu arkadın kademe başına ölçülmüş tepe VRAM / süre medyanları (sahne verisi).

    Ekranda yazan hiçbir sayı ezberden gelmesin: hepsi runs.json'daki koşuların
    medyanı — ama yalnız SON PROMISE_WINDOW koşunun, yani bugün olan davranışın.
    3 örnekten azı "kesin" değil: `provisional` işaretiyle döner, arayüz onu
    önemsizleştirir. Hiç ölçülmemiş kademe ise anahtarda hiç yoktur.
    """
    rows_by_tier: dict[str, list] = {}
    for run in load_runs()[-limit:]:
        peak = run.get("peak_vram_mb")
        tier = run.get("quality")
        if run.get("backend") != BACKEND:
            continue
        if not (tier and isinstance(peak, (int, float)) and peak > 0):
            continue
        rows_by_tier.setdefault(tier, []).append(run)
    out = {}
    for tier, rows in rows_by_tier.items():
        window = rows[-PROMISE_WINDOW:]
        peaks = [round(float(r["peak_vram_mb"]) / 1024.0, 2) for r in window]
        times = [round(float(r["total_time"]), 1) for r in window
                 if isinstance(r.get("total_time"), (int, float)) and r["total_time"] > 0]
        out[tier] = {"runs": len(window),
                     "provisional": len(window) < 3,
                     "peak_gb": round(float(median(peaks)), 2),
                     "worst_gb": round(max(peaks), 2),
                     "median_time_s": round(float(median(times)), 1) if times else None,
                     "target": _shape(window, "resolution"),
                     "native": _shape(window, "native_resolution")}
    return out


@app.get("/api/info")
async def get_info():
    """Ürün sözünün sayısal hâli: tavan, ölçülmüş tepeler, sözleşme eşikleri."""
    return {
        "backend": BACKEND,
        "vram_ceiling_gb": VRAM_CEILING_GB,
        "tier_ladder": list(TIER_LADDER),
        "measured": measured_by_tier(),
        "contract": {
            "min_blend_px": quality_math.MIN_BLEND_PX,
            "tile_px": [quality_math.TILE_MIN_PX, quality_math.TILE_STRIDE_PX],
            "seam_periods": list(quality_math.seam_periods()),
            "seam_z_limit": quality_math.SEAM_Z_LIMIT,
            "rel_detail_floor": quality_math.REL_DETAIL_FLOOR,
            "detail_mp_cap": detail_mp_cap(),
        },
    }

@app.get("/api/system")
async def get_system():
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "Motor hazır değil"})
    try:
        info = engine.system_info()
    except Exception:
        logger.exception("Sistem bilgisi okunamadı")
        return JSONResponse(status_code=500, content={"error": "Sistem bilgisi alınamadı."})
    info["backend_env"] = BACKEND
    if BACKEND != "qwen":
        info["tiers"] = FLUX_TIERS
        info["auto_steps"] = FLUX_AUTO_STEPS
        info["dims"] = FLUX_DIMS
    return info


@app.get("/api/runs")
async def get_runs(limit: int = Query(default=100, ge=1, le=300)):
    return [public_telemetry(run) for run in reversed(load_runs()[-limit:])]


@app.get("/api/math")
async def get_math():
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(math_executor, inference_math.snapshot)
    except Exception:
        logger.exception("Matematik verisi hazırlanamadı")
        return JSONResponse(status_code=500, content={"error": "Matematik verisi hazırlanamadı."})


@app.get("/api/resident-plan")
async def get_resident_plan(
    budget_gb: float = Query(default=5.0, ge=0.5, le=16.0),
    width: int = Query(default=1024, ge=256, le=8192),
    height: int = Query(default=576, ge=256, le=8192),
    bits: int = Query(default=4, ge=2, le=4),
    strict: bool = True,
):
    return resident_math.plan(budget_gb, width, height, bits, strict)


@app.post("/api/math/solve")
async def solve_math(req: MathSolveRequest):
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            math_executor,
            lambda: inference_math.solve(
                req.budget_gb,
                req.max_time_s,
                req.min_native_mp,
                req.steps,
                req.warm,
                req.objective,
            ),
        )
    except Exception:
        logger.exception("Kısıt çözücü çalışamadı")
        return JSONResponse(status_code=500, content={"error": "Plan şu anda çözülemiyor."})


@app.post("/api/cache/clear")
async def clear_cache():
    if engine is None:
        return JSONResponse(status_code=503, content={"error": "Motor hazır değil"})
    loop = asyncio.get_running_loop()
    try:
        cleared = await loop.run_in_executor(executor, engine.clear_cache)
    except Exception:
        logger.exception("Önbellek temizlenemedi")
        return JSONResponse(status_code=500, content={"error": "Önbellek temizlenemedi."})
    return {"cleared": cleared}

@app.get("/api/styles")
async def get_styles():
    return [
        {"key": k, "name": v["name"], "desc": v["desc"]}
        for k, v in STYLE_PRESETS.items()
    ]

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
META_FIELDS = ("seed", "prompt", "style_name", "aspect_ratio", "quality",
               "quality_report", "peak_vram_mb", "total_time", "created_at",
               # Sözleşme paneli geçmiş karesine bakınca da aynı cümleyi kurmalı:
               # kademe adı değil, çizilen ızgara ve fiilen örtüşme sorgulanır.
               "resolution", "native_resolution", "vae_blend_px")


def read_sidecar(filename: str) -> dict:
    data = read_json(os.path.join(META_DIR, f"{filename}.json"), {})
    return data if isinstance(data, dict) else {}


def remove_sidecar(filename: str) -> None:
    try:
        os.remove(os.path.join(META_DIR, f"{filename}.json"))
    except FileNotFoundError:
        return
    except OSError as exc:
        logger.warning("Görsel metadatası silinemedi: %s", exc)


@app.get("/api/gallery")
async def get_gallery():
    if not os.path.exists(GALLERY_DIR):
        return []
    files = [
        f for f in os.listdir(GALLERY_DIR)
        if f.lower().endswith(IMAGE_EXTS)
    ]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(GALLERY_DIR, x)), reverse=True)
    items = []
    for f in files[:40]:
        meta = read_sidecar(f)
        item = {"filename": f, "url": f"/images/{f}"}
        item.update({k: meta.get(k) for k in META_FIELDS})
        items.append(item)
    return items

@app.delete("/api/gallery/{filename}")
async def delete_image(filename: str):
    filename = os.path.basename(filename)
    path = os.path.join(GALLERY_DIR, filename)
    if not filename.lower().endswith(IMAGE_EXTS) or not os.path.isfile(path):
        return JSONResponse(status_code=404, content={"status": "error", "message": "Dosya bulunamadı."})
    try:
        os.remove(path)
        remove_sidecar(filename)
        return {"status": "success", "message": f"{filename} silindi."}
    except OSError:
        logger.exception("Görsel silinemedi: %s", filename)
        return JSONResponse(status_code=500, content={"status": "error", "message": "Görsel silinemedi."})


@app.delete("/api/gallery")
async def clear_gallery():
    deleted_count = 0
    failed_count = 0
    for filename in os.listdir(GALLERY_DIR):
        if not filename.lower().endswith(IMAGE_EXTS):
            continue
        try:
            os.remove(os.path.join(GALLERY_DIR, filename))
            remove_sidecar(filename)
            deleted_count += 1
        except OSError:
            failed_count += 1
            logger.exception("Galeri öğesi silinemedi: %s", filename)
    if failed_count:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Bazı görseller silinemedi.",
                "deleted_count": deleted_count,
                "failed_count": failed_count,
            },
        )
    return {"status": "success", "deleted_count": deleted_count, "failed_count": 0}


@app.post("/api/optimize_prompt")
async def optimize_prompt(req: OptimizeRequest):
    if req.style not in STYLE_PRESETS:
        raise HTTPException(status_code=422, detail="Bilinmeyen stil.")
    return {"optimized": compile_physically_grounded_prompt(req.prompt, req.style)}

@app.get("/images/{filename}")
async def get_image(filename: str):
    filename = os.path.basename(filename)
    if not filename.lower().endswith(IMAGE_EXTS):
        return JSONResponse(status_code=404, content={"error": "Dosya bulunamadı"})
    path = os.path.join(GALLERY_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"error": "Dosya bulunamadı"})

@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()

    send_failed = False

    def send_sync(payload):
        nonlocal send_failed
        if send_failed:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop)
            future.result(timeout=1.0)
        except FutureTimeoutError:
            future.cancel()
            logger.warning("Adım iletisi 1 sn içinde teslim edilemedi; ilerleme iletileri durduruldu.")
            send_failed = True
        except (RuntimeError, OSError):
            logger.info("İstemci bağlantıyı kapattı; ilerleme iletileri durduruldu.")
            send_failed = True

    def make_callbacks(index, count):
        def step_cb(info):
            payload = dict(info)
            payload["variant"] = index
            payload["variant_count"] = count
            send_sync(payload)
        def status_cb(message):
            prefix = f"Varyasyon {index + 1}/{count} • " if count > 1 else ""
            send_sync({"type": "status", "message": prefix + message})
        return step_cb, status_cb

    try:
        while True:
            data = await websocket.receive_text()
            try:
                req = GenerateRequest.model_validate(json.loads(data))
            except (json.JSONDecodeError, ValidationError):
                await websocket.send_json({
                    "type": "error",
                    "message": "Geçersiz üretim isteği. Alanları denetleyip yeniden deneyin.",
                })
                continue

            if req.style not in STYLE_PRESETS:
                await websocket.send_json({"type": "error", "message": "Bilinmeyen stil."})
                continue

            prompt = req.prompt
            style_key = req.style
            aspect_ratio = req.aspect_ratio
            quality, vram_guard = guard_quality(req.quality, aspect_ratio)
            detail_cap = detail_mp_cap() if req.detail_pass else None
            if quality is None:
                await websocket.send_json({
                    "type": "error",
                    "message": f"{req.quality} kademesi {VRAM_CEILING_GB:.1f} GB VRAM "
                               "tavanına sığmıyor. Daha düşük kademe seçin ya da tavanı "
                               "HYPERFLUX_VRAM_CEILING_GB ile yükseltin.",
                    "guard": vram_guard,
                })
                continue
            enable_crpf = req.enable_crpf
            enable_geo = req.enable_geo
            steps_override = req.steps or None
            refine_steps = req.refine_steps
            refine_sigma = req.refine_sigma
            sigmas = req.sigmas
            count = req.count
            base_seed = req.seed if req.seed >= 0 else random.randint(0, 2 ** 31 - 2)
            tag = req.tag

            await websocket.send_json({"type": "status", "message": "Motor hazırlanıyor..."})

            try:
                variants = []
                for i in range(count):
                    variant_seed = (base_seed + i) % (2 ** 31 - 1)
                    step_cb, status_cb = make_callbacks(i, count)

                    def run_inference(cb=step_cb, st=status_cb, s=variant_seed,
                                      prompt=prompt, style_key=style_key,
                                      aspect_ratio=aspect_ratio, quality=quality,
                                      enable_crpf=enable_crpf, enable_geo=enable_geo,
                                      steps_override=steps_override,
                                      refine_steps=refine_steps,
                                      refine_sigma=refine_sigma, sigmas=sigmas,
                                      req=req, detail_cap=detail_cap):
                        kwargs = {
                            "prompt": prompt,
                            "style_key": style_key,
                            "aspect_ratio": aspect_ratio,
                            "quality": quality,
                            "seed": s,
                            "enable_geo": enable_geo,
                            "steps_override": steps_override,
                            "step_callback": cb,
                        }
                        if BACKEND == "classic":
                            kwargs["enable_crpf"] = enable_crpf
                        elif BACKEND == "qwen":
                            kwargs["status_callback"] = st
                        elif BACKEND == "fast":
                            kwargs["enable_crpf"] = enable_crpf
                            kwargs["status_callback"] = st
                            kwargs["refine_steps"] = refine_steps
                            kwargs["refine_sigma"] = refine_sigma
                            kwargs["sigmas"] = sigmas
                            # Detay geçişi ölçümlü olarak yalnız FLUX yolunda bağlı;
                            # Qwen kademe zinciri ayrı doğrulanmayı bekliyor.
                            kwargs["detail_pass"] = req.detail_pass
                            # Tavan sözü ızgaradan korunur: kademe etiketi tepeyi
                            # söylemiyor, ölçülmüş eşit söylüyor.
                            kwargs["detail_max_mp"] = detail_cap
                        return engine.generate(**kwargs)

                    telemetry = await loop.run_in_executor(
                        executor, lambda: attach_quality(run_inference()))
                    telemetry["variant_index"] = i
                    telemetry.update({
                        "tag": tag,
                        "prompt": prompt,
                        "style_key": style_key,
                        "backend": BACKEND,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
                    if vram_guard:
                        telemetry["vram_guard"] = vram_guard
                        await websocket.send_json({
                            "type": "status",
                            "message": guard_status(vram_guard),
                            "guard": vram_guard,
                        })
                    append_run(telemetry)
                    public = public_telemetry(telemetry)
                    variants.append(public)
                    if count > 1:
                        await websocket.send_json({
                            "type": "variant", "index": i, "count": count,
                            "telemetry": public,
                        })

                await websocket.send_json({
                    "type": "complete",
                    "telemetry": variants[-1],
                    "variants": variants,
                    "count": count,
                    "total_time": round(sum(v["total_time"] for v in variants), 2),
                })
            except Exception:
                logger.exception("WebSocket üretimi tamamlanamadı")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Üretim tamamlanamadı. Motor günlüklerini denetleyin.",
                    })
                except Exception:
                    return
    except WebSocketDisconnect:
        logger.info("Üretim istemcisi bağlantıyı kapattı")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app.server:app", host="127.0.0.1",
                port=int(os.environ.get("HYPERFLUX_PORT", "7870")), reload=False)
