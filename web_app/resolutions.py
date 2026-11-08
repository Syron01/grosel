"""Çözünürlük matrisi — GPU bağımlılığı olmayan tek kaynak.

KÇM matematik katmanı bu modülü torch kurulmadan da okuyabilsin diye
engine.py'den ayrı tutuluyor.
"""

ASPECT_RATIOS = ("16:9", "1:1", "9:16", "4:3", "3:4", "2:3", "21:9")

RESOLUTION_MATRIX = {
    "420p": {
        "steps": 3,
        "upscale": False,
        "16:9": (640, 368),
        "1:1":  (512, 512),
        "9:16": (368, 640),
        "4:3":  (640, 480),
        "3:4":  (480, 640),
        "2:3":  (512, 768),
        "21:9": (768, 336),
    },
    "720p": {
        "steps": 4,
        "upscale": False,
        "16:9": (896, 512),
        "1:1":  (704, 704),
        "9:16": (512, 896),
        "4:3":  (800, 608),
        "3:4":  (608, 800),
        "2:3":  (640, 960),
        "21:9": (1024, 448),
    },
    "1080p": {
        "steps": 4,
        "upscale": True,
        "target_res": {
            "16:9": (1920, 1080),
            "1:1":  (1440, 1440),
            "9:16": (1080, 1920),
            "4:3":  (1440, 1080),
            "3:4":  (1080, 1440),
            "2:3":  (1080, 1620),
            "21:9": (2520, 1080),
        },
        "16:9": (1280, 720),
        "1:1":  (1024, 1024),
        "9:16": (720, 1280),
        "4:3":  (1088, 816),
        "3:4":  (816, 1088),
        "2:3":  (832, 1248),
        "21:9": (1456, 624),
    },
    "2K": {
        "steps": 4,
        "upscale": True,
        "target_res": {
            "16:9": (2048, 1152),
            "1:1":  (1536, 1536),
            "9:16": (1152, 2048),
            "4:3":  (1600, 1200),
            "3:4":  (1200, 1600),
            "2:3":  (1440, 2160),
            "21:9": (2560, 1088),
        },
        "16:9": (1280, 720),
        "1:1":  (1024, 1024),
        "9:16": (720, 1280),
        "4:3":  (1088, 816),
        "3:4":  (816, 1088),
        "2:3":  (1024, 1536),
        "21:9": (1456, 624),
    },
    "4K": {
        "steps": 4,
        "upscale": True,
        "target_res": {
            "16:9": (3840, 2160),
            "1:1":  (2160, 2160),
            "9:16": (2160, 3840),
            "4:3":  (2880, 2160),
            "3:4":  (2160, 2880),
            "2:3":  (2160, 3240),
            "21:9": (3840, 1632),
        },
        "16:9": (1280, 720),
        "1:1":  (1024, 1024),
        "9:16": (720, 1280),
        "4:3":  (1088, 816),
        "3:4":  (816, 1088),
        "2:3":  (1280, 1920),
        "21:9": (1456, 624),
    }
}


def get_dimensions(quality: str, ratio: str):
    q_data = RESOLUTION_MATRIX.get(quality, RESOLUTION_MATRIX["1080p"])
    steps = q_data["steps"]
    upscale = q_data.get("upscale", False)

    base_w, base_h = q_data.get(ratio, (1024, 576))
    target_w, target_h = (base_w, base_h)

    if upscale and "target_res" in q_data:
        target_w, target_h = q_data["target_res"].get(ratio, (base_w * 2, base_h * 2))

    return base_w, base_h, target_w, target_h, steps, upscale


def detail_target(base_w: int, base_h: int, target_w: int, target_h: int,
                  max_scale: float = 2.0, max_mp: float = 2.2, enabled: bool = True):
    """Detay geçişinin üreteceği ızgara; 16'nın katına katlanır, hedefi aşmaz.

    Model bu ızgarada doğrudan örnekleme yapar; latent sonradan büyütülmez.
    Adım süresi ve VRAM alanla büyüdüğü için tavan ölçekten çok megapiksel ile
    konur (ölçüm: 2.2 MP @ 3.4 s/adım, 5.24 GB).
    """
    if not enabled:
        return None
    scale = max(target_w / base_w, target_h / base_h)
    if scale <= 1.0:
        return None
    area_scale = (max_mp * 1_000_000 / (base_w * base_h)) ** 0.5
    scale = min(scale, max_scale, area_scale)
    if scale <= 1.0:
        return None
    w = min(int(base_w * scale) // 16 * 16, target_w // 16 * 16)
    h = min(int(base_h * scale) // 16 * 16, target_h // 16 * 16)
    if w <= base_w or h <= base_h:
        return None
    return w, h


# Tepe VRAM sözünün payı: eşitlik tam tavanda değil, altında kurulur. Ölçüm
# gürültüsü + soğuk/ılık farkı 0.1 GB'ı bulabiliyor.
CEILING_MARGIN = 0.97


def fit_mp_ceiling(points: list, ceiling_mb: float,
                   margin: float = CEILING_MARGIN) -> float | None:
    """(MP, MB) ölçümlerinden tavana sığan en büyük üretim ızgarasını MP bulur.

    Neden bu: kademe etiketi tepeyi söylemiyor — aynı 1080p, detay geçişi
    kapalıyken 0.92 MP/4.45 GB, açıkken 2.06 MP/5.12 GB. Yani tavan ya
    ölçülmüş eşitlikle korunacak ya da söz uydurma kalacak.

    En küçük kareler çizgisi >= 3 ölçüm ve >= 2 farklı ızgara ister; slope <= 0
    ise (soğuk ilk koşu gibi kirlenmiş veri) None döner: motor kendi SR_MAX_MP
    değerinde kalır, iddia kurulmaz. Çizgi denoise fazından çizilir — birleşik
    tepe ilk koşuda metin kodalayıcıyla şişiyor (ölçüldü: 5036 MB encode vs
    ılık 4554 MB).
    """
    pts = [(float(x), float(y)) for x, y in points if x and y]
    if len(pts) < 3 or len({round(x, 3) for x, _ in pts}) < 2:
        return None
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    if slope <= 0:
        return None
    intercept = (sy - slope * sx) / n
    max_mb = ceiling_mb * margin
    if intercept >= max_mb:
        return None
    # slope birimi MB / MP, tavan MB cinsinden: bölüm doğrudan MP verir.
    return round((max_mb - intercept) / slope, 3)
