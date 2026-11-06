"""Modelden bağımsız kalite sözleşmesi: plan değişmezleri + ölçülmüş kanıt.

Kaliteyi model değil boru hattı üretir. Sözleşme iki katmandan kurulur:

1. Değişmezler (plan denetimi, kesin): boru hattının kendine verdiği sözler. Ara
   latent'in yeniden ölçeklenmemesi, VAE blend'inin yeterince geniş olması, stil
   chip'inin kullanıcının yazdığı ortamla çelişmemesi, VRAM tavanı. Yanlışsa hata
   budur; görüntüye bakmayı beklemez.
2. Ölçütler (kare denetimi, destekleyici): yalnızca doğrulanmış olanlar kaldı.
   Kenar-genişliği ve halo tahmincileri bulanık karede *düşse* de güvenilir sıralama
   vermediği için sözleşmeden çıkarıldı; yerine pütürden arınmış doku ölçütü geldi.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Ölçüm en fazla bu kadar piksel üzerinde yapılır; fazlalıkta kare küçültülür ve
# birim native piksele geri çevrilir, böylece eşikler çözünürlükten bağımsız kalır.
MAX_ANALYSIS_PX = 2_600_000


def load_luma(path: str) -> tuple[np.ndarray, float]:
    """Gri kanal 0..1 + 'analiz pikeli başına native piksel' katsayısı."""
    im = Image.open(path).convert("L")
    w, h = im.size
    if w * h > MAX_ANALYSIS_PX:
        s = (MAX_ANALYSIS_PX / (w * h)) ** 0.5
        im = im.resize((max(32, int(w * s)), max(32, int(h * s))), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0, w / im.size[0]


def noise_index(a: np.ndarray) -> float:
    """Pütür indeksi 0..1: korelasyonsuz gren 1.0'a, düzgün yapı 0.0'a yaklaşır.

    İki-atlama farkı varyansının komşu-fark varyansına oranından çıkar: gren iki
    ölçümde de aynı gücü verdiği için oran 1'e, gerçek yapı 4'e kaçar. İçerik
    yoğunluğundan bağımsız; 3 Kasım 2026'da bulanıklaştırma/USM merdiveninde
    tek yönlü doğrulandı (0.054 → 0.142 → 0.566 → 0.646).

    8-bit karede kuantum tabanı ~0.4: akışkan bir rama float'ta 0.000, PNG'de
    0.392 okunur. Eşikler bu tabanın üstünden kurulur; tek başına "pürüzsüz"
    kanıtı sayılmaz, yalnızca gren ağırlığını kısmak için kullanılır.
    """
    out = []
    for ax in (0, 1):
        if ax:
            d1, d2 = a[1:, :] - a[:-1, :], a[2:, :] - a[:-2, :]
            n = min(d1.shape[0], d2.shape[0])
            d1, d2 = d1[:n], d2[:n]
        else:
            d1, d2 = a[:, 1:] - a[:, :-1], a[:, 2:] - a[:, :-2]
            n = min(d1.shape[1], d2.shape[1])
            d1, d2 = d1[:, :n], d2[:, :n]
        v = float((d1 * d1).mean())
        if v > 1e-8:
            out.append(float((d2 * d2).mean()) / v)
    ratio = float(np.mean(out)) if out else 4.0
    return round(float(np.clip((1.0 - ratio / 4.0) / 0.75, 0.0, 1.0)), 3)


def laplacian_var(path: str) -> float:
    """Ham yüksek frekans enerjisi (1e6 biriminde); pütürü de sayar."""
    a, _ = load_luma(path)
    l = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
         - 4 * a[1:-1, 1:-1])
    return round(float(l.var()) * 1e6, 1)


def _laplacian(a: np.ndarray) -> np.ndarray:
    return (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
            - 4 * a[1:-1, 1:-1])


def _median_5x5(a: np.ndarray) -> np.ndarray:
    """Ayrılabilir 5'li medyan kaskadı: scipy istemeden tek karelik detay ayıklama.

    Tam 5x5 medyanın yaklaşıkı; ölçüm 3 Kasım'da scipy sürümüyle %10 içinde
    aynı sıralamayı verdi (27.5'e karşılık 25.0), eşikler bu sürümle kalibre.
    """
    m = np.median(np.lib.stride_tricks.sliding_window_view(a, 5, axis=0), axis=-1)
    return np.median(np.lib.stride_tricks.sliding_window_view(m, 5, axis=1), axis=-1)


def detail_index(path: str, long_side: int = 1024) -> float:
    """Eşlenmiş gösterim genişliğinde korelant detay (pütür ayıklanmış).

    Kare 1024 kenara ölçeklenir (küçükse yukarı, büyükse aşağı — izgara sabit
    kalsın ki kademe karşılaştırmaları anlamlı olsun), 5x5 medyanla gren silinir,
    sonra Laplacian varyansı ölçülür; yani yalnızca gerçek yapı kalır. Ölçülen
    dağılım (3 Kasım 2026): 420p/8 taslak 27.5 · 1080p 2268 · 2K native 2407 ·
    FLUX Studio 2963-3000.
    """
    im = Image.open(path).convert("L")
    w, h = im.size
    s = long_side / max(w, h)
    im = im.resize((max(32, int(round(w * s))), max(32, int(round(h * s)))),
                   Image.LANCZOS)
    a = np.asarray(im, dtype=np.float32) / 255.0
    return round(float(_laplacian(_median_5x5(a)).var()) * 1e6, 1)


def structure_index(path: str) -> float:
    """Pütürden arınmış doku: ham HF enerjisi × (1 - pütür indeksi).

    Bulanıklaştırınca iki faktör birden düştüğü için tek yönlü duyarlı; gren
    ekleyince lap_var artar ama (1 - pütür) söner, yani pütür sahte puan üretemez.
    """
    a, _ = load_luma(path)
    return round(laplacian_var(path) * (1.0 - noise_index(a)), 1)


def seam_periods(blend_px: int | None = None) -> tuple[int, ...]:
    """Diki periyodu karo GEOMETRISINDEN gelir, hazır demetten değil.

    Eskiden (192, 384, 512) deneniyordu. 384 aralıklı gerçek diki, 512 katı
    sütunların dörtte birine de denk düşüyor; o sütunlar örneklem başına az
    olduğu için 512'de okunan z sıçrama-kaçak (leakage) ürünüydü. Aynı karede
    ölçüldü: 384 -> 2.08 (temel, temiz) ama 512 -> 3.20 (sahte ihlal). Geometri
    başka temel frekans üretemediği için artık yalnız stride sınanır; alarm ya
    gerçek dikidir ya hiç.
    """
    periods = {TILE_STRIDE_PX}
    if blend_px:
        actual = TILE_MIN_PX - int(blend_px)
        if actual >= 8:
            periods.add(actual)
    return tuple(sorted(periods))


def seam_z(path: str, periods: tuple[int, ...] | None = None) -> dict:
    """Diki skoru: tile aralığındaki sütun sıçraması, faz kaydırmış kontrolüne göre.

    z < 3 artefakt yok; 3-6 kuşkulu; > 6 görünür diki. Periyot, analiz ölçeğine
    göre ölçeklenir ki 4K karede de 2K karede de aynı eşik geçerli olsun.
    """
    a, scale = load_luma(path)
    native_w = Image.open(path).size[0]
    d = np.abs(np.diff(a.mean(axis=0)))
    w = a.shape[1]
    out = {}
    for T in (periods or seam_periods()):
        t = max(8, int(round(T * w / native_w)))
        idx = np.arange(t, w - 1, t)
        if len(idx) < 3:
            continue
        sel = np.arange(1, len(d))
        base = float(d[sel[sel % t != 0]].mean())
        if base <= 1e-9:
            continue
        real = float(d[idx - 1].mean()) / base
        ratios = np.array([float(d[np.arange(o, w - 1, t)].mean()) / base
                           for o in range(1, t) if len(range(o, w - 1, t)) > 3])
        out[T] = round((real - ratios.mean()) / (ratios.std() + 1e-9), 2)
    return out


@dataclass
class QualityReport:
    """Sözleşme sonucu: ölçütler, değişmez ihlalleri, uyarılar, kayıpsız onarımlar."""

    metrics: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    actions: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict:
        return {"metrics": {k: (v if not isinstance(v, dict)
                                 else {str(a): b for a, b in v.items()})
                            for k, v in self.metrics.items()},
                "failures": self.failures, "warnings": self.warnings,
                "actions": self.actions, "ok": self.ok}


SEAM_Z_LIMIT = 3.0
# VAE karosu başına düşen örtüşme; 3 Kasım'da 64 px'de diki ölçüldü, 128 px'de kayboldu.
MIN_BLEND_PX = 96
# Yapısal detay denetimi GÖRELİ: aynı kademe, stil ve arkadaki önceki karelerin
# medyanının %60'ına düşen kare uyarı alır. Mutlak eşik kurulmadı, çünkü ölçüm
# içerik yoğunluğuna da bağlı: sentetik kontrolde seyrek ama keskin sahne 101,
# gerçek üretilmiş kare 2268 okundu — mutlak taban düz stilleri haksız
# cezalandırırdı. (edge_width_px ve halo_index de aynı nedenle atılmıştı:
# kontrollerde tek yönlü değillerdi.) Bu ölçüt bulanıklık merdiveninde tek yönlü:
# 2268 → 1377 → 437 → 159 (r = 0,1,2,3).
REL_DETAIL_FLOOR = 0.6
TILE_MIN_PX = 512
TILE_STRIDE_PX = 384


def tiling_kwargs() -> dict:
    """Sözleşmenin karo geometrisi: 512 px karo, 384 px adım → 128 px onusme.

    Qwen'in diffusers varsayilanlari 256/192 (64 px onusme) olculen diki kusurunun
    kaynagi idi; z=4.11 -> 0.17 bu kuralla indi. FLUX2'nin kendi varsayilani 1024
    px karodur: onusme daha genis ama decode tepesi dort kat buyuk. Tavan sozu
    icin karo kucultulur, onusme esigin altina dusurulmez.
    """
    return {"tile_sample_min_height": TILE_MIN_PX, "tile_sample_min_width": TILE_MIN_PX,
            "tile_sample_stride_height": TILE_STRIDE_PX,
            "tile_sample_stride_width": TILE_STRIDE_PX}


def blend_px() -> int:
    """Sozlesmenin hedef onusme pikseli (kurulmus degil, istenen)."""
    return TILE_MIN_PX - TILE_STRIDE_PX


def blend_px_of(vae) -> int | None:
    """VAE'nin FIILEN onusme pikseli. Telemetri buradan yazilir, sabitten degil."""
    minimum = getattr(vae, "tile_sample_min_height", None)
    stride = getattr(vae, "tile_sample_stride_height", None)
    if minimum and stride:
        return int(minimum - stride)
    size = getattr(vae, "tile_sample_min_size", None)
    factor = getattr(vae, "tile_overlap_factor", None)
    if size and factor is not None:
        return int(round(size * factor))
    return None


def apply_tiling(vae):
    """Karo sozlesmesini her VAE API'sine uygular; kurulan onusme pikselini dondurur.

    diffusers'ta iki farkli yuzey var: Qwen `enable_tiling(min, stride)` aliyor,
    FLUX2 ise argsiz `enable_tiling()` + `tile_sample_min_size`/`tile_overlap_factor`
    oznitelikleri kullanuyor. Motor-agnostik bir soz tek imzaya baglanamaz; hangisi
    varsa onun karsiligi kurulur. Latent izgarasi ornekliden yenilir ki karo boyutu
    degisince gizli uzay kaymasin.
    """
    kwargs = tiling_kwargs()
    if "tile_sample_min_height" in inspect.signature(vae.enable_tiling).parameters:
        vae.enable_tiling(**kwargs)
        return blend_px_of(vae)

    vae.enable_tiling()
    if hasattr(vae, "tile_sample_min_size"):
        ratio = (vae.tile_sample_min_size / vae.tile_latent_min_size
                 if getattr(vae, "tile_latent_min_size", 0) else 8.0)
        vae.tile_sample_min_size = kwargs["tile_sample_min_height"]
        if getattr(vae, "tile_latent_min_size", 0):
            vae.tile_latent_min_size = int(kwargs["tile_sample_min_height"] / ratio)
        vae.tile_overlap_factor = 1 - kwargs["tile_sample_stride_height"] / TILE_MIN_PX
    return blend_px_of(vae)


def plan_invariants(*, native_w: int = 0, native_h: int = 0, target_w: int = 0,
                    target_h: int = 0, vae_blend_px: int | None = None,
                    latent_interpolated: bool = False,
                    peak_vram_mb: float | None = None,
                    vram_ceiling_mb: float | None = None) -> tuple[list, list]:
    """Boru hattının kendine verdiği sözleri denetler; (ihlal, onarım) döner.

    Ölçütlerden farklı olarak burada yanlış pozitif yok: plan ya native grid'dir ya
    değildir. Ghosting geri gelmesi ancak bu kural bozulursa mümkün.
    """
    fails, fixes = [], []
    if latent_interpolated:
        fails.append("latent: hedef boyuta ara kare üzerinden geçildi")
        fixes.append("native grid: latent'i doğrudan hedef kovada aç")
    if vae_blend_px is not None and vae_blend_px < MIN_BLEND_PX:
        fails.append(f"vae_blend={vae_blend_px}px<{MIN_BLEND_PX}px")
        fixes.append("karo örtüşmesini genişlet (kayıpsız, kaliteyi düşürmez)")
    if target_w and native_w and target_w < native_w:
        fails.append("hedef native'dan küçük: çözünürlük çöpe gidiyor")
        fixes.append("native'ı hedefe eşitle")
    if peak_vram_mb and vram_ceiling_mb and peak_vram_mb > vram_ceiling_mb:
        fails.append(f"vram={peak_vram_mb / 1024:.2f}GB>tavan={vram_ceiling_mb / 1024:.2f}GB")
        fixes.append("faz sürekliliği: denoise sırasında VAE'yi, decode sırasında "
                     "transformer'ı CPU'ya al; sonra native kovayı bir kademe indir")
    return fails, fixes


def evaluate(path: str, baseline_detail: float | None = None, **plan) -> QualityReport:
    """Kareyi ölç, plan değişmezleriyle birlikte sözleşmeden geçir.

    baseline_detail: aynı kademe/stil/arkadaki önceki karelerin medyan detay
    indeksi. Verilmezse göreli denetim hiç kurulmaz — tahmin edilmez.
    """
    a, _ = load_luma(path)
    detail = detail_index(path)
    rep = QualityReport(metrics={
        "structure_index": structure_index(path),
        "detail_index": detail,
        "noise_index": noise_index(a),
        "lap_var": laplacian_var(path),
        "seam_z": seam_z(path, seam_periods(plan.get("vae_blend_px"))),
    })
    for period, z in rep.metrics["seam_z"].items():
        if z > SEAM_Z_LIMIT:
            rep.failures.append(f"seam_z@{period}={z}>{SEAM_Z_LIMIT}")
            rep.actions.append("karo örtüşmesini genişlet (kayıpsız)")
    if baseline_detail and detail < REL_DETAIL_FLOOR * baseline_detail:
        rep.warnings.append(
            f"detail={detail} < {REL_DETAIL_FLOOR:.2f}x taban={baseline_detail}: "
            "bu kademenin alıştığı yapısal detayın altına düşüldü")
        rep.actions.append("kademe kovasını büyüt ya da detay geçişini aç "
                           "(model hatası değil, boru hattı kararı)")
    if plan:
        f, x = plan_invariants(**plan)
        rep.failures += f
        for action in x:
            if action not in rep.actions:
                rep.actions.append(action)
    return rep
