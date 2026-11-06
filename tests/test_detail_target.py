"""Uretken detay gecisinin olcek politikasi (torch'suz)."""

import pytest

from web_app.resolutions import (RESOLUTION_MATRIX, detail_target, fit_mp_ceiling,
                                 get_dimensions)

# RTX 5060'da, fast arkadında ölçülen (ızgara MP, denoise tepesi MB).
MEASURED = [(0.459, 4281.9), (0.922, 4554.0), (0.922, 4554.0), (2.058, 5241.6)]


def test_detay_gecisi_yoksa_none():
    assert detail_target(1024, 576, 1024, 576) is None
    assert detail_target(1024, 576, 2048, 1152, enabled=False) is None


def test_16_katina_katlanir():
    w, h = detail_target(1280, 720, 1920, 1080)
    assert w % 16 == 0 and h % 16 == 0
    assert (w, h) == (1920, 1072)


def test_alan_tavani_olcegi_sinirlandirir():
    # 4K 21:9: taban 0.91 MP; 2.0x izin verse 3.6 MP cikardi, alan tavani 2.2 MP'de tutar.
    w, h = detail_target(1456, 624, 3840, 1632, max_scale=2.0, max_mp=2.2)
    assert w * h <= 2.2 * 1_000_000
    assert (w, h) == (2256, 960)


def test_hedeften_buyuk_uretilmez():
    w, h = detail_target(1280, 720, 1440, 816, max_scale=2.0, max_mp=8.0)
    assert w <= 1440 and h <= 816


def test_olcek_tavani_alan_tavanindan_daha_sinirlayici_olabilir():
    w, h = detail_target(1000, 1000, 4000, 4000, max_scale=1.5, max_mp=100.0)
    assert (w, h) == (1488, 1488)


def test_cizim_izgarasi_olculen_tavana_gore_budanir():
    """Kademe adı tepeyi söylemiyor: 1080p detay açıkken 2.06 MP / 5.12 GB idi.

    Eşitlik ızgarayı 1.60 MP'de kesince aynı kademe 4.8 GB'da kalıyor — söz
    uydurma değil, ölçülmüş çizgiden geliyor.
    """
    cap = fit_mp_ceiling(MEASURED, 5.0 * 1024)
    assert cap == pytest.approx(1.604, abs=0.02)
    w, h = detail_target(1280, 720, 1920, 1080, max_mp=cap)
    assert (w, h) == (1680, 944)
    assert w * h / 1e6 <= cap


def test_yeterince_farkli_izgara_yoksa_iddia_kurulmaz():
    assert fit_mp_ceiling([], 5.0 * 1024) is None
    assert fit_mp_ceiling([(0.922, 4554.0)] * 3, 5.0 * 1024) is None
    assert fit_mp_ceiling([(0.5, 5000.0), (1.0, 4000.0), (2.0, 3000.0)], 5.0 * 1024) is None


def test_taban_baslica_tavani_asiyorsa_izgara_care_degil():
    """Yerleşim ağırlıkları tek başına tavanı taşıyorsa küçültme çözmez: None."""
    assert fit_mp_ceiling([(0.5, 6000.0), (0.6, 6100.0), (0.7, 6200.0)], 5.0 * 1024) is None


def test_tum_kademeler_icin_politika_tutarli():
    """Detay gecisi yalniz alan tavanina sigen kademelerde var olur; sigmiyorsa
    taban zaten tavani asmistir ve bunu telemetri detail_scale=1.0 olarak bildirir."""
    for quality, data in RESOLUTION_MATRIX.items():
        for ratio in data:
            if ratio in ("steps", "upscale", "target_res"):
                continue
            base_w, base_h, target_w, target_h, _, upscale = get_dimensions(quality, ratio)
            got = detail_target(base_w, base_h, target_w, target_h)
            if not upscale or (base_w, base_h) == (target_w, target_h):
                assert got is None, (quality, ratio)
            elif base_w * base_h > 2.2 * 1_000_000:
                assert got is None, (quality, ratio)
            else:
                w, h = got
                assert w % 16 == 0 and h % 16 == 0
                assert base_w < w <= target_w and base_h < h <= target_h
                assert w * h <= 2.2 * 1_000_000
