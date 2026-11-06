"""VRAM tavanı uçuş öncesi denetim: ızgara budanır, kademe indirilir, söz kırılmaz.

Denetim ÖLÇÜMLE kurulur: runs.json'daki aynı arkada + aynı kademe + aynı oranın
medyanı. Ölçüm yoksa Atelier NF4 eşitliğini kullanır (muhafazakâr), Studio'da
iddia kurulmaz — tahmin uydurup kalite fatura edilmez.
"""
import pytest

from web_app import server


def _runs(peak_mb, tier, ratio="16:9", n=4, backend=None):
    backend = backend or server.BACKEND
    return [{"quality": tier, "aspect_ratio": ratio, "backend": backend,
             "peak_vram_mb": peak_mb} for _ in range(n)]


@pytest.fixture
def history(monkeypatch):
    def apply(runs):
        monkeypatch.setattr(server, "load_runs", lambda: runs)
        monkeypatch.setattr(server, "VRAM_CEILING_GB", 5.0)
    return apply


def test_olcum_yoksa_studio_iddia_kurmaz(history, monkeypatch):
    history([])
    monkeypatch.setattr(server, "BACKEND", "fast")
    assert server.predict_peak_gb("1080p", "16:9") is None
    assert server.guard_quality("1080p", "16:9") == ("1080p", None)


def test_bakendi_bilinmeyen_eski_kayit_kullanimaz(history):
    runs = [{"quality": "1080p", "aspect_ratio": "16:9", "peak_vram_mb": 6 * 1024}
            for _ in range(6)]
    history(runs)
    assert server.measured_peak_gb("1080p", "16:9") is None


def test_tavan_asiliyorsa_kademe_bire_iner(history):
    history(_runs(6.2 * 1024, "1080p") + _runs(3.4 * 1024, "720p"))
    tier, guard = server.guard_quality("1080p", "16:9")
    assert tier == "720p"
    assert guard["steps_down"] == 1 and guard["used"] == "720p"
    assert guard["predicted_peak_gb"] == pytest.approx(3.4)


def test_temiz_kademe_donmez(history):
    history(_runs(4.6 * 1024, "1080p"))
    tier, guard = server.guard_quality("1080p", "16:9")
    assert tier == "1080p" and guard is None


def test_hicbiri_sigmazsa_uretim_baslamaz(history):
    history(_runs(7.5 * 1024, "420p"))
    tier, guard = server.guard_quality("420p", "16:9")
    assert tier is None
    assert guard["reason"] == "hiçbir kademe tavana sığmıyor"


def test_tavan_kapatildiginda_denetim_sessiz_gecer(history, monkeypatch):
    history(_runs(9.0 * 1024, "1080p"))
    monkeypatch.setattr(server, "VRAM_CEILING_GB", 0.0)
    assert server.guard_quality("1080p", "16:9") == ("1080p", None)


def test_bilinmeyen_kademe_pas_gecer(history):
    history(_runs(9.0 * 1024, "8K"))
    assert server.guard_quality("8K", "16:9") == ("8K", None)


def test_uc_adet_ornek_yetersiz_se_tavan_denetim_yapmaz(history):
    history(_runs(6.5 * 1024, "1080p", n=2))
    assert server.measured_peak_gb("1080p", "16:9") is None
    tier, guard = server.guard_quality("1080p", "16:9")
    assert tier == "1080p" and guard is None


def test_kademe_tablosu_olculeni_verir_az_ornegi_isaretler(history):
    runs = _runs(4 * 1024, "1080p", n=3) + [{"quality": "2K", "aspect_ratio": "16:9",
                                             "backend": server.BACKEND,
                                             "peak_vram_mb": 6.2 * 1024,
                                             "resolution": "2048x1152",
                                             "native_resolution": "1760x992",
                                             "total_time": 260.0}]
    history(runs)
    table = server.measured_by_tier()
    assert table["1080p"]["runs"] == 3 and table["1080p"]["provisional"] is False
    assert table["2K"]["provisional"] is True
    assert table["2K"]["worst_gb"] == pytest.approx(6.2)
    assert table["2K"]["native"] == "1760x992"
    assert table["2K"]["median_time_s"] == pytest.approx(260.0)
    assert "420p" not in table, "hiç ölçülmemiş kademe tabloda uydurulmaz"


def test_kademe_tablosu_boru_hatti_neslini_tasi(history):
    """Panel güncel davranışı söyler: eski neslin 6 GB'lık koşuları pencere dışında kalır.

    Detay tavanı gibi bir boru hattı değişikliğinden sonra tüm geçmişi medyanda
    eritmek, sahnedeki kartın bugün ne çizdiğini gizlerdi.
    """
    history(_runs(6.0 * 1024, "1080p", n=5) + _runs(4.8 * 1024, "1080p", n=5))
    table = server.measured_by_tier()
    assert table["1080p"]["runs"] == 5
    assert table["1080p"]["worst_gb"] == pytest.approx(4.8)


def test_kademe_tablosu_baska_arkadin_kosulunu_bu_sozun_kaniti_yapmaz(history):
    other = "classic" if server.BACKEND != "classic" else "fast"
    history(_runs(4 * 1024, "1080p", backend=other))
    assert server.measured_by_tier() == {}


def _grid(w, h, denoise_mb, n=3, backend=None):
    backend = backend or server.BACKEND
    return [{"quality": "1080p", "aspect_ratio": "16:9", "backend": backend,
             "native_resolution": f"{w}x{h}", "denoise_peak_mb": denoise_mb,
             "peak_vram_mb": denoise_mb} for _ in range(n)]


def test_tavan_sozi_cizim_izgarasindan_korunur(history):
    """Kademe adı tepeyi söylemiyor: 1080p detay açıkken 2.06 MP, kapalıyken 0.92 MP.

    Ölçülmüş üç ızgara çizgiyi kuruyor; tavanın altına kalan en büyük ızgara
    1.6 MP — yani söz, kademe indirilmeden de tutuyor.
    """
    history(_grid(896, 512, 4281.9, n=1) + _grid(1280, 720, 4554.0)
            + _grid(1920, 1072, 5241.6))
    assert server.detail_mp_cap() == pytest.approx(1.604, abs=0.03)


def test_izgara_basi_tavana_sigmiyorsa_cap_kurulmaz(history):
    """Ağırlık yerleşimi tek başına tavanı aşıyorsa ızgarayı küçültmek çare değil."""
    history(_grid(1280, 720, 6200.0) + _grid(1920, 1072, 6800.0))
    assert server.detail_mp_cap() is None


def test_tavan_kapatilsa_izgara_da_budaklanmaz(history, monkeypatch):
    history(_grid(1280, 720, 4554.0) + _grid(1920, 1072, 5241.6))
    monkeypatch.setattr(server, "VRAM_CEILING_GB", 0.0)
    assert server.detail_mp_cap() is None


def test_inilen_kademe_olculmemisse_cumlde_sayi_uydurulmaz(history):
    """Canlıda yakalanan çökme: tahmini tepe None iken f-string format hatası
    üretimi 'motor günlüğünü denetleyin' ile düşürüyordu."""
    history(_runs(6.2 * 1024, "1080p") + _runs(3.4 * 1024, "720p", n=1))
    tier, guard = server.guard_quality("1080p", "16:9")
    assert (tier, guard["used"]) == ("720p", "720p")
    assert guard["measured"] is False
    assert "henüz ölçülmedi" in server.guard_status(guard)


def test_olculmus_indirme_sayiyi_soyler(history):
    history(_runs(6.2 * 1024, "1080p") + _runs(3.4 * 1024, "720p"))
    tier, guard = server.guard_quality("1080p", "16:9")
    assert guard["measured"] is True
    assert "tahmini tepe 3.40 GB" in server.guard_status(guard)
