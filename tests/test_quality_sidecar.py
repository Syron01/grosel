"""Sunucu bağlantısı: telemetri + sidecar kalite bloğu.

GPU'yu yeniden başlatmadan doğrulanan kısım: attach_quality'nin planı doğru
kurması, sözleşmeyi kareye uygulaması, sonucu sidecar'a eklemesi ve ölçüm
batarsa iç hata detayını kullanıcıya sızdırmaması.
"""
import json

import numpy as np
from PIL import Image

from web_app import server


def _frame(tmp_path, noise=0.0, seed=5):
    a = np.full((256, 448), 120.0)
    a[:, 200:] = 40.0
    if noise:
        a += np.random.default_rng(seed).normal(0, noise, a.shape)
    p = tmp_path / "frame.png"
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L").save(p)
    return str(p)


def _tel(path, tmp_path, monkeypatch, peak_mb=2.41 * 1024):
    monkeypatch.setattr(server, "META_DIR", str(tmp_path / ".meta"))
    monkeypatch.setattr(server, "load_runs", lambda: [])
    (tmp_path / ".meta").mkdir(exist_ok=True)
    return {"filepath": path, "native_resolution": "448x256",
            "resolution": "448x256", "vae_blend_px": server.quality_math.blend_px(),
            "quality": "720p", "style_key": "editorial", "peak_vram_mb": peak_mb}


def _history(detail, n=6, style="editorial", tier="720p"):
    return [{"quality": tier, "style_key": style,
             "quality_report": {"metrics": {"detail_index": detail}}} for _ in range(n)]


def test_goreli_taban_ayni_kademe_ve_stilden_toplanir(monkeypatch):
    monkeypatch.setattr(server, "load_runs",
                        lambda: _history(2000.0) + _history(50.0, style="çizgi"))
    assert server.detail_baseline({"quality": "720p", "style_key": "editorial"}) == 2000.0
    # eşik altı örnek: taban uydurulmaz
    monkeypatch.setattr(server, "load_runs", lambda: _history(2000.0, n=2))
    assert server.detail_baseline({"quality": "720p", "style_key": "editorial"}) is None
    assert server.detail_baseline({"quality": "2K", "style_key": "editorial"}) is None


def test_attach_quality_yumusak_kareyi_tarihine_kiyaslayarak_uyarir(tmp_path, monkeypatch):
    path = _frame(tmp_path)
    tel = _tel(path, tmp_path, monkeypatch)
    monkeypatch.setattr(server, "load_runs", lambda: _history(1e6))
    report = server.attach_quality(tel)["quality_report"]
    assert report["warnings"], "tarihine göre çökmüş detay görünmeli"
    assert report["ok"] is True, "uyarı çizimi iptal ettirmez"


def test_attach_quality_telemetriyi_ve_sidecari_doldurur(tmp_path, monkeypatch):
    path = _frame(tmp_path)
    tel = server.attach_quality(_tel(path, tmp_path, monkeypatch))
    assert tel["quality_report"]["ok"] is True, tel["quality_report"]["failures"]
    assert tel["quality_report"]["metrics"]["structure_index"] > 0
    assert tel["quality"] == "720p", "kademe etiketi ezilmemeli (runs.json ona göre gruplanıyor)"
    sidecar = tmp_path / ".meta" / "frame.png.json"
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["quality_report"]["ok"] is True


def test_attach_quality_grenli_kareyi_cezalandirmaz(tmp_path, monkeypatch):
    """Grenli karede pütür yüksek ama sözleşme yalnız diki/plan bozuksa patlar."""
    path = _frame(tmp_path, noise=8.0)
    tel = server.attach_quality(_tel(path, tmp_path, monkeypatch))
    assert tel["quality_report"]["ok"] is True


def test_attach_quality_vram_tavanini_plan_degismezine_yazar(tmp_path, monkeypatch):
    path = _frame(tmp_path)
    tel = _tel(path, tmp_path, monkeypatch, peak_mb=6 * 1024)
    q = server.attach_quality(tel)["quality_report"]
    assert any("vram=" in f for f in q["failures"])
    assert q["actions"], "tavan aşımında kayıpsız onarım önerisi beklenir"


def test_attach_quality_kare_yoksa_ici_yapi_sizdirmaz(tmp_path, monkeypatch):
    tel = _tel(str(tmp_path / "yok.png"), tmp_path, monkeypatch)
    q = server.attach_quality(tel)["quality_report"]
    assert q["ok"] is None
    blob = json.dumps(q)
    assert "Traceback" not in blob and "web_app" not in blob


def test_attach_quality_bozuk_png_uretim_durmaaz(tmp_path, monkeypatch):
    path = tmp_path / "bozuk.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n bozuk")
    q = server.attach_quality(_tel(str(path), tmp_path, monkeypatch))["quality_report"]
    assert q["ok"] is None
    assert "ölçüm yapılamadı" in json.dumps(q, ensure_ascii=False)
