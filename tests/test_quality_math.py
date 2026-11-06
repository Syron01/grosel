"""Kalite sözleşmesi testleri: ölçütler sentetik kontrollere karşı tek yönlü olmalı.

Gerçek kareler galeride duruyor ama test onlara bağımlı olamaz; burada bilinen
pozitif/negatif örnekler üretiliyor (bulanıklaştırma, diki, gren, plato kenarı).
"""
import numpy as np
from PIL import Image, ImageFilter

from web_app import quality_math as qm


def _write(tmp_path, name, arr):
    p = tmp_path / name
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L").save(p)
    return str(p)


def _blur(a, radius):
    return np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")
                      .filter(ImageFilter.GaussianBlur(radius)), dtype=np.float64)


def _step_field(seed=7):
    """Platoları düz, kenarları keskin sentetik sahne: sözleşme için temiz referans."""
    rng = np.random.default_rng(seed)
    a = np.full((512, 512), 120.0)
    for x0 in (64, 180, 300, 420):
        a[:, x0:] = rng.choice([40.0, 200.0])
    a += rng.normal(0, 1.2, a.shape)          # hafif gren, gerçekçi olsun diye
    return a


def test_yapi_indeksi_bulaniklikla_tek_yonlu_duser(tmp_path):
    a = _step_field()
    sharp = _write(tmp_path, "sharp.png", a)
    soft1 = _write(tmp_path, "soft1.png", _blur(a, 1.5))
    soft2 = _write(tmp_path, "soft2.png", _blur(a, 3.0))
    s = [qm.structure_index(p) for p in (sharp, soft1, soft2)]
    assert s[0] > s[1] > s[2], f"yapi indeksi bulanikliga tepki vermedi: {s}"


def test_putur_indeksi_greni_yapiyi_ayirir(tmp_path):
    """Gren ~1.0, düz alan 0.0; 8-bit karede kuantum tabanı ~0.4'ü geçmez.

    Taban ölçüldü: 256 karelik akışkan rampa float'ta 0.000, PNG'e yazılınca
    0.392 okunuyor. Eşik bu tabanın üstünde tutulur ki gerçek 8-bit karelerde
    test çökmesin.
    """
    flat = np.full((256, 256), 128.0)
    rng = np.random.default_rng(1)
    grainy = flat + rng.normal(0, 12, flat.shape)
    smooth = np.tile(np.linspace(40, 200, 256), (256, 1))
    g = qm.noise_index(qm.load_luma(_write(tmp_path, "g.png", grainy))[0])
    s = qm.noise_index(qm.load_luma(_write(tmp_path, "s.png", smooth))[0])
    assert g > 0.9
    assert s < g * 0.5
    assert qm.noise_index(qm.load_luma(_write(tmp_path, "f.png", flat))[0]) == 0.0


def test_diki_sentetik_bantlamayi_yakalar(tmp_path):
    a = np.full((256, 1024), 100.0)
    rng = np.random.default_rng(3)
    a += rng.normal(0, 2, a.shape)
    clean = _write(tmp_path, "clean.png", a)
    banded = a.copy()
    for x in range(0, 1024, 192):              # tile araliginda sabit siralama
        banded[:, x:x + 96] += 6.0
    banded_path = _write(tmp_path, "banded.png", banded)
    assert qm.seam_z(clean, periods=(192,)).get(192, 99) < qm.seam_z(banded_path, periods=(192,))[192]
    assert qm.seam_z(banded_path, periods=(192,))[192] > qm.SEAM_Z_LIMIT


def test_diki_periyodu_geometriden_gelir_sizinti_elenir(tmp_path):
    """384 aralıklı gerçek diki 512 katı sütunların 1/4'üne de denk düşüyor;
    eskiden aynı kare 384'te 2.08 (temiz) ama 512'de 3.20 (ihlal) okunuyordu.

    Hazır demet yerine geometrinin üretebileceği temel frekans sınanır: sentetik
    384 dikisi kendi periyodunda güçlü yakalanır (ölçüldü: 14.1), komşu periyot
    artık sözleşmeyi kırmaz.
    """
    a = np.full((256, 1920), 100.0)
    a += np.random.default_rng(3).normal(0, 2, a.shape)
    banded = a.copy()
    for x in range(0, 1920, 384):
        banded[:, x:x + 96] += 6.0
    path = _write(tmp_path, "banded384.png", banded)
    assert qm.seam_periods(128) == (384,)
    assert qm.seam_z(path)[384] > qm.SEAM_Z_LIMIT
    assert set(qm.seam_z(path)) == {384}


def test_kucuk_karede_yeterli_karo_tekrari_yoksa_sonuc_bos(tmp_path):
    """640 px genişlikte 384 adımlı yalnız ~1.7 karo var: sayı uydurmak yerine ölçülmez."""
    a = np.full((256, 640), 100.0)
    a += np.random.default_rng(1).normal(0, 2, a.shape)
    assert qm.seam_z(_write(tmp_path, "small.png", a)) == {}


def test_plan_degismezleri_ihlali_gorur():
    fails, fixes = qm.plan_invariants(native_w=1536, native_h=864, target_w=1920,
                                      target_h=1080, vae_blend_px=64,
                                      peak_vram_mb=6 * 1024, vram_ceiling_mb=5 * 1024)
    assert any("vae_blend" in f for f in fails)
    assert any("vram=" in f for f in fails)
    assert any("karo örtüşmesini" in x for x in fixes)


def test_plan_degismezleri_temiz_plani_vuramaz():
    fails, _ = qm.plan_invariants(native_w=1760, native_h=992, target_w=1920,
                                  target_h=1080, vae_blend_px=qm.blend_px(),
                                  peak_vram_mb=2.41 * 1024,
                                  vram_ceiling_mb=5 * 1024)
    assert fails == []


def test_hedef_native_kuculse_bozulma_sinyali():
    fails, _ = qm.plan_invariants(native_w=2048, native_h=1152, target_w=1280,
                                  target_h=720)
    assert any("çözünürlük çöpe" in f for f in fails)


def test_evaluate_keskin_sentetik_kareye_gecer(tmp_path):
    rep = qm.evaluate(_write(tmp_path, "sharp.png", _step_field()))
    assert rep.ok, rep.failures
    assert {"structure_index", "detail_index", "noise_index", "lap_var",
            "seam_z"} <= rep.metrics.keys()
    assert rep.metrics["detail_index"] > 0
    assert rep.warnings == [], "taban yoksa göreli uyarı da yok"
    assert rep.as_dict()["ok"] is True          # JSON'a girecek bicim sabit kalsin


def test_detay_uyarisi_goreli_tabandan_gelir(tmp_path):
    """Sözleşme 'yumuşak kare'yi ancak bir önceki kademeyle kıyaslayınca görür."""
    path = _write(tmp_path, "sharp.png", _step_field())
    detail = qm.detail_index(path)
    rep = qm.evaluate(path, baseline_detail=detail / qm.REL_DETAIL_FLOOR * 3)
    assert rep.warnings and rep.actions
    assert rep.ok, "düşük detay stiller meşru olabilir; üretilen çizimi iptal ettirmez"
    tam = qm.evaluate(path, baseline_detail=detail)
    assert tam.warnings == []


def test_detay_indeksi_bulaniklikla_tek_yonlu_duser(tmp_path):
    a = _step_field(seed=11)
    values = [qm.detail_index(_write(tmp_path, f"d{i}.png", _blur(a, r) if r else a))
              for i, r in enumerate((0, 1, 2, 3))]
    assert all(x > y for x, y in zip(values, values[1:])), values


def test_tiling_kurali_varsayilandan_genis():
    kw = qm.tiling_kwargs()
    assert kw["tile_sample_min_height"] - kw["tile_sample_stride_height"] >= qm.MIN_BLEND_PX
