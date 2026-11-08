"""Prompt derleyicisi kullanıcı isteğiyle çelişen çapa ekleyemez."""

import pytest

from web_app.engine import compile_physically_grounded_prompt as compile
from web_app.engine import detect_medium

STOREFRONT = (
    'Editorial architectural photograph of a neighborhood storefront at blue hour. '
    'Above the entrance, a warm-white neon sign in hand-bent cursive reads exactly "OpenRouter", '
    'the only text in the scene. Large windows reveal a cozy lived-in interior. '
    'Straight-on composition, quiet street with no people.'
)


def test_istenen_tabela_yazisi_silinmez():
    compiled = compile(STOREFRONT, "pure_raw")
    assert "zero legible text" not in compiled
    assert 'reads exactly "OpenRouter" (letters O p e n R o u t e r in that order' in compiled


def test_heceleme_tirnagin_hemen_ardina_gomer():
    compiled = compile(STOREFRONT, "pure_raw")
    assert compiled.index("OpenRouter") < compiled.index("O p e n")


def test_cok_kelimeli_alinti_soze_bozunur():
    compiled = compile('A shop sign that reads exactly "Cafe Milano".', "pure_raw")
    assert "letters C a f e, then M i l a n o in that order" in compiled
    assert "one unbroken word" not in compiled


def test_uzun_metinde_heceleme_yapilmaz():
    from web_app.engine import spelling_anchor
    assert spelling_anchor("OpenRouter") != ""
    assert spelling_anchor("Supercalifragilisticexpialidocious") == ""


def test_kural_disi_alinti_genel_capaya_duser():
    long_text = "OpenRouter Coffee Roasters Limited Edition"
    compiled = compile(f'A banner reads exactly "{long_text}".', "pure_raw")
    assert "in that order" not in compiled
    assert "render the quoted word exactly as written" in compiled


def test_olumsuz_ornekte_anatomi_capasi_cikmaz():
    assert "five distinct slender fingers" not in compile(STOREFRONT, "pure_raw")


def test_hyphenated_belirtec_onzeyne_tetik_lemaz():
    assert "five distinct slender fingers" not in compile("A hand-bent cursive sign on a wall.", "pure_raw")


def test_turkce_olumsuzluk_de_tetik_lemez():
    assert "five distinct slender fingers" not in compile("İnsan olmayan boş bir sokak.", "pure_raw")


def test_sahne_olmayan_sey_uydurulmaz():
    # "interior" tek başına şömine çapasını tetiklememeliydi.
    assert "fireplace" not in compile(STOREFRONT, "pure_raw")
    assert "fireplace" in compile("A cozy living room with a fireplace at dusk.", "pure_raw")


def test_gercek_insan_sahnesine_anatomi_capasi_eklenmez():
    # A/B (seed 7102, 1920x1072): çapa ellerin büyümesine ve parmakların
    # birleşmesine yol açtı; kaldırıldı.
    prompt = "Portrait of a young woman, hands resting on a table."
    assert compile(prompt, "pure_raw") == prompt


def test_geo_kapatilincak_yalniz_stil_eklenir():
    assert compile(STOREFRONT, "pure_raw", enable_geo=False) == STOREFRONT


def test_matbu_nokta_virgul_dizgi_bozuklugu_yok():
    compiled = compile(STOREFRONT, "matte_chiaroscuro")
    assert ".," not in compiled
    assert ",." not in compiled
    assert ", ," not in compiled


@pytest.mark.parametrize("style", ["pure_raw", "cinematic_35mm", "anime_concept"])
def test_cikti_hic_bos_degil(style):
    assert compile(STOREFRONT, style).strip()


def test_fotograf_diyen_prompte_boyamali_stil_eklenmez():
    compiled = compile("Editorial photograph of a quiet street at dusk.", "matte_chiaroscuro")
    assert "Atmospheric concept art painting" not in compiled
    assert "painterly textured finish" not in compiled


def test_kullanici_boyama_dediyse_stil_kalir():
    compiled = compile("A watercolor painting of a fox in snow.", "matte_chiaroscuro")
    assert compiled.startswith("Atmospheric concept art painting")
    assert compiled.endswith("soft ambient shadow transitions")


def test_uygun_ortamda_stil_eklenir():
    compiled = compile("Editorial photograph of a quiet street at dusk.", "cinematic_35mm")
    assert compiled.startswith("Cinematic 35mm film still")


def test_ortam_sozu_yoksa_stil_yine_uygulanir():
    assert compile("A red bicycle leaning on a wall.", "matte_chiaroscuro").startswith(
        "Atmospheric concept art painting")


def test_celikide_fotograf_once_gelir():
    # "photorealistic painting" gibi cümlelerde kullanıcı aslında fotoğrafik sonuç istiyor.
    assert detect_medium("A photorealistic painting of a harbor.") == "photo"
    assert "Atmospheric concept art painting" not in compile(
        "A photorealistic painting of a harbor.", "matte_chiaroscuro")
