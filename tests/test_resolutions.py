import pytest

from web_app.resolutions import ASPECT_RATIOS, RESOLUTION_MATRIX, get_dimensions


@pytest.mark.parametrize("quality", sorted(RESOLUTION_MATRIX))
@pytest.mark.parametrize("ratio", ASPECT_RATIOS)
def test_every_tier_has_a_grid_aligned_ratio(quality, ratio):
    base_w, base_h, target_w, target_h, steps, upscale = get_dimensions(quality, ratio)
    assert base_w % 16 == 0 and base_h % 16 == 0
    assert target_w >= base_w and target_h >= base_h
    assert 3 <= steps <= 4
    assert upscale == (quality in ("1080p", "2K", "4K"))


def test_advertised_1080p_is_a_upscaled_native_720p_render():
    assert get_dimensions("1080p", "16:9") == (1280, 720, 1920, 1080, 4, True)
    assert get_dimensions("420p", "16:9") == (640, 368, 640, 368, 3, False)


def test_unknown_quality_falls_back_to_1080p():
    assert get_dimensions("8K", "16:9") == get_dimensions("1080p", "16:9")


def test_unknown_ratio_falls_back_and_scales_target():
    assert get_dimensions("1080p", "1:8") == (1024, 576, 2048, 1152, 4, True)
    assert get_dimensions("720p", "1:8") == (1024, 576, 1024, 576, 4, False)


def test_target_matrices_only_exist_where_upscaling_happens():
    for quality, data in RESOLUTION_MATRIX.items():
        if data.get("upscale"):
            assert set(data["target_res"]) == set(ASPECT_RATIOS)
        else:
            assert "target_res" not in data
