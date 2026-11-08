import pytest

from web_app.resident_math import (
    A_INTERCEPT_GB,
    A_MP_SLOPE_GB,
    NF4_GB,
    TE_STREAM_PEAK_GB,
    VAE_PEAK_GB,
    activation_gb,
    plan,
)


def phase_peaks(payload):
    return {p["id"]: p["peak_gb"] for p in payload["phases"]}


def test_model_matches_measured_denoise_peaks():
    assert abs(plan(5.0, 1024, 576)["peak_gb"] - 3.82) < 0.02
    assert abs(plan(5.0, 1280, 704)["peak_gb"] - 3.96) < 0.02


def test_resolution_is_snapped_to_the_vae_grid():
    assert plan(5.0, 1000, 566)["resolution"] == "992x544"
    assert plan(5.0, 64, 64)["resolution"] == "256x256"


def test_denoise_is_the_binding_phase_at_every_budget():
    payload = plan(5.0, 1280, 704)
    peaks = phase_peaks(payload)
    assert payload["peak_gb"] == max(peaks.values())
    assert peaks["denoise"] > peaks["encode"] > peaks["decode"]
    assert payload["headroom_gb"] == round(payload["budget_gb"] - payload["peak_gb"], 3)


def test_feasibility_follows_the_inequality():
    assert plan(4.0, 1280, 704)["feasible"] is True
    assert plan(3.9, 1280, 704)["feasible"] is False


def test_strict_phases_keep_the_transformer_out_of_encode_and_decode():
    strict = plan(5.0, 1024, 576, strict=True)
    loose = plan(5.0, 1024, 576, strict=False)
    assert phase_peaks(strict)["encode"] == TE_STREAM_PEAK_GB
    assert phase_peaks(strict)["decode"] == VAE_PEAK_GB
    assert phase_peaks(loose)["encode"] == pytest.approx(TE_STREAM_PEAK_GB + NF4_GB[4])
    assert loose["peak_gb"] > strict["peak_gb"]
    assert loose["ram"]["runtime_gb"] < strict["ram"]["runtime_gb"]


def test_activation_grows_monotonically_with_megapixels():
    sizes = [(1024, 576), (1280, 704), (1536, 864), (1920, 1080)]
    values = [activation_gb(w, h) for w, h in sizes]
    assert values == sorted(values)
    assert activation_gb(1280, 704) == pytest.approx(A_INTERCEPT_GB + A_MP_SLOPE_GB * (1280 * 704 / 1e6))


def test_max_native_mp_is_the_feasibility_boundary():
    payload = plan(5.0, 1024, 576)
    limit = payload["max_native_mp"]
    at_limit = int((limit * 1e6) ** 0.5) // 32 * 32
    assert plan(5.0, at_limit, at_limit)["feasible"] is True
    assert plan(5.0, 2048, 2048)["feasible"] is False


def test_budget_below_the_transformer_has_no_native_ceiling():
    assert plan(3.0, 1024, 576)["max_native_mp"] is None


def test_bit_options_are_ordered_by_footprint():
    peaks = {o["bits"]: o["peak_gb"] for o in plan(2.0, 256, 256)["options"]}
    assert peaks[2] < peaks[3] < peaks[4]
    states = {o["bits"]: o["feasible"] for o in plan(2.0, 256, 256)["options"]}
    assert states == {4: False, 3: False, 2: True}


def test_only_4_bit_is_supported_and_other_widths_fall_back():
    assert plan(5.0, 1024, 576, bits=8)["bits"] == 4
    assert plan(5.0, 1024, 576, bits=8)["supported"] is True
    assert plan(5.0, 1024, 576, bits=2)["supported"] is False
