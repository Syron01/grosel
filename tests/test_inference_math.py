import numpy as np

from web_app.inference_math import (LinearFit, Plan, _constraints_ok, _fit_linear, _loocv,
                                    _pareto, _solve, _violation)


def make_plan(name, time_s, peak_gb, native_mp, mode="fast"):
    return Plan(filename=None, prompt=name, quality="1080p", steps=4, resolution="1280x720",
                native_mp=native_mp, target_mp=native_mp, cold=False, offload=False,
                total_time=time_s, peak_gb=peak_gb, clip=None, mode=mode, tag=name)


def test_pareto_keeps_only_undominated_plans():
    slow = make_plan("yavas", 6.0, 4.5, 0.9)
    fast = make_plan("hizli", 4.0, 4.4, 0.9)
    quality = make_plan("kalite", 7.0, 4.6, 1.04)
    frontier = _pareto([slow, fast, quality])
    assert [p.tag for p in frontier] == ["hizli", "kalite"]


def test_pareto_ignores_modes_that_are_not_benchmarked():
    frontier = _pareto([make_plan("a", 4.0, 4.0, 0.9), make_plan("b", 5.0, 4.0, 0.9, mode="qwen")])
    assert [p.tag for p in frontier] == ["a"]


def test_constraints_are_inclusive_and_violation_is_zero_inside():
    plan = make_plan("sinir", 4.0, 4.4, 0.9)
    assert _constraints_ok(plan, 4.4, 4.0, 0.9) is True
    assert _violation(plan, 4.4, 4.0, 0.9) == 0.0
    assert _constraints_ok(plan, 4.3, 4.0, 0.9) is False
    assert _violation(plan, 4.3, 4.0, 0.9) > 0.0


def test_solver_picks_the_objective_correct_plan():
    plans = [make_plan("hizli", 3.0, 4.6, 0.59), make_plan("kalite", 6.0, 4.4, 0.92)]
    assert _solve(plans, 5.0, 8.0, 0.4, "time")["chosen"]["tag"] == "hizli"
    assert _solve(plans, 5.0, 8.0, 0.4, "quality")["chosen"]["tag"] == "kalite"
    assert _solve(plans, 5.0, 8.0, 0.4, "vram")["chosen"]["tag"] == "kalite"


def test_solver_reports_the_closest_plan_when_constraints_cannot_be_met():
    near_miss = make_plan("zaman_asimi", 9.0, 4.4, 0.90)
    low_quality = make_plan("kalite_eksik", 3.0, 4.4, 0.20)
    result = _solve([near_miss, low_quality], 5.0, 8.0, 0.45, "quality")
    assert result["status"] == "infeasible"
    assert result["chosen"]["tag"] == "zaman_asimi"
    assert result["violations"] == {"vram_gb": 0.0, "time_s": 1.0, "native_mp": 0.0}


def test_solver_surfaces_binding_constraints():
    plans = [make_plan("dar", 4.0, 4.99, 0.9), make_plan("pahali", 9.0, 4.2, 1.2)]
    result = _solve(plans, 5.0, 8.0, 0.45, "quality")
    assert result["status"] == "optimal"
    assert "VRAM bütçesi" in result["binding"]


def test_ridge_fit_recovers_a_known_linear_law():
    x = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 5.0], [4.0, 3.0], [5.0, 8.0]])
    y = 2.0 + 3.0 * x[:, 0] - 1.0 * x[:, 1]
    fit = _fit_linear(np.column_stack([np.ones(len(x)), x]), y)
    assert isinstance(fit, LinearFit)
    assert fit.r2 > 0.999
    assert np.allclose([fit.predict([1.0, *row]) for row in x], y, atol=1e-2)


def test_constant_feature_column_acts_as_the_intercept():
    x = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0], [1.0, 8.0]])
    y = 5.0 + 1.5 * x[:, 1]
    fit = _fit_linear(x, y)
    assert fit.r2 > 0.999
    assert np.allclose([fit.predict(row) for row in x], y, atol=1e-2)


def test_loocv_reports_generalisation_not_training_fit():
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    rows = np.column_stack([np.ones(5), x])
    linear = _loocv(rows, 2.0 * x[:, 0] + 1.0)
    assert linear["n"] == 5
    assert linear["r2"] > 0.99

    noise = np.random.default_rng(7).normal(0, 4.0, size=5)
    scrambled = _loocv(rows, noise)
    assert scrambled["r2"] < 0.5


def test_loocv_is_conservative_below_four_samples():
    assert _loocv([[1.0], [2.0]], [1.0, 2.0]) == {"r2": 0.0, "mae": 0.0, "n": 2}
