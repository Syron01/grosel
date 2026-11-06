import math

import pytest
from pydantic import ValidationError

from web_app.schemas import GenerateRequest, MathSolveRequest, OptimizeRequest


def valid(**overrides):
    payload = {"prompt": "matte chiaroscuro studio shot"}
    payload.update(overrides)
    return GenerateRequest.model_validate(payload)


def test_defaults_are_the_advertised_safe_path():
    req = valid()
    assert req.quality == "1080p"
    assert req.aspect_ratio == "16:9"
    assert req.count == 1
    assert req.steps == 0
    assert req.seed == -1


def test_prompt_is_stripped_and_required():
    assert valid(prompt="  gece sokagi  ").prompt == "gece sokagi"
    with pytest.raises(ValidationError):
        valid(prompt="     ")
    with pytest.raises(ValidationError):
        GenerateRequest.model_validate({"style": "pure_raw"})


def test_prompt_length_bounds():
    assert valid(prompt="a" * 4000).prompt == "a" * 4000
    with pytest.raises(ValidationError):
        valid(prompt="a" * 4001)


@pytest.mark.parametrize("field,value", [
    ("quality", "8K"),
    ("aspect_ratio", "5:4"),
    ("count", 0),
    ("count", 5),
    ("steps", 49),
    ("steps", -1),
    ("refine_steps", 5),
    ("refine_sigma", 0.01),
    ("refine_sigma", 0.61),
    ("seed", -2),
    ("seed", 2 ** 31 - 1),
    ("sigmas", []),
    ("sigmas", [1.5]),
    ("sigmas", [0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1]),
    ("tag", "u" * 25),
    ("style", "s" * 65),
])
def test_out_of_range_input_is_rejected(field, value):
    with pytest.raises(ValidationError):
        valid(**{field: value})


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        valid(**{"prompt": "ok", "pipeline_kwargs": {"num_inference_steps": 9999}})


def test_boundary_values_are_accepted():
    assert valid(sigmas=[0, 1]).sigmas == [0.0, 1.0]
    assert valid(seed=2 ** 31 - 2).seed == 2 ** 31 - 2
    assert valid(count=4).count == 4


@pytest.mark.parametrize("field,value", [
    ("budget_gb", float("nan")),
    ("budget_gb", 20.0),
    ("max_time_s", 0.0),
    ("min_native_mp", 5.0),
    ("steps", 5),
    ("objective", "speed"),
])
def test_math_solver_constraints_are_bounded(field, value):
    with pytest.raises(ValidationError):
        MathSolveRequest.model_validate({field: value})


def test_math_solver_accepts_nan_free_payload():
    req = MathSolveRequest.model_validate({"steps": 6, "objective": "vram", "warm": False})
    assert math.isfinite(req.budget_gb)
    assert req.steps == 6 and req.warm is False


def test_optimize_request_bounds():
    assert OptimizeRequest.model_validate({"prompt": "tokyo rain"}).style == "matte_chiaroscuro"
    with pytest.raises(ValidationError):
        OptimizeRequest.model_validate({"prompt": "  "})
