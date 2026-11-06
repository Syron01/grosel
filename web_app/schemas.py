from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Prompt = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
]
Sigma = Annotated[float, Field(ge=0, le=1)]


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: Prompt
    style: str = Field(default="matte_chiaroscuro", max_length=64)


class MathSolveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget_gb: float = Field(default=5.0, ge=0.5, le=8.0)
    max_time_s: float = Field(default=8.0, ge=0.5, le=60.0)
    min_native_mp: float = Field(default=0.45, ge=0.1, le=2.0)
    steps: Literal[3, 4, 6] = 4
    warm: bool = True
    objective: Literal["time", "vram", "quality"] = "quality"


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: Prompt
    style: str = Field(default="matte_chiaroscuro", max_length=64)
    aspect_ratio: Literal["16:9", "1:1", "9:16", "4:3", "3:4", "2:3", "21:9"] = "16:9"
    quality: Literal["420p", "720p", "1080p", "2K", "4K"] = "1080p"
    enable_crpf: bool = True
    enable_geo: bool = True
    steps: int = Field(default=0, ge=0, le=48)
    refine_steps: int = Field(default=0, ge=0, le=4)
    detail_pass: bool = True
    refine_sigma: float = Field(default=0.15, ge=0.02, le=0.60)
    sigmas: list[Sigma] | None = Field(default=None, min_length=1, max_length=8)
    count: int = Field(default=1, ge=1, le=4)
    seed: int = Field(default=-1, ge=-1, le=2**31 - 2)
    tag: str = Field(default="studio", max_length=24)
