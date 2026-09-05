"""Pydantic configuration models.

CLI flags and YAML files (FR-8.3) both parse into these same objects, so there
is exactly one config code path rather than two that drift. Validation lives
here, which means a bad config is rejected with a field-level message before any
slow work starts.

CPU-safe: no CUDA or TensorRT imports.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "BenchmarkConfig",
    "BuildConfig",
    "CalibrationAlgorithm",
    "CalibrationConfig",
    "ConfigError",
    "PipelineConfig",
    "Precision",
    "ShapeProfile",
    "TritonConfig",
]


class ConfigError(ValueError):
    """Raised for config that is syntactically valid but semantically wrong."""


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


class CalibrationAlgorithm(str, Enum):
    """INT8 calibration algorithms (FR-3.3)."""

    ENTROPY = "entropy"  # IInt8EntropyCalibrator2, TensorRT's default
    MINMAX = "minmax"  # IInt8MinMaxCalibrator
    PERCENTILE = "percentile"  # custom


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


# Matches "1x3x224x224" — the shape syntax used by --dynamic-shapes.
_SHAPE_RE = re.compile(r"^\d+(x\d+)*$")


class ShapeProfile(_Base):
    """A TensorRT optimization profile for one input (FR-4.2).

    ``min``/``opt``/``max`` are full shapes of equal rank. ``opt`` is what
    TensorRT tunes tactics for, so it should be the shape you actually serve
    most of the time, not the midpoint of the range.
    """

    input_name: str
    min: tuple[int, ...]
    opt: tuple[int, ...]
    max: tuple[int, ...]

    @model_validator(mode="after")
    def _check_consistent(self) -> ShapeProfile:
        ranks = {len(self.min), len(self.opt), len(self.max)}
        if len(ranks) != 1:
            raise ValueError(
                f"min/opt/max shapes for input {self.input_name!r} have different ranks "
                f"({len(self.min)}, {len(self.opt)}, {len(self.max)}); they must all describe "
                f"the same tensor."
            )
        for axis, (lo, mid, hi) in enumerate(zip(self.min, self.opt, self.max, strict=True)):
            if not (lo <= mid <= hi):
                raise ValueError(
                    f"Input {self.input_name!r} axis {axis}: expected min <= opt <= max, "
                    f"got {lo} <= {mid} <= {hi}."
                )
            if lo < 1:
                raise ValueError(
                    f"Input {self.input_name!r} axis {axis}: min dimension must be at least 1, "
                    f"got {lo}."
                )
        return self

    @property
    def dynamic_axes(self) -> tuple[int, ...]:
        return tuple(i for i, (lo, hi) in enumerate(zip(self.min, self.max, strict=True)) if lo != hi)

    @classmethod
    def parse(cls, spec: str) -> ShapeProfile:
        """Parse the CLI form ``name:1x3x224x224,4x3x224x224,16x3x224x224``.

        The three shapes are min, opt, max in that order (FR-4.2).
        """
        if ":" not in spec:
            raise ConfigError(
                f"Malformed shape profile {spec!r}. Expected "
                f"'input_name:MINxSHAPE,OPTxSHAPE,MAXxSHAPE', for example "
                f"'input_0:1x3x224x224,4x3x224x224,16x3x224x224'."
            )
        name, _, shapes_part = spec.partition(":")
        name = name.strip()
        if not name:
            raise ConfigError(f"Shape profile {spec!r} has an empty input name.")

        parts = [p.strip() for p in shapes_part.split(",")]
        if len(parts) != 3:
            raise ConfigError(
                f"Shape profile for {name!r} needs exactly 3 comma-separated shapes "
                f"(min, opt, max); got {len(parts)}: {shapes_part!r}."
            )
        parsed: list[tuple[int, ...]] = []
        for label, part in zip(("min", "opt", "max"), parts, strict=True):
            if not _SHAPE_RE.match(part):
                raise ConfigError(
                    f"Shape profile for {name!r}: {label} shape {part!r} is not a valid shape. "
                    f"Use dimensions joined by 'x', for example '1x3x224x224'."
                )
            parsed.append(tuple(int(d) for d in part.split("x")))
        return cls(input_name=name, min=parsed[0], opt=parsed[1], max=parsed[2])


class CalibrationConfig(_Base):
    """INT8 calibration settings (FR-3.2, FR-3.3)."""

    data_path: Path | None = None
    algorithm: CalibrationAlgorithm = CalibrationAlgorithm.ENTROPY
    batch_size: int = Field(default=8, ge=1)
    num_batches: int = Field(default=64, ge=1)
    percentile: float = Field(default=99.99, gt=0, le=100)
    cache_path: Path | None = None

    @model_validator(mode="after")
    def _percentile_only_for_percentile(self) -> CalibrationConfig:
        if self.algorithm is not CalibrationAlgorithm.PERCENTILE and self.percentile != 99.99:
            raise ValueError(
                f"percentile={self.percentile} was set but algorithm is "
                f"{self.algorithm.value!r}; percentile only applies to the 'percentile' "
                f"algorithm."
            )
        return self


class BuildConfig(_Base):
    """TensorRT engine build settings (FR-5.1)."""

    precision: Precision = Precision.FP16
    workspace_gb: float = Field(default=4.0, gt=0)
    shape_profiles: tuple[ShapeProfile, ...] = ()
    auto_mixed_precision: bool = False
    mixed_precision_layers: int = Field(default=5, ge=0)
    timing_cache: Path | None = None
    use_cache: bool = True
    calibration: CalibrationConfig | None = None

    @model_validator(mode="after")
    def _int8_needs_calibration(self) -> BuildConfig:
        if self.precision is Precision.INT8 and (
            self.calibration is None or self.calibration.data_path is None
        ):
            raise ValueError(
                "precision='int8' requires calibration data. Pass --calibration-data "
                "pointing at representative inputs; INT8 without calibration produces an "
                "engine whose accuracy is effectively unpredictable."
            )
        if self.auto_mixed_precision and self.precision is not Precision.INT8:
            raise ValueError(
                f"auto_mixed_precision only applies to INT8 builds, but precision is "
                f"{self.precision.value!r}."
            )
        return self

    @property
    def is_dynamic(self) -> bool:
        return bool(self.shape_profiles)


class BenchmarkConfig(_Base):
    """Benchmark methodology (FR-7.1, NFR-1.3)."""

    backends: tuple[str, ...] = ("pytorch", "onnxruntime", "tensorrt")
    batch_sizes: tuple[int, ...] = (1, 4, 16)
    warmup_iters: int = Field(default=20, ge=20)
    measure_iters: int = Field(default=200, ge=1)
    validation_data: Path | None = None
    accuracy_metric: str = "auto"

    @field_validator("backends")
    @classmethod
    def _known_backends(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        known = {"pytorch", "onnxruntime", "tensorrt"}
        unknown = set(v) - known
        if unknown:
            raise ValueError(
                f"Unknown benchmark backend(s): {', '.join(sorted(unknown))}. "
                f"Available: {', '.join(sorted(known))}."
            )
        if not v:
            raise ValueError("At least one benchmark backend must be selected.")
        return v


class TritonConfig(_Base):
    """Triton model repository generation (FR-6.1, FR-6.2)."""

    model_name: str | None = None
    model_version: int = Field(default=1, ge=1)
    max_batch_size: int = Field(default=16, ge=0)
    dynamic_batching: bool = True
    preferred_batch_sizes: tuple[int, ...] = (4, 8)
    max_queue_delay_us: int = Field(default=100, ge=0)
    instance_count: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _preferred_within_max(self) -> TritonConfig:
        if self.max_batch_size and self.preferred_batch_sizes:
            too_big = [b for b in self.preferred_batch_sizes if b > self.max_batch_size]
            if too_big:
                raise ValueError(
                    f"preferred_batch_sizes {too_big} exceed max_batch_size "
                    f"{self.max_batch_size}; Triton would reject this config."
                )
        return self


class PipelineConfig(_Base):
    """A full `trtship ship` run (FR-8.1, FR-8.3)."""

    model: Path | None = None
    model_name: str | None = None
    example_input: tuple[int, ...] | None = None
    output_dir: Path = Path("./trtship_output")
    build: BuildConfig = BuildConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    triton: TritonConfig = TritonConfig()
    verify_tolerance: float = Field(default=1e-3, gt=0)

    @classmethod
    def from_yaml(cls, path: Path | str) -> PipelineConfig:
        """Load from a YAML file, with errors that name the file (FR-8.3)."""
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text())
        except yaml.YAMLError as exc:
            raise ConfigError(f"{p} is not valid YAML: {exc}") from exc
        if raw is None:
            raise ConfigError(f"{p} is empty.")
        if not isinstance(raw, dict):
            raise ConfigError(
                f"{p} must contain a YAML mapping at the top level, got "
                f"{type(raw).__name__}."
            )
        return cls.model_validate(raw)

    def to_dict(self) -> dict[str, Any]:
        """Manifest-safe view (NFR-8.1)."""
        return self.model_dump(mode="json")
