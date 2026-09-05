"""Tests for the pydantic config layer (FR-8.3)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trtship.core.config import (
    BenchmarkConfig,
    BuildConfig,
    CalibrationAlgorithm,
    CalibrationConfig,
    ConfigError,
    PipelineConfig,
    Precision,
    ShapeProfile,
    TritonConfig,
)

# ---- ShapeProfile parsing (FR-4.2 CLI syntax) --------------------------


def test_parse_the_documented_cli_form() -> None:
    p = ShapeProfile.parse("input:1x3x224x224,4x3x224x224,16x3x224x224")
    assert p.input_name == "input"
    assert p.min == (1, 3, 224, 224)
    assert p.opt == (4, 3, 224, 224)
    assert p.max == (16, 3, 224, 224)
    assert p.dynamic_axes == (0,)


def test_static_profile_reports_no_dynamic_axes() -> None:
    p = ShapeProfile.parse("x:1x3,1x3,1x3")
    assert p.dynamic_axes == ()


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("no-colon-here", "Expected 'input_name:"),
        (":1x3,1x3,1x3", "empty input name"),
        ("x:1x3,1x3", "exactly 3 comma-separated shapes"),
        ("x:1x3,1x3,1x3,1x3", "exactly 3 comma-separated shapes"),
        ("x:1x3,bad,1x3", "not a valid shape"),
    ],
)
def test_malformed_profiles_explain_the_expected_form(spec: str, expected: str) -> None:
    with pytest.raises(ConfigError, match=expected):
        ShapeProfile.parse(spec)


def test_min_opt_max_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="min <= opt <= max"):
        ShapeProfile.parse("x:8x3,4x3,16x3")


def test_ranks_must_match() -> None:
    with pytest.raises(ValidationError, match="different ranks"):
        ShapeProfile(input_name="x", min=(1, 3), opt=(1, 3, 4), max=(1, 3, 4))


def test_zero_min_dimension_rejected() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        ShapeProfile.parse("x:0x3,1x3,1x3")


# ---- BuildConfig invariants --------------------------------------------


def test_int8_without_calibration_data_is_rejected() -> None:
    """INT8 with no calibration is the classic silently-bad engine."""
    with pytest.raises(ValidationError, match="requires calibration data"):
        BuildConfig(precision=Precision.INT8)


def test_int8_with_calibration_data_is_accepted(tmp_path) -> None:
    cfg = BuildConfig(
        precision=Precision.INT8,
        calibration=CalibrationConfig(data_path=tmp_path),
    )
    assert cfg.precision is Precision.INT8


def test_fp16_needs_no_calibration() -> None:
    assert BuildConfig(precision=Precision.FP16).calibration is None


def test_auto_mixed_precision_only_applies_to_int8() -> None:
    with pytest.raises(ValidationError, match="only applies to INT8"):
        BuildConfig(precision=Precision.FP16, auto_mixed_precision=True)


def test_shape_profiles_make_the_build_dynamic() -> None:
    cfg = BuildConfig(shape_profiles=(ShapeProfile.parse("x:1x3,4x3,16x3"),))
    assert cfg.is_dynamic


def test_default_build_is_static() -> None:
    assert not BuildConfig().is_dynamic


def test_unknown_config_keys_are_rejected() -> None:
    """extra='forbid' turns a typo into an error instead of a silent no-op."""
    with pytest.raises(ValidationError):
        BuildConfig(precison="fp16")  # type: ignore[call-arg]


# ---- CalibrationConfig --------------------------------------------------


def test_percentile_setting_requires_the_percentile_algorithm() -> None:
    with pytest.raises(ValidationError, match="only applies to the 'percentile'"):
        CalibrationConfig(algorithm=CalibrationAlgorithm.ENTROPY, percentile=99.0)


def test_percentile_algorithm_accepts_a_percentile() -> None:
    cfg = CalibrationConfig(algorithm=CalibrationAlgorithm.PERCENTILE, percentile=99.9)
    assert cfg.percentile == 99.9


# ---- BenchmarkConfig (NFR-1.3) -----------------------------------------


def test_warmup_floor_is_enforced() -> None:
    """NFR-1.3 mandates >= 20 discarded warmup iterations."""
    with pytest.raises(ValidationError):
        BenchmarkConfig(warmup_iters=5)


def test_unknown_backend_lists_the_valid_ones() -> None:
    with pytest.raises(ValidationError, match="Unknown benchmark backend"):
        BenchmarkConfig(backends=("pytorch", "tensorflow"))


# ---- TritonConfig (FR-6.2) ---------------------------------------------


def test_preferred_batch_sizes_cannot_exceed_max_batch_size() -> None:
    with pytest.raises(ValidationError, match="exceed max_batch_size"):
        TritonConfig(max_batch_size=4, preferred_batch_sizes=(4, 8))


def test_triton_defaults_are_self_consistent() -> None:
    cfg = TritonConfig()
    assert all(b <= cfg.max_batch_size for b in cfg.preferred_batch_sizes)


# ---- YAML loading (FR-8.3) ---------------------------------------------


def test_yaml_round_trip(tmp_path) -> None:
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        """
        model_name: resnet18
        output_dir: ./out
        build:
          precision: fp16
          workspace_gb: 2.0
        benchmark:
          batch_sizes: [1, 8]
        triton:
          max_batch_size: 8
        """
    )
    cfg = PipelineConfig.from_yaml(path)
    assert cfg.model_name == "resnet18"
    assert cfg.build.precision is Precision.FP16
    assert cfg.benchmark.batch_sizes == (1, 8)
    assert cfg.triton.max_batch_size == 8


def test_missing_yaml_names_the_path(tmp_path) -> None:
    with pytest.raises(ConfigError, match="Config file not found"):
        PipelineConfig.from_yaml(tmp_path / "nope.yaml")


def test_invalid_yaml_names_the_file(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        PipelineConfig.from_yaml(path)


def test_empty_yaml_is_rejected(tmp_path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    with pytest.raises(ConfigError, match="is empty"):
        PipelineConfig.from_yaml(path)


def test_non_mapping_yaml_is_rejected(tmp_path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ConfigError, match="mapping at the top level"):
        PipelineConfig.from_yaml(path)


def test_pipeline_config_is_manifest_serializable() -> None:
    """NFR-8.1: the manifest must capture the full config as JSON."""
    import json

    d = PipelineConfig(model_name="x").to_dict()
    json.dumps(d)  # must not raise
    assert d["build"]["precision"] == "fp16"
