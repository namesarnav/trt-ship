"""Integration tests that validate the GPU machine itself.

These are the first thing test-gpu.yml runs. If the self-hosted runner is
misprovisioned — wrong torch build, missing TensorRT, a card that cannot do the
precisions this project claims — that should fail here with a clear message,
not thirty minutes later inside an engine build.

Everything here is marked ``gpu`` and skipped on CPU-only machines, which is
every run on the development Mac.
"""

from __future__ import annotations

import pytest

from trtship.env import detect

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def env():
    e = detect()
    if not e.cuda_available:
        pytest.skip("no CUDA device — these tests describe the GPU runner only")
    return e


def test_cuda_is_visible_to_torch(env) -> None:
    assert env.gpus, "CUDA reports available but no devices were enumerated"
    assert env.cuda_version is not None


def test_tensorrt_is_installed(env) -> None:
    """The single most common runner misconfiguration."""
    assert env.has_tensorrt, (
        "CUDA is available but TensorRT is not installed on this runner. "
        'Install with: pip install "trtship[gpu]"'
    )


def test_tensorrt_imports_and_reports_a_version(env) -> None:
    """Guarded import — this module must stay importable on the Mac."""
    import tensorrt as trt

    assert trt.__version__.startswith("10."), (
        f"trtship targets TensorRT 10.x, runner has {trt.__version__}. "
        f"See docs/compatibility.md."
    )


def test_a_builder_can_actually_be_constructed(env) -> None:
    """Catches a working import over a broken CUDA/driver pairing."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    assert builder is not None


def test_engine_building_stage_is_reported_available(env) -> None:
    from trtship.env import Stage

    assert Stage.BUILD in env.available_stages()
    assert env.can_build_engines


def test_card_supports_the_precisions_this_project_claims(env) -> None:
    """Fails loudly rather than letting Phase 4 discover it.

    Not xfail: if the card cannot do INT8, the scope claim needs changing, and a
    silently-expected failure would let that slide.
    """
    gpu = env.primary_gpu
    assert gpu is not None
    assert gpu.supports("fp16"), f"{gpu.name} (sm {gpu.sm}) cannot do FP16"
    assert gpu.supports("int8"), (
        f"{gpu.name} (sm {gpu.sm}) does not support INT8. The INT8 calibration "
        f"work in Phase 4 cannot be validated on this hardware — update "
        f"docs/compatibility.md and rescope."
    )


def test_docker_present_for_the_triton_smoke_test(env) -> None:
    if not env.docker_available:
        pytest.skip("Docker unavailable — Triton smoke test (FR-6.3) cannot run here")
    from trtship.env import Stage

    assert Stage.SMOKE_TEST in env.available_stages()
