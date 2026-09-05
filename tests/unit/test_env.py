"""Tests for environment detection (NFR-2.3).

These run on whatever machine invokes them, so they assert on invariants rather
than on a particular GPU being present.
"""

from __future__ import annotations

import dataclasses

from trtship.env import Environment, GpuInfo, Stage, detect


def _env(**overrides) -> Environment:
    base = Environment(
        platform="Linux",
        machine="x86_64",
        python_version="3.11.9",
        torch_version="2.4.0",
        cuda_available=True,
        cuda_version="12.4",
        driver_version="550.54",
        tensorrt_version="10.2.0",
        onnx_version="1.16.1",
        onnxruntime_version="1.18.1",
        onnxruntime_gpu=True,
        docker_available=True,
        gpus=[GpuInfo(0, "NVIDIA RTX 4090", (8, 9), 24564)],
    )
    return dataclasses.replace(base, **overrides)


def test_detect_runs_on_this_machine() -> None:
    env = detect()
    assert env.platform
    assert env.python_version
    assert isinstance(env.gpus, list)


def test_detect_is_json_serializable() -> None:
    """The environment block of the run manifest (NFR-8.1)."""
    import json

    json.dumps(detect().to_dict())


# ---- precision gating by compute capability ----------------------------


def test_ada_supports_fp8() -> None:
    gpu = GpuInfo(0, "RTX 4090", (8, 9), 24564)
    assert gpu.supports("fp8")
    assert gpu.supports("int8")
    assert gpu.supports_sparsity


def test_pascal_has_int8_but_no_fp8_and_no_sparsity() -> None:
    gpu = GpuInfo(0, "GTX 1080 Ti", (6, 1), 11264)
    assert gpu.supports("int8")
    assert not gpu.supports("fp8")
    assert not gpu.supports_sparsity


def test_old_card_lacks_int8() -> None:
    gpu = GpuInfo(0, "GTX 960", (5, 2), 4096)
    assert gpu.supports("fp32")
    assert not gpu.supports("fp16")
    assert not gpu.supports("int8")


def test_supported_precisions_lists_only_what_the_card_can_do() -> None:
    turing = GpuInfo(0, "RTX 2080", (7, 5), 8192)
    assert turing.supported_precisions == ["fp32", "fp16", "int8"]


# ---- stage availability -------------------------------------------------


def test_full_gpu_machine_can_run_everything() -> None:
    assert set(_env().available_stages()) == set(Stage)


def test_mac_reports_cpu_stages_only() -> None:
    """The actual development machine: arm64 macOS, no CUDA, no TensorRT."""
    env = _env(
        platform="Darwin",
        machine="arm64",
        cuda_available=False,
        cuda_version=None,
        driver_version=None,
        tensorrt_version=None,
        onnxruntime_gpu=False,
        gpus=[],
    )
    stages = set(env.available_stages())
    assert stages == {Stage.EXPORT, Stage.VERIFY, Stage.TRITON_CONFIG, Stage.REPORT}
    assert not env.can_build_engines


def test_mac_blocker_explains_that_macos_has_no_tensorrt_at_all() -> None:
    env = _env(platform="Darwin", cuda_available=False, tensorrt_version=None, gpus=[])
    joined = " ".join(env.blockers())
    assert "macOS has no CUDA support" in joined
    assert "Linux/Windows machine with an NVIDIA GPU" in joined


def test_cuda_without_tensorrt_suggests_the_extra() -> None:
    env = _env(tensorrt_version=None)
    assert any('pip install "trtship[gpu]"' in b for b in env.blockers())
    assert not env.can_build_engines


def test_missing_onnxruntime_removes_the_verify_stage() -> None:
    env = _env(onnxruntime_version=None, onnxruntime_gpu=False)
    assert Stage.VERIFY not in env.available_stages()
    assert Stage.EXPORT in env.available_stages()


def test_no_docker_removes_only_the_smoke_test() -> None:
    env = _env(docker_available=False)
    stages = set(env.available_stages())
    assert Stage.SMOKE_TEST not in stages
    assert Stage.BUILD in stages


def test_no_torch_blocks_everything_loudly() -> None:
    env = _env(torch_version=None)
    assert any("no stage can run" in b for b in env.blockers())


def test_healthy_machine_reports_no_blockers() -> None:
    assert _env().blockers() == []


# ---- cache fingerprint (FR-5.3) ----------------------------------------


def test_fingerprint_changes_with_tensorrt_version() -> None:
    assert _env().cache_fingerprint() != _env(tensorrt_version="10.3.0").cache_fingerprint()


def test_fingerprint_changes_with_gpu_architecture() -> None:
    other = _env(gpus=[GpuInfo(0, "A100", (8, 0), 40960)])
    assert _env().cache_fingerprint() != other.cache_fingerprint()


def test_fingerprint_ignores_python_version() -> None:
    """A Python upgrade must not invalidate every cached engine."""
    assert _env().cache_fingerprint() == _env(python_version="3.12.4").cache_fingerprint()
