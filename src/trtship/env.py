"""Environment and capability detection (NFR-2.3).

trtship is developed on machines that cannot run it end to end: macOS/arm64 has
no CUDA and no TensorRT wheel at all. Rather than fail deep inside a build with
an ImportError, every entry point asks this module what the current machine can
actually do, and stages that cannot run say so up front.

Nothing here imports ``tensorrt`` at module scope. Probing is done through
``importlib.util.find_spec`` and guarded local imports so that ``import trtship``
succeeds on a laptop with no NVIDIA hardware.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["Environment", "GpuInfo", "Stage", "detect"]


class Stage(str, Enum):
    """Pipeline stages, used to report what this machine can run."""

    EXPORT = "export"
    VERIFY = "verify"
    CALIBRATE = "calibrate"
    BUILD = "build"
    BENCHMARK = "benchmark"
    TRITON_CONFIG = "triton-config"
    SMOKE_TEST = "smoke-test"
    REPORT = "report"


# Compute-capability floors for each TensorRT precision. Sourced from NVIDIA's
# support matrix; the point of encoding them is that `doctor` can tell you in
# week 1 that your card cannot do a thing, rather than you finding out in a
# calibration run in week 10.
_PRECISION_FLOORS: dict[str, tuple[float, str]] = {
    "fp32": (0.0, "always available"),
    "fp16": (5.3, "Maxwell 5.3+"),
    "int8": (6.1, "Pascal 6.1+ (dp4a); fast INT8 tensor cores from Turing 7.5"),
    "fp8": (8.9, "Ada 8.9 / Hopper 9.0+"),
}
_SPARSITY_FLOOR = 8.0  # Ampere


def _module_version(name: str) -> str | None:
    """Version of an installed module, without importing heavy ones needlessly."""
    if importlib.util.find_spec(name) is None:
        return None
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        try:
            mod = importlib.import_module(name)
            return str(getattr(mod, "__version__", "unknown"))
        except Exception:
            return "unknown"


@dataclass(frozen=True)
class GpuInfo:
    """One CUDA device."""

    index: int
    name: str
    compute_capability: tuple[int, int]
    total_memory_mb: int

    @property
    def sm(self) -> float:
        major, minor = self.compute_capability
        return float(f"{major}.{minor}")

    def supports(self, precision: str) -> bool:
        floor, _ = _PRECISION_FLOORS.get(precision.lower(), (float("inf"), ""))
        return self.sm >= floor

    @property
    def supported_precisions(self) -> list[str]:
        return [p for p in _PRECISION_FLOORS if self.supports(p)]

    @property
    def supports_sparsity(self) -> bool:
        return self.sm >= _SPARSITY_FLOOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "total_memory_mb": self.total_memory_mb,
            "supported_precisions": self.supported_precisions,
            "supports_sparsity": self.supports_sparsity,
        }


@dataclass(frozen=True)
class Environment:
    """Everything about this machine that affects what trtship can do.

    Also the environment block of the run manifest (NFR-8.1) and one input to
    the engine cache key (FR-5.3) — an engine built against a different TensorRT
    version or a different compute capability must not be reused.
    """

    platform: str
    machine: str
    python_version: str
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    driver_version: str | None
    tensorrt_version: str | None
    onnx_version: str | None
    onnxruntime_version: str | None
    onnxruntime_gpu: bool
    docker_available: bool
    gpus: list[GpuInfo] = field(default_factory=list)

    # ---- capability queries ----------------------------------------------

    @property
    def primary_gpu(self) -> GpuInfo | None:
        return self.gpus[0] if self.gpus else None

    @property
    def has_tensorrt(self) -> bool:
        return self.tensorrt_version is not None

    @property
    def can_build_engines(self) -> bool:
        return self.cuda_available and self.has_tensorrt

    def available_stages(self) -> list[Stage]:
        """Stages this machine can actually run."""
        stages: list[Stage] = [Stage.TRITON_CONFIG, Stage.REPORT]
        if self.torch_version:
            stages.insert(0, Stage.EXPORT)
            if self.onnxruntime_version:
                stages.insert(1, Stage.VERIFY)
        if self.can_build_engines:
            stages += [Stage.BUILD, Stage.CALIBRATE, Stage.BENCHMARK]
            if self.docker_available:
                stages.append(Stage.SMOKE_TEST)
        return sorted(set(stages), key=lambda s: list(Stage).index(s))

    def blockers(self) -> list[str]:
        """Human-readable reasons a stage is unavailable.

        Written to be actionable: each line says what is missing and what it
        costs you, so `doctor` never leaves the user guessing (NFR-4.3).
        """
        out: list[str] = []
        if self.torch_version is None:
            out.append("PyTorch is not installed — no stage can run. Install with: pip install torch")
        if not self.cuda_available:
            if self.platform == "Darwin":
                out.append(
                    "No CUDA: macOS has no CUDA support at all, and NVIDIA ships no TensorRT "
                    "build for it. Engine build, calibration, benchmarking and the Triton smoke "
                    "test must run on a Linux/Windows machine with an NVIDIA GPU."
                )
            else:
                out.append(
                    "No CUDA device visible to PyTorch. Check `nvidia-smi` and that your torch "
                    "build is the CUDA one, not the CPU-only wheel."
                )
        elif not self.has_tensorrt:
            out.append(
                "CUDA is available but TensorRT is not installed. Install with: "
                'pip install "trtship[gpu]"'
            )
        if self.onnxruntime_version is None:
            out.append(
                "onnxruntime is not installed — export verification (FR-2.4) will be skipped. "
                'Install with: pip install "trtship[export]"'
            )
        if not self.docker_available:
            out.append(
                "Docker was not found — the Triton smoke test (FR-6.3) cannot run here."
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "machine": self.machine,
            "python": self.python_version,
            "torch": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda": self.cuda_version,
            "driver": self.driver_version,
            "tensorrt": self.tensorrt_version,
            "onnx": self.onnx_version,
            "onnxruntime": self.onnxruntime_version,
            "onnxruntime_gpu": self.onnxruntime_gpu,
            "docker": self.docker_available,
            "gpus": [g.to_dict() for g in self.gpus],
        }

    def cache_fingerprint(self) -> str:
        """The environment half of the engine cache key (FR-5.3).

        Deliberately narrow: only the things that actually invalidate a built
        engine. Adding the Python version here would cause spurious rebuilds.
        """
        gpu = self.primary_gpu
        return "|".join(
            [
                f"trt={self.tensorrt_version}",
                f"cuda={self.cuda_version}",
                f"sm={gpu.compute_capability if gpu else None}",
            ]
        )


def _detect_gpus() -> tuple[list[GpuInfo], bool, str | None]:
    """Enumerate CUDA devices via torch. Returns (gpus, cuda_available, cuda_version)."""
    if importlib.util.find_spec("torch") is None:
        return [], False, None
    try:
        import torch
    except Exception:
        return [], False, None

    if not torch.cuda.is_available():
        return [], False, None

    cuda_version = getattr(torch.version, "cuda", None)
    gpus: list[GpuInfo] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(
            GpuInfo(
                index=i,
                name=props.name,
                compute_capability=(props.major, props.minor),
                total_memory_mb=props.total_memory // (1024 * 1024),
            )
        )
    return gpus, True, cuda_version


def _detect_driver() -> str | None:
    """NVIDIA driver version via nvidia-smi, if present."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def _detect_docker() -> bool:
    """Whether a Docker daemon is reachable — not merely whether the CLI exists."""
    if shutil.which("docker") is None:
        return False
    try:
        out = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


def detect() -> Environment:
    """Probe this machine. Safe to call anywhere, on any platform."""
    gpus, cuda_available, cuda_version = _detect_gpus()
    return Environment(
        platform=platform.system(),
        machine=platform.machine(),
        python_version=".".join(str(v) for v in sys.version_info[:3]),
        torch_version=_module_version("torch"),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        driver_version=_detect_driver(),
        tensorrt_version=_module_version("tensorrt"),
        onnx_version=_module_version("onnx"),
        onnxruntime_version=_module_version("onnxruntime") or _module_version("onnxruntime-gpu"),
        onnxruntime_gpu=_module_version("onnxruntime-gpu") is not None,
        docker_available=_detect_docker(),
        gpus=gpus,
    )
