"""trtship — PyTorch to TensorRT deployment.

Importing this package must never import ``tensorrt``. Development happens on
macOS/arm64 where TensorRT does not exist, so the CPU-safe core has to import
cleanly there; ``tests/unit/test_no_tensorrt_import.py`` enforces it.

Public names are resolved lazily so that ``import trtship`` stays cheap and does
not pull in torch until something actually needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.0.1.dev0"

if TYPE_CHECKING:
    from trtship.core.config import BuildConfig, PipelineConfig, Precision
    from trtship.core.model_spec import ModelSpec, TensorSpec
    from trtship.env import Environment, detect

__all__ = [
    "BuildConfig",
    "Environment",
    "ModelSpec",
    "PipelineConfig",
    "Precision",
    "TensorSpec",
    "__version__",
    "detect",
]

_LAZY: dict[str, str] = {
    "ModelSpec": "trtship.core.model_spec",
    "TensorSpec": "trtship.core.model_spec",
    "PipelineConfig": "trtship.core.config",
    "BuildConfig": "trtship.core.config",
    "Precision": "trtship.core.config",
    "Environment": "trtship.env",
    "detect": "trtship.env",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
