"""Enforces the rule that makes CPU-machine development possible.

``import trtship`` must never transitively import ``tensorrt``. macOS/arm64 has
no TensorRT wheel, so the moment some module grows a top-level ``import
tensorrt``, the entire package — including the parts that need no GPU at all —
stops importing on the development machine.

This is checked in a subprocess with ``tensorrt`` actively blocked, because
checking ``sys.modules`` in-process would pass on any machine that simply does
not have TensorRT installed, which is exactly the machine that cannot detect the
regression.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Modules that must import with TensorRT unavailable.
CPU_SAFE_MODULES = [
    "trtship",
    "trtship.core",
    "trtship.core.config",
    "trtship.core.model_spec",
    "trtship.env",
    "trtship.cli",
]

_BLOCK_AND_IMPORT = textwrap.dedent(
    """
    import sys

    class _BlockTensorRT:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "tensorrt" or fullname.startswith("tensorrt."):
                raise ImportError(
                    "tensorrt import was blocked by test_no_tensorrt_import"
                )
            return None

    sys.meta_path.insert(0, _BlockTensorRT())

    import importlib
    for name in {modules!r}:
        importlib.import_module(name)

    assert "tensorrt" not in sys.modules, (
        "tensorrt ended up in sys.modules after importing the CPU-safe core"
    )
    print("OK")
    """
)


def test_cpu_safe_modules_import_without_tensorrt() -> None:
    script = _BLOCK_AND_IMPORT.format(modules=CPU_SAFE_MODULES)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        "The CPU-safe core failed to import with tensorrt unavailable.\n"
        "Something grew a top-level `import tensorrt`; move it inside the "
        "function that needs it.\n\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_importing_trtship_does_not_import_tensorrt_here() -> None:
    """Cheap in-process check, meaningful on a machine that does have TensorRT."""
    script = textwrap.dedent(
        """
        import sys
        import trtship
        assert "tensorrt" not in sys.modules, sorted(
            m for m in sys.modules if "tensorrt" in m
        )
        print("OK")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr
