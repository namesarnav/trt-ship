"""The internal representation of "a model to be optimized".

Every downstream stage (export, calibrate, build, benchmark, deploy) operates on
a :class:`ModelSpec` and never on a raw ``nn.Module``. That indirection is what
lets a second source format (a bare ONNX file with no PyTorch model behind it,
say) be added by changing only ingestion.

This module is CPU-safe: it imports ``torch`` but never ``tensorrt``.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

__all__ = ["ModelSpec", "ModelSpecError", "TensorSpec"]


class ModelSpecError(ValueError):
    """Raised when a model and its example inputs are inconsistent.

    Carries a message written for the user, not a PyTorch traceback (FR-1.3).
    """


# TensorRT, ONNX and Triton all name dtypes differently. We keep the torch name
# as the canonical string because it round-trips through the run manifest and is
# the one a PyTorch user recognises; each backend maps outward from here.
_TORCH_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
_NAMES_TO_TORCH_DTYPE = {v: k for k, v in _TORCH_DTYPE_NAMES.items()}


def dtype_to_str(dtype: torch.dtype) -> str:
    """Canonical string name for a torch dtype."""
    try:
        return _TORCH_DTYPE_NAMES[dtype]
    except KeyError:
        raise ModelSpecError(
            f"dtype {dtype} is not supported by trtship. Supported dtypes: "
            f"{', '.join(sorted(_TORCH_DTYPE_NAMES.values()))}."
        ) from None


def str_to_dtype(name: str) -> torch.dtype:
    """Inverse of :func:`dtype_to_str`."""
    try:
        return _NAMES_TO_TORCH_DTYPE[name]
    except KeyError:
        raise ModelSpecError(
            f"Unknown dtype name {name!r}. Supported: "
            f"{', '.join(sorted(_NAMES_TO_TORCH_DTYPE))}."
        ) from None


@dataclass(frozen=True)
class TensorSpec:
    """One model input or output.

    A ``None`` entry in ``shape`` marks a dynamic dimension. Static-shape builds
    (FR-4.1) require every entry to be concrete; dynamic builds (FR-4.2) resolve
    the ``None`` entries against an optimization profile.
    """

    name: str
    shape: tuple[int | None, ...]
    dtype: str

    @property
    def is_dynamic(self) -> bool:
        return any(d is None for d in self.shape)

    @property
    def dynamic_axes(self) -> tuple[int, ...]:
        return tuple(i for i, d in enumerate(self.shape) if d is None)

    @property
    def rank(self) -> int:
        return len(self.shape)

    def torch_dtype(self) -> torch.dtype:
        return str_to_dtype(self.dtype)

    def resolve(self, **axis_sizes: int) -> tuple[int, ...]:
        """Fill dynamic axes by index, e.g. ``spec.resolve(**{"0": 4})``.

        Raises if any dynamic axis is left unresolved, rather than silently
        substituting 1 and producing an engine that is wrong in a way that only
        shows up under load.
        """
        out: list[int] = []
        for i, d in enumerate(self.shape):
            if d is not None:
                out.append(d)
                continue
            key = str(i)
            if key not in axis_sizes:
                raise ModelSpecError(
                    f"Input {self.name!r} has a dynamic dimension at axis {i} that was "
                    f"not given a size. Pass a value for axis {i}, or build a static "
                    f"engine instead."
                )
            out.append(axis_sizes[key])
        return tuple(out)

    @classmethod
    def from_tensor(cls, name: str, tensor: torch.Tensor) -> TensorSpec:
        return cls(name=name, shape=tuple(tensor.shape), dtype=dtype_to_str(tensor.dtype))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TensorSpec:
        return cls(
            name=d["name"],
            shape=tuple(None if s is None else int(s) for s in d["shape"]),
            dtype=d["dtype"],
        )


@dataclass
class ModelSpec:
    """A model plus everything downstream stages need to know about it.

    Construct with :meth:`from_module`, which validates the example inputs
    against the model's forward signature and infers output metadata.
    """

    name: str
    example_inputs: tuple[torch.Tensor, ...]
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    module: nn.Module | None = None
    source: str = "nn.Module"
    metadata: dict[str, Any] = field(default_factory=dict)
    _weights_hash: str | None = field(default=None, repr=False)

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        example_inputs: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        name: str | None = None,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        source: str = "nn.Module",
    ) -> ModelSpec:
        """Build a spec, validating inputs and inferring output metadata.

        Runs one forward pass under ``torch.no_grad()`` to learn the output
        shapes and dtypes, which Triton config generation (FR-6.1) needs.
        """
        if not isinstance(module, nn.Module):
            raise ModelSpecError(
                f"Expected a torch.nn.Module, got {type(module).__name__}. If you have a "
                f"checkpoint on disk, load it first or pass its path to trtship.load()."
            )

        inputs_t = _as_tuple(example_inputs)
        if not inputs_t:
            raise ModelSpecError("example_inputs is empty; at least one input tensor is required.")
        for i, t in enumerate(inputs_t):
            if not isinstance(t, torch.Tensor):
                raise ModelSpecError(
                    f"example_inputs[{i}] is a {type(t).__name__}, not a torch.Tensor. "
                    f"trtship needs concrete tensors to trace shapes and dtypes."
                )

        _check_forward_arity(module, inputs_t)

        resolved_name = name or type(module).__name__
        in_names = input_names or [f"input_{i}" for i in range(len(inputs_t))]
        if len(in_names) != len(inputs_t):
            raise ModelSpecError(
                f"Got {len(in_names)} input_names for {len(inputs_t)} example inputs; "
                f"these must match one-to-one."
            )
        in_specs = [TensorSpec.from_tensor(n, t) for n, t in zip(in_names, inputs_t, strict=True)]

        out_specs = _infer_outputs(module, inputs_t, output_names)

        return cls(
            name=resolved_name,
            example_inputs=inputs_t,
            inputs=in_specs,
            outputs=out_specs,
            module=module,
            source=source,
        )

    # ---- derived properties ----------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.example_inputs[0].device

    @property
    def has_dynamic_inputs(self) -> bool:
        return any(s.is_dynamic for s in self.inputs)

    def weights_hash(self) -> str:
        """Stable hash of the model's parameters and buffers.

        One determinant of the engine cache key (FR-5.2). Hashing raw tensor
        bytes in a fixed (sorted) key order makes this reproducible across
        processes; ``state_dict`` ordering alone is not guaranteed to be.
        """
        if self._weights_hash is not None:
            return self._weights_hash
        if self.module is None:
            raise ModelSpecError(
                "Cannot hash weights: this ModelSpec has no module attached "
                "(it was probably restored from a manifest). Supply the hash explicitly."
            )
        h = hashlib.sha256()
        state = self.module.state_dict()
        for key in sorted(state):
            tensor = state[key]
            h.update(key.encode("utf-8"))
            h.update(str(tuple(tensor.shape)).encode("utf-8"))
            h.update(str(tensor.dtype).encode("utf-8"))
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        digest = h.hexdigest()
        self._weights_hash = digest
        return digest

    # ---- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Manifest-safe view. Drops live torch objects (NFR-8.1)."""
        return {
            "name": self.name,
            "source": self.source,
            "inputs": [s.to_dict() for s in self.inputs],
            "outputs": [s.to_dict() for s in self.outputs],
            "weights_hash": self._weights_hash,
            "metadata": self.metadata,
        }


def _as_tuple(x: torch.Tensor | tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if isinstance(x, torch.Tensor):
        return (x,)
    return tuple(x)


def _check_forward_arity(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
    """Fail fast on obvious arity mismatches (FR-1.3).

    We only check what a signature can tell us. ``*args`` forwards and models
    with many defaulted keyword arguments are common enough that a strict check
    would produce false failures, so the trial forward in ``_infer_outputs`` is
    the real gate; this just catches the common case with a better message.
    """
    try:
        sig = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return

    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return

    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [p for p in positional if p.default is inspect.Parameter.empty]

    if len(inputs) > len(positional):
        names = ", ".join(p.name for p in positional) or "(none)"
        raise ModelSpecError(
            f"{type(module).__name__}.forward accepts {len(positional)} positional "
            f"argument(s) ({names}) but {len(inputs)} example input(s) were given."
        )
    if len(inputs) < len(required):
        missing = ", ".join(p.name for p in required[len(inputs) :])
        raise ModelSpecError(
            f"{type(module).__name__}.forward requires {len(required)} positional "
            f"argument(s) but only {len(inputs)} example input(s) were given. "
            f"Missing: {missing}."
        )


def _infer_outputs(
    module: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    output_names: list[str] | None,
) -> list[TensorSpec]:
    """Run one forward pass to learn output shapes and dtypes."""
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            result = module(*inputs)
    except Exception as exc:
        in_desc = ", ".join(f"{tuple(t.shape)}:{t.dtype}" for t in inputs)
        raise ModelSpecError(
            f"{type(module).__name__}.forward raised {type(exc).__name__} on the example "
            f"inputs [{in_desc}]. The model must run successfully in PyTorch before it can "
            f"be exported.\n  Underlying error: {exc}"
        ) from exc
    finally:
        if was_training:
            module.train()

    tensors = _flatten_outputs(result)
    names = output_names or [f"output_{i}" for i in range(len(tensors))]
    if len(names) != len(tensors):
        raise ModelSpecError(
            f"Got {len(names)} output_names but the model produced {len(tensors)} output tensor(s)."
        )
    return [TensorSpec.from_tensor(n, t) for n, t in zip(names, tensors, strict=True)]


def _flatten_outputs(result: Any) -> tuple[torch.Tensor, ...]:
    """Flatten a forward return into tensors, in a stable order.

    Handles the shapes real models actually return: a tensor, a tuple/list, or a
    dict-like (HuggingFace ``ModelOutput`` included, which is a dataclass whose
    ``None`` fields must be skipped).
    """
    if isinstance(result, torch.Tensor):
        return (result,)
    if isinstance(result, (list, tuple)):
        out: list[torch.Tensor] = []
        for item in result:
            out.extend(_flatten_outputs(item))
        return tuple(out)
    if hasattr(result, "items"):  # dict / HF ModelOutput
        out = []
        for _, v in result.items():
            if v is None:
                continue
            out.extend(_flatten_outputs(v))
        return tuple(out)
    raise ModelSpecError(
        f"Model returned {type(result).__name__}, which trtship cannot interpret as "
        f"tensor outputs. Wrap the model so its forward returns a tensor, a tuple of "
        f"tensors, or a dict of tensors."
    )
