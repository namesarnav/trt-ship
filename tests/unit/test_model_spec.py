"""Tests for the ModelSpec seam. CPU-only by design."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from trtship.core.model_spec import ModelSpec, ModelSpecError, TensorSpec


class Tiny(nn.Module):
    def __init__(self, in_features: int = 4, out_features: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TwoInput(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


class MultiOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x * 2, x.sum(dim=-1)


class DictOutput(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        return {"logits": x * 2, "hidden": None, "pooled": x.mean(dim=-1)}


# ---- TensorSpec ---------------------------------------------------------


def test_tensor_spec_from_tensor() -> None:
    spec = TensorSpec.from_tensor("input_0", torch.zeros(2, 3, dtype=torch.float32))
    assert spec.shape == (2, 3)
    assert spec.dtype == "float32"
    assert not spec.is_dynamic
    assert spec.rank == 2


def test_tensor_spec_dynamic_axes() -> None:
    spec = TensorSpec(name="x", shape=(None, 3, 224, 224), dtype="float32")
    assert spec.is_dynamic
    assert spec.dynamic_axes == (0,)
    assert spec.resolve(**{"0": 8}) == (8, 3, 224, 224)


def test_tensor_spec_resolve_rejects_unresolved_axis() -> None:
    spec = TensorSpec(name="x", shape=(None, 3), dtype="float32")
    with pytest.raises(ModelSpecError, match="dynamic dimension at axis 0"):
        spec.resolve()


def test_tensor_spec_round_trips_through_dict() -> None:
    spec = TensorSpec(name="x", shape=(None, 3, 224, 224), dtype="float16")
    assert TensorSpec.from_dict(spec.to_dict()) == spec


def test_unsupported_dtype_names_the_dtype() -> None:
    with pytest.raises(ModelSpecError, match="complex64"):
        TensorSpec.from_tensor("x", torch.zeros(2, dtype=torch.complex64))


# ---- ModelSpec construction --------------------------------------------


def test_from_module_infers_inputs_and_outputs() -> None:
    spec = ModelSpec.from_module(Tiny(), torch.randn(2, 4))
    assert spec.name == "Tiny"
    assert [s.shape for s in spec.inputs] == [(2, 4)]
    assert [s.shape for s in spec.outputs] == [(2, 3)]
    assert spec.outputs[0].dtype == "float32"


def test_from_module_accepts_a_bare_tensor_or_a_tuple() -> None:
    one = ModelSpec.from_module(Tiny(), torch.randn(2, 4))
    many = ModelSpec.from_module(Tiny(), (torch.randn(2, 4),))
    assert [s.shape for s in one.inputs] == [s.shape for s in many.inputs]


def test_multi_output_models_flatten_in_order() -> None:
    spec = ModelSpec.from_module(MultiOutput(), torch.randn(2, 4))
    assert [s.name for s in spec.outputs] == ["output_0", "output_1"]
    assert [s.shape for s in spec.outputs] == [(2, 4), (2,)]


def test_dict_outputs_skip_none_fields() -> None:
    """HuggingFace ModelOutput routinely carries None fields; they are not tensors."""
    spec = ModelSpec.from_module(DictOutput(), torch.randn(2, 4))
    assert len(spec.outputs) == 2


def test_two_input_model() -> None:
    spec = ModelSpec.from_module(TwoInput(), (torch.randn(2, 4), torch.randn(2, 4)))
    assert len(spec.inputs) == 2


def test_module_stays_in_its_original_training_mode() -> None:
    """Inferring outputs runs a forward pass; it must not silently flip .train()."""
    model = Tiny()
    model.train()
    ModelSpec.from_module(model, torch.randn(2, 4))
    assert model.training is True


# ---- FR-1.3: fail fast, with a message that names the mismatch ---------


def test_too_many_inputs_names_the_arity() -> None:
    with pytest.raises(ModelSpecError, match="accepts 1 positional argument"):
        ModelSpec.from_module(Tiny(), (torch.randn(2, 4), torch.randn(2, 4)))


def test_too_few_inputs_names_the_missing_argument() -> None:
    with pytest.raises(ModelSpecError, match="Missing: b"):
        ModelSpec.from_module(TwoInput(), torch.randn(2, 4))


def test_shape_mismatch_surfaces_as_a_trtship_error_not_a_torch_traceback() -> None:
    with pytest.raises(ModelSpecError) as exc:
        ModelSpec.from_module(Tiny(in_features=4), torch.randn(2, 99))
    message = str(exc.value)
    assert "forward raised" in message
    assert "(2, 99)" in message  # the offending shape is named


def test_non_module_input_is_rejected_clearly() -> None:
    with pytest.raises(ModelSpecError, match=r"Expected a torch\.nn\.Module"):
        ModelSpec.from_module("not a model", torch.randn(2, 4))  # type: ignore[arg-type]


def test_non_tensor_example_input_is_rejected_clearly() -> None:
    with pytest.raises(ModelSpecError, match=r"not a torch\.Tensor"):
        ModelSpec.from_module(Tiny(), ([1, 2, 3],))  # type: ignore[arg-type]


def test_empty_example_inputs_rejected() -> None:
    with pytest.raises(ModelSpecError, match="at least one input tensor"):
        ModelSpec.from_module(Tiny(), ())


def test_mismatched_input_names_rejected() -> None:
    with pytest.raises(ModelSpecError, match="one-to-one"):
        ModelSpec.from_module(Tiny(), torch.randn(2, 4), input_names=["a", "b"])


# ---- weights hashing (cache key input, FR-5.2) -------------------------


def test_weights_hash_is_stable_for_identical_weights() -> None:
    torch.manual_seed(0)
    a = ModelSpec.from_module(Tiny(), torch.randn(2, 4))
    b = ModelSpec.from_module(Tiny(), torch.randn(2, 4))
    b.module.load_state_dict(a.module.state_dict())  # type: ignore[union-attr]
    assert a.weights_hash() == ModelSpec.from_module(
        b.module, torch.randn(2, 4)  # type: ignore[arg-type]
    ).weights_hash()


def test_weights_hash_changes_when_weights_change() -> None:
    spec = ModelSpec.from_module(Tiny(), torch.randn(2, 4))
    before = spec.weights_hash()
    with torch.no_grad():
        spec.module.fc.weight.add_(1.0)  # type: ignore[union-attr]
    spec._weights_hash = None
    assert spec.weights_hash() != before


def test_weights_hash_without_a_module_explains_itself() -> None:
    spec = ModelSpec(
        name="detached",
        example_inputs=(torch.randn(1, 4),),
        inputs=[TensorSpec("x", (1, 4), "float32")],
        outputs=[TensorSpec("y", (1, 3), "float32")],
        module=None,
    )
    with pytest.raises(ModelSpecError, match="restored from a manifest"):
        spec.weights_hash()


# ---- manifest serialization (NFR-8.1) ----------------------------------


def test_to_dict_drops_live_torch_objects() -> None:
    spec = ModelSpec.from_module(Tiny(), torch.randn(2, 4))
    d = spec.to_dict()
    assert "module" not in d
    assert "example_inputs" not in d
    assert d["inputs"][0]["shape"] == [2, 4]
