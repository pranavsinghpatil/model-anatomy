import ast
from types import SimpleNamespace

import pytest
import torch

from model_anatomy.src.static_flow import StaticFlowAnalyzer, StaticValue, combine
from model_anatomy.src.static_tensors import analyze_tensor_factory_call

native_ones = torch.ones


class FactoryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(is_encoder_decoder=False)
        self.layer = torch.nn.Linear(2, 2, device="meta")

    def forward(self, x):
        return torch.ones(size=[x.shape[0], 3])


class AliasedFactoryModel(FactoryModel):
    def forward(self, x):
        return native_ones((x.shape[0], 3))


class FullKeywordFactoryModel(FactoryModel):
    def forward(self, x):
        return torch.full(size=(x.shape[0], 3), fill_value=1.0)


class MethodFactoryModel(FactoryModel):
    def forward(self, x):
        return x.new_ones((2, 3))


class LikeFactoryModel(FactoryModel):
    def forward(self, x):
        return torch.ones_like(x)


class LikeFromFactoryModel(FactoryModel):
    def forward(self, x):
        return torch.ones_like(torch.ones((2, 3)))


class LikeFromTupleModel(FactoryModel):
    def forward(self, x):
        dimensions = (2, 3)
        return torch.ones_like(dimensions)


class ShadowedTorchModel(FactoryModel):
    def forward(self, torch, x):
        return torch.ones((x.shape[0], 3))


def ones(input):
    raise AssertionError("Source helpers must not execute")


class ShadowedFactoryModel(FactoryModel):
    def forward(self, x):
        mask = ones(x)
        if mask is None:
            return x
        return self.layer(x)


class DynamicStarFactoryModel(FactoryModel):
    def forward(self, x, dims):
        return torch.ones(*dims)


class DynamicKwFactoryModel(FactoryModel):
    def forward(self, x, size):
        kwargs = {"size": size}
        return torch.ones(**kwargs)


class InvalidSizeFactoryModel(FactoryModel):
    def forward(self, x):
        return torch.ones((-1, 3))


class NestedCallFactoryModel(FactoryModel):
    def forward(self, x):
        return torch.ones((self.layer(x).shape[0], 3))


class ShapeChangedFactoryModel(FactoryModel):
    def forward(self, x):
        mask = torch.ones((2, 3)).unsqueeze(0)
        if mask.dim() == 2:
            return x
        return self.layer(x)


def analyze(model):
    paths = {path: SimpleNamespace(kind="backbone") for path, _ in model.named_modules()}
    return StaticFlowAnalyzer(model, "", paths).analyze()


def test_native_factory_rank_from_size_keyword_and_symbolic_dimensions():
    flow = analyze(FactoryModel())

    assert flow.outputs.materialized is True
    assert flow.outputs.rank == 2
    assumption = next(item for item in flow.profile_assumptions if item["predicate"] == "torch.ones")
    assert "Callable identity matched torch.ones" in assumption["basis"]
    assert "result rank is 2" in assumption["basis"]


def test_imported_native_factory_alias_is_verified_by_identity():
    flow = analyze(AliasedFactoryModel())

    assert flow.outputs.materialized is True
    assert flow.outputs.rank == 2
    assert any(item["predicate"] == "torch.ones" for item in flow.profile_assumptions)


def test_full_factory_accepts_size_and_fill_keywords():
    flow = analyze(FullKeywordFactoryModel())

    assert flow.outputs.materialized is True
    assert flow.outputs.rank == 2


def test_shadowed_torch_name_is_not_resolved_from_module_globals():
    flow = analyze(ShadowedTorchModel())

    assert flow.outputs.materialized is False
    assert any("callable identity did not match native PyTorch" in warning for warning in flow.warnings)


def test_tensor_method_factory_uses_known_receiver_and_explicit_size():
    flow = analyze(MethodFactoryModel())

    assert flow.outputs.materialized is True
    assert flow.outputs.rank == 2
    assert any(item["predicate"] == "torch.new_ones" for item in flow.profile_assumptions)


def test_like_factory_keeps_tensor_identity_but_requires_known_input_rank_for_rank_fact():
    flow = analyze(LikeFactoryModel())

    assert flow.outputs.materialized is True
    assert flow.outputs.rank is None


def test_like_factory_uses_rank_only_from_a_known_tensor_input():
    flow = analyze(LikeFromFactoryModel())

    assert flow.outputs.materialized is True
    assert flow.outputs.rank == 2


def test_like_factory_rejects_tuple_input_even_when_its_length_is_known():
    flow = analyze(LikeFromTupleModel())

    assert flow.outputs.materialized is False
    assert any("tensor input identity was not established" in warning for warning in flow.warnings)


def test_shadowed_factory_name_does_not_establish_tensor_identity():
    flow = analyze(ShadowedFactoryModel())

    assert flow.outputs.materialized is False
    assert any("callable identity did not match native PyTorch" in warning for warning in flow.warnings)
    assert any("Conditional source path is unresolved: mask is None" in warning for warning in flow.warnings)


def test_dynamic_star_factory_arguments_remain_unknown_and_warn():
    flow = analyze(DynamicStarFactoryModel())

    assert flow.outputs.materialized is False
    assert flow.outputs.rank is None
    assert any("dynamic *args/**kwargs" in warning for warning in flow.warnings)


def test_dynamic_keyword_expansion_remains_unknown_and_warns():
    flow = analyze(DynamicKwFactoryModel())

    assert flow.outputs.materialized is False
    assert flow.outputs.rank is None
    assert any("dynamic *args/**kwargs" in warning for warning in flow.warnings)


def test_known_negative_dimension_is_not_reported_as_a_tensor_factory():
    flow = analyze(InvalidSizeFactoryModel())

    assert flow.outputs.materialized is False
    assert flow.outputs.rank is None
    assert any("invalid size" in warning for warning in flow.warnings)


def test_factory_argument_dependencies_are_interpreted_once():
    flow = analyze(NestedCallFactoryModel())

    assert [event.module_path for event in flow.events] == ["layer"]
    assert flow.outputs.rank == 2


def test_shape_changing_method_uses_proven_native_output_rank():
    flow = analyze(ShapeChangedFactoryModel())

    assert flow.warnings == []
    assert [event.kind for event in flow.events] == ["tensor_creation", "tensor_operation", "module"]
    assert flow.events[1].metadata["output_port_specs"]["output"]["shape"] == [1, 2, 3]


@pytest.mark.parametrize(
    "name,extra",
    [("ones", ""), ("zeros", ""), ("empty", ""), ("full", ", 1.0")],
)
@pytest.mark.parametrize("shape", [(), (3,), (2, 4)])
def test_native_factory_rank_matches_meta_tensor_rank(name, extra, shape):
    source = f"torch.{name}({shape!r}{extra}, device='meta')"
    call = ast.parse(source, mode="eval").body
    fact = analyze_tensor_factory_call(
        call,
        {"torch": torch},
        False,
        lambda _expression: None,
        lambda _expression: (False, None),
    )

    assert fact is not None and fact.native
    assert fact.rank == getattr(torch, name)(shape, *((1.0,) if name == "full" else ()), device="meta").ndim


@pytest.mark.parametrize("name", ["ones", "zeros", "empty"])
@pytest.mark.parametrize("dimensions", [(3,), (2, 4)])
def test_variadic_native_factory_rank_matches_meta_tensor_rank(name, dimensions):
    source = f"torch.{name}({', '.join(map(str, dimensions))}, device='meta')"
    call = ast.parse(source, mode="eval").body
    fact = analyze_tensor_factory_call(
        call,
        {"torch": torch},
        False,
        lambda _node: None,
        lambda _node: (False, None),
    )

    assert fact is not None and fact.native
    assert fact.rank == getattr(torch, name)(*dimensions, device="meta").ndim


def test_full_factory_rejects_scalar_size_accepted_by_variadic_factories():
    call = ast.parse("torch.full(3, 1.0)", mode="eval").body
    fact = analyze_tensor_factory_call(
        call,
        {"torch": torch},
        False,
        lambda _node: None,
        lambda _node: (False, None),
    )

    assert fact is not None and not fact.native
    assert fact.reason == "full requires a tuple or list size"


def test_merge_with_none_does_not_convert_tensor_dependencies_into_identity_evidence():
    tensor = StaticValue(refs=[("input:x", "output")], materialized=True, rank=2)
    result = combine([tensor, StaticValue(constant=None)], preserve_tensor_facts=True)
    assert result.refs == tensor.refs
    assert result.materialized is False
    assert result.rank is None
    assert result.identity_unknown is True


@pytest.mark.parametrize(
    "source",
    [
        "torch.ones_like(x, 1)",
        "torch.full_like(x, 1, 2)",
        "torch.full_like(x, 1, fill_value=2)",
        "torch.full((2, 3), 1, fill_value=2)",
        "x.new_full((2, 3), 1, fill_value=2)",
    ],
)
def test_invalid_factory_argument_binding_does_not_establish_a_tensor(source):
    call = ast.parse(source, mode="eval").body
    fact = analyze_tensor_factory_call(
        call, {"torch": torch}, True, lambda _node: None, lambda _node: (True, 2)
    )
    assert fact is not None and not fact.native


def test_tiny_native_t5_keeps_sequence_length_unresolved_without_mask_rank_warning():
    from model_anatomy.src.registry import instantiate_native, load_native_config

    model, _ = instantiate_native(
        load_native_config(
            {
                "model_type": "t5",
                "architectures": ["T5ForConditionalGeneration"],
                "vocab_size": 64,
                "d_model": 16,
                "d_kv": 8,
                "d_ff": 32,
                "num_layers": 1,
                "num_decoder_layers": 1,
                "num_heads": 2,
            }
        )
    )
    paths = {path: SimpleNamespace(kind="backbone") for path, _ in model.named_modules()}
    flow = StaticFlowAnalyzer(model, "decoder", paths).analyze()

    assert any("sequence_length != 1" in warning for warning in flow.warnings)
    assert not any("attention_mask.dim() == 4" in warning for warning in flow.warnings)
