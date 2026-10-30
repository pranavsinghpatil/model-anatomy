"""Tests for conservative tensor-only operator contracts."""

import pytest
import torch

from model_anatomy.src.ir import TensorSpec
from model_anatomy.src.tensor_contracts import tensor_outputs


def spec(shape, dtype="F32"):
    return TensorSpec(shape, dtype, "test")


@pytest.mark.parametrize(
    ("shape", "chunks", "dim", "expected"),
    [
        ([10, 3], 4, 0, [[3, 3], [3, 3], [3, 3], [1, 3]]),
        ([5, 3], 4, -2, [[2, 3], [2, 3], [1, 3]]),
        ([1, 2], 4, 0, [[1, 2]]),
        ([0, 3], 4, 0, [[0, 3]] * 4),
    ],
)
def test_chunk_ports_match_torch_contract(shape, chunks, dim, expected):
    outputs = tensor_outputs("chunk", {"input": spec(shape)}, {"chunks": chunks, "dim": dim})

    reference = torch.chunk(torch.empty(shape, device="meta"), chunks, dim=dim)
    assert outputs is not None
    assert list(outputs) == [str(index) for index in range(len(reference))]
    assert [outputs[str(index)].shape for index in range(len(reference))] == [list(item.shape) for item in reference]
    assert [outputs[str(index)].shape for index in range(len(reference))] == expected


def test_chunk_refuses_symbolic_split_and_invalid_signatures():
    assert tensor_outputs("chunk", {"input": spec(["B", 8])}, {"chunks": 2, "dim": 0}) is None
    assert tensor_outputs("chunk", {"input": spec([4, 8])}, {"chunks": 0, "dim": 0}) is None
    assert tensor_outputs("chunk", {"input": spec([4, 8])}, {"chunks": 2, "dim": 2}) is None
    assert tensor_outputs("chunk", {"input": spec([4, 8]), "extra": spec([4, 8])}, {"chunks": 2, "dim": 0}) is None


def test_isinf_preserves_shape_and_returns_bool():
    result = tensor_outputs("isinf", {"input": spec(["B", 4], "F16")}, {})
    assert result is not None
    assert result["output"].shape == ["B", 4]
    assert result["output"].dtype == "BOOL"
    assert tensor_outputs("isinf", {"input": spec([2], "I64")}, {}) is None
    assert tensor_outputs("isinf", {"input": spec([2])}, {"dim": 0}) is None


@pytest.mark.parametrize(
    ("inputs", "parameters", "expected"),
    [
        ({"input": spec(["B", 4], "BF16")}, {"min": -1.0, "max": 1.0}, ["B", 4]),
        ({"input": spec([3], "F32")}, {"min": 0}, [3]),
        ({"input": spec([3], "I32")}, {"min": 0}, None),
        ({"input": spec([3], "F32")}, {"min": True}, None),
        ({"input": spec([3], "F32")}, {}, None),
        ({"input": spec([3], "F32")}, {"min": float("inf")}, None),
    ],
)
def test_scalar_clamp_only_claims_floating_contract(inputs, parameters, expected):
    output = tensor_outputs("clamp", inputs, parameters)
    if expected is None:
        assert output is None
    else:
        assert output is not None
        assert output["output"].shape == expected
        assert output["output"].dtype == inputs["input"].dtype


def test_where_uses_strict_dtype_and_proven_broadcasting():
    inputs = {
        "condition": spec(["B", 1], "BOOL"),
        "input": spec([1, 8], "F32"),
        "other": spec(["B", 8], "F32"),
    }
    output = tensor_outputs("where", inputs, {})
    assert output is not None
    assert output["output"].shape == ["B", 8]
    assert output["output"].dtype == "F32"

    incompatible = {**inputs, "input": spec([3, 8], "F32"), "other": spec([4, 8], "F32")}
    assert tensor_outputs("where", incompatible, {}) is None
    uncertain = {**inputs, "input": spec(["S", 8], "F32")}
    assert tensor_outputs("where", uncertain, {}) is None
    assert tensor_outputs("where", {**inputs, "condition": spec(["B", 1], "I8")}, {}) is None
    assert tensor_outputs("where", {**inputs, "other": spec(["B", 8], "F16")}, {}) is None
    assert tensor_outputs("where", inputs, {"broadcast": True}) is None


@pytest.mark.parametrize(
    ("shape", "dim", "keepdim", "expected"),
    [
        ([2, 3, 4], 1, False, [2, 4]),
        ([2, 3, 4], -1, True, [2, 3, 1]),
        (["B", 3, "S"], (0, 2), False, [3]),
        ([2, 3], None, False, []),
        ([2, 3], (), True, [2, 3]),
    ],
)
def test_any_shape_and_bool_dtype_match_torch(shape, dim, keepdim, expected):
    inputs = {"input": spec(shape, "I64")}
    params = {"dim": dim, "keepdim": keepdim}
    output = tensor_outputs("any", inputs, params)
    assert output is not None
    assert output["output"].shape == expected
    assert output["output"].dtype == "BOOL"

    if all(type(size) is int for size in shape):
        reference = torch.any(torch.empty(shape, dtype=torch.int64, device="meta"), dim=dim, keepdim=keepdim)
        assert output["output"].shape == list(reference.shape)


def test_any_rejects_invalid_axes_and_unknown_shape():
    assert tensor_outputs("any", {"input": spec([2, 3])}, {"dim": 2}) is None
    assert tensor_outputs("any", {"input": spec([2, 3])}, {"dim": (0, 0)}) is None
    assert tensor_outputs("any", {"input": spec(None)}, {"dim": 0}) is None
    assert tensor_outputs("any", {"input": spec([2, 3])}, {"keepdim": 1}) is None
    assert tensor_outputs("any", {"input": spec([2, 3])}, {"dim": "all"}) is None


def test_unknown_method_and_invalid_metadata_stay_unknown():
    assert tensor_outputs("reshape", {"input": spec([2, 3])}, {}) is None
    assert tensor_outputs("where", {"input": spec([2, 3])}, {}) is None
    assert tensor_outputs("isinf", {"input": spec([2], "custom")}, {}) is None


@pytest.mark.parametrize(
    ("method", "shape", "parameters", "expected"),
    [
        ("reshape", [2, 3, 4], {"dimensions": [6, 4]}, [6, 4]),
        ("view", ["B", 4, 8], {"dimensions": [-1, 4, 8]}, ["B", 4, 8]),
        ("reshape", [2, 3, 4], {"dimensions": [-1, 4]}, [6, 4]),
        ("transpose", ["B", 3, 4], {"dimensions": [1, -1]}, ["B", 4, 3]),
        ("unsqueeze", ["B", 4], {"dim": -1}, ["B", 4, 1]),
        ("squeeze", ["B", 1, 4], {"dim": 1}, ["B", 4]),
        ("squeeze", ["B", 3, 4], {"dim": 1}, ["B", 3, 4]),
        ("flatten", ["B", 1, 4, 3], {"start_dim": 1, "end_dim": 2}, ["B", 4, 3]),
        ("flatten", ["B", 2, 3], {"start_dim": 0, "end_dim": 2}, None),
    ],
)
def test_shape_operations_keep_only_provable_dimensions(method, shape, parameters, expected):
    output = tensor_outputs(method, {"input": spec(shape)}, parameters)
    if expected is None:
        assert output is None
    else:
        assert output is not None
        assert output["output"].shape == expected


def test_shape_operations_match_torch_meta_shapes():
    tensor = torch.empty((2, 3, 4), device="meta")
    cases = [
        ("reshape", {"dimensions": [6, 4]}, tensor.reshape(6, 4)),
        ("view", {"dimensions": [2, 12]}, tensor.view(2, 12)),
        ("transpose", {"dimensions": [0, -1]}, tensor.transpose(0, -1)),
        ("unsqueeze", {"dim": 1}, tensor.unsqueeze(1)),
        ("squeeze", {"dim": 1}, torch.empty((2, 1, 4), device="meta").squeeze(1)),
        ("flatten", {"start_dim": 1, "end_dim": 2}, tensor.flatten(1, 2)),
        ("cast", {"dtype": "F16"}, tensor.to(dtype=torch.float16)),
    ]
    for method, parameters, expected in cases:
        input_shape = list(tensor.shape) if method != "squeeze" else [2, 1, 4]
        output = tensor_outputs(method, {"input": spec(input_shape)}, parameters)
        assert output is not None
        assert output["output"].shape == list(expected.shape)


def test_shape_operations_reject_invalid_or_unprovable_requests():
    assert tensor_outputs("reshape", {"input": spec([2, 3])}, {"dimensions": [5]}) is None
    assert tensor_outputs("view", {"input": spec(["B", 4])}, {"dimensions": [-1, 2]}) is None
    assert tensor_outputs("reshape", {"input": spec([2, 3])}, {"dimensions": [-1, -1]}) is None
    assert tensor_outputs("transpose", {"input": spec([2, 3])}, {"dimensions": [0, 2]}) is None
    assert tensor_outputs("unsqueeze", {"input": spec([2, 3])}, {"dim": 3}) is None
    assert tensor_outputs("squeeze", {"input": spec(["B", 3])}, {"dim": 0}) is None
    assert tensor_outputs("flatten", {"input": spec([2, 3])}, {"start_dim": 1, "end_dim": 0}) is None
    assert tensor_outputs("cast", {"input": spec([2])}, {"dtype": "custom"}) is None


def test_reshape_cancels_only_explicitly_positive_symbolic_factors():
    inputs = {"input": spec(["B", "S"])}
    assert tensor_outputs("reshape", inputs, {"dimensions": [-1, "S"]}) is None
    output = tensor_outputs(
        "reshape", inputs, {"dimensions": [-1, "S"], "positive_symbols": ["S"]}
    )
    assert output is not None
    assert output["output"].shape == ["B", "S"]


def test_reshape_rejects_zero_denominator_and_invalid_positive_symbol_claims():
    assert tensor_outputs(
        "view", {"input": spec(["B", 0])}, {"dimensions": [-1, 0]}
    ) is None
    assert tensor_outputs(
        "reshape",
        {"input": spec(["B", "S"])},
        {"dimensions": [-1, "S"], "positive_symbols": ["S", "S"]},
    ) is None
    assert tensor_outputs(
        "reshape",
        {"input": spec(["B", "S"])},
        {"dimensions": [-1, "S"], "positive_symbols": ["S! bad"]},
    ) is None


def test_cast_preserves_known_dtype_even_when_shape_is_unknown():
    output = tensor_outputs("cast", {"input": spec(None)}, {"dtype": "F16"})
    assert output is not None
    assert output["output"].shape is None
    assert output["output"].dtype == "F16"


def test_clone_preserves_only_known_tensor_facts():
    output = tensor_outputs("clone", {"input": spec(["B", 1, "S"], "BF16")}, {})
    assert output is not None
    assert output["output"].shape == ["B", 1, "S"]
    assert output["output"].dtype == "BF16"
    assert output["output"].semantic_name == "clone"
    assert tensor_outputs("clone", {"input": spec([2], "custom")}, {}) is None
    assert tensor_outputs("clone", {"input": spec([2])}, {"memory_format": "contiguous"}) is None


@pytest.mark.parametrize(
    ("input_shape", "dimensions", "expected"),
    [
        ([1, 1, 4, 8], ["B", 1, -1, -1], ["B", 1, 4, 8]),
        (["B", 1, "S", "S"], ["B", 4, -1, -1], ["B", 4, "S", "S"]),
        ([1, 4], [2, 4], [2, 4]),
    ],
)
def test_expand_preserves_or_proves_broadcast_dimensions(input_shape, dimensions, expected):
    positive = [axis for axis in dimensions if isinstance(axis, str)]
    output = tensor_outputs(
        "expand",
        {"input": spec(input_shape)},
        {"dimensions": dimensions, "positive_symbols": positive},
    )
    assert output is not None
    assert output["output"].shape == expected


def test_expand_rejects_invalid_or_unrelated_dimensions():
    inputs = {"input": spec(["B", 2, 4])}
    assert tensor_outputs("expand", inputs, {"dimensions": ["B", 3, 4], "positive_symbols": ["B"]}) is None
    assert tensor_outputs("expand", inputs, {"dimensions": ["B", 2], "positive_symbols": ["B"]}) is None
    assert tensor_outputs("expand", inputs, {"dimensions": ["C", 2, 4], "positive_symbols": ["B"]}) is None
    assert tensor_outputs("expand", inputs, {"dimensions": ["B", 2, 4], "positive_symbols": []}) is None
    assert tensor_outputs("expand", inputs, {"dimensions": ["B", -1, 4, -1], "positive_symbols": ["B"]}) is None
    assert tensor_outputs("expand", {"input": spec(None)}, {"dimensions": ["B"], "positive_symbols": ["B"]}) is None


def test_t5_slice_contract_tracks_new_axes_and_same_symbol_prefix_slice():
    expanded = tensor_outputs(
        "slice",
        {"input": spec(["B", 1, "S", "S"], "F32")},
        {
            "indices": [
                {"start": None, "stop": None, "step": None},
                {"start": None, "stop": None, "step": None},
                {"start": None, "stop": None, "step": None},
                {"start": None, "stop": "S", "step": None},
            ]
        },
    )
    assert expanded is not None
    assert expanded["output"].shape == ["B", 1, "S", "S"]
    mask_view = tensor_outputs(
        "slice",
        {"input": spec(["B", "S"], "BOOL")},
        {
            "indices": [
                {"start": None, "stop": None, "step": None},
                None,
                None,
                {"start": None, "stop": None, "step": None},
            ]
        },
    )
    assert mask_view is not None
    assert mask_view["output"].shape == ["B", 1, 1, "S"]
    assert tensor_outputs("slice", {"input": spec(["B", "S"])}, {"indices": [None]}) is None


def test_t5_slice_contract_rejects_boolean_start_and_step():
    inputs = {"input": spec(["B", "S"], "BOOL")}
    assert tensor_outputs(
        "slice",
        inputs,
        {"indices": [{"start": True, "stop": None, "step": None}, {"start": None, "stop": None, "step": None}]},
    ) is None
    assert tensor_outputs(
        "slice",
        inputs,
        {"indices": [{"start": None, "stop": None, "step": True}, {"start": None, "stop": None, "step": None}]},
    ) is None


def test_masked_fill_requires_boolean_broadcastable_mask_and_known_scalar():
    inputs = {"input": spec(["B", 1, "S", "S"], "F32"), "mask": spec(["B", 1, 1, "S"], "BOOL")}
    output = tensor_outputs("masked_fill", inputs, {"value": -3.4e38})
    assert output is not None
    assert output["output"].shape == inputs["input"].shape
    assert output["output"].dtype == "F32"
    assert tensor_outputs("masked_fill", inputs, {"value": None}) is None
    assert tensor_outputs("masked_fill", {**inputs, "mask": spec(["B", 1, 1, "S"], "I32")}, {"value": 0}) is None
    assert tensor_outputs("masked_fill", {**inputs, "mask": spec(["B", 2, 1, "S"], "BOOL")}, {"value": 0}) is None
    assert tensor_outputs("masked_fill", inputs, {"value": float("inf")}) is None


@pytest.mark.parametrize(
    ("dtype", "value"),
    [("F16", 65505), ("BF16", 3.4e38), ("F32", 3.5e38), ("F64", 1.8e308)],
)
def test_masked_fill_rejects_values_outside_input_float_range(dtype, value):
    inputs = {
        "input": spec(["B", "S"], dtype),
        "mask": spec(["B", "S"], "BOOL"),
    }
    assert tensor_outputs("masked_fill", inputs, {"value": value}) is None


def test_masked_fill_rejects_non_floating_input():
    inputs = {"input": spec(["B", "S"], "I32"), "mask": spec(["B", "S"], "BOOL")}
    assert tensor_outputs("masked_fill", inputs, {"value": -1}) is None


def test_native_t5_mask_algebra_requires_proven_broadcast_and_dtypes():
    integer = spec(["T"], "I64")
    positions = spec(["T", 1], "I64")
    comparison = tensor_outputs("gt", {"left": integer, "right": positions}, {})
    assert comparison is not None and comparison["output"].shape == ["T", "T"]
    assert comparison["output"].dtype == "BOOL"
    assert tensor_outputs("gt", {"left": spec(["T"], "F32"), "right": positions}, {}) is None
    assert tensor_outputs("gt", {"left": integer, "right": spec([1, "K"], "I64")}, {}) is None

    product = tensor_outputs(
        "float_bool_mul",
        {"left": spec(["T", "T"], "F32"), "right": comparison["output"]},
        {},
    )
    assert product is not None and product["output"].dtype == "F32"
    assert tensor_outputs("float_bool_mul", {"left": integer, "right": comparison["output"]}, {}) is None
    assert tensor_outputs("float_bool_mul", {"left": spec(["T", "T"], "F32"), "right": integer}, {}) is None

    equal = tensor_outputs("eq_scalar", {"input": spec(["B", 1, "T", "T"], "F32")}, {"scalar": 0})
    assert equal is not None and equal["output"].dtype == "BOOL"
    assert tensor_outputs("eq_scalar", {"input": spec(["T"], "F16")}, {"scalar": float("inf")}) is None
    assert tensor_outputs("scalar_left_sub", {"input": spec(["B", "T"], "F32")}, {"scalar": 1.0}) is not None
    assert tensor_outputs("scalar_left_sub", {"input": integer}, {"scalar": 1.0}) is None
    assert tensor_outputs("triu", {"input": spec(["T", "T"], "F32")}, {"diagonal": 1}) is not None
    assert tensor_outputs("triu", {"input": spec(["T"], "F32")}, {"diagonal": 1}) is None
    assert tensor_outputs("triu", {"input": spec(["T", "T"], "F32")}, {"diagonal": True}) is None
