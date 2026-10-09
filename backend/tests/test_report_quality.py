"""Report assessment distinguishes actual extraction gaps from method scope."""

import math

import pytest

from model_anatomy.service.jobs import ir_summary


def complete_report():
    return {
        "architecture": "NativeModel",
        "root_graph": {"id": "overview"},
        "metadata": {
            "engine_version": "recorded-engine",
            "limitations": ["No forward execution", "Fixed-resolution inputs only"],
            "analysis_report": {
                "semantic_coverage": 1,
                "module_semantic_coverage": 1,
                "checkpoint_parameter_coverage": 1,
                "tensor_shape_coverage": 1,
                "unknown_nodes": 0,
                "unmapped_checkpoint_tensors": [],
                "warnings": [],
                "mode": "static",
                "forward_executed": False,
                "runtime_trace_available": False,
            },
        },
    }


def test_complete_report_with_method_scope_has_no_extraction_warnings():
    data = complete_report()
    summary = ir_summary(data)
    assert summary["result_status"] == "completed"
    assert summary["warnings"] == summary["partial_reasons"] == []
    assert summary["limitations"] == data["metadata"]["limitations"]
    assert summary["quality"]["forward_executed"] is False
    assert summary["engine_version"] == "recorded-engine"


def test_actual_warnings_are_not_removed_even_when_the_same_text_is_a_limitation():
    data = complete_report()
    data["metadata"]["analysis_report"]["warnings"] = ["No forward execution"] * 2
    summary = ir_summary(data)
    assert summary["result_status"] == "partial"
    assert summary["warnings"] == ["No forward execution"]
    assert "No forward execution" in summary["limitations"]


@pytest.mark.parametrize("value", [None, True, "1", -1, 1.01, math.nan, math.inf, 10**400])
def test_invalid_or_unrecorded_coverage_never_becomes_success(value):
    data = complete_report()
    data["metadata"]["analysis_report"]["tensor_shape_coverage"] = value
    summary = ir_summary(data)
    assert summary["result_status"] == "partial"
    assert summary["quality"]["tensor_shape_coverage"] is None
    assert summary["partial_reasons"] == ["Graph connection shapes: coverage was not established."]


def test_real_missing_shapes_and_unmapped_tensors_have_specific_reasons():
    data = complete_report()
    report = data["metadata"]["analysis_report"]
    report.update(tensor_shape_coverage=0.95, unknown_nodes=2, unmapped_checkpoint_tensors=["extra.weight"])
    summary = ir_summary(data)
    assert summary["result_status"] == "partial"
    assert summary["partial_reasons"] == [
        "Graph connection shapes: coverage is incomplete.",
        "2 unclassified graph components.",
        "1 unmapped checkpoint tensor.",
    ]
    assert summary["quality"]["unmapped_checkpoint_tensors"] == 1


def test_absent_analysis_report_is_unknown_not_complete():
    data = complete_report()
    del data["metadata"]["analysis_report"]
    summary = ir_summary(data)
    assert summary["result_status"] == "partial"
    assert len(summary["partial_reasons"]) == 6
    assert all(value is None for value in summary["quality"].values())


@pytest.mark.parametrize("value", [None, True, "0", -1, 0.5])
def test_invalid_unknown_component_count_prevents_completion(value):
    data = complete_report()
    data["metadata"]["analysis_report"]["unknown_nodes"] = value
    summary = ir_summary(data)
    assert summary["result_status"] == "partial"
    assert summary["quality"]["unknown_nodes"] is None
    assert summary["partial_reasons"] == ["Unknown component count: check was not established."]


@pytest.mark.parametrize("value", [None, 0, {}, "extra.weight"])
def test_invalid_unmapped_tensor_list_prevents_completion(value):
    data = complete_report()
    data["metadata"]["analysis_report"]["unmapped_checkpoint_tensors"] = value
    summary = ir_summary(data)
    assert summary["result_status"] == "partial"
    assert summary["quality"]["unmapped_checkpoint_tensors"] is None
    assert summary["partial_reasons"] == ["Unmapped tensor count: check was not established."]
