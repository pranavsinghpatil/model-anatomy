from pathlib import Path

import pytest

from model_anatomy.service import views
from model_anatomy.service.errors import ServiceError


def test_views_do_not_mutate_stored_report():
    source = Path(__file__).resolve().parent / "fixtures/gpt2-report.json"
    ir = views.load_ir(source.read_bytes())
    before = ir.to_json()
    graph = views.graph_view(ir, ir.root_graph.id)
    views.node_view(ir, ir.root_graph.id, graph["nodes"][0]["id"])
    assert ir.to_json() == before


def test_artifact_checkpoint_availability_uses_full_report_not_page():
    source = Path(__file__).resolve().parent / "fixtures/gpt2-report.json"
    report = views.load_ir(source.read_bytes())
    assert len(report.tensors) == 160
    page = views.artifact_view(report, offset=160, limit=50)
    assert page["tensors"] == []
    assert page["checkpoint_available"] is True


def test_artifact_view_preserves_explicit_checkpoint_unavailable():
    source = Path(__file__).resolve().parent / "fixtures/gpt2-report.json"
    report = views.load_ir(source.read_bytes())
    report.metadata["inspection"] = {"checkpoint_available": False}
    assert views.artifact_view(report, offset=160)["checkpoint_available"] is False


@pytest.mark.parametrize("data", [b"not-json", b"{}", b'{"schema_version":"invalid"}'])
def test_invalid_stored_report_fails_honestly(data):
    with pytest.raises(ServiceError) as exc:
        views.load_ir(data)
    assert exc.value.code == "invalid_artifact"
