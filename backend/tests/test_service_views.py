from pathlib import Path

import pytest

from model_anatomy.service import views
from model_anatomy.service.errors import ServiceError


def test_views_do_not_mutate_stored_report():
    source = Path(__file__).resolve().parents[2] / "output/gpt2/model.json"
    ir = views.load_ir(source.read_bytes())
    before = ir.to_json()
    graph = views.graph_view(ir, ir.root_graph.id)
    views.node_view(ir, ir.root_graph.id, graph["nodes"][0]["id"])
    assert ir.to_json() == before


@pytest.mark.parametrize("data", [b"not-json", b"{}", b'{"schema_version":"invalid"}'])
def test_invalid_stored_report_fails_honestly(data):
    with pytest.raises(ServiceError) as exc:
        views.load_ir(data)
    assert exc.value.code == "invalid_artifact"
