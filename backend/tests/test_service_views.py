import json
from pathlib import Path
from unittest.mock import patch

from model_anatomy.api.settings import Settings
from model_anatomy.service import views


def test_prepared_read_cache_invalidates_when_snapshot_changes(tmp_path):
    source = Path(__file__).resolve().parents[2] / "output/gpt2/model.json"
    directory = tmp_path / "gpt2"
    directory.mkdir()
    path = directory / "model.json"
    path.write_bytes(source.read_bytes())
    settings = Settings(prepared_dir=tmp_path)
    views._prepared_snapshot.cache_clear()
    with patch.object(views, "load_ir", wraps=views.load_ir) as load:
        first, ir = views.prepared(settings, "example-gpt2")
        second, again = views.prepared(settings, "example-gpt2")
        assert load.call_count == 1
        assert first == second and ir is again
        before = ir.to_json()
        graph = views.graph_view(ir, ir.root_graph.id)
        views.node_view(ir, ir.root_graph.id, graph["nodes"][0]["id"])
        assert ir.to_json() == before
        data = json.loads(path.read_text(encoding="utf-8"))
        data["metadata"]["analysis_report"]["warnings"].append("Snapshot changed")
        path.write_text(json.dumps(data), encoding="utf-8")
        updated, _ = views.prepared(settings, "example-gpt2")
        assert "Snapshot changed" in updated["warnings"]
        assert load.call_count == 2
    views._prepared_snapshot.cache_clear()
