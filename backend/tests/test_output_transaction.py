from pathlib import Path

import pytest

from model_anatomy.src import output_transaction, render
from model_anatomy.src.ir import Graph, ModelIR, Node
from model_anatomy.src.output_transaction import publish_files


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def test_publish_replaces_and_removes_obsolete_files(tmp_path):
    first = tmp_path / "reports" / "first.json"
    removed = tmp_path / "reports" / "old.mmd"
    _write(first, "old")
    _write(removed, "stale")

    publish_files({first: "new", tmp_path / "config" / "config.json": "{}"}, {removed})

    assert first.read_text(encoding="utf-8") == "new"
    assert (tmp_path / "config" / "config.json").read_text(encoding="utf-8") == "{}"
    assert not removed.exists()
    assert not list(tmp_path.rglob(".anatomy-*"))


def test_late_replace_failure_restores_all_existing_outputs(tmp_path, monkeypatch):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    _write(first, "old first")
    _write(second, "old second")
    replace = output_transaction.os.replace
    failed = False

    def fail_second_stage(source, target):
        nonlocal failed
        if Path(target) == second and ".anatomy-stage-" in str(source) and not failed:
            failed = True
            raise OSError("simulated late rename failure")
        return replace(source, target)

    monkeypatch.setattr(output_transaction.os, "replace", fail_second_stage)
    with pytest.raises(OSError, match="simulated late rename failure"):
        publish_files({first: "new first", second: "new second"}, set())

    assert first.read_text(encoding="utf-8") == "old first"
    assert second.read_text(encoding="utf-8") == "old second"
    assert not list(tmp_path.glob(".anatomy-*"))


def test_obsolete_unlink_failure_restores_published_files(tmp_path, monkeypatch):
    first, obsolete = tmp_path / "first.json", tmp_path / "old.mmd"
    _write(first, "old first")
    _write(obsolete, "old graph")
    unlink = Path.unlink

    def fail_obsolete(path, *args, **kwargs):
        if path == obsolete:
            raise OSError("simulated obsolete cleanup failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_obsolete)
    with pytest.raises(OSError, match="simulated obsolete cleanup failure"):
        publish_files({first: "new first"}, {obsolete})

    assert first.read_text(encoding="utf-8") == "old first"
    assert obsolete.read_text(encoding="utf-8") == "old graph"


def test_stage_write_failure_keeps_previous_outputs(tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    _write(first, "old")
    open_path = Path.open

    def fail_stage(path, *args, **kwargs):
        if ".anatomy-stage-" in path.name:
            raise OSError("simulated stage write failure")
        return open_path(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_stage)
    with pytest.raises(OSError, match="simulated stage write failure"):
        publish_files({first: "new"}, set())

    assert first.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".anatomy-*"))


def test_overlapping_resolved_targets_are_rejected(tmp_path):
    first = tmp_path / "first.json"
    aliased = tmp_path / "unused" / ".." / "first.json"

    with pytest.raises(ValueError, match="paths overlap"):
        publish_files({first: "one", aliased: "two"}, set())

    assert not first.exists()


def test_existing_symlink_target_is_rejected(tmp_path):
    referent = tmp_path / "referent.json"
    link = tmp_path / "link.json"
    _write(referent, "preserve")
    try:
        link.symlink_to(referent)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="cannot be symlinks"):
        publish_files({link: "replace"}, set())

    assert referent.read_text(encoding="utf-8") == "preserve"
    assert link.is_symlink()


def test_renderer_rolls_back_complete_report_when_late_publish_fails(tmp_path, monkeypatch):
    output, config, preview = tmp_path / "reports", tmp_path / "config", tmp_path / "preview"
    ir = ModelIR(
        model_id="example/model",
        architecture="ExampleModel",
        root_graph=Graph("example_root", [Node("input", "Input", "input")], []),
        metadata={
            "revision": "a" * 40,
            "source_config": {"model_type": "example", "value": 1},
            "artifacts": {"metadata_bytes_read": 12},
        },
    )
    first_paths = render.write_outputs(ir, output, config, preview)
    before = {path: path.read_bytes() for path in first_paths}
    ir.metadata["source_config"]["value"] = 2
    replace = output_transaction.os.replace
    failed = False
    expected_late_path = preview / "REPORTS_PREVIEW.md"

    def fail_preview(source, target):
        nonlocal failed
        if Path(target) == expected_late_path and ".anatomy-stage-" in str(source) and not failed:
            failed = True
            raise OSError("simulated preview publish failure")
        return replace(source, target)

    monkeypatch.setattr(output_transaction.os, "replace", fail_preview)
    with pytest.raises(OSError, match="simulated preview publish failure"):
        render.write_outputs(ir, output, config, preview)

    assert {path: path.read_bytes() for path in first_paths} == before
