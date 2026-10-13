import json
import sqlite3
from pathlib import Path

import pytest

from model_anatomy.service.debug import format_event, inspect_analysis, plain, read_events, scan_log
from model_anatomy.service.errors import ServiceError
from model_anatomy.service.storage import LocalStorage


def test_event_view_is_readable_and_does_not_echo_unlisted_private_fields():
    text = format_event(
        {
            "event": "analysis_outcome",
            "analysis_id": "a" * 32,
            "status": "partial",
            "cached": True,
            "warning_count": 38,
            "body": "private-body",
            "cookie": "private-cookie",
        },
        "worker",
    )
    assert "status=partial" in text and "warning_count=38" in text and "cached=True" in text
    assert "private-body" not in text and "private-cookie" not in text
    assert "\x1b" not in plain("model\x1b[31m\nnext")


def test_reader_skips_third_party_notices_and_waits_for_a_complete_record(tmp_path):
    path = tmp_path / "worker.log"
    path.write_bytes(b'library notice\n{"event":"analysis_started"}\n{"event":"analysis_finished"')
    events, position = read_events(path)
    assert [event["event"] for event in events] == ["analysis_started"]
    with path.open("ab") as handle:
        handle.write(b"}\n")
    events, _ = read_events(path, position)
    assert [event["event"] for event in events] == ["analysis_finished"]


def test_reader_recovers_after_truncation_and_rejects_oversized_lines(tmp_path):
    path = tmp_path / "worker.log"
    path.write_text('{"event":"a"}\n')
    assert read_events(path, 100)[0] == [{"event": "a"}]
    path.write_bytes(b"x" * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="budget"):
        read_events(path)


@pytest.mark.parametrize("repeat", [1, 40])
def test_same_file_truncate_and_rewrite_restarts_at_the_new_prefix(tmp_path, repeat):
    path = tmp_path / "worker.log"
    path.write_text('{"event":"old_record"}\n' * 10)
    _, cursor, _ = scan_log(path)
    path.write_text('{"event":"replacement"}\n' * repeat)
    events, _, more = scan_log(path, cursor)
    assert len(events) == repeat and all(event["event"] == "replacement" for event in events)
    assert more is False


def test_appending_retains_position_and_replacement_identity_restarts(tmp_path):
    path = tmp_path / "worker.log"
    path.write_text('{"event":"first"}\n')
    _, cursor, _ = scan_log(path)
    with path.open("a") as handle:
        handle.write('{"event":"second"}\n')
    events, cursor, _ = scan_log(path, cursor)
    assert events == [{"event": "second"}]
    path.rename(tmp_path / "old.log")
    path.write_text('{"event":"rotated"}\n')
    assert scan_log(path, cursor)[0] == [{"event": "rotated"}]


def test_shorter_replacement_still_drains_all_bounded_chunks(tmp_path, monkeypatch):
    from model_anatomy.service import debug

    path = tmp_path / "worker.log"
    path.write_text('{"event":"old"}\n' * 200)
    _, cursor, _ = scan_log(path)
    path.write_text('{"event":"new"}\n' * 100)
    monkeypatch.setattr(debug, "MAX_READ_BYTES", 64)
    events, _, more = scan_log(path, cursor)
    assert 0 < len(events) < 100
    assert more is True


def test_unicode_line_and_direction_controls_cannot_forge_terminal_output():
    text = plain("safe\u2028next\u2029line\u202eevil\u2066hidden\u2069\u200f\u061c")
    assert text == "safe next line evil hidden   "


def saved_analysis(tmp_path):
    report = json.loads((Path(__file__).resolve().parents[2] / "output/gpt2/model.json").read_bytes())
    report["metadata"]["analysis_report"]["tensor_shape_coverage"] = 0.5
    report["metadata"]["analysis_report"]["warnings"] = ["Opaque call native_transform: unverified"]
    report["root_graph"]["nodes"].append({"id": "isolated_audio", "module_path": "audio_tower"})
    digest = LocalStorage(tmp_path / "objects").put(json.dumps(report).encode())
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as connection:
        connection.execute(
            "CREATE TABLE analysis_jobs (id TEXT,model_id TEXT,revision TEXT,engine_version TEXT,"
            "status TEXT,stage TEXT,cached INTEGER,blob_digest TEXT)"
        )
        connection.execute(
            "INSERT INTO analysis_jobs VALUES(?,?,?,?,?,?,?,?)",
            ("a" * 32, report["model_id"], "b" * 40, "test-engine", "partial", "partial", 1, digest),
        )
    return digest


def test_saved_analysis_explains_gaps_without_mutating_database_or_report(tmp_path):
    digest = saved_analysis(tmp_path)
    database = (tmp_path / "jobs.sqlite3").read_bytes()
    blob = LocalStorage(tmp_path / "objects").path(digest)
    before = blob.read_bytes()
    output = "\n".join(inspect_analysis(tmp_path, "a" * 32))
    assert "status: partial" in output and "tensor_shape_coverage: 50.0%" in output
    assert "audio_tower" in output and "Opaque call native_transform" in output
    assert database == (tmp_path / "jobs.sqlite3").read_bytes()
    assert before == blob.read_bytes()


def test_invalid_or_missing_analysis_does_not_create_a_database(tmp_path):
    with pytest.raises(ValueError, match="Analysis ID"):
        inspect_analysis(tmp_path, "../../invalid")
    with pytest.raises(sqlite3.OperationalError):
        inspect_analysis(tmp_path, "a" * 32)
    assert not (tmp_path / "jobs.sqlite3").exists()


def test_corrupt_report_fails_integrity_check(tmp_path):
    digest = saved_analysis(tmp_path)
    LocalStorage(tmp_path / "objects").path(digest).write_bytes(b"corrupt")
    with pytest.raises(ServiceError, match="integrity"):
        inspect_analysis(tmp_path, "a" * 32)
