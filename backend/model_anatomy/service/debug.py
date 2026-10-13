"""Local operator diagnostics: read-only saved analyses and readable JSON log tails."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .errors import ServiceError
from .jobs import ir_summary
from .storage import LocalStorage

MAX_READ_BYTES = 1024 * 1024
MAX_LINE_BYTES = 64 * 1024


def plain(value: object, limit: int = 240) -> str:
    return "".join(
        " " if unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"} else character
        for character in str(value)
    )[:limit]


@dataclass(frozen=True)
class LogCursor:
    identity: tuple[int, int]
    position: int
    head: bytes
    tail: bytes


def checkpoint(path: Path, position: int) -> tuple[bytes, bytes]:
    with path.open("rb") as handle:
        head = handle.read(min(position, 128))
        handle.seek(max(0, position - 128))
        tail = handle.read(min(position, 128))
    return head, tail


def scan_log(path: Path, cursor: LogCursor | None = None) -> tuple[list[dict], LogCursor, bool]:
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino)
    start = cursor.position if cursor and cursor.identity == identity else 0
    if start > stat.st_size or cursor and checkpoint(path, start) != (cursor.head, cursor.tail):
        start = 0
    events, position = read_events(path, start)
    head, tail = checkpoint(path, position)
    return events, LogCursor(identity, position, head, tail), position < stat.st_size and position > start


def format_event(event: dict, source: str) -> str:
    fields = [plain(event.get("time", "")), source, plain(event.get("level", "INFO")), plain(event["event"])]
    for name in ("analysis_id", "request_id", "model_id", "stage", "status", "code", "engine_version"):
        if event.get(name) is not None:
            fields.append(f"{name}={plain(event[name])}")
    for name in ("cached", "duration_ms", "warning_count", "exception_type"):
        if event.get(name) is not None:
            fields.append(f"{name}={plain(event[name])}")
    frames = event.get("frames")
    for frame in frames[:12] if isinstance(frames, list) else []:
        if isinstance(frame, dict):
            fields.append(
                f"at {plain(frame.get('file'))}:{plain(frame.get('line'))} {plain(frame.get('function'))}"
            )
    return " | ".join(fields)


def read_events(path: Path, position: int = 0) -> tuple[list[dict], int]:
    events = []
    with path.open("rb") as handle:
        handle.seek(position if position <= path.stat().st_size else 0)
        started = handle.tell()
        while handle.tell() - started < MAX_READ_BYTES:
            before = handle.tell()
            line = handle.readline(MAX_LINE_BYTES + 1)
            if len(line) > MAX_LINE_BYTES:
                raise ValueError("Log line exceeds the diagnostic reading budget")
            if not line or not line.endswith(b"\n"):
                return events, before
            try:
                event = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if isinstance(event, dict) and isinstance(event.get("event"), str):
                events.append(event)
        return events, handle.tell()


def inspect_analysis(data_dir: Path, analysis_id: str) -> list[str]:
    if not re.fullmatch(r"[0-9a-f]{32}", analysis_id):
        raise ValueError("Analysis ID must contain 32 lowercase hexadecimal characters")
    database = (data_dir / "jobs.sqlite3").resolve()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        job = connection.execute(
            "SELECT model_id, revision, engine_version, status, stage, cached, blob_digest "
            "FROM analysis_jobs WHERE id = ?",
            (analysis_id,),
        ).fetchone()
    if job is None:
        raise ValueError("Analysis was not found in this local database")
    lines = [f"ANALYSIS {analysis_id}"]
    lines.extend(f"{key}: {plain(job[key])}" for key in job.keys() if key != "blob_digest")  # noqa: SIM118 - sqlite Row iterates values
    if not job["blob_digest"]:
        return [*lines, "No saved report yet. Use the event history to inspect progress/failure."]
    root = data_dir / "objects"
    if not root.is_dir():
        raise ValueError("Local private report storage is unavailable")
    report = json.loads(LocalStorage(root).get(job["blob_digest"]))
    summary = ir_summary(report)
    lines += ["", f"Recorded report outcome: {summary['result_status']}", "PARTIAL CHECKS"]
    lines.extend(f"- {plain(reason)}" for reason in summary["partial_reasons"])
    if not summary["partial_reasons"]:
        lines.append("- No incomplete architecture checks recorded; inspect warnings and result kind.")
    for name, value in summary["quality"].items():
        if name.endswith("coverage"):
            lines.append(f"  {name}: {value:.1%}" if value is not None else f"  {name}: unknown")
    graph = report.get("root_graph", {})
    connected = {edge[key] for edge in graph.get("edges", []) for key in ("source", "target")}
    isolated = [
        node.get("module_path") or node["id"]
        for node in graph.get("nodes", [])
        if node["id"] not in connected
    ]
    relations = Counter(edge.get("relation", "unknown") for edge in graph.get("edges", []))
    lines += ["", f"ROOT GRAPH: {len(graph.get('nodes', []))} nodes; relations={dict(relations)}"]
    if isolated:
        lines.append("No recorded root relationship: " + ", ".join(plain(path) for path in isolated))
        lines.append(
            "These components are not proven execution connections. Do not connect them by guesswork."
        )
    lines += ["", f"EXTRACTION WARNINGS ({len(summary['warnings'])})"]
    lines.extend(f"{index}. {plain(warning, 600)}" for index, warning in enumerate(summary["warnings"], 1))
    lines.append("\nRead-only diagnostics: no weights, inference, job mutation or cache invalidation.")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".local/app-v1"))
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--analysis-id")
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--requests", action="store_true", help="Include routine per-request timing events")
    args = parser.parse_args()
    try:
        if args.analysis_id:
            print("\n".join(inspect_analysis(args.data_dir, args.analysis_id)), flush=True)
        directory = args.log_dir or args.data_dir / "native-runtime"
        paths = {name: directory / f"{name}-error.log" for name in ("api", "worker")}
        positions: dict[str, LogCursor] = {}
        print(f"\nEVENT HISTORY: {directory} (UTC timestamps; Ctrl+C stops follow)", flush=True)
        while True:
            found = False
            unread = False
            for name, path in paths.items():
                if not path.is_file():
                    continue
                found = True
                try:
                    events, cursor, more = scan_log(path, positions.get(name))
                except FileNotFoundError:
                    continue  # Rotation can briefly remove the path between existence and read.
                unread |= more
                positions[name] = cursor
                for event in events:
                    if event["event"] == "request_finished" and not args.requests:
                        continue
                    if not args.analysis_id or event.get("analysis_id") == args.analysis_id:
                        print(format_event(event, name), flush=True)
            if not found and not args.follow:
                print("No API/worker log files found. Supply --log-dir for another launch profile.")
            if not args.follow and not unread:
                break
            if not unread:
                time.sleep(0.5)
    except KeyboardInterrupt:
        return
    except (ValueError, OSError, sqlite3.Error, ServiceError, KeyError, TypeError) as exc:
        parser.exit(
            1, f"Diagnostics unavailable ({type(exc).__name__}); check data directory and analysis ID.\n"
        )


if __name__ == "__main__":
    main()
