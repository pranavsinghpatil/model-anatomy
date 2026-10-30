"""Test public control policy and the real cross-process SQLite stop lock."""

from __future__ import annotations

import importlib.util
import io
import json
import multiprocessing
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend"))
sys.path.insert(0, str(ROOT / "deploy"))
import demo_control as CONTROL
from support import interactive_demo as INTERACTIVE
from support import sandbox_demo_control as REMOTE
from support.recovery_window import RECOVERY_END


class StopReservationTests(unittest.TestCase):
    def setUp(self) -> None:
        (ROOT / ".tmp").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".tmp", prefix="demo-stop-")
        self.addCleanup(self.temporary.cleanup)
        self.control = Path(self.temporary.name)
        (self.control / "data").mkdir()
        self.database = self.control / "data/jobs.sqlite3"
        config = {
            "profile": "sandbox-sqlite-demo", "data_dir": str(self.control / "data"),
            "environment": {"MODEL_ANATOMY_STORAGE_BACKEND": "local", "MODEL_ANATOMY_DATABASE_URL": "sqlite:///" + str(self.database)},
        }
        (self.control / "runtime.json").write_text(json.dumps(config))
        with sqlite3.connect(self.database) as connection:
            connection.execute("CREATE TABLE analysis_jobs (status TEXT)")
            connection.commit()

    def test_any_visitors_queued_or_running_job_refuses_stop(self) -> None:
        for status in ["queued", "running"]:
            with self.subTest(status=status):
                with sqlite3.connect(self.database) as connection:
                    connection.execute("DELETE FROM analysis_jobs")
                    connection.execute("INSERT INTO analysis_jobs VALUES (?)", (status,))
                REMOTE.hold_stop(self.control, "session-1", "test", lease_seconds=0.1)
                result = REMOTE.read_json(self.control / "stop-test.json")
                self.assertEqual(result["state"], "busy")
                self.assertEqual(result[status], 1)
                self.assertFalse(REMOTE.stopping(self.control, "session-1"))

    def test_idle_check_blocks_a_concurrent_submission_until_stop_lease_ends(self) -> None:
        process = multiprocessing.get_context("fork").Process(
            target=REMOTE.hold_stop,
            args=(self.control, "session-1", "a" * 32), kwargs={"lease_seconds": 1.2},
        )
        process.start()
        self.addCleanup(lambda: process.kill() if process.is_alive() else None)
        deadline = time.monotonic() + 3
        result = self.control / "stop-reservation.json"
        while not result.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(result.exists())
        self.assertTrue(REMOTE.stopping(self.control, "session-1"))
        self.assertFalse(REMOTE.stopping(self.control, "new-session"))
        with sqlite3.connect(self.database, timeout=0.05) as competing, \
                self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
            competing.execute("BEGIN IMMEDIATE")
        process.join(timeout=3)
        self.assertEqual(process.exitcode, 0)
        self.assertTrue(REMOTE.stopping(self.control, "session-1"))
        with sqlite3.connect(self.database, timeout=0.05) as competing:
            competing.execute("BEGIN IMMEDIATE")
            competing.execute("INSERT INTO analysis_jobs VALUES ('queued')")
        self.assertEqual(REMOTE.status(self.control, "new-session")["queued"], 1)

    def test_missing_or_wrong_profile_database_never_authorizes_stop(self) -> None:
        self.database.unlink()
        REMOTE.hold_stop(self.control, "session-1", "missing", lease_seconds=0.1)
        self.assertEqual(REMOTE.read_json(self.control / "stop-missing.json")["state"], "unavailable")
        self.assertFalse(self.database.exists())

    def test_start_does_not_override_an_existing_stop_reservation(self) -> None:
        (self.control / "stop-reservation.json").write_text(json.dumps({
            "session_id": "session-1", "expires_at": time.time() + 30,
        }))
        with patch.object(REMOTE.subprocess, "Popen") as spawn:
            self.assertEqual(REMOTE.start(self.control, "session-1"), {"state": "stopping"})
            spawn.assert_not_called()


class ControlPolicyTests(unittest.IsolatedAsyncioTestCase):
    def box(self, status: str):
        runtime = SimpleNamespace(id="session-1", stop=AsyncMock())
        return SimpleNamespace(status=status, current_session=runtime, session=AsyncMock(return_value=runtime))

    async def test_status_and_stop_on_stopped_demo_never_acquire_a_session(self) -> None:
        for action in ["status", "stop"]:
            box = self.box("stopped")
            code, value = await CONTROL.operate(box, {}, action)
            self.assertEqual(code, 200)
            self.assertEqual(value["state"], "stopped")
            box.session.assert_not_awaited()
            box.current_session.stop.assert_not_awaited()

    async def test_busy_reservation_never_calls_provider_stop(self) -> None:
        box = self.box("running")
        with patch.object(CONTROL, "remote", AsyncMock(return_value={"state": "busy", "queued": 1, "running": 0})):
            code, value = await CONTROL.operate(box, {}, "stop")
        self.assertEqual(code, 409)
        self.assertFalse(value["stop_allowed"])
        box.session.assert_not_awaited()
        box.current_session.stop.assert_not_awaited()

    async def test_stop_requires_reservation_and_targets_the_inspected_session(self) -> None:
        box = self.box("running")
        with patch.object(CONTROL, "remote", AsyncMock(return_value={"state": "reserved"})) as remote:
            code, value = await CONTROL.operate(box, {}, "stop")
        self.assertEqual(code, 202)
        self.assertEqual(value["state"], "stopping")
        remote.assert_awaited_once_with(box.current_session, {}, "reserve-stop")
        box.current_session.stop.assert_awaited_once()
        box.session.assert_not_awaited()

    async def test_start_resumes_only_when_stopped_and_does_not_extend_running_session(self) -> None:
        for status in ["stopped", "running"]:
            box = self.box(status)
            with patch.object(CONTROL, "start_allowed", return_value=True), \
                    patch.object(CONTROL, "remote", AsyncMock(return_value={"state": "starting"})):
                code, _ = await CONTROL.operate(box, {}, "start")
            self.assertEqual(code, 202)
            self.assertEqual(box.session.await_count, int(status == "stopped"))

    async def test_start_during_snapshotting_or_stopping_never_resumes(self) -> None:
        for status in ["stopping", "snapshotting"]:
            box = self.box(status)
            with patch.object(CONTROL, "start_allowed", return_value=True):
                _, result = await CONTROL.operate(box, {}, "start")
            self.assertEqual(result["state"], "stopping")
            box.session.assert_not_awaited()

    async def test_expired_start_cannot_acquire_a_session(self) -> None:
        box = self.box("stopped")
        with patch.object(CONTROL, "start_allowed", return_value=False), self.assertRaises(CONTROL.ControlError):
            await CONTROL.operate(box, {}, "start")
        box.session.assert_not_awaited()
        self.assertEqual(CONTROL.CONTROL_END, RECOVERY_END)

    async def test_invalid_or_missing_stop_reservation_fails_closed(self) -> None:
        for state in ["unavailable", "conflict"]:
            box = self.box("running")
            with patch.object(CONTROL, "remote", AsyncMock(return_value={"state": state})), \
                    self.assertRaises(CONTROL.ControlError):
                await CONTROL.operate(box, {}, "stop")
            box.current_session.stop.assert_not_awaited()

    async def test_idle_starting_services_can_be_stopped_after_global_db_check(self) -> None:
        box = self.box("running")
        with patch.object(CONTROL, "remote", AsyncMock(return_value={"state": "running", "queued": 0, "running": 0})), \
                patch.object(CONTROL, "healthy", return_value=False):
            _, result = await CONTROL.operate(box, {}, "status")
        self.assertEqual(result["state"], "starting")
        self.assertTrue(result["stop_allowed"])
        box.session.assert_not_awaited()

    def test_same_origin_json_post_is_required_and_body_cannot_choose_a_target(self) -> None:
        headers = {"Origin": "https://demo.vercel.app", "Content-Type": "application/json", "Sec-Fetch-Site": "same-origin"}
        with patch.object(CONTROL, "configuration", return_value={"public_origin": "https://demo.vercel.app"}):
            self.assertEqual(CONTROL.validate_request("POST", headers, b'{"action":"start"}'), "start")
            self.assertEqual(CONTROL.validate_request("GET", {}, b""), "status")
            for changed, body in [
                ({**headers, "Origin": "https://elsewhere.invalid"}, b'{"action":"stop"}'),
                ({"Content-Type": "application/json"}, b'{"action":"stop"}'),
                ({**headers, "Content-Type": "text/plain"}, b'{"action":"stop"}'),
                (headers, b'{"action":"stop","sandbox_name":"other"}'),
                (headers, b'{"action":"restart"}'),
                (headers, b'[]'),
            ]:
                with self.subTest(body=body), self.assertRaises(CONTROL.ControlError):
                    CONTROL.validate_request("POST", changed, body)


class ProductionPayloadTests(unittest.TestCase):
    def archive(self) -> bytes:
        import tarfile

        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w") as archive:
            for name in [
                "package.json", "package-lock.json", "index.html", "api/demo.py", "demo_control/__init__.py",
                "src/components/DemoControls.tsx", "requirements.txt", ".python-version",
            ]:
                content = b"source"
                item = tarfile.TarInfo("frontend/" + name)
                item.size = len(content)
                archive.addfile(item, io.BytesIO(content))
            content = b'DEMO_CONTROL_PROTOCOL = "visitor-controls-v1"\n'
            item = tarfile.TarInfo("backend/model_anatomy/service/jobs.py")
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))
        return output.getvalue()

    def test_partial_frontend_only_merge_is_rejected_before_provisioning(self) -> None:
        payload = self.archive().replace(b"visitor-controls-v1", b"visitor-controls-v0")
        candidate = {"sandbox_name": "new", "commit": "a" * 40, "app_dir": "/vercel/model-anatomy", "demo_url": "https://sb-new.vercel.run"}
        with self.assertRaisesRegex(RuntimeError, "backend demo admission protection"):
            INTERACTIVE.frontend_files(payload, candidate, "https://demo.vercel.app")

    def test_only_api_is_proxied_and_controls_and_frontend_stay_outside_sandbox(self) -> None:
        candidate = {"sandbox_name": "new", "commit": "a" * 40, "app_dir": "/vercel/model-anatomy", "demo_url": "https://sb-new.vercel.run"}
        files = INTERACTIVE.frontend_files(self.archive(), candidate, "https://demo.vercel.app")
        generated = {item["file"]: json.loads(item["data"]) for item in files if item["encoding"] == "utf-8"}
        config = generated["frontend/vercel.json"]
        self.assertEqual(config["functions"]["api/demo.py"]["maxDuration"], 300)
        self.assertEqual(config["rewrites"][0], {"source": "/api/v1/:path*", "destination": "https://sb-new.vercel.run/api/v1/:path*"})
        import re
        spa = re.compile("^" + config["rewrites"][-1]["source"] + "$")
        for route in ["/", "/app", "/catalog", "/app/analyses/saved"]:
            self.assertIsNotNone(spa.fullmatch(route))
        for route in ["/api/demo", "/api/v1/analyses", "/health", "/assets/app.js"]:
            self.assertIsNone(spa.fullmatch(route))
        deployment = generated["frontend/demo_control/deployment.json"]
        self.assertEqual(deployment["sandbox_name"], "new")
        for key in ["project_id", "team_id", "token"]:
            self.assertNotIn(key, deployment)

    def test_credentials_are_private_production_settings_without_vite_prefixes(self) -> None:
        response = {"created": [{"key": "DEMO_VERCEL_" + key} for key in ["TOKEN", "PROJECT_ID", "TEAM_ID"]]}
        routes = SimpleNamespace(project_id="prj_test", team_id="team_test", token="test-token", request=MagicMock(return_value=response))
        INTERACTIVE.Production(routes).configure_credentials()
        args = routes.request.call_args
        self.assertEqual(args.args[0], "POST")
        self.assertEqual(args.kwargs, {"upsert": "true"})
        settings = args.args[2]
        self.assertEqual({item["key"] for item in settings}, {"DEMO_VERCEL_TOKEN", "DEMO_VERCEL_PROJECT_ID", "DEMO_VERCEL_TEAM_ID"})
        self.assertTrue(all(item["type"] == "sensitive" and item["target"] == ["production"] for item in settings))

    def test_partial_environment_failure_is_rejected_without_logging_values(self) -> None:
        routes = SimpleNamespace(project_id="prj_test", team_id="team_test", token="test-token", request=MagicMock(return_value={
            "created": [], "failed": [{"error": {"value": "do-not-log"}}],
        }))
        with self.assertRaisesRegex(RuntimeError, "private Production control settings") as result:
            INTERACTIVE.Production(routes).configure_credentials()
        self.assertNotIn("do-not-log", str(result.exception))

    def test_superseded_deployment_cannot_control_its_old_sandbox(self) -> None:
        project = {"id": "prj_test", "accountId": "team_test", "targets": {"production": {"id": "dpl_current"}}}
        deployment = {"projectId": "prj_test", "meta": {"modelAnatomyProtocol": INTERACTIVE.PROTOCOL, "modelAnatomySandbox": "current"}}
        replies = []
        for value in [project, deployment]:
            reply = MagicMock()
            reply.__enter__.return_value.read.return_value = json.dumps(value).encode()
            replies.append(reply)
        with patch.dict("os.environ", {"DEMO_VERCEL_TOKEN": "token", "DEMO_VERCEL_PROJECT_ID": "prj_test", "DEMO_VERCEL_TEAM_ID": "team_test"}), \
                patch.object(CONTROL, "urlopen", side_effect=replies), self.assertRaises(CONTROL.ControlError):
            CONTROL.require_current_deployment({"sandbox_name": "old", "commit": "a" * 40, "sandbox_origin": "https://sb-old.vercel.run"})


class FunctionHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = importlib.util.spec_from_file_location("demo_function", ROOT / "frontend/api/demo.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Cannot load the Vercel Function handler")
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.request = SimpleNamespace(
            headers={}, rfile=io.BytesIO(), wfile=io.BytesIO(),
            send_response=MagicMock(), send_header=MagicMock(), end_headers=MagicMock(),
        )

    def test_sdk_failure_is_redacted_and_never_cached(self) -> None:
        with patch.object(self.module, "dispatch", AsyncMock(side_effect=RuntimeError("private-token-and-destination"))):
            self.module.handler.respond(self.request, "GET")
        self.request.send_response.assert_called_once_with(503)
        self.request.send_header.assert_any_call("Cache-Control", "no-store")
        self.assertNotIn(b"private-token-and-destination", self.request.wfile.getvalue())

    def test_get_only_dispatches_status(self) -> None:
        with patch.object(self.module, "dispatch", AsyncMock(return_value=(200, CONTROL.response("stopped")))) as dispatch:
            self.module.handler.respond(self.request, "GET")
        dispatch.assert_awaited_once_with("status")
        self.assertEqual(json.loads(self.request.wfile.getvalue())["state"], "stopped")

    def test_oversized_post_never_dispatches_a_cloud_operation(self) -> None:
        self.request.headers = {"Content-Length": "10000"}
        with patch.object(self.module, "dispatch", AsyncMock()) as dispatch:
            self.module.handler.respond(self.request, "POST")
        dispatch.assert_not_awaited()
        self.request.send_response.assert_called_once_with(413)


if __name__ == "__main__":
    unittest.main()
