"""Exercise source isolation, published-target recovery, and failed demo rollouts."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))
SPEC = importlib.util.spec_from_file_location("github_demo", ROOT / "deploy/05_github_demo.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load the workflow controller")
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)
from support import recovery_window
from support.credentials import (
    access_token,
    bind_destination,
    destination,
    load_private_environment,
    without_destination,
)
from support.project_routes import ProjectRoutes, sandbox_origin
from support.recovery_window import RECOVERY_END, disable_expired_workflow, recovery_allowed


class ArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        (ROOT / ".tmp").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="workflow-source-", dir=ROOT / ".tmp")
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.git("init", "-qb", "main")
        self.git("config", "user.name", "Deployment Test")
        self.git("config", "user.email", "deployment-test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        hooks = self.repo / ".git/empty-hooks"
        hooks.mkdir()
        self.git("config", "core.hooksPath", str(hooks))
        (self.repo / "frontend").mkdir()
        for name in ["pyproject.toml", "uv.lock", "alembic.ini", "frontend/package.json"]:
            (self.repo / name).write_text("main source")
        self.git("add", ".")
        self.git("commit", "-qm", "Create main fixture")
        self.main = self.git("rev-parse", "HEAD")
        self.git("switch", "-qc", "feat/deploy")
        (self.repo / "pyproject.toml").write_text("deployment source")
        self.git("commit", "-qam", "Change deployment fixture")

    def git(self, *args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True, stderr=subprocess.DEVNULL).strip()

    def test_selected_commit_ignores_checked_out_branch_and_untracked_credentials(self) -> None:
        (self.repo / ".env").write_text("NOT_FOR_UPLOAD=test")
        revision, payload, _count = DEMO.PREPARE.committed_source(self.repo, "main", self.main)
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            self.assertNotIn(".env", archive.getnames())
            self.assertEqual(archive.extractfile("pyproject.toml").read(), b"main source")
        self.assertEqual(revision, self.main)

    def test_main_and_deploy_are_independently_selectable(self) -> None:
        main, _payload, _count = DEMO.PREPARE.committed_source(self.repo, "main")
        deploy, _payload, _count = DEMO.PREPARE.committed_source(self.repo, "feat/deploy")
        self.assertEqual(main, self.main)
        self.assertNotEqual(main, deploy)

    def test_private_files_generated_state_and_symlinks_are_rejected(self) -> None:
        for name in [".env.production", ".tmp/state.json", "model.safetensors", "escape"]:
            with self.subTest(name=name):
                path = self.repo / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if name == "escape":
                    path.symlink_to("/etc/passwd")
                else:
                    path.write_text("invalid deployment fixture")
                self.git("add", "-f", name)
                self.git("commit", "-qm", "Track invalid fixture")
                with self.assertRaises(RuntimeError):
                    DEMO.PREPARE.committed_source(self.repo)
                self.git("reset", "--hard", "HEAD~1")

    def test_explicit_revision_must_be_a_sha(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "full Git commit SHA"):
            DEMO.PREPARE.committed_source(self.repo, "main", "--exec=invalid")


class AuthenticationTests(unittest.TestCase):
    def test_ci_never_prompts_when_the_secret_is_missing(self) -> None:
        with patch.dict("os.environ", {"CI": "true"}, clear=True), patch("getpass.getpass") as prompt:
            with self.assertRaisesRegex(RuntimeError, "repository secret"):
                access_token()
            prompt.assert_not_called()

    def test_explicit_token_does_not_use_cli_login(self) -> None:
        with patch.dict("os.environ", {"CI": "true", "VERCEL_TOKEN": " test-token "}, clear=True):
            self.assertEqual(access_token(), "test-token")

    def test_ci_requires_destination_secrets_and_never_reads_the_local_file(self) -> None:
        with patch.dict("os.environ", {"CI": "true"}, clear=True), patch.object(Path, "read_text") as read:
            with self.assertRaisesRegex(RuntimeError, "VERCEL_PROJECT_ID"):
                destination()
            read.assert_not_called()

    def test_private_file_does_not_override_injected_environment(self) -> None:
        (ROOT / ".tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as folder:
            private = Path(folder) / ".env"
            private.write_text('VERCEL_PROJECT_ID=prj_local\nVERCEL_TEAM_ID="team_local"\nVERCEL_TOKEN=\n')
            with patch.dict("os.environ", {"VERCEL_PROJECT_ID": "prj_injected"}, clear=True):
                load_private_environment(private)
                with patch("support.credentials.load_private_environment"):
                    self.assertEqual(destination(), {"project_id": "prj_injected", "team_id": "team_local"})

    def test_destination_mismatch_is_rejected_without_disclosing_ids(self) -> None:
        with patch.dict("os.environ", {
            "CI": "true", "VERCEL_PROJECT_ID": "prj_selected", "VERCEL_TEAM_ID": "team_selected",
        }, clear=True), self.assertRaisesRegex(RuntimeError, "another configured"):
            bind_destination({"project_id": "prj_another"})

    def test_diagnostic_json_recursively_omits_private_destination_ids(self) -> None:
        clean = without_destination({"project_id": "prj_private", "previous": {
            "team_id": "team_private", "sandbox_name": "demo",
        }, "items": [{"project_id": "prj_private"}]})
        self.assertEqual(clean, {"previous": {"sandbox_name": "demo"}, "items": [{}]})


class RecoveryWindowTests(unittest.TestCase):
    def test_new_york_december_16_is_inclusive_but_december_17_is_excluded(self) -> None:
        new_york_offset = timezone(timedelta(hours=-5))
        self.assertTrue(recovery_allowed(datetime(2026, 12, 16, 23, 59, 59, tzinfo=new_york_offset)))
        self.assertFalse(recovery_allowed(datetime(2026, 12, 17, 0, 0, 0, tzinfo=new_york_offset)))
        self.assertFalse(recovery_allowed(RECOVERY_END))

    def test_requested_december_1_through_15_window_remains_enabled(self) -> None:
        new_york_offset = timezone(timedelta(hours=-5))
        for day in [1, 15]:
            with self.subTest(day=day):
                self.assertTrue(recovery_allowed(datetime(2026, 12, day, 0, 0, tzinfo=new_york_offset)))
                self.assertTrue(recovery_allowed(datetime(2026, 12, day, 23, 59, 59, tzinfo=new_york_offset)))

    def test_disabling_targets_only_the_recovery_workflow(self) -> None:
        context = unittest.mock.MagicMock()
        context.__enter__.return_value.status = 204
        with patch.dict("os.environ", {"GITHUB_REPOSITORY": "owner/repo", "GITHUB_TOKEN": "test-token"}, clear=True), \
                patch("support.recovery_window.urlopen", return_value=context) as request:
            disable_expired_workflow()
            self.assertEqual(request.call_args.args[0].method, "PUT")
            self.assertEqual(request.call_args.args[0].full_url,
                             "https://api.github.com/repos/owner/repo/actions/workflows/recover-demo.yml/disable")

    def test_window_guard_does_not_disable_an_active_schedule(self) -> None:
        with patch.object(recovery_window, "recovery_allowed", return_value=True), \
                patch.object(recovery_window, "disable_expired_workflow") as disable, \
                patch("sys.argv", ["recovery_window.py", "--disable-expired-workflow"]), \
                patch.dict("os.environ", {}, clear=True):
            recovery_window.main()
            disable.assert_not_called()

    def test_expired_controller_never_reads_local_profiles_or_secrets(self) -> None:
        with patch.object(DEMO, "recovery_allowed", return_value=False), \
                patch.object(DEMO, "destination") as configured, patch.object(Path, "read_text") as read, \
                patch("sys.argv", ["05_github_demo.py", "recover"]):
            DEMO.main()
            configured.assert_not_called()
            read.assert_not_called()


def route_document(origin: str = "https://sb-old.vercel.run") -> dict:
    return {
        "id": "route_demo", "name": "Model Anatomy Sandbox", "enabled": True,
        "rawSrc": "^/(.*)$", "srcSyntax": "regex",
        "route": {"src": "^/(.*)$", "dest": origin + "/$1"},
    }


class RouteApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.routes = ProjectRoutes("prj_test", "team_test", "test-token", "Model Anatomy Sandbox")
        self.rule = route_document()
        self.versions = {"versions": [{"id": "v1", "isLive": True}]}

    def test_unpublished_edits_cannot_be_accidentally_promoted(self) -> None:
        with patch.object(self.routes, "request", return_value={
            "versions": [*self.versions["versions"], {"id": "other-edit", "isStaging": True}],
        }) as request:
            with self.assertRaisesRegex(RuntimeError, "Unpublished"):
                self.routes.live()
            self.assertEqual(request.call_count, 1)

    def test_wrong_project_owner_or_domain_fails_preflight(self) -> None:
        cases = [
            [{"id": "prj_test", "accountId": "team_other"}],
            [{"id": "prj_test", "accountId": "team_test", "targets": {"production": {"id": "dpl_test"}}},
             {"domains": [{"name": "another.vercel.app"}]}],
        ]
        for responses in cases:
            with self.subTest(responses=responses), patch.object(self.routes, "request", side_effect=responses), \
                    self.assertRaises(RuntimeError):
                self.routes.check_project("https://demo.vercel.app")

    def test_non_demo_destinations_and_path_placeholders_are_rejected(self) -> None:
        for dest in ["https://example.com/$1", "https://sb-old.vercel.run/:path*", "https://sb-old.vercel.run/$1?x=y"]:
            with self.subTest(dest=dest), self.assertRaises(RuntimeError):
                sandbox_origin(dest)

    def test_missing_duplicate_conditional_disabled_and_redirect_rules_are_rejected(self) -> None:
        cases = [[], [self.rule, self.rule]]
        for modify in [
            lambda r: r.update(enabled=False),
            lambda r: r["route"].update(status=307),
            lambda r: r["route"].update(methods=["GET"]),
            lambda r: r["route"].update(has=[{"type": "host", "value": "example.com"}]),
            lambda r: r.update(rawSrc="/:path*", srcSyntax="path-to-regexp"),
        ]:
            rule = copy.deepcopy(self.rule)
            modify(rule)
            cases.append([rule])
        for rules in cases:
            with self.subTest(rules=rules), patch.object(self.routes, "request", side_effect=[
                self.versions, {"routes": rules},
            ]), self.assertRaises(RuntimeError):
                self.routes.live()

    def test_changed_production_version_is_preserved(self) -> None:
        baseline = {"version_id": "v1", "rule": self.rule, "origin": "https://sb-old.vercel.run"}
        with patch.object(self.routes, "live", return_value={**baseline, "version_id": "v-other"}), \
                patch.object(self.routes, "request") as request:
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                self.routes.publish(baseline, "https://sb-new.vercel.run")
            request.assert_not_called()

    def test_publish_updates_only_the_named_rule_and_uses_numbered_capture(self) -> None:
        baseline = {"version_id": "v1", "rule": self.rule, "origin": "https://sb-old.vercel.run"}
        published = {**baseline, "version_id": "v2", "origin": "https://sb-new.vercel.run"}
        with patch.object(self.routes, "live", side_effect=[baseline, published]), \
                patch.object(self.routes, "request", side_effect=[
                    {"version": {"id": "v2"}},
                    {"versions": [{"id": "v1", "isLive": True}, {"id": "v2", "isStaging": True}]},
                    {"version": {"id": "v2"}},
                ]) as request:
            self.assertEqual(self.routes.publish(baseline, published["origin"]), published)
            patch_call = request.call_args_list[0]
            self.assertEqual(patch_call.args[:2], ("PATCH", "/v1/projects/prj_test/routes/route_demo"))
            self.assertEqual(patch_call.args[2]["route"]["srcSyntax"], "regex")
            self.assertEqual(patch_call.args[2]["route"]["route"]["dest"], "https://sb-new.vercel.run/$1")
            self.assertEqual(request.call_args_list[-1].args[2], {"id": "v2", "action": "promote"})

    def test_concurrent_edit_after_staging_is_not_promoted(self) -> None:
        baseline = {"version_id": "v1", "rule": self.rule, "origin": "https://sb-old.vercel.run"}
        with patch.object(self.routes, "live", return_value=baseline), \
                patch.object(self.routes, "request", side_effect=[
                    {"version": {"id": "v2"}},
                    {"versions": [{"id": "v1", "isLive": True}, {"id": "user-edit", "isStaging": True}]},
                ]) as request:
            with self.assertRaisesRegex(RuntimeError, "nothing was promoted"):
                self.routes.publish(baseline, "https://sb-new.vercel.run")
            self.assertFalse(any(call.args[0] == "POST" for call in request.call_args_list))


class FakeRoutes:
    def __init__(self, events):
        self.events = events
        self.target = {"origin": "https://sb-old.vercel.run", "version_id": "v1"}

    def check_project(self, _origin):
        return None

    def live(self):
        return copy.deepcopy(self.target)

    def publish(self, baseline, origin):
        if baseline != self.live():
            raise RuntimeError("conflict")
        self.events.append(("publish", origin))
        self.target = {"origin": origin, "version_id": self.target["version_id"] + ".next"}
        return self.live()


class RolloutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        (ROOT / ".tmp").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="workflow-rollout-", dir=ROOT / ".tmp")
        self.addCleanup(self.directory.cleanup)
        self.work = Path(self.directory.name)
        self.args = argparse.Namespace(action="deploy", source_repo=ROOT, source_branch="main", work_dir=self.work)
        self.events = []
        self.routes = FakeRoutes(self.events)
        self.config = {"project_id": "prj_test", "team_id": "team_test", "route_name": "demo", "public_origin": "https://demo.vercel.app"}
        self.previous = {"sandbox_name": "old", "commit": "a" * 40, "prepared": True}
        self.fresh = {"sandbox_name": "new", "commit": "b" * 40, "prepared": True}
        self.failed_origin = None

    async def prepare(self, args, _revision, _payload):
        self.events.append(("prepare", "new"))
        DEMO.save(args.state, self.fresh)

    async def start(self, args, prepared):
        name = prepared["sandbox_name"]
        self.events.append(("start", name))
        DEMO.save(args.output, {"demo_url": f"https://sb-{name}.vercel.run"})

    async def stop(self, state):
        self.events.append(("stop", state["sandbox_name"]))

    def verify(self, origin, _expected=None):
        self.events.append(("verify", origin))
        if origin == self.failed_origin:
            raise RuntimeError("verification fixture failed")
        return {"catalog_total": 0, "health": {"storage_backend": "local"}}

    async def run_controller(self):
        with patch.object(DEMO, "ProjectRoutes", return_value=self.routes), \
                patch.object(DEMO, "access_token", return_value="test-token"), \
                patch.object(DEMO, "current_sandbox", AsyncMock(return_value=self.previous)), \
                patch.object(DEMO, "record_baseline", return_value={"health": {"storage_backend": "local"}, "catalog": {"total": 2}}), \
                patch.object(DEMO.PREPARE, "committed_source", return_value=("b" * 40, b"source", 1)), \
                patch.object(DEMO.PREPARE, "prepare", side_effect=self.prepare), \
                patch.object(DEMO.START, "start", side_effect=self.start), \
                patch.object(DEMO, "stop_sandbox", side_effect=self.stop), \
                patch.object(DEMO, "verify", side_effect=self.verify), \
                patch.object(DEMO.asyncio, "sleep", AsyncMock()), \
                patch.object(DEMO, "recovery_allowed", return_value=True), \
                patch.dict("os.environ", {"GITHUB_STEP_SUMMARY": ""}):
            await DEMO.execute(self.args, self.config)

    async def test_fresh_deploy_verifies_proxy_before_stopping_previous(self) -> None:
        await self.run_controller()
        self.assertLess(self.events.index(("verify", "https://sb-new.vercel.run")), self.events.index(("publish", "https://sb-new.vercel.run")))
        self.assertLess(self.events.index(("verify", "https://demo.vercel.app")), self.events.index(("stop", "old")))
        self.assertEqual(self.routes.live()["origin"], "https://sb-new.vercel.run")
        self.assertNotIn(("stop", "new"), self.events)

    async def test_unhealthy_candidate_never_replaces_or_stops_the_previous(self) -> None:
        self.failed_origin = "https://sb-new.vercel.run"
        with self.assertRaises(RuntimeError):
            await self.run_controller()
        self.assertEqual(self.routes.live()["origin"], "https://sb-old.vercel.run")
        self.assertNotIn(("stop", "old"), self.events)
        self.assertIn(("stop", "new"), self.events)

    async def test_failed_proxy_verification_restores_old_target_and_stops_candidate(self) -> None:
        self.failed_origin = "https://demo.vercel.app"
        with self.assertRaisesRegex(RuntimeError, "previous CDN target was restored"):
            await self.run_controller()
        self.assertEqual(self.routes.live()["origin"], "https://sb-old.vercel.run")
        self.assertNotIn(("stop", "old"), self.events)
        self.assertIn(("stop", "new"), self.events)

    async def test_start_resumes_published_sandbox_without_new_source_or_cutover(self) -> None:
        self.args.action = "start"
        self.args.source_repo = None
        await self.run_controller()
        self.assertIn(("start", "old"), self.events)
        self.assertFalse(any(event[0] in {"prepare", "publish", "stop"} for event in self.events))

    async def test_publication_error_preserves_previous_and_stops_unpublished_candidate(self) -> None:
        with patch.object(self.routes, "publish", side_effect=RuntimeError("API fixture unavailable")), \
                self.assertRaisesRegex(RuntimeError, "API fixture unavailable"):
            await self.run_controller()
        self.assertNotIn(("stop", "old"), self.events)
        self.assertIn(("stop", "new"), self.events)

    async def test_uncertain_publication_never_stops_a_potentially_published_candidate(self) -> None:
        original = self.routes.publish

        def uncertain(baseline, origin):
            original(baseline, origin)
            raise RuntimeError("Connection lost after publication")

        with patch.object(self.routes, "publish", side_effect=uncertain), \
                self.assertRaisesRegex(RuntimeError, "Connection lost"):
            await self.run_controller()
        self.assertFalse(any(event[0] == "stop" for event in self.events))
        self.assertFalse(json.loads((self.work / "result.json").read_text()).get("ready"))

    async def test_scheduled_recovery_never_restarts_an_active_or_transitioning_sandbox(self) -> None:
        self.args.action = "recover"
        for status in ["running", "pending", "stopping", "snapshotting"]:
            with self.subTest(status=status):
                self.events.clear()
                self.previous["status"] = status
                await self.run_controller()
                self.assertFalse(any(event[0] in {"prepare", "start", "publish", "stop"} for event in self.events))

    async def test_scheduled_recovery_resumes_stopped_sandbox_without_new_source(self) -> None:
        self.args.action = "recover"
        self.previous["status"] = "stopped"
        await self.run_controller()
        self.assertIn(("start", "old"), self.events)
        self.assertFalse(any(event[0] in {"prepare", "publish", "stop"} for event in self.events))

    async def test_scheduled_recovery_cannot_resume_a_failed_or_unknown_environment(self) -> None:
        self.args.action = "recover"
        self.previous["status"] = "failed"
        with self.assertRaisesRegex(RuntimeError, "not in a recoverable"):
            await self.run_controller()
        self.assertFalse(self.events)

    async def test_expired_recovery_makes_no_cloud_requests(self) -> None:
        self.args.action = "recover"
        with patch.object(DEMO, "recovery_allowed", return_value=False), \
                patch.object(DEMO, "ProjectRoutes") as routes, patch.object(DEMO, "access_token") as token:
            await DEMO.execute(self.args, self.config)
            routes.assert_not_called()
            token.assert_not_called()


class PublicVerificationTests(unittest.TestCase):
    def test_html_instead_of_javascript_is_rejected(self) -> None:
        response = SimpleNamespace(
            status=200, headers=SimpleNamespace(get_content_type=lambda: "text/html"),
        )
        context = unittest.mock.MagicMock()
        context.__enter__.return_value = response
        with patch.object(DEMO.START, "public_checks", return_value={}), \
                patch.object(DEMO, "read_url", return_value='<script src="/assets/app.js"></script>'), \
                patch.object(DEMO, "urlopen", return_value=context), \
                self.assertRaisesRegex(RuntimeError, "content type"):
            DEMO.verify("https://demo.vercel.app")

    def test_wrong_sandbox_identity_is_rejected_even_when_health_passes(self) -> None:
        response = SimpleNamespace(
            status=200, headers=SimpleNamespace(get_content_type=lambda: "text/javascript"),
        )
        context = unittest.mock.MagicMock()
        context.__enter__.return_value = response
        with patch.object(DEMO.START, "public_checks", return_value={}), \
                patch.object(DEMO, "read_url", side_effect=[
                    '<script src="/assets/app.js"></script>', {"commit": "a" * 40, "sandbox_name": "old"},
                ]), patch.object(DEMO, "urlopen", return_value=context), \
                self.assertRaisesRegex(RuntimeError, "another Sandbox"):
            DEMO.verify("https://demo.vercel.app", {"commit": "b" * 40, "sandbox_name": "new"})


class DiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_published_route_wins_over_a_more_recent_failed_candidate(self) -> None:
        boxes = [SimpleNamespace(
            name=name, tags={"app": "model-anatomy", "purpose": "demo", "commit": "a" * 40},
            cwd="/vercel", status="running",
            routes=[SimpleNamespace(port=8080, system=False, url=f"https://sb-{name}.vercel.run")],
        ) for name in ["failed-new", "published-old"]]

        async def query(**_kwargs):
            for box in boxes:
                yield box

        class Session:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        sdk = SimpleNamespace(query_sandboxes=query, get_sandbox=AsyncMock(side_effect=boxes))
        models = SimpleNamespace(**{key: SimpleNamespace for key in [
            "SandboxCredentials", "SandboxQueryByCreatedAt", "SandboxServiceOptions", "TagFilter",
        ]})
        with patch.dict(sys.modules, {
            "vercel": SimpleNamespace(sandbox=sdk), "vercel.api": SimpleNamespace(session=lambda **_kwargs: Session()),
            "vercel.sandbox": models,
        }):
            state = await DEMO.current_sandbox({"project_id": "prj_test", "team_id": "team_test"}, "https://sb-published-old.vercel.run")
        self.assertEqual(state["sandbox_name"], "published-old")
        self.assertEqual(state["app_dir"], "/vercel/model-anatomy")
        self.assertEqual(sdk.get_sandbox.await_count, 2)


if __name__ == "__main__":
    unittest.main()
