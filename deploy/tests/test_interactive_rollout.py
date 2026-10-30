"""Exercise frontend publication, legacy routing migration, and rollback order."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))
import test_demo_controls as fixtures
from support import interactive_demo as INTERACTIVE
from support.project_routes import ProjectRoutes


class Routes:
    def __init__(self, events):
        self.events = events
        self.value = {
            "version_id": "v1", "origin": "https://sb-old.vercel.run",
            "rule": {"id": "route", "enabled": True},
        }

    def check_project(self, _origin, **_kwargs):
        pass

    def live(self, **_kwargs):
        return copy.deepcopy(self.value)

    def set_enabled(self, baseline, enabled):
        if self.value != baseline:
            raise RuntimeError("routing changed")
        self.events.append("enable-legacy" if enabled else "disable-legacy")
        self.value["rule"]["enabled"] = enabled
        self.value["version_id"] += "x"
        return self.live()


class Production:
    def __init__(self, events):
        self.events = events
        self.deployment = {"id": "dpl_before", "meta": {}}
        self.fail_create_response = False

    def current(self, **_kwargs):
        return copy.deepcopy(self.deployment)

    def create(self, _archive, candidate, _config, _branch):
        self.events.append("publish-production")
        self.deployment = {"id": "dpl_new", "meta": {
            "modelAnatomyProtocol": INTERACTIVE.PROTOCOL,
            "modelAnatomySandbox": candidate["sandbox_name"], "modelAnatomySourceCommit": candidate["commit"],
            "modelAnatomySandboxOrigin": candidate["demo_url"],
        }}
        if self.fail_create_response:
            raise OSError("lost response")
        return self.current()

    def wait(self, _deployment_id):
        self.events.append("build-ready")
        return self.current()

    def cancel_pending(self, _deployment_id):
        self.events.append("check-build")

    def restore(self, previous_id, expected_id):
        if self.deployment["id"] != expected_id:
            raise RuntimeError("another deployment")
        self.events.append("restore-production")
        self.deployment = {"id": previous_id, "meta": {}}


class InteractiveRolloutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        self.routes = Routes(self.events)
        self.production = Production(self.events)
        self.previous = {"sandbox_name": "old", "commit": "a" * 40, "app_dir": "/vercel/model-anatomy", "prepared": True}
        self.candidate = {"sandbox_name": "new", "commit": "b" * 40, "app_dir": "/vercel/model-anatomy", "prepared": True}
        self.config = {"project_id": "prj_test", "team_id": "team_test", "route_name": "Demo", "public_origin": "https://demo.vercel.app"}
        self.folder = tempfile.TemporaryDirectory(dir=ROOT / ".tmp", prefix="interactive-rollout-")
        self.addCleanup(self.folder.cleanup)
        self.args = SimpleNamespace(action="deploy", work_dir=Path(self.folder.name), source_repo=Path("source"), source_branch="feat/deploy")

        def save(path, value):
            path.write_text(json.dumps(value))

        async def prepare(args, _revision, _payload):
            self.events.append("prepare-sandbox")
            save(args.state, self.candidate)

        async def start(args, _candidate):
            self.events.append("start-sandbox")
            save(args.output, {"demo_url": "https://sb-new.vercel.run"})

        async def stop(state):
            self.events.append("stop-" + state["sandbox_name"])

        self.ops = SimpleNamespace(
            current_sandbox=AsyncMock(return_value=self.previous),
            record_baseline=MagicMock(return_value={"health": {"storage_backend": "local"}, "catalog": {"total": 2}}),
            save=save, source_revision=MagicMock(return_value="b" * 40),
            PREPARE=SimpleNamespace(
                committed_source=MagicMock(return_value=("b" * 40, fixtures.ProductionPayloadTests().archive(), 8)),
                prepare=prepare,
            ), START=SimpleNamespace(start=start),
            verify=MagicMock(return_value={"catalog_total": 0}), stop_sandbox=stop, append_summary=MagicMock(),
        )
        for name, value in [("ProjectRoutes", self.routes), ("Production", self.production)]:
            context = patch.object(INTERACTIVE, name, return_value=value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(INTERACTIVE, "access_token", return_value="test-token")
        context.start()
        self.addCleanup(context.stop)
        context = patch.object(INTERACTIVE, "verify_entrypoint", side_effect=self.verify)
        self.check = context.start()
        self.addCleanup(context.stop)

    def verify(self, _config, _candidate, _ops):
        self.events.append("verify-public-controls")
        return {"catalog_total": 0, "controls": True}

    async def test_cutover_checks_frontend_and_controls_before_stopping_old_sandbox(self):
        await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.assertEqual(self.events, [
            "prepare-sandbox", "start-sandbox", "publish-production", "build-ready",
            "disable-legacy", "verify-public-controls", "stop-old",
        ])

    async def test_failed_public_checks_restore_production_and_legacy_rule_before_cleanup(self):
        self.check.side_effect = RuntimeError("controls unavailable")
        with patch.object(INTERACTIVE.asyncio, "sleep", AsyncMock()), self.assertRaises(RuntimeError):
            await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.assertEqual(self.events[-4:], ["check-build", "restore-production", "enable-legacy", "stop-new"])
        self.assertNotIn("stop-old", self.events)

    async def test_lost_create_response_preserves_potentially_published_candidate(self):
        self.production.fail_create_response = True
        with self.assertRaises(OSError):
            await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.assertEqual(self.production.current()["id"], "dpl_new")
        self.assertNotIn("stop-new", self.events)
        self.assertNotIn("stop-old", self.events)

    async def test_start_recovers_production_target_instead_of_disabled_legacy_destination(self):
        self.args.action = "start"
        self.routes.value["rule"]["enabled"] = False
        self.production.deployment = {"id": "dpl_current", "meta": {
            "modelAnatomyProtocol": INTERACTIVE.PROTOCOL,
            "modelAnatomySandbox": "new", "modelAnatomySourceCommit": "b" * 40,
            "modelAnatomySandboxOrigin": "https://sb-new.vercel.run",
        }}
        self.ops.current_sandbox.return_value = self.candidate
        await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.ops.current_sandbox.assert_awaited_once_with(self.config, "https://sb-new.vercel.run")
        self.assertEqual(self.events, ["start-sandbox", "verify-public-controls"])
        self.ops.PREPARE.committed_source.assert_not_called()

    async def test_recover_is_a_noop_without_reading_tokens_or_cloud(self):
        self.args.action = "recover"
        with patch.object(INTERACTIVE, "access_token") as token:
            await INTERACTIVE.execute(self.args, {}, self.ops)
            token.assert_not_called()
        self.assertEqual(self.events, [])

    async def test_first_deployment_needs_no_old_sandbox_or_cdn_rule(self):
        self.production.deployment = None
        self.routes.value = {"version_id": None, "rule": None, "origin": None}
        await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.assertEqual(self.events, [
            "prepare-sandbox", "start-sandbox", "publish-production", "build-ready", "verify-public-controls",
        ])
        self.ops.current_sandbox.assert_not_awaited()

    async def test_new_project_placeholder_requires_explicit_initialization(self):
        self.routes.value = {"version_id": None, "rule": None, "origin": None}
        with self.assertRaisesRegex(RuntimeError, "Initialize project"):
            await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.assertEqual(self.events, [])
        self.args.initialize_project = True
        await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.assertNotIn("disable-legacy", self.events)
        self.assertNotIn("stop-old", self.events)

    async def test_placeholder_restored_when_first_public_verification_fails(self):
        self.routes.value = {"version_id": None, "rule": None, "origin": None}
        self.args.initialize_project = True
        self.check.side_effect = RuntimeError("controls unavailable")
        with patch.object(INTERACTIVE.asyncio, "sleep", AsyncMock()), self.assertRaises(RuntimeError):
            await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.assertEqual(self.events[-3:], ["check-build", "restore-production", "stop-new"])

    async def test_first_production_is_preserved_when_no_rollback_target_exists(self):
        self.production.deployment = None
        self.routes.value = {"version_id": None, "rule": None, "origin": None}
        self.check.side_effect = RuntimeError("controls unavailable")
        with patch.object(INTERACTIVE.asyncio, "sleep", AsyncMock()), self.assertRaises(RuntimeError):
            await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.assertNotIn("stop-new", self.events)
        self.assertNotIn("restore-production", self.events)

    async def test_interactive_update_does_not_require_a_legacy_rule(self):
        self.routes.value = {"version_id": None, "rule": None, "origin": None}
        self.production.deployment["meta"] = {
            "modelAnatomyProtocol": INTERACTIVE.PROTOCOL,
            "modelAnatomySandbox": "old", "modelAnatomySourceCommit": "a" * 40,
            "modelAnatomySandboxOrigin": "https://sb-old.vercel.run",
        }
        await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.assertEqual(self.events[-1], "stop-old")
        self.assertNotIn("disable-legacy", self.events)

    async def test_unknown_protocol_cannot_be_overwritten_with_initialization_flag(self):
        self.routes.value = {"version_id": None, "rule": None, "origin": None}
        self.production.deployment["meta"] = {"modelAnatomyProtocol": "another-protocol"}
        self.args.initialize_project = True
        with self.assertRaises(RuntimeError):
            await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.assertEqual(self.events, [])

    async def test_start_on_empty_project_does_not_create_or_resume_any_sandbox(self):
        self.production.deployment = None
        self.routes.value = {"version_id": None, "rule": None, "origin": None}
        self.args.action = "start"
        with self.assertRaisesRegex(RuntimeError, "Deploy Demo first"):
            await INTERACTIVE.execute(self.args, self.config, self.ops)
        self.assertEqual(self.events, [])


class RouteMigrationTests(unittest.TestCase):
    def test_empty_project_has_no_legacy_rule(self):
        routes = ProjectRoutes("prj_test", "team_test", "token", "Demo")
        with patch.object(routes, "request", side_effect=[{"versions": []}, {"routes": []}]):
            self.assertEqual(routes.live(allow_missing=True), {"version_id": None, "rule": None, "origin": None})

    def test_live_version_with_no_legacy_rule_is_valid_for_interactive_deployment(self):
        routes = ProjectRoutes("prj_test", "team_test", "token", "Demo")
        with patch.object(routes, "request", side_effect=[{"versions": [{"id": "v1", "isLive": True}]}, {"routes": []}]):
            self.assertIsNone(routes.live(allow_missing=True)["rule"])

    def test_unrelated_active_rules_fail_before_any_changes(self):
        routes = ProjectRoutes("prj_test", "team_test", "token", "Demo")
        with patch.object(routes, "request", side_effect=[{"versions": []}, {"routes": [{"name": "Other", "enabled": True}]}]), self.assertRaisesRegex(RuntimeError, "Unrelated"):
            routes.live(allow_missing=True)

    def test_git_connected_project_must_be_disconnected(self):
        routes = ProjectRoutes("prj_test", "team_test", "token", "Demo")
        with patch.object(routes, "request", return_value={"id": "prj_test", "accountId": "team_test", "link": {"type": "github"}}), self.assertRaisesRegex(RuntimeError, "Disconnect"):
            routes.check_project("https://demo.vercel.app", require_production=False)

    def test_empty_unlinked_project_accepts_its_own_assigned_domain(self):
        routes = ProjectRoutes("prj_test", "team_test", "token", "Demo")
        with patch.object(routes, "request", side_effect=[
            {"id": "prj_test", "accountId": "team_test", "targets": None}, {"domains": [{"name": "demo.vercel.app"}]},
        ]):
            routes.check_project("https://demo.vercel.app", require_production=False)

    def test_disabled_rule_can_only_be_read_when_explicitly_allowed(self):
        routes = ProjectRoutes("prj_test", "team_test", "token", "Demo")
        rule = {"id": "r1", "name": "Demo", "enabled": False, "rawSrc": "^/(.*)$", "srcSyntax": "regex", "route": {"src": "^/(.*)$", "dest": "https://sb-old.vercel.run/$1"}}
        replies = [{"versions": [{"id": "v1", "isLive": True}]}, {"routes": [rule]}]
        with patch.object(routes, "request", side_effect=replies):
            self.assertFalse(routes.live(allow_disabled=True)["rule"]["enabled"])
        with patch.object(routes, "request", side_effect=replies), self.assertRaises(RuntimeError):
            routes.live()


if __name__ == "__main__":
    unittest.main()
