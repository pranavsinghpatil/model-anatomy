"""Offline checks for committed source uploads and Project selection."""

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))
SPEC = importlib.util.spec_from_file_location("production_entrypoint", ROOT / "deploy/04_create_production_entrypoint.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load the deployment helper")
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class SourceUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        (ROOT / ".tmp").mkdir(exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(prefix="entrypoint-test-", dir=ROOT / ".tmp")
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)
        self.git("init", "-q", "-b", "feat/deploy")
        self.git("config", "user.name", "Deployment Test")
        self.git("config", "user.email", "deployment-test@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        hooks = self.repo / ".git/empty-hooks"
        hooks.mkdir()
        self.git("config", "core.hooksPath", str(hooks))
        frontend = self.repo / "frontend"
        frontend.mkdir()
        (frontend / "package.json").write_text(json.dumps({"scripts": {"build": "vite build"}}))
        (frontend / "package-lock.json").write_text("{}")
        (frontend / "index.html").write_text("committed frontend")
        (frontend / "icon.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\x00")
        (self.repo / "private-profile.json").write_text("not frontend source")
        self.git("add", "-f", ".")
        self.git("commit", "-qm", "Create deployment fixture")
        self.commit = self.git("rev-parse", "HEAD")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.repo, check=True, capture_output=True, text=True,
        ).stdout.strip()

    def test_uses_named_branch_commit_instead_of_working_tree_or_checked_out_branch(self) -> None:
        self.git("switch", "-qc", "other-branch")
        (self.repo / "frontend/index.html").write_text("other branch")
        self.git("commit", "-qam", "Change other branch")
        (self.repo / "frontend/index.html").write_text("uncommitted change")
        (self.repo / "frontend/.env").write_text("UNTRACKED_TEST_VALUE=not-for-upload")
        commit, files, _total = HELPER.frontend_files(self.repo)
        contents = {item["file"]: base64.b64decode(item["data"]) for item in files}
        self.assertEqual(commit, self.commit)
        self.assertEqual(contents["frontend/index.html"], b"committed frontend")
        self.assertNotIn("frontend/.env", contents)
        self.assertNotIn("private-profile.json", contents)
        self.assertEqual(contents["frontend/icon.png"], b"\x89PNG\r\n\x1a\n\xff\x00")

    def test_rejects_tracked_private_environment_files(self) -> None:
        (self.repo / "frontend/.env.production").write_text("TEST_VALUE=not-for-upload")
        self.git("add", "-f", ".")
        self.git("commit", "-qm", "Track an invalid environment fixture")
        with self.assertRaisesRegex(RuntimeError, "private environment"):
            HELPER.frontend_files(self.repo)

    def test_payload_has_explicit_target_and_provenance_without_git_identity(self) -> None:
        commit, files, _total = HELPER.frontend_files(self.repo)
        payload = HELPER.deployment_payload("prj_test", "test-project", commit, files)
        self.assertEqual(payload["project"], "prj_test")
        self.assertEqual(payload["target"], "production")
        self.assertEqual(payload["meta"]["modelAnatomySourceCommit"], self.commit)
        self.assertNotIn("gitSource", payload)
        self.assertNotIn("gitMetadata", payload)


class ProjectSelectionTests(unittest.TestCase):
    def test_rejects_other_accounts_links_and_existing_production(self) -> None:
        invalid = [
            {"id": "prj_other", "accountId": "team_test"},
            {"id": "prj_test", "accountId": "team_other"},
            {"id": "prj_test", "accountId": "team_test", "link": {"type": "github"}},
            {"id": "prj_test", "accountId": "team_test", "targets": {"production": {"readyState": "READY"}}},
            {"id": "prj_test", "accountId": "team_test", "targets": {"production": {"readyState": "BUILDING"}}},
        ]
        for project in invalid:
            with self.subTest(project=project), self.assertRaises(RuntimeError):
                HELPER.check_project(project, "prj_test", "team_test")

    def test_accepts_the_unlinked_project_and_a_failed_initial_build(self) -> None:
        HELPER.check_project({"id": "prj_test", "accountId": "team_test"}, "prj_test", "team_test")
        HELPER.check_project({
            "id": "prj_test", "accountId": "team_test", "targets": {"production": {"readyState": "ERROR"}},
        }, "prj_test", "team_test")


if __name__ == "__main__":
    unittest.main()
