"""Check private account setup and account-independent deployment configuration."""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy"))
import test_github_demo

DEMO = test_github_demo.DEMO
CREATE = DEMO.helper("create_project", "01_vercel_create_project.py")


class AccountSetupTests(unittest.TestCase):
    def setUp(self):
        (ROOT / ".tmp").mkdir(exist_ok=True)
        folder = tempfile.TemporaryDirectory(dir=ROOT / ".tmp", prefix="account-setup-")
        self.addCleanup(folder.cleanup)
        self.private = Path(folder.name) / ".env"
        self.config = Path(folder.name) / "settings.json"
        self.config.write_text('{"route_name":"Model Anatomy Sandbox","interactive_controls":true}')
        environment = patch.dict(os.environ, {
            "CI": "true", "VERCEL_PROJECT_ID": "prj_teammate", "VERCEL_TEAM_ID": "team_teammate",
            "VERCEL_TOKEN": "private-test-token", "DEMO_PUBLIC_ORIGIN": "https://teammate-demo.vercel.app/",
        }, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_selected_account_and_domain_come_only_from_environment(self):
        value = DEMO.load_configuration(self.config)
        self.assertEqual(value["project_id"], "prj_teammate")
        self.assertEqual(value["team_id"], "team_teammate")
        self.assertEqual(value["public_origin"], "https://teammate-demo.vercel.app")
        self.assertNotIn("private-test-token", json.dumps(value))

    def test_no_fallback_to_another_operators_domain(self):
        del os.environ["DEMO_PUBLIC_ORIGIN"]
        with self.assertRaisesRegex(RuntimeError, "DEMO_PUBLIC_ORIGIN"):
            DEMO.load_configuration(self.config)

    def test_account_specific_settings_are_rejected_in_tracked_json(self):
        for key, value in [("project_id", "prj_other"), ("team_id", "team_other"), ("token", "private"), ("public_origin", "https://other.vercel.app")]:
            with self.subTest(key=key):
                self.config.write_text(json.dumps({key: value}))
                with self.assertRaisesRegex(RuntimeError, "account-specific"):
                    DEMO.load_configuration(self.config)

    def test_private_env_can_supply_an_operators_own_domain(self):
        self.private.write_text('DEMO_PUBLIC_ORIGIN="https://private-profile.vercel.app"\n')
        del os.environ["DEMO_PUBLIC_ORIGIN"]
        del os.environ["CI"]
        from support.credentials import load_private_environment
        load_private_environment(self.private)
        self.assertEqual(DEMO.load_configuration(self.config)["public_origin"], "https://private-profile.vercel.app")

    def test_project_creation_saves_identifiers_without_logging_credentials(self):
        del os.environ["VERCEL_PROJECT_ID"]
        self.private.write_text('# Private operator profile\nVERCEL_PROJECT_ID=prj_your_project_id\nVERCEL_TOKEN="keep-this-value"\nDEMO_PUBLIC_ORIGIN=https://teammate-demo.vercel.app\n')
        response = io.StringIO(json.dumps({"id": "prj_created", "accountId": "team_teammate", "name": "teammate-demo"}))
        captured = io.StringIO()
        with patch.object(CREATE, "urlopen", return_value=contextlib.closing(response)) as request, contextlib.redirect_stdout(captured):
            CREATE.create_project("teammate-demo", private_env=self.private)
        self.assertEqual(json.loads(request.call_args.args[0].data), {"name": "teammate-demo"})
        profile = self.private.read_text()
        self.assertIn("VERCEL_PROJECT_ID=prj_created", profile)
        self.assertIn("VERCEL_TEAM_ID=team_teammate", profile)
        self.assertIn('VERCEL_TOKEN="keep-this-value"', profile)
        self.assertIn("DEMO_PUBLIC_ORIGIN=https://teammate-demo.vercel.app", profile)
        self.assertEqual(stat.S_IMODE(self.private.stat().st_mode), 0o600)
        for private_value in ["prj_created", "team_teammate", "private-test-token", "keep-this-value"]:
            self.assertNotIn(private_value, captured.getvalue())

    def test_project_creation_refuses_to_overwrite_an_existing_profile(self):
        with patch.object(CREATE, "urlopen") as request, self.assertRaisesRegex(RuntimeError, "already configured"):
            CREATE.create_project("teammate-demo", private_env=self.private)
        request.assert_not_called()
        self.assertFalse(self.private.exists())


if __name__ == "__main__":
    unittest.main()
