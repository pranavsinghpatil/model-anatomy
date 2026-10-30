"""Static runtime-contract and provider-validation tests; no inference."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from infra import preflight


class PreflightTests(unittest.TestCase):
    def test_default_schema_check_uses_root_alembic_config(self):
        with (
            patch.object(preflight, "check_database") as database,
            patch.object(preflight, "check_bucket"),
            patch.object(preflight, "check_models"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(preflight.main([]), 0)
        database.assert_called_once_with("alembic.ini", skip_schema=False)

    def test_skip_schema_is_explicit_and_reported(self):
        output = io.StringIO()
        with (
            patch.object(preflight, "check_database") as database,
            patch.object(preflight, "check_bucket"),
            patch.object(preflight, "check_models"),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(preflight.main(["--skip-schema"]), 0)
        database.assert_called_once_with("alembic.ini", skip_schema=True)
        self.assertIn("connectivity only; schema skipped", output.getvalue())

    def test_provider_errors_do_not_leak_secrets(self):
        output = io.StringIO()
        with (
            patch.object(preflight, "check_database", side_effect=RuntimeError("postgres://SECRET@host/db")),
            patch.object(preflight, "check_bucket", side_effect=RuntimeError("KEY_SECRET")),
            patch.object(preflight, "check_models", side_effect=RuntimeError("TOKEN_SECRET")),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(preflight.main([]), 1)
        self.assertNotIn("SECRET", output.getvalue())
        self.assertEqual(output.getvalue().count("FAIL:"), 3)

    def test_static_never_contacts_providers(self):
        with (
            patch.object(preflight, "check_runtime_contract"),
            patch.object(preflight, "read_json") as request,
            patch.object(preflight, "check_database") as database,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(preflight.main(["--static"]), 0)
        request.assert_not_called()
        database.assert_not_called()

        with (
            patch.object(preflight, "check_runtime_contract"),
            patch.object(preflight, "read_json") as request,
            patch.object(preflight, "check_database") as database,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(preflight.main(["--offline"]), 0)
        request.assert_not_called()
        database.assert_not_called()

    def test_runtime_contract_requires_components_and_excludes_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "backend/model_anatomy/api/app.py",
                "backend/model_anatomy/service/worker.py",
                "infra/healthcheck.py",
                "alembic.ini",
                "migrations/versions/001_initial.py",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            preflight.check_runtime_contract(root)
            (root / "output").mkdir()
            with self.assertRaisesRegex(ValueError, "must not contain"):
                preflight.check_runtime_contract(root)

    def test_model_must_be_exact_catalog_match(self):
        values = {"NEBIUS_MODEL": "nvidia/Nemotron-test", "NEBIUS_API_KEY": "secret"}
        with patch.object(preflight, "required", side_effect=values.__getitem__):
            with patch.object(
                preflight, "read_json", return_value={"data": [{"id": "nvidia/Nemotron-test"}]}
            ):
                preflight.check_models()
            with (
                patch.object(preflight, "read_json", return_value={"data": [{"id": "nvidia/Other"}]}),
                self.assertRaises(ValueError),
            ):
                preflight.check_models()
        with (
            patch.object(preflight, "required", return_value="other/Nemotron-test"),
            self.assertRaises(ValueError),
        ):
            preflight.check_models()

    def test_public_or_ambiguous_bucket_is_rejected(self):
        values = {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_KEY": "secret",
            "SUPABASE_BUCKET": "private",
        }
        with patch.object(preflight, "required", side_effect=values.__getitem__):
            for public in (True, None, "false", 0):
                with (
                    patch.object(preflight, "read_json", return_value={"id": "private", "public": public}),
                    self.assertRaises(ValueError),
                ):
                    preflight.check_bucket()
            with patch.object(preflight, "read_json", return_value={"id": "private", "public": False}):
                preflight.check_bucket()

    def test_unsafe_provider_urls_are_rejected(self):
        for value in (
            "http://host",
            "https://user:secret@host",
            "https://host?key=secret",
            "https://host:444",
        ):
            with self.assertRaises(ValueError):
                preflight.https_base(value)
        with self.assertRaises(ValueError):
            preflight.https_base("https://untrusted.example/v1", allowed_hosts=preflight.TOKEN_FACTORY_HOSTS)


if __name__ == "__main__":
    unittest.main()
