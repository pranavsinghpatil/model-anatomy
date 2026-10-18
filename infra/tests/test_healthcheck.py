"""Health probes detect stale leases and suppress credential-bearing errors."""

import contextlib
import io
import time
import unittest
from unittest.mock import MagicMock, patch

from infra import healthcheck


class HealthcheckTests(unittest.TestCase):
    def test_worker_requires_a_fresh_lease(self):
        process = MagicMock()
        process.read_bytes.return_value = b"python\0-m\0model_anatomy.service.worker\0"
        psycopg = MagicMock()
        connection = psycopg.connect.return_value.__enter__.return_value
        with (
            patch.object(healthcheck.Path, "glob", return_value=[process]),
            patch.dict("sys.modules", {"psycopg": psycopg}),
            patch.dict(healthcheck.os.environ, {"MODEL_ANATOMY_DATABASE_URL": "postgresql://fake/db"}),
        ):
            for row in (None, (None, time.time() + 100), ("owner", time.time() - 100)):
                connection.execute.return_value.fetchone.return_value = row
                with self.assertRaises(RuntimeError):
                    healthcheck.check_worker()
            connection.execute.return_value.fetchone.return_value = ("owner", time.time() + 100)
            healthcheck.check_worker()
        self.assertIn("default_transaction_read_only=on", psycopg.connect.call_args.kwargs["options"])
        self.assertTrue(connection.execute.call_args.args[0].startswith("SELECT "))

    def test_missing_worker_does_not_connect(self):
        psycopg = MagicMock()
        with (
            patch.object(healthcheck.Path, "glob", return_value=[]),
            patch.dict("sys.modules", {"psycopg": psycopg}),
            self.assertRaises(RuntimeError),
        ):
            healthcheck.check_worker()
        psycopg.connect.assert_not_called()

    def test_probe_errors_are_quiet(self):
        output = io.StringIO()
        with (
            patch.object(healthcheck.sys, "argv", ["healthcheck.py", "worker"]),
            patch.object(healthcheck, "check_worker", side_effect=RuntimeError("postgres://SECRET@host")),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
        ):
            self.assertEqual(healthcheck.main(), 1)
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
