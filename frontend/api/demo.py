"""Public demo controls; credentials and SDK operations stay in this Function."""

import asyncio
import json
from http.server import BaseHTTPRequestHandler

from demo_control import ControlError, dispatch, validate_request


class handler(BaseHTTPRequestHandler):
    def respond(self, method: str) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 128:
                raise ControlError(413, "Request body exceeds the demo control limit.")
            body = self.rfile.read(length) if method == "POST" else b""
            action = validate_request(method, dict(self.headers), body)
            status, payload = asyncio.run(dispatch(action))
        except ControlError as exc:
            status, payload = exc.status, {"error": {"message": exc.message}}
        except Exception:  # noqa: BLE001 - Never return credential-bearing SDK exceptions.
            status, payload = 503, {"error": {"message": "Demo controls are unavailable. Try again shortly."}}
        content = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        self.respond("GET")

    def do_POST(self):
        self.respond("POST")

    def log_message(self, *_args):
        pass
