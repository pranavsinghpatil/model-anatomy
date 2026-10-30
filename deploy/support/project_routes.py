"""Publish the existing demo rewrite through Vercel's project routes API."""

from __future__ import annotations

import copy
import json
import re
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen


def sandbox_origin(destination: str) -> str:
    parsed = urlsplit(destination)
    if (
        parsed.scheme != "https"
        or not re.fullmatch(r"sb-[a-z0-9-]+\.vercel\.run", parsed.netloc)
        or parsed.path != "/$1"
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("The demo rewrite must target https://sb-....vercel.run/$1")
    return f"https://{parsed.netloc}"


class ProjectRoutes:
    def __init__(self, project_id: str, team_id: str, token: str, route_name: str) -> None:
        self.project_id = project_id
        self.team_id = team_id
        self.token = token
        self.route_name = route_name
        self.path = f"/v1/projects/{quote(project_id, safe='')}/routes"

    def request(self, method: str, path: str, body: dict | list | None = None, **query) -> dict:
        url = "https://api.vercel.com" + path + "?" + urlencode({"teamId": self.team_id, **query})
        request = Request(
            url, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as exc:
            # Provider error bodies can contain request details; keep them out of Actions logs.
            raise RuntimeError(f"Vercel API {method} failed with HTTP {exc.code}") from None

    def check_project(self, public_origin: str, *, require_production: bool = True) -> None:
        project = self.request("GET", f"/v9/projects/{quote(self.project_id, safe='')}")
        if project.get("id") != self.project_id or project.get("accountId") != self.team_id:
            raise RuntimeError("The Vercel project does not belong to the configured team")
        if project.get("link"):
            raise RuntimeError("Disconnect this project's Git integration before using manual Actions deployments")
        if require_production and not (project.get("targets") or {}).get("production"):
            raise RuntimeError("Create the regular Production entry point before using these workflows")
        domains = self.request("GET", f"/v9/projects/{quote(self.project_id, safe='')}/domains")
        if urlsplit(public_origin).hostname not in {item.get("name") for item in domains.get("domains", [])}:
            raise RuntimeError("The public demo domain is not assigned to this Vercel project")

    def live(self, *, allow_disabled: bool = False, allow_missing: bool = False) -> dict:
        versions = self.request("GET", self.path + "/versions")["versions"]
        if any(item.get("isStaging") for item in versions):
            raise RuntimeError("Unpublished CDN routing edits exist; publish or discard them in Vercel first")
        live = [item for item in versions if item.get("isLive")]
        if len(live) != 1 and not (allow_missing and not versions):
            raise RuntimeError("Expected exactly one published CDN routing version")
        version_id = live[0]["id"] if live else None
        document = self.request("GET", self.path, **({"versionId": version_id} if version_id else {}))
        matches = [item for item in document["routes"] if item.get("name") == self.route_name]
        if allow_missing:
            if any(item.get("enabled", True) and item.get("name") != self.route_name for item in document["routes"]):
                raise RuntimeError("Unrelated active CDN rules were preserved; inspect them before deploying")
            if not matches:
                return {"version_id": version_id, "rule": None, "origin": None}
        if len(matches) != 1:
            raise RuntimeError("Expected exactly one existing CDN rule with the configured name")
        rule = matches[0]
        route = rule["route"]
        if (
            (rule.get("enabled", True) is not True and not allow_disabled)
            or rule.get("srcSyntax", "regex") != "regex"
            or rule.get("rawSrc", route.get("src")) != "^/(.*)$"
            or any(route.get(key) for key in set(route) - {"src", "dest", "caseSensitive", "respectOriginCacheControl"})
        ):
            raise RuntimeError("The demo CDN rule must be an enabled, unconditional ^/(.*)$ Regex Rewrite")
        origin = sandbox_origin(route.get("dest", ""))
        return {"version_id": live[0]["id"], "rule": copy.deepcopy(rule), "origin": origin}

    def publish(self, baseline: dict, origin: str) -> dict:
        return self._publish(baseline, origin, enabled=True, allow_disabled=False)

    def set_enabled(self, baseline: dict, enabled: bool) -> dict:
        """Migrate the legacy catch-all only; preserve its reversible target."""
        return self._publish(baseline, baseline["origin"], enabled=enabled, allow_disabled=True)

    def _publish(self, baseline: dict, origin: str, *, enabled: bool, allow_disabled: bool) -> dict:
        sandbox_origin(origin + "/$1")
        options = {"allow_disabled": True} if allow_disabled else {}
        if self.live(**options) != baseline:
            raise RuntimeError("CDN routing changed during this run; the existing route was preserved")
        rule = baseline["rule"]
        response = self.request("PATCH", self.path + "/" + quote(rule["id"], safe=""), {
            "route": {
                "name": rule["name"],
                "description": "Forward all requests to the active Model Anatomy demo Sandbox.",
                "enabled": enabled, "srcSyntax": "regex",
                "route": {
                    "src": "^/(.*)$", "dest": origin + "/$1",
                    "caseSensitive": rule["route"].get("caseSensitive", False),
                    "respectOriginCacheControl": True,
                },
            },
        })
        version_id = response["version"]["id"]
        versions = self.request("GET", self.path + "/versions")["versions"]
        staged = [item["id"] for item in versions if item.get("isStaging")]
        live = [item["id"] for item in versions if item.get("isLive")]
        if staged != [version_id] or live != [baseline["version_id"]]:
            raise RuntimeError("CDN routing changed while staging; nothing was promoted")
        self.request("POST", self.path + "/versions", {"id": version_id, "action": "promote"})
        published = self.live(**options)
        if published["version_id"] != version_id or published["origin"] != origin:
            raise RuntimeError("The new CDN routing version could not be confirmed")
        if published["rule"].get("enabled", True) is not enabled:
            raise RuntimeError("The CDN rule activation could not be confirmed")
        return published
