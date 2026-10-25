"""Guard reviewed engine dependency constraints and all platform lock entries.

The security minimums cover the advisories reviewed for PR #59, including
https://osv.dev/vulnerability/PYSEC-2026-2286 (torch fixed in 2.10.0) and
https://osv.dev/vulnerability/GHSA-xrqw-3rrv-vx5w (transformers fixed in 5.10.0).
This is a regression guard, not an ongoing vulnerability scan or safety claim.
"""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from model_anatomy.src.version import NATIVE_TRANSFORMERS_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
MIN_FIXED = {"torch": Version("2.10.0"), "transformers": Version("5.10.0")}


def _read_toml(name: str) -> dict:
    with (REPO_ROOT / name).open("rb") as handle:
        return tomllib.load(handle)


def _engine_requirements() -> dict[str, Requirement]:
    dependencies = _read_toml("pyproject.toml")["project"]["optional-dependencies"]["engine"]
    return {requirement.name: requirement for requirement in map(Requirement, dependencies)}


def test_torch_manifest_declares_reviewed_security_floor():
    requirement = _engine_requirements()["torch"]
    lower_bounds = [
        Version(specifier.version)
        for specifier in requirement.specifier
        if specifier.operator in {">=", "==", "~="}
    ]
    assert lower_bounds and max(lower_bounds) >= MIN_FIXED["torch"]


@pytest.mark.parametrize("affected", ["2.4.0", "2.5.0", "2.6.0", "2.8.0", "2.9.1"])
def test_torch_manifest_excludes_reviewed_affected_releases(affected):
    assert Version(affected) not in _engine_requirements()["torch"].specifier


def test_transformers_manifest_preserves_audited_source_contract():
    assert Version(NATIVE_TRANSFORMERS_VERSION) >= MIN_FIXED["transformers"]
    assert _engine_requirements()["transformers"].specifier == SpecifierSet(
        f"=={NATIVE_TRANSFORMERS_VERSION}"
    )


@pytest.mark.parametrize("name", ["torch", "transformers"])
def test_every_platform_lock_entry_satisfies_manifest_and_security_minimum(name):
    requirement = _engine_requirements()[name]
    packages = [package for package in _read_toml("uv.lock")["package"] if package["name"] == name]
    assert packages, f"{name} must be resolved in uv.lock"
    for package in packages:
        version = Version(package["version"])
        assert version >= MIN_FIXED[name]
        assert version in requirement.specifier


@pytest.mark.parametrize("name", ["torch", "transformers"])
def test_lock_root_requirement_metadata_matches_manifest(name):
    root = next(
        package for package in _read_toml("uv.lock")["package"] if package["name"] == "model-anatomy"
    )
    entries = [entry for entry in root["metadata"]["requires-dist"] if entry["name"] == name]
    assert entries, f"{name} must be recorded in the lock's project metadata"
    for entry in entries:
        assert SpecifierSet(entry["specifier"]) == _engine_requirements()[name].specifier
