"""Guard the dependency floors that block known CVEs.

torch/transformers are bumped to CVE-fixed minimums in pyproject.toml:
- torch >= 2.5.0 (PYSEC-2026-2286, PYSEC-2024-259,
  GHSA-53q9-r3pm-6pq6, PYSEC-2025-41, PYSEC-2025-203,
  PYSEC-2025-204, PYSEC-2025-205, PYSEC-2025-207,
  PYSEC-2025-208, PYSEC-2025-209)
- transformers >= 4.48.0 (GHSA-hxxf-235m-72v3,
  GHSA-qxrp-vhvm-j765, GHSA-wrfc-pvp9-mr9g,
  PYSEC-2024-227, PYSEC-2024-228, PYSEC-2024-229,
  PYSEC-2026-2288, GHSA-xrqw-3rrv-vx5w,
  PYSEC-2026-3929, GHSA-29pf-2h5f-8g72,
  PYSEC-2026-2289, GHSA-fgcw-684q-jj6r,
  PYSEC-2025-40)
"""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MIN_FIXED = {
    "torch": (2, 5, 0),
    "transformers": (4, 48, 0),
}

_VERSION_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def _declared_floor(package: str) -> tuple[int, ...]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
        project = tomllib.load(fh)["project"]
    for dep in project["dependencies"]:
        name = re.split(r"[<>=~!]", dep, maxsplit=1)[0].strip()
        if name == package:
            match = _VERSION_RE.search(dep)
            if match is None:
                raise AssertionError(f"no version found for {package} in {dep!r}")
            return tuple(int(part) for part in match.groups() if part is not None)
    raise AssertionError(f"{package} not declared in pyproject.toml")


def test_declared_torch_floor_blocks_cves():
    assert _declared_floor("torch") >= MIN_FIXED["torch"]


def test_declared_transformers_floor_blocks_cves():
    assert _declared_floor("transformers") >= MIN_FIXED["transformers"]
