"""Discover a deterministic-size, metadata-only 250 repository HF matrix.

The first 120 cases are copied byte-for-byte as JSON objects from the validated
100-series manifest. New cases are selected from Hugging Face's public model
API; no file contents or weights are downloaded. Run from any directory.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "benchmarks" / "model-matrix-100.json"
DEST = ROOT / "benchmarks" / "model-matrix-250.json"
TARGET = 250
ADDITIONS = TARGET - 120
USER_AGENT = "model-anatomy-benchmark-discovery/2.0"

# One query per intended task family keeps the additions balanced. The API's
# pipeline_tag supplies the declared task; category records the benchmark
# coverage intent, not successful reconstruction support.
PIPELINE_TAGS = [
    "text-generation",
    "text2text-generation",
    "fill-mask",
    "text-classification",
    "token-classification",
    "question-answering",
    "summarization",
    "translation",
    "sentence-similarity",
    "feature-extraction",
    "image-classification",
    "object-detection",
    "image-segmentation",
    "depth-estimation",
    "image-to-text",
    "visual-question-answering",
    "image-text-to-text",
    "video-classification",
    "automatic-speech-recognition",
    "text-to-speech",
    "audio-classification",
    "audio-to-audio",
    "text-to-audio",
    "zero-shot-audio-classification",
    "text-to-image",
    "image-to-image",
    "image-to-video",
    "text-to-video",
    "unconditional-image-generation",
    "zero-shot-image-classification",
]

# These API searches add specialist repository types that don't have a stable
# Transformers pipeline tag. Search hits are accepted only when the returned
# repository ID or metadata tags support the specialist intent.
SPECIALIST_SEARCHES = {
    "time-series-forecasting": ("chronos", "timeseries", "time-series", "moirai", "ttm"),
    "protein-model": ("esm2", "prot_bert", "ankh", "protein-language", "protein"),
    "tabular-model": ("tabpfn", "tabular", "table-prediction"),
    "robotics-policy": ("openvla", "lerobot", "robotics", "act_aloha", "pi0"),
}

SPECIALIST_MODEL_PATTERNS = {
    "time-series-forecasting": re.compile(
        r"chronos|timesfm|moirai|timeseries|time-series|tspulse|patchtst|ttm",
        re.IGNORECASE,
    ),
    "protein-model": re.compile(
        r"(?:^|[-_])(?:esm\d*|prot_bert|ankh)(?:$|[-_])",
        re.IGNORECASE,
    ),
    "tabular-model": re.compile(
        r"tabpfn|tabular|kumo[-_]tabular",
        re.IGNORECASE,
    ),
    "robotics-policy": re.compile(
        r"openvla|lerobot|robotics|act[-_]aloha|(?:^|[-_])pi0(?:$|[-_])",
        re.IGNORECASE,
    ),
}


def request_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def full_models(url: str) -> list[dict[str, Any]]:
    result = request_json(url)
    return result if isinstance(result, list) else []


def fetch_pipeline(tag: str) -> tuple[str, list[dict[str, Any]], str | None]:
    query = urllib.parse.urlencode(
        {
            "pipeline_tag": tag,
            "limit": 100,
            "sort": "downloads",
            "direction": "-1",
            "full": "true",
        }
    )
    url = "https://huggingface.co/api/models?" + query
    try:
        return tag, full_models(url), None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return tag, [], f"pipeline_tag={tag}: {exc}"


def fetch_specialist(item: tuple[str, str]) -> tuple[str, list[dict[str, Any]], str | None]:
    category, term = item
    query = urllib.parse.urlencode(
        {
            "search": term,
            "limit": 100,
            "sort": "downloads",
            "direction": "-1",
            "full": "true",
        }
    )
    url = "https://huggingface.co/api/models?" + query
    try:
        return category, full_models(url), None
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return category, [], f"search={term}: {exc}"


def valid_model(model: dict[str, Any]) -> bool:
    model_id, sha = model.get("id"), model.get("sha")
    return (
        isinstance(model_id, str)
        and "/" in model_id
        and isinstance(sha, str)
        and re.fullmatch(r"[0-9a-f]{40}", sha) is not None
        and isinstance(model.get("siblings"), list)
    )


def model_case(model: dict[str, Any], category: str, note: str | None = None) -> dict[str, Any]:
    case: dict[str, Any] = {
        "model_id": model["id"],
        "category": category,
        "revision": model["sha"],
    }
    task = model.get("pipeline_tag")
    if task:
        case["declared_task"] = task
    details = [note, f"HF API verified; library_name={model.get('library_name') or 'unspecified'}"]
    details = [detail for detail in details if detail]
    case["notes"] = "; ".join(details)
    return case


def supports_specialist(model: dict[str, Any], category: str, terms: tuple[str, ...]) -> bool:
    """Require model-name evidence and ignore the repository's author name."""
    repository_name = model.get("id", "").split("/", 1)[-1].lower()
    pattern = SPECIALIST_MODEL_PATTERNS.get(category)
    if pattern is None:
        evidence = " ".join(
            [
                repository_name,
                " ".join(model.get("tags") or []),
                " ".join(sibling.get("rfilename", "") for sibling in model.get("siblings", [])),
            ]
        ).lower()
        return any(term.lower() in evidence for term in terms)
    if category == "protein-model" and re.search(r"ner|detect", repository_name, re.IGNORECASE):
        return False
    return pattern.search(repository_name) is not None


def verify_reference(case: dict[str, Any]) -> tuple[str, str | None]:
    """Check an inherited artifact link at its exact pinned API revision."""
    reference = case.get("reference")
    if not reference:
        return case["model_id"], None
    parsed = urllib.parse.urlparse(reference)
    prefix, separator, artifact = parsed.path.partition("/blob/")
    parts = artifact.split("/", 1)
    if parsed.scheme != "https" or parsed.netloc != "huggingface.co" or not separator or len(parts) != 2:
        return case["model_id"], "malformed Hugging Face artifact URL"
    model_id = prefix.lstrip("/")
    revision, filename = parts
    if model_id != case["model_id"] or revision != case["revision"] or not filename:
        return case["model_id"], "artifact URL identity or revision differs from case"
    endpoint = (
        "https://huggingface.co/api/models/"
        + urllib.parse.quote(model_id, safe="/")
        + "/revision/"
        + revision
    )
    try:
        model = request_json(endpoint)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return case["model_id"], f"revision metadata request failed: {exc}"
    siblings = {item.get("rfilename") for item in model.get("siblings", [])}
    if model.get("id") != model_id or model.get("sha") != revision or filename not in siblings:
        return case["model_id"], "artifact path is absent at its pinned revision"
    return case["model_id"], None


def build_candidate_pools(
    results: list[tuple[str, list[dict[str, Any]], str | None]],
    existing_ids: set[str],
    specialist_searches: dict[str, tuple[str, ...]],
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    """Merge query results into stable, ranked, de-duplicated category pools."""
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    rejections: Counter[str] = Counter()
    for category, models, error in results:
        if error:
            print("REJECT", error)
        terms = specialist_searches.get(category)
        for model in models:
            if not valid_model(model):
                rejections["missing canonical ID, API SHA, or siblings"] += 1
                continue
            if model["id"] in existing_ids:
                rejections["already present in baseline"] += 1
                continue
            if terms and not supports_specialist(model, category, terms):
                rejections["specialist search lacks matching metadata evidence"] += 1
                continue
            if not terms and model.get("pipeline_tag") != category:
                rejections["pipeline tag mismatch"] += 1
                continue
            category_models = merged.setdefault(category, {})
            previous = category_models.get(model["id"])
            if previous is None or model_rank(model) < model_rank(previous):
                category_models[model["id"]] = model
            else:
                rejections["duplicate candidate across query pools"] += 1

    pools = {category: sorted(models.values(), key=model_rank) for category, models in merged.items()}
    return pools, rejections


def model_rank(model: dict[str, Any]) -> tuple[int, str, str]:
    """Rank by downloads, then canonical ID and revision for stable ties."""
    downloads = model.get("downloads")
    if not isinstance(downloads, int):
        downloads = 0
    return -downloads, model["id"], model["sha"]


def select_additions(
    pool_by_category: dict[str, list[dict[str, Any]]],
    categories: list[str],
    existing_ids: set[str],
    addition_count: int,
    author_limit: int = 8,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Select round-robin with separate scan cursors and accepted-case caps."""
    selected: list[dict[str, Any]] = []
    selected_ids = set(existing_ids)
    authors: Counter[str] = Counter()
    accepted_by_category: Counter[str] = Counter()
    scan_positions: Counter[str] = Counter()
    rejections: Counter[str] = Counter()

    while len(selected) < addition_count:
        changed = False
        for category in categories:
            if len(selected) >= addition_count:
                break
            category_limit = (
                8
                if category
                in {
                    "text-generation",
                    "text2text-generation",
                    "feature-extraction",
                    "text-to-image",
                    "automatic-speech-recognition",
                }
                else 6
            )
            if accepted_by_category[category] >= category_limit:
                continue
            candidates = pool_by_category.get(category, [])
            while scan_positions[category] < len(candidates):
                model = candidates[scan_positions[category]]
                scan_positions[category] += 1
                model_id = model["id"]
                author = model_id.split("/", 1)[0].lower()
                if model_id in selected_ids:
                    rejections["duplicate candidate across query pools"] += 1
                    continue
                if authors[author] >= author_limit:
                    rejections["author diversity cap"] += 1
                    continue
                selected_ids.add(model_id)
                authors[author] += 1
                accepted_by_category[category] += 1
                note = None
                if category in SPECIALIST_SEARCHES:
                    note = f"intended coverage: {category} (HF metadata search evidence)"
                selected.append(model_case(model, category, note))
                changed = True
                break
        if not changed:
            break
    return selected, rejections


def combine_cases(
    base_cases: list[dict[str, Any]],
    additions: list[dict[str, Any]],
    target: int = TARGET,
) -> list[dict[str, Any]]:
    """Retain baseline case objects and enforce exact size and unique IDs."""
    cases = base_cases + additions
    ids = [case["model_id"].casefold() for case in cases]
    if len(cases) != target or len(set(ids)) != target:
        raise SystemExit(
            f"Internal exact-count/uniqueness check failed: {len(cases)} cases, {len(set(ids))} IDs"
        )
    return cases


def main() -> None:
    base_data = json.loads(BASE.read_text(encoding="utf-8"))
    base_cases = base_data["cases"]
    if len(base_cases) != 120 or len({case["model_id"] for case in base_cases}) != 120:
        raise SystemExit(f"Expected exactly 120 distinct baseline cases; found {len(base_cases)}")
    existing = {case["model_id"] for case in base_cases}

    jobs = [("pipeline", tag) for tag in PIPELINE_TAGS]
    jobs += [
        ("specialist", (category, term)) for category, terms in SPECIALIST_SEARCHES.items() for term in terms
    ]
    results: list[tuple[str, list[dict[str, Any]], str | None]] = []
    # Keep remote discovery concurrency bounded to four requests.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for kind, job in jobs:
            fetch = fetch_pipeline if kind == "pipeline" else fetch_specialist
            futures.append(pool.submit(fetch, job))
        results = [future.result() for future in futures]

    pool_by_category, rejections = build_candidate_pools(results, existing, SPECIALIST_SEARCHES)
    categories = list(PIPELINE_TAGS) + list(SPECIALIST_SEARCHES)
    selected, selection_rejections = select_additions(pool_by_category, categories, existing, ADDITIONS)
    rejections.update(selection_rejections)

    if len(selected) != ADDITIONS:
        counts = {category: len(pool_by_category.get(category, [])) for category in categories}
        raise SystemExit(
            f"Could find only {len(selected)} distinct additions; "
            f"need {ADDITIONS}. Candidate counts: {counts}"
        )

    cases = combine_cases(base_cases, selected)
    for case in cases:
        if re.fullmatch(r"[0-9a-f]{40}", case["revision"]) is None:
            raise SystemExit(f"Invalid pinned revision: {case['model_id']}")
        reference = case.get("reference")
        if reference:
            parsed = urllib.parse.urlparse(reference)
            path_revision = parsed.path.split("/blob/", 1)[-1].split("/", 1)[0]
            if parsed.netloc != "huggingface.co" or path_revision != case["revision"]:
                raise SystemExit(f"Invalid pinned artifact reference: {case['model_id']}")

    references = [case for case in cases if case.get("reference")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        reference_results = list(pool.map(verify_reference, references))
    reference_failures = [(model_id, error) for model_id, error in reference_results if error]
    if reference_failures:
        raise SystemExit(f"Pinned artifact reference verification failed: {reference_failures}")

    manifest = {
        "schema_version": 1,
        "discovered_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "discovery": {
            "source": "Hugging Face public model metadata API",
            "baseline_manifest": "model-matrix-100.json",
            "baseline_cases_preserved": 120,
            "new_cases_metadata_verified": len(selected),
            "artifact_references_verified_at_pinned_revision": len(references),
            "weights_downloaded": False,
            "new_case_selection": (
                "balanced round-robin by declared pipeline task and "
                "evidence-filtered specialist searches; max eight additions per author"
            ),
            "rejections": dict(sorted(rejections.items())),
            "category_counts": dict(sorted(Counter(case["category"] for case in cases).items())),
        },
        "cases": cases,
    }
    DEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE {DEST} ({len(cases)} unique repositories; {len(selected)} new API-verified cases)")
    print("REJECTIONS", dict(rejections))


if __name__ == "__main__":
    main()
