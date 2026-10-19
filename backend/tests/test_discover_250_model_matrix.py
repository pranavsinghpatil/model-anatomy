from __future__ import annotations

import pytest
from scripts.discover_250_model_matrix import (
    build_candidate_pools,
    combine_cases,
    select_additions,
    supports_specialist,
)


def api_model(
    model_id: str,
    *,
    downloads: int = 0,
    pipeline_tag: str | None = None,
    tags: list[str] | None = None,
    revision: str = "a" * 40,
) -> dict[str, object]:
    return {
        "id": model_id,
        "sha": revision,
        "downloads": downloads,
        "pipeline_tag": pipeline_tag,
        "tags": tags or [],
        "siblings": [],
    }


def test_candidate_pools_are_stably_ranked_and_merge_duplicate_search_hits() -> None:
    lower = api_model("org/prot_bert-lower", downloads=10, tags=["protein"])
    upper = api_model("org/esm2-upper", downloads=20, tags=["protein"])
    duplicate = api_model("org/esm2-upper", downloads=20, tags=["protein"])
    results_a = [
        ("protein-model", [lower, upper], None),
        ("protein-model", [duplicate], None),
    ]
    results_b = list(reversed(results_a))
    specialist_searches = {"protein-model": ("protein",)}

    pools_a, _ = build_candidate_pools(results_a, set(), specialist_searches)
    pools_b, _ = build_candidate_pools(results_b, set(), specialist_searches)

    assert [model["id"] for model in pools_a["protein-model"]] == [
        "org/esm2-upper",
        "org/prot_bert-lower",
    ]
    assert pools_a == pools_b


@pytest.mark.parametrize(
    ("model_id", "category", "terms", "tags"),
    [
        (
            "tabularisai/multilingual-sentiment-analysis",
            "tabular-model",
            ("tabular",),
            ["text-classification", "sentiment"],
        ),
        (
            "tabularisai/ModernFinBERT",
            "tabular-model",
            ("tabular",),
            ["text-classification", "finance"],
        ),
        (
            "OpenMed/OpenMed-NER-ProteinDetect-SuperClinical-141M",
            "protein-model",
            ("protein",),
            ["token-classification", "biomedical-ner"],
        ),
    ],
)
def test_specialist_filter_rejects_author_and_task_name_false_matches(
    model_id: str,
    category: str,
    terms: tuple[str, ...],
    tags: list[str],
) -> None:
    assert not supports_specialist(api_model(model_id, tags=tags), category, terms)


@pytest.mark.parametrize(
    ("model_id", "category", "terms"),
    [
        ("facebook/esm2_t6_8M_UR50D", "protein-model", ("protein",)),
        ("Rostlab/prot_bert", "protein-model", ("protein",)),
        ("ElnaggarLab/ankh-base", "protein-model", ("protein",)),
        ("Prior-Labs/TabPFN-v2-clf", "tabular-model", ("tabpfn",)),
        ("amazon/chronos-t5-small", "time-series-forecasting", ("chronos",)),
    ],
)
def test_specialist_filter_accepts_trusted_model_name_patterns(
    model_id: str,
    category: str,
    terms: tuple[str, ...],
) -> None:
    assert supports_specialist(api_model(model_id), category, terms)


def test_selection_scans_past_author_cap_and_enforces_author_category_and_id_caps() -> None:
    same_author = [api_model(f"crowded/model-{index}", downloads=100 - index) for index in range(8)]
    category_one = same_author + [
        api_model(f"other-{index}/model", downloads=10 - index) for index in range(8)
    ]
    shared = api_model("shared/model", downloads=30)
    category_two = [shared, api_model("second/model", downloads=20)]
    category_three = [api_model(f"third-{index}/model", downloads=10 - index) for index in range(8)]

    selected, _ = select_additions(
        {
            "text-classification": category_one,
            "token-classification": category_two,
            "question-answering": category_three,
        },
        ["text-classification", "token-classification", "question-answering"],
        {"shared/model"},
        addition_count=20,
        author_limit=2,
    )

    ids = [case["model_id"] for case in selected]
    authors = [model_id.split("/", 1)[0] for model_id in ids]
    categories = [case["category"] for case in selected]
    assert len(ids) == len(set(ids))
    assert "shared/model" not in ids
    assert len(ids) == 13
    assert max(authors.count(author) for author in set(authors)) <= 2
    assert categories.count("text-classification") == 6
    assert categories.count("token-classification") == 1
    assert categories.count("question-answering") == 6


def test_combine_cases_preserves_baseline_and_enforces_exact_250_unique_ids() -> None:
    baseline = [
        {"model_id": f"baseline/model-{index}", "category": "text", "notes": f"kept-{index}"}
        for index in range(120)
    ]
    additions = [{"model_id": f"added/model-{index}", "category": "vision"} for index in range(130)]

    cases = combine_cases(baseline, additions)

    assert cases[:120] == baseline
    assert len(cases) == 250
    assert len({case["model_id"] for case in cases}) == 250

    with pytest.raises(SystemExit, match="exact-count/uniqueness"):
        combine_cases(baseline, additions[:-1])
