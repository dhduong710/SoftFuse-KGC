from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

COMPARE_PATH = ROOT / "outputs/fuzzy_retrieval/retrieval_valid_compare.json"
CASE_PATH = ROOT / "outputs/fuzzy_retrieval/retrieval_case_samples_main.json"

VARIANT_PATHS = {
    "soft_support_fuzzy_retrieval_v1": ROOT / "dataset/setting_a/fuzzy_retrieval/valid_fuzzy_retrieval_v1.json",
    "soft_support_fuzzy_retrieval_tight": ROOT / "dataset/setting_a/fuzzy_retrieval/valid_soft_support_fuzzy_retrieval_tight.json",
    "soft_support_fuzzy_retrieval_directplus": ROOT / "dataset/setting_a/fuzzy_retrieval/valid_soft_support_fuzzy_retrieval_directplus.json",
}

OUT_MAIN_JSON = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_fuzzy_retrieval_main.json"
OUT_MAIN_MANIFEST = ROOT / "dataset/setting_a/fuzzy_retrieval/retrieval_main_manifest.json"
OUT_DECISION = ROOT / "outputs/fuzzy_retrieval/retrieval_main_decision.json"
OUT_REPORT = ROOT / "outputs/fuzzy_retrieval/reports/fuzzy_retrieval_main_selection.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def count_bucket(bucket_obj: Dict[str, Any], bucket_name: str) -> int:
    return int(bucket_obj["bucket_counts"].get(bucket_name, 0))


def build_variant_record(
    name: str,
    metrics: Dict[str, Any],
    soft_metrics: Dict[str, Any],
    case_obj: Dict[str, Any],
) -> Dict[str, Any]:
    anchor_caution_count = count_bucket(case_obj, "tight_anchor_caution_same_rank") if name == "soft_support_fuzzy_retrieval_tight" else 0
    redundancy_count = count_bucket(case_obj, "directplus_redundant_vs_v1") if name == "soft_support_fuzzy_retrieval_directplus" else 0

    eligible = (
        metrics["worsened_vs_soft_support"] == 0
        and metrics["candidate_coverage_preserved_rate"] >= 0.999
        and metrics["mrr_like"] >= soft_metrics["mrr_like"] - 0.001
    )

    return {
        "variant_name": name,
        "eligible": eligible,
        "metrics": {
            "mrr_like": metrics["mrr_like"],
            "hits1_like": metrics["hits1_like"],
            "hits3_like": metrics["hits3_like"],
            "hits10_like": metrics["hits10_like"],
            "avg_gold_rank": metrics["avg_gold_rank"],
            "avg_subgraph_size": metrics["avg_subgraph_size"],
            "avg_triple_score": metrics["avg_triple_score"],
            "direct_shortcut_path_rate": metrics["direct_shortcut_path_rate"],
            "contradiction_path_rate": metrics["contradiction_path_rate"],
            "candidate_coverage_preserved_rate": metrics["candidate_coverage_preserved_rate"],
            "avg_query_touch_count": metrics["avg_query_touch_count"],
            "avg_non_direct_query_touch_count": metrics["avg_non_direct_query_touch_count"],
            "worsened_vs_soft_support": metrics["worsened_vs_soft_support"],
            "improved_vs_soft_support": metrics["improved_vs_soft_support"],
            "selected_subgraph_identical_to_v1_rate": metrics["selected_subgraph_identical_to_v1_rate"],
        },
        "selection_features": {
            "shortcut_gain_vs_soft_support": round(
                soft_metrics["direct_shortcut_path_rate"] - metrics["direct_shortcut_path_rate"], 6
            ),
            "subgraph_size_gain_vs_soft_support": round(
                soft_metrics["avg_subgraph_size"] - metrics["avg_subgraph_size"], 6
            ),
            "mrr_delta_vs_soft_support": round(
                metrics["mrr_like"] - soft_metrics["mrr_like"], 6
            ),
            "redundancy_count": redundancy_count,
            "anchor_caution_count": anchor_caution_count,
        },
    }


def choose_main_variant(candidates: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    eligible = [c for c in candidates if c["eligible"]]
    if not eligible:
        raise RuntimeError("No eligible retrieval variant found.")

    ranked = sorted(
        eligible,
        key=lambda c: (
            c["metrics"]["direct_shortcut_path_rate"],            # lower is better
            c["metrics"]["avg_subgraph_size"],                    # lower is better
            c["metrics"]["selected_subgraph_identical_to_v1_rate"],  # lower redundancy better
            c["selection_features"]["anchor_caution_count"],      # lower caution better
            -c["metrics"]["mrr_like"],                            # higher is better
        )
    )
    return ranked[0]["variant_name"], ranked


def main() -> None:
    compare = load_json(COMPARE_PATH)
    case_obj = load_json(CASE_PATH)

    rows = compare["rows"]
    soft_metrics = rows["soft_support_raw"]

    candidate_names = [
        "soft_support_fuzzy_retrieval_v1",
        "soft_support_fuzzy_retrieval_tight",
        "soft_support_fuzzy_retrieval_directplus",
    ]

    candidate_records = [
        build_variant_record(name, rows[name], soft_metrics, case_obj)
        for name in candidate_names
    ]

    selected_variant, ranked_candidates = choose_main_variant(candidate_records)

    selected_rows = load_json(VARIANT_PATHS[selected_variant])
    assert isinstance(selected_rows, list) and len(selected_rows) == 500, "Selected main variant must have 500 rows."

    # Canonicalize output row name in copied artifact
    main_rows = []
    for row in selected_rows:
        copied = dict(row)
        copied["variant_name"] = "soft_support_fuzzy_retrieval_main"
        copied["selected_source_variant"] = selected_variant
        main_rows.append(copied)

    main_manifest = {
        "stage": "fuzzy_retrieval_main_selection",
        "canonical_row_name": "soft_support_fuzzy_retrieval_main",
        "selected_source_variant": selected_variant,
        "num_rows": len(main_rows),
        "selection_rule": [
            "eligible if no ranking-like collapse vs soft_support_raw",
            "eligible if candidate coverage preserved",
            "prefer lower direct_shortcut_path_rate",
            "then prefer smaller avg_subgraph_size",
            "then prefer lower redundancy vs v1",
            "then prefer lower anchor caution count",
        ],
        "ranked_candidates": ranked_candidates,
    }

    decision = {
        "stage": "fuzzy_retrieval_main_selection",
        "decision": "SELECT_MAIN_RETRIEVAL_ROW",
        "main_input_row": "soft_support_raw",
        "selected_main_row": "soft_support_fuzzy_retrieval_main",
        "selected_source_variant": selected_variant,
        "rejected_variants": [x for x in candidate_names if x != selected_variant],
        "main_reason": [
            "selected row preserves ranking-like proxy relative to soft_support_raw",
            "selected row reduces direct shortcut evidence more strongly than v1",
            "selected row keeps candidate coverage preserved",
            "selected row offers a cleaner trade-off than the remaining variants",
        ],
        "selected_metrics": rows[selected_variant],
        "delta_vs_v1": compare["delta_vs_v1"].get(selected_variant, {}),
        "delta_vs_soft_support": compare["delta_vs_soft_support"][selected_variant],
        "case_review_support": {
            "tight_same_rank_cleaner_than_v1": count_bucket(case_obj, "tight_same_rank_cleaner_than_v1"),
            "tight_preserved_backbone_gain": count_bucket(case_obj, "tight_preserved_backbone_gain"),
            "tight_anchor_caution_same_rank": count_bucket(case_obj, "tight_anchor_caution_same_rank"),
            "directplus_redundant_vs_v1": count_bucket(case_obj, "directplus_redundant_vs_v1"),
            "raw_bottleneck_failure": count_bucket(case_obj, "raw_bottleneck_failure"),
        },
    }

    save_json(OUT_MAIN_JSON, main_rows)
    save_json(OUT_MAIN_MANIFEST, main_manifest)
    save_json(OUT_DECISION, decision)

    report_lines = [
        "# Fuzzy Retrieval Main Row Selection",
        "",
        "## Decision",
        f"- selected_main_row: `soft_support_fuzzy_retrieval_main`",
        f"- selected_source_variant: `{selected_variant}`",
        "",
        "## Why selected",
        "- ranking-like proxy is preserved relative to soft_support_raw",
        "- direct shortcut evidence is reduced more than in v1",
        "- candidate coverage stays preserved",
        "- this variant provides the cleanest overall trade-off on valid",
        "",
        "## Ranked candidates",
        json.dumps(ranked_candidates, ensure_ascii=False, indent=2),
        "",
        "## Selected metrics",
        json.dumps(rows[selected_variant], ensure_ascii=False, indent=2),
        "",
        "## Delta vs soft_support_raw",
        json.dumps(compare["delta_vs_soft_support"][selected_variant], ensure_ascii=False, indent=2),
        "",
        "## Case review support",
        json.dumps(decision["case_review_support"], ensure_ascii=False, indent=2),
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 80)
    print("FUZZY RETRIEVAL MAIN ROW SELECTED")
    print("main_json :", OUT_MAIN_JSON)
    print("manifest  :", OUT_MAIN_MANIFEST)
    print("decision  :", OUT_DECISION)
    print("report    :", OUT_REPORT)
    print("=" * 80)
    print(json.dumps({
        "selected_source_variant": selected_variant,
        "selected_main_row": "soft_support_fuzzy_retrieval_main",
        "selected_metrics": rows[selected_variant],
        "case_review_support": decision["case_review_support"],
    }, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
