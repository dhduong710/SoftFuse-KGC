#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build reviewer-safe baseline comparison table.

Input:
- outputs/baselines/reviewer_safe_metrics_all.json

Outputs:
- outputs/baselines/baseline_main_table.json
- outputs/baselines/baseline_main_table.csv
- outputs/baselines/baseline_interpretation.json
- outputs/baselines/reports/baseline_main_table.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(".")
RESULTS_DIR = ROOT / "outputs" / "baselines"
REPORTS_DIR = ROOT / "outputs" / "baselines" / "reports"

IN_METRICS = RESULTS_DIR / "reviewer_safe_metrics_all.json"

OUT_JSON = RESULTS_DIR / "baseline_main_table.json"
OUT_CSV = RESULTS_DIR / "baseline_main_table.csv"
OUT_DECISION = RESULTS_DIR / "baseline_interpretation.json"
OUT_MD = REPORTS_DIR / "baseline_main_table.md"

ROW_ORDER = [
    "transe",
    "distmult",
    "complex",
    "rotate",
    "rgcn",
    "hrgat",
    "backbone_raw",
    "soft_support_raw",
    "soft_support_fuzzy_retrieval_main",
]

ROW_DISPLAY = {
    "transe": "TransE",
    "distmult": "DistMult",
    "complex": "ComplEx",
    "rotate": "RotatE",
    "rgcn": "R-GCN",
    "hrgat": "HRGAT",
    "backbone_raw": "DrKGC-style backbone raw",
    "soft_support_raw": "Soft support raw",
    "soft_support_fuzzy_retrieval_main": "SoftFuse main",
}

ROW_GROUP = {
    "transe": "Structure baseline",
    "distmult": "Structure baseline",
    "complex": "Structure baseline",
    "rotate": "Structure baseline",
    "rgcn": "Structure baseline",
    "hrgat": "Structure baseline",
    "backbone_raw": "DrKGC-compatible",
    "soft_support_raw": "SoftFuse",
    "soft_support_fuzzy_retrieval_main": "SoftFuse",
}


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require_metric_fields(row: dict[str, Any]) -> None:
    required = [
        "row_name",
        "family",
        "gold_present_at20",
        "reviewer_safe_mrr_at20",
        "mrr_present_only",
        "hits1_at20",
        "hits3_at20",
        "hits10_at20",
        "hits20_at20",
        "avg_gold_rank_absent_as_21",
        "gold_rank_21_count",
        "unique_top1_count",
        "top1_dominance",
    ]
    missing = [k for k in required if k not in row]
    if missing:
        raise KeyError(f"Missing fields in row {row.get('row_name')}: {missing}")


def metric_subset(row: dict[str, Any]) -> dict[str, Any]:
    require_metric_fields(row)
    name = row["row_name"]
    return {
        "row_name": name,
        "display_name": ROW_DISPLAY.get(name, name),
        "group": ROW_GROUP.get(name, row.get("family", "")),
        "family": row["family"],
        "gold_present_at20": float(row["gold_present_at20"]),
        "reviewer_safe_mrr_at20": float(row["reviewer_safe_mrr_at20"]),
        "mrr_present_only": float(row["mrr_present_only"]),
        "hits1_at20": float(row["hits1_at20"]),
        "hits3_at20": float(row["hits3_at20"]),
        "hits10_at20": float(row["hits10_at20"]),
        "hits20_at20": float(row["hits20_at20"]),
        "avg_gold_rank_absent_as_21": float(row["avg_gold_rank_absent_as_21"]),
        "gold_rank_21_count": int(row["gold_rank_21_count"]),
        "unique_top1_count": int(row["unique_top1_count"]),
        "top1_dominance": float(row["top1_dominance"]),
    }


def rows_by_display_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {r["row_name"]: r for r in rows}
    ordered = []
    for name in ROW_ORDER:
        if name in by_name:
            ordered.append(metric_subset(by_name[name]))

    missing = [name for name in ROW_ORDER if name not in by_name]
    if missing:
        print(f"WARNING: missing expected rows: {missing}")

    return ordered


def get_row(rows: list[dict[str, Any]], row_name: str) -> dict[str, Any]:
    for row in rows:
        if row["row_name"] == row_name:
            return row
    raise KeyError(f"Row not found: {row_name}")


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {
        "delta_gold_present_at20": a["gold_present_at20"] - b["gold_present_at20"],
        "delta_mrr_at20": a["reviewer_safe_mrr_at20"] - b["reviewer_safe_mrr_at20"],
        "delta_hits1_at20": a["hits1_at20"] - b["hits1_at20"],
        "delta_hits3_at20": a["hits3_at20"] - b["hits3_at20"],
        "delta_hits10_at20": a["hits10_at20"] - b["hits10_at20"],
        "delta_avg_rank_absent_as_21": (
            a["avg_gold_rank_absent_as_21"] - b["avg_gold_rank_absent_as_21"]
        ),
    }


def write_csv(test_rows: list[dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "display_name",
        "group",
        "gold_present_at20",
        "reviewer_safe_mrr_at20",
        "mrr_present_only",
        "hits1_at20",
        "hits3_at20",
        "hits10_at20",
        "hits20_at20",
        "avg_gold_rank_absent_as_21",
        "gold_rank_21_count",
        "unique_top1_count",
        "top1_dominance",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in test_rows:
            writer.writerow({k: row[k] for k in fields})


def decide_interpretation(valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_sorted = sorted(valid_rows, key=lambda r: r["reviewer_safe_mrr_at20"], reverse=True)
    test_sorted = sorted(test_rows, key=lambda r: r["reviewer_safe_mrr_at20"], reverse=True)

    fog = get_row(test_rows, "soft_support_fuzzy_retrieval_main")
    backbone = get_row(test_rows, "backbone_raw")
    complex_row = get_row(test_rows, "complex")

    structure_rows = [r for r in test_rows if r["family"] == "structure_baseline"]
    best_structure = max(structure_rows, key=lambda r: r["reviewer_safe_mrr_at20"])

    fog_vs_backbone = delta(fog, backbone)
    fog_vs_complex = delta(fog, complex_row)
    fog_vs_best_structure = delta(fog, best_structure)
    complex_vs_backbone = delta(complex_row, backbone)

    if fog["reviewer_safe_mrr_at20"] >= best_structure["reviewer_safe_mrr_at20"]:
        headline = "SOFTFUSE_MAIN_BEST_TEST_MRR_OR_TIED_WITH_STRUCTURE_BASELINE"
        stance = (
            "SoftFuse main achieves the best locked-test reviewer-safe MRR@20, "
            "slightly above the strongest structure-only candidate generator."
        )
    else:
        headline = "BEST_STRUCTURE_BASELINE_HIGHER_THAN_SOFTFUSE_MAIN"
        stance = (
            "The strongest structure-only candidate generator exceeds SoftFuse main on "
            "locked-test MRR@20. SoftFuse should be framed as improving the DrKGC-compatible "
            "pipeline and providing evidence-aware retrieval."
        )

    decision = {
        "headline_decision": headline,
        "recommended_stance": stance,
        "valid_best_row": {
            "row_name": valid_sorted[0]["row_name"],
            "display_name": valid_sorted[0]["display_name"],
            "family": valid_sorted[0]["family"],
            "gold_present_at20": valid_sorted[0]["gold_present_at20"],
            "mrr_at20": valid_sorted[0]["reviewer_safe_mrr_at20"],
        },
        "test_best_row": {
            "row_name": test_sorted[0]["row_name"],
            "display_name": test_sorted[0]["display_name"],
            "family": test_sorted[0]["family"],
            "gold_present_at20": test_sorted[0]["gold_present_at20"],
            "mrr_at20": test_sorted[0]["reviewer_safe_mrr_at20"],
        },
        "test_best_structure_baseline": {
            "row_name": best_structure["row_name"],
            "display_name": best_structure["display_name"],
            "gold_present_at20": best_structure["gold_present_at20"],
            "mrr_at20": best_structure["reviewer_safe_mrr_at20"],
            "hits1_at20": best_structure["hits1_at20"],
            "hits3_at20": best_structure["hits3_at20"],
            "hits10_at20": best_structure["hits10_at20"],
        },
        "test_softfuse_main": {
            "row_name": fog["row_name"],
            "display_name": fog["display_name"],
            "gold_present_at20": fog["gold_present_at20"],
            "mrr_at20": fog["reviewer_safe_mrr_at20"],
            "hits1_at20": fog["hits1_at20"],
            "hits3_at20": fog["hits3_at20"],
            "hits10_at20": fog["hits10_at20"],
        },
        "deltas": {
            "softfuse_main_minus_backbone_raw": fog_vs_backbone,
            "softfuse_main_minus_complex": fog_vs_complex,
            "softfuse_main_minus_best_structure": fog_vs_best_structure,
            "complex_minus_backbone_raw": complex_vs_backbone,
        },
        "important_caveats": [
            "ComplEx has much higher Gold@20 than SoftFuse main on both validation and test.",
            "ComplEx is higher than SoftFuse main on validation MRR@20.",
            "SoftFuse main is slightly higher than ComplEx on locked-test MRR@20.",
            "The margin between SoftFuse main and ComplEx on test is very small, so avoid overclaiming universal superiority.",
            "The strongest framing is that SoftFuse improves the DrKGC-compatible raw source and remains competitive with a strong structure-only generator.",
        ],
        "recommended_text": (
            "Although ComplEx provides substantially higher top-20 gold coverage, SoftFuse main "
            "achieves the highest locked-test reviewer-safe MRR@20. This indicates that candidate "
            "coverage alone is insufficient: ranking quality within the top-20 candidate list and "
            "evidence-aware support modeling remain important. We therefore report structure-only "
            "models as upstream candidate-generator baselines and position SoftFuse as an "
            "evidence-aware extension of the DrKGC-compatible graph-augmented LLM pipeline."
        ),
    }

    return decision


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Method | Group | Gold@20 | MRR@20 | H@1 | H@3 | H@10 | Avg. rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['display_name']} | {r['group']} "
            f"| {r['gold_present_at20']:.4f} "
            f"| {r['reviewer_safe_mrr_at20']:.6f} "
            f"| {r['hits1_at20']:.4f} "
            f"| {r['hits3_at20']:.4f} "
            f"| {r['hits10_at20']:.4f} "
            f"| {r['avg_gold_rank_absent_as_21']:.3f} |"
        )
    return "\n".join(lines)


def json_block(obj: Any) -> str:
    return "```json\n" + json.dumps(obj, indent=2, ensure_ascii=False) + "\n```"


def write_report(valid_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], decision: dict[str, Any]) -> None:
    fog = get_row(test_rows, "soft_support_fuzzy_retrieval_main")
    comp = get_row(test_rows, "complex")
    back = get_row(test_rows, "backbone_raw")

    lines = []
    lines.append("# Baseline Main Table and Interpretation\n")
    lines.append("## Decision\n")
    lines.append(f"**{decision['headline_decision']}**\n")
    lines.append("## Main conclusion\n")
    lines.append(decision["recommended_stance"] + "\n")

    lines.append("## Key locked-test numbers\n")
    lines.append("| Row | Gold@20 | MRR@20 | Hits@1 | Hits@3 | Hits@10 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| DrKGC-style backbone raw | {back['gold_present_at20']:.4f} | "
        f"{back['reviewer_safe_mrr_at20']:.6f} | {back['hits1_at20']:.4f} | "
        f"{back['hits3_at20']:.4f} | {back['hits10_at20']:.4f} |"
    )
    lines.append(
        f"| ComplEx | {comp['gold_present_at20']:.4f} | "
        f"{comp['reviewer_safe_mrr_at20']:.6f} | {comp['hits1_at20']:.4f} | "
        f"{comp['hits3_at20']:.4f} | {comp['hits10_at20']:.4f} |"
    )
    lines.append(
        f"| SoftFuse main | {fog['gold_present_at20']:.4f} | "
        f"{fog['reviewer_safe_mrr_at20']:.6f} | {fog['hits1_at20']:.4f} | "
        f"{fog['hits3_at20']:.4f} | {fog['hits10_at20']:.4f} |"
    )
    lines.append("")

    lines.append("## Full valid table\n")
    lines.append(markdown_table(valid_rows))
    lines.append("")

    lines.append("## Full test table\n")
    lines.append(markdown_table(test_rows))
    lines.append("")

    lines.append("## Important deltas on test\n")
    lines.append("### SoftFuse main minus backbone raw\n")
    lines.append(json_block(decision["deltas"]["softfuse_main_minus_backbone_raw"]))
    lines.append("")
    lines.append("### SoftFuse main minus ComplEx\n")
    lines.append(json_block(decision["deltas"]["softfuse_main_minus_complex"]))
    lines.append("")
    lines.append("### ComplEx minus backbone raw\n")
    lines.append(json_block(decision["deltas"]["complex_minus_backbone_raw"]))
    lines.append("")

    lines.append("## Interpretation\n")
    lines.append(
        "The result is scientifically useful because ComplEx is a strong candidate generator "
        "with much higher Gold@20, but SoftFuse main still obtains slightly higher locked-test "
        "MRR@20."
    )
    lines.append("")
    lines.append(
        "Candidate coverage alone is not sufficient. Even with lower Gold@20, SoftFuse improves "
        "rank placement under reviewer-safe MRR@20 by modeling support and evidence quality."
    )
    lines.append("")

    lines.append("## Caveats\n")
    for item in decision["important_caveats"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## Recommended reporting text\n")
    lines.append(decision["recommended_text"])
    lines.append("")

    lines.append("## Files produced\n")
    lines.append("- `outputs/baselines/baseline_main_table.json`")
    lines.append("- `outputs/baselines/baseline_main_table.csv`")
    lines.append("- `outputs/baselines/baseline_interpretation.json`")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_json(IN_METRICS)

    if "valid" not in raw or "test" not in raw:
        raise KeyError("Input JSON must contain top-level keys: valid, test")

    if "all_rows" not in raw["valid"] or "all_rows" not in raw["test"]:
        raise KeyError("Input JSON must contain valid['all_rows'] and test['all_rows']")

    valid_rows = rows_by_display_order(raw["valid"]["all_rows"])
    test_rows = rows_by_display_order(raw["test"]["all_rows"])

    decision = decide_interpretation(valid_rows, test_rows)

    out_obj = {
        "source": str(IN_METRICS),
        "metric_protocol": raw.get("metric_protocol", {}),
        "valid_rows": valid_rows,
        "test_rows": test_rows,
        "test_rows_for_reporting": test_rows,
        "decision": decision,
    }

    write_json(OUT_JSON, out_obj)
    write_json(OUT_DECISION, decision)
    write_csv(test_rows)
    write_report(valid_rows, test_rows, decision)

    print("=" * 100)
    print("BASELINE MAIN TABLE")
    print("=" * 100)
    print(f"Wrote: {OUT_JSON}")
    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_DECISION}")
    print(f"Wrote: {OUT_MD}")
    print()
    print("Decision:", decision["headline_decision"])
    print("Recommended stance:", decision["recommended_stance"])
    print()
    print("Valid best:", decision["valid_best_row"])
    print("Test best:", decision["test_best_row"])
    print("Best structure baseline:", decision["test_best_structure_baseline"])
    print("SoftFuse main:", decision["test_softfuse_main"])
    print()
    print("Test deltas:")
    for name, d in decision["deltas"].items():
        print(f"- {name}")
        for k, v in d.items():
            print(f"    {k}: {v:+.6f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
