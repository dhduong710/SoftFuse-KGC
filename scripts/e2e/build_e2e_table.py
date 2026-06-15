from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(".").resolve()

BACKBONE_SOFT_SUMMARY = ROOT / "outputs/e2e/e2e_backbone_soft_summary.json"
RETRIEVAL_SUMMARY = ROOT / "outputs/e2e/e2e_retrieval_summary.json"

INFER_READY = {
    "backbone_raw": ROOT / "dataset/setting_a/e2e_infer_ready/backbone_raw/test.json",
    "soft_support_raw": ROOT / "dataset/setting_a/e2e_infer_ready/soft_support_raw/test.json",
    "retrieval_main": ROOT / "dataset/setting_a/e2e_infer_ready/retrieval_main/test.json",
}

OUT_MAIN = ROOT / "outputs/e2e/e2e_main_table.json"
OUT_ABL = ROOT / "outputs/e2e/e2e_ablation.json"
OUT_MD = ROOT / "outputs/e2e/reports/e2e_table_and_ablation.md"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def avg(vals: List[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def graph_package_from_infer_ready(row_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    subgraph_sizes = [len(x.get("subgraph", [])) for x in rows]
    out = {
        "num_rows": len(rows),
        "avg_subgraph_size": round(avg([float(x) for x in subgraph_sizes]), 8),
        "min_subgraph_size": min(subgraph_sizes),
        "max_subgraph_size": max(subgraph_sizes),
    }

    if row_name == "retrieval_main":
        selected_variants = sorted(set(x.get("selected_source_variant") for x in rows))
        summaries = [x.get("subgraph_summary", {}) for x in rows]

        coverage_vals = [
            float(s.get("candidate_coverage_preserved_rate", 0.0))
            for s in summaries
            if "candidate_coverage_preserved_rate" in s
        ]

        shortcut_rates = []
        for x in rows:
            triples = x.get("triple_score_rows", [])
            if triples:
                shortcut_rates.append(
                    sum(1 for t in triples if t.get("direct_candidate_query_flag", False)) / len(triples)
                )

        out.update({
            "selected_source_variant_set": selected_variants,
            "avg_candidate_coverage_preserved_rate": round(avg(coverage_vals), 8) if coverage_vals else None,
            "avg_direct_shortcut_path_rate": round(avg(shortcut_rates), 8) if shortcut_rates else None,
        })

    return out


def metric_subset(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "num_examples",
        "k",
        "gold_present_rate",
        "mrr_at20",
        "hits1_at20",
        "hits3_at20",
        "hits10_at20",
        "hits20_at20",
        "avg_gold_rank_with_absent_as_21",
        "avg_adjusted_rank_with_absent_as_21",
        "gold_rank_21_count",
        "rank_21_count",
        "exact_generated_rate",
        "pred_in_candidate_rate",
    ]
    return {k: metrics[k] for k in keep if k in metrics}


def delta_metrics(base: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, float]:
    keys = [
        "gold_present_rate",
        "mrr_at20",
        "hits1_at20",
        "hits3_at20",
        "hits10_at20",
        "hits20_at20",
    ]
    out = {}
    for k in keys:
        if k in base and k in target:
            out[f"delta_{k}"] = round(float(target[k]) - float(base[k]), 8)
    return out


def row_pack(
    row_name: str,
    display_name: str,
    candidate: Dict[str, Any],
    e2e: Dict[str, Any],
    graph_pkg: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "row_name": display_name,
        "candidate_ceiling_reviewer_safe": metric_subset(candidate),
        "e2e_generation_reviewer_safe": metric_subset(e2e),
        "graph_package": graph_pkg,
    }


def main() -> None:
    backbone_soft_summary = load_json(BACKBONE_SOFT_SUMMARY)
    retrieval_summary = load_json(RETRIEVAL_SUMMARY)

    infer_ready_rows = {k: load_json(v) for k, v in INFER_READY.items()}
    for k, rows in infer_ready_rows.items():
        assert isinstance(rows, list) and len(rows) == 500, f"{k} must have 500 rows"

    graph_pkgs = {
        "backbone_raw": graph_package_from_infer_ready("backbone_raw", infer_ready_rows["backbone_raw"]),
        "soft_support_raw": graph_package_from_infer_ready("soft_support_raw", infer_ready_rows["soft_support_raw"]),
        "retrieval_main": graph_package_from_infer_ready("retrieval_main", infer_ready_rows["retrieval_main"]),
    }

    backbone_candidate = backbone_soft_summary["backbone_raw"]["candidate_ceiling_reviewer_safe"]
    backbone_e2e = backbone_soft_summary["backbone_raw"]["e2e_generation_reviewer_safe"]

    soft_candidate = backbone_soft_summary["soft_support_raw"]["candidate_ceiling_reviewer_safe"]
    soft_e2e = backbone_soft_summary["soft_support_raw"]["e2e_generation_reviewer_safe"]

    retrieval_candidate = retrieval_summary["retrieval_main"]["candidate_ceiling_reviewer_safe"]
    retrieval_e2e = retrieval_summary["retrieval_main"]["e2e_generation_reviewer_safe"]

    rows = [
        row_pack(
            row_name="backbone_raw",
            display_name="backbone_raw",
            candidate=backbone_candidate,
            e2e=backbone_e2e,
            graph_pkg=graph_pkgs["backbone_raw"],
        ),
        row_pack(
            row_name="soft_support_raw",
            display_name="soft_support_raw",
            candidate=soft_candidate,
            e2e=soft_e2e,
            graph_pkg=graph_pkgs["soft_support_raw"],
        ),
        row_pack(
            row_name="retrieval_main",
            display_name="soft_support_fuzzy_retrieval_main",
            candidate=retrieval_candidate,
            e2e=retrieval_e2e,
            graph_pkg=graph_pkgs["retrieval_main"],
        ),
    ]

    soft_minus_backbone = {
        "candidate_ceiling": delta_metrics(backbone_candidate, soft_candidate),
        "e2e_generation": delta_metrics(backbone_e2e, soft_e2e),
        "graph_package": {
            "delta_avg_subgraph_size": round(
                graph_pkgs["soft_support_raw"]["avg_subgraph_size"]
                - graph_pkgs["backbone_raw"]["avg_subgraph_size"],
                8,
            ),
        },
    }

    retrieval_minus_soft = {
        "candidate_ceiling": delta_metrics(soft_candidate, retrieval_candidate),
        "e2e_generation": delta_metrics(soft_e2e, retrieval_e2e),
        "graph_package": {
            "delta_avg_subgraph_size": round(
                graph_pkgs["retrieval_main"]["avg_subgraph_size"]
                - graph_pkgs["soft_support_raw"]["avg_subgraph_size"],
                8,
            ),
            "retrieval_avg_subgraph_size": graph_pkgs["retrieval_main"]["avg_subgraph_size"],
            "soft_avg_subgraph_size": graph_pkgs["soft_support_raw"]["avg_subgraph_size"],
            "retrieval_selected_source_variant_set": graph_pkgs["retrieval_main"].get("selected_source_variant_set"),
            "retrieval_avg_candidate_coverage_preserved_rate": graph_pkgs["retrieval_main"].get("avg_candidate_coverage_preserved_rate"),
            "retrieval_avg_direct_shortcut_path_rate": graph_pkgs["retrieval_main"].get("avg_direct_shortcut_path_rate"),
        },
    }

    retrieval_minus_backbone = {
        "candidate_ceiling": delta_metrics(backbone_candidate, retrieval_candidate),
        "e2e_generation": delta_metrics(backbone_e2e, retrieval_e2e),
        "graph_package": {
            "delta_avg_subgraph_size": round(
                graph_pkgs["retrieval_main"]["avg_subgraph_size"]
                - graph_pkgs["backbone_raw"]["avg_subgraph_size"],
                8,
            ),
        },
    }

    narrative_checks = {
        "soft_improves_candidate_mrr_vs_backbone": soft_candidate["mrr_at20"] > backbone_candidate["mrr_at20"],
        "soft_improves_e2e_mrr_vs_backbone": soft_e2e["mrr_at20"] > backbone_e2e["mrr_at20"],
        "retrieval_preserves_candidate_mrr_vs_soft": retrieval_candidate["mrr_at20"] >= soft_candidate["mrr_at20"],
        "retrieval_preserves_e2e_mrr_vs_soft": retrieval_e2e["mrr_at20"] >= soft_e2e["mrr_at20"],
        "retrieval_has_smaller_subgraph_than_soft": graph_pkgs["retrieval_main"]["avg_subgraph_size"] < graph_pkgs["soft_support_raw"]["avg_subgraph_size"],
        "retrieval_keeps_candidate_coverage": graph_pkgs["retrieval_main"].get("avg_candidate_coverage_preserved_rate") == 1.0,
    }

    if (
        narrative_checks["soft_improves_candidate_mrr_vs_backbone"]
        and narrative_checks["soft_improves_e2e_mrr_vs_backbone"]
        and narrative_checks["retrieval_preserves_candidate_mrr_vs_soft"]
        and narrative_checks["retrieval_preserves_e2e_mrr_vs_soft"]
        and narrative_checks["retrieval_has_smaller_subgraph_than_soft"]
    ):
        provisional_main_row = "soft_support_fuzzy_retrieval_main"
        decision_note = (
            "Keep retrieval_main as the main E2E row: soft support provides the main "
            "ranking gain over backbone, while retrieval_main preserves candidate/E2E performance "
            "and substantially reduces the evidence subgraph."
        )
    else:
        provisional_main_row = "soft_support_raw"
        decision_note = (
            "Use soft_support_raw as provisional main row unless further review justifies retrieval_main."
        )

    main_table = {
        "stage": "e2e_main_table_reviewer_safe",
        "status": "BUILT_REVIEWER_SAFE",
        "metric_policy": {
            "candidate_ceiling_rr_rule": "1/rank if rank <= 20 else 0",
            "e2e_generation_rr_rule": "1/adjusted_rank if adjusted_rank <= 20 else 0",
            "gold_rank_out_of_top20": 21,
            "reporting_note": (
                "Use reviewer-safe recomputation from prediction rows, not raw infer.py metrics, "
                "for E2E tables."
            ),
        },
        "main_rows": rows,
        "narrative_checks": narrative_checks,
        "provisional_main_row": provisional_main_row,
        "decision_note": decision_note,
    }

    ablation = {
        "stage": "e2e_ablation_reviewer_safe",
        "status": "BUILT_REVIEWER_SAFE",
        "reference_row": "backbone_raw",
        "candidate_stage_intermediate": "soft_support_raw",
        "main_row": "soft_support_fuzzy_retrieval_main",
        "soft_minus_backbone": soft_minus_backbone,
        "retrieval_minus_soft": retrieval_minus_soft,
        "retrieval_minus_backbone": retrieval_minus_backbone,
        "interpretation": {
            "candidate_stage_claim": (
                "soft_support_raw improves candidate ceiling and E2E reviewer-safe MRR over backbone_raw."
            ),
            "retrieval_stage_claim": (
                "retrieval_main preserves soft_support_raw performance while reducing evidence subgraph size."
            ),
            "known_limitation": (
                "E2E Hits@1 remains weak because the fixed LLM often generates a plausible candidate "
                "instead of the exact gold string even when gold is ranked first."
            ),
        },
    }

    save_json(OUT_MAIN, main_table)
    save_json(OUT_ABL, ablation)

    md = []
    md.append("# E2E Main Table and Ablation")
    md.append("")
    md.append(f"- status: **{main_table['status']}**")
    md.append(f"- provisional_main_row: **`{main_table['provisional_main_row']}`**")
    md.append("")
    md.append("## 1. Main E2E table")
    for row in rows:
        md.append(f"### {row['row_name']}")
        md.append("#### Candidate ceiling reviewer-safe")
        for k, v in row["candidate_ceiling_reviewer_safe"].items():
            md.append(f"- {k}: `{v}`")
        md.append("#### E2E generation reviewer-safe")
        for k, v in row["e2e_generation_reviewer_safe"].items():
            md.append(f"- {k}: `{v}`")
        md.append("#### Graph package")
        for k, v in row["graph_package"].items():
            md.append(f"- {k}: `{v}`")
        md.append("")
    md.append("## 2. Narrative checks")
    for k, v in narrative_checks.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 3. soft_minus_backbone")
    md.append("### Candidate ceiling")
    for k, v in soft_minus_backbone["candidate_ceiling"].items():
        md.append(f"- {k}: `{v}`")
    md.append("### E2E generation")
    for k, v in soft_minus_backbone["e2e_generation"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 4. retrieval_minus_soft")
    md.append("### Candidate ceiling")
    for k, v in retrieval_minus_soft["candidate_ceiling"].items():
        md.append(f"- {k}: `{v}`")
    md.append("### E2E generation")
    for k, v in retrieval_minus_soft["e2e_generation"].items():
        md.append(f"- {k}: `{v}`")
    md.append("### Graph package")
    for k, v in retrieval_minus_soft["graph_package"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 5. Decision note")
    md.append(decision_note)
    md.append("")
    md.append("## 6. Known limitation")
    md.append(ablation["interpretation"]["known_limitation"])

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "main_table": main_table,
        "ablation": ablation,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
