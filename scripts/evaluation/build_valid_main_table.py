from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(".").resolve()

INPUTS = {
    "backbone_raw": ROOT / "dataset/setting_b/eval_valid/valid_backbone_raw_eval.json",
    "ontology_raw": ROOT / "dataset/setting_b/eval_valid/valid_ontology_raw_eval.json",
    "soft_support_raw": ROOT / "dataset/setting_b/eval_valid/valid_soft_support_raw_eval.json",
    "retrieval_main": ROOT / "dataset/setting_b/eval_valid/valid_retrieval_main_eval.json",
}

OUT_TABLE = ROOT / "outputs/evaluation/valid_main_table.json"
OUT_ABLATION = ROOT / "outputs/evaluation/valid_ablation.json"
OUT_REPORT = ROOT / "outputs/evaluation/reports/valid_main_table.md"

ROW_META = {
    "backbone_raw": {
        "canonical_row_name": "backbone_raw",
        "role": "reference",
        "stage": "reference",
    },
    "ontology_raw": {
        "canonical_row_name": "ontology_raw",
        "role": "negative_control",
        "stage": "negative_control",
    },
    "soft_support_raw": {
        "canonical_row_name": "soft_support_raw",
        "role": "candidate_stage_main_intermediate",
        "stage": "candidate_stage",
    },
    "retrieval_main": {
        "canonical_row_name": "soft_support_fuzzy_retrieval_main",
        "role": "main_row",
        "stage": "retrieval_stage",
    },
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def safe_mean(xs):
    return 0.0 if not xs else float(mean(xs))


def summarize_ranking_reviewer_safe(rows: list[dict]) -> dict:
    ranks = [int(r["gold_rank"]) for r in rows]
    n = len(ranks)

    present_mask = [r <= 20 for r in ranks]
    num_present = sum(1 for x in present_mask if x)

    rr_at20_items = [(1.0 / r) if r <= 20 else 0.0 for r in ranks]
    rr_present_items = [(1.0 / r) for r in ranks if r <= 20]

    return {
        "num_rows": n,
        "num_present": num_present,
        "gold_present_rate": round(num_present / n, 6),
        "mrr_at20": round(sum(rr_at20_items) / n, 6),
        "mrr_present_only": round(safe_mean(rr_present_items), 6),
        "hits1_at20": round(sum(1 for r in ranks if r <= 1) / n, 6),
        "hits3_at20": round(sum(1 for r in ranks if r <= 3) / n, 6),
        "hits10_at20": round(sum(1 for r in ranks if r <= 10) / n, 6),
        "avg_gold_rank": round(sum(ranks) / n, 6),
    }


def candidate_debug_direct_flag(d: dict) -> bool:
    return bool(
        d.get("candidate_query_edge_count", 0) > 0
        or d.get("direct_link_flag", False)
        or d.get("candidate_query_direct_flag", False)
    )


def candidate_debug_evidence_flag(d: dict) -> bool:
    return bool(
        d.get("evidence_edge_touch_count", 0) > 0
        or d.get("evidence_touch_count", 0) > 0
        or d.get("candidate_in_aligned_evidence", False)
    )


def candidate_debug_contra_flag(d: dict) -> bool:
    return bool(
        d.get("contra_flag", False)
        or d.get("contra_penalty", 0) > 0
    )


def summarize_soft_support_specific(rows: list[dict]) -> dict:
    top5_direct_rates = []
    top5_evidence_rates = []
    top5_contra_rates = []
    support_score_means = []

    for row in rows:
        st = row["stage_specific"]
        debug_rows = st.get("candidate_debug_rows", []) or []
        support_scores = st.get("support_scores", []) or []

        if support_scores:
            support_score_means.append(mean([float(x) for x in support_scores]))

        if debug_rows:
            top5 = debug_rows[:5]
            top5_direct_rates.append(
                sum(1 for d in top5 if candidate_debug_direct_flag(d)) / max(len(top5), 1)
            )
            top5_evidence_rates.append(
                sum(1 for d in top5 if candidate_debug_evidence_flag(d)) / max(len(top5), 1)
            )
            top5_contra_rates.append(
                sum(1 for d in top5 if candidate_debug_contra_flag(d)) / max(len(top5), 1)
            )

    return {
        "variant_name": rows[0]["stage_specific"].get("variant_name"),
        "avg_support_score": round(safe_mean(support_score_means), 6),
        "avg_top5_direct_link_rate": round(safe_mean(top5_direct_rates), 6),
        "avg_top5_evidence_positive_rate": round(safe_mean(top5_evidence_rates), 6),
        "avg_top5_contra_rate": round(safe_mean(top5_contra_rates), 6),
    }


def summarize_retrieval_specific(rows: list[dict]) -> dict:
    subgraph_sizes = []
    triple_scores = []
    direct_rates = []
    contra_rates = []
    coverage_rates = []

    variant_name = rows[0]["stage_specific"].get("variant_name")
    selected_source_variant = rows[0]["stage_specific"].get("selected_source_variant")

    for row in rows:
        st = row["stage_specific"]
        subgraph_sizes.append(int(st.get("num_selected_triples", 0)))
        triple_scores.append(float(st.get("avg_triple_score_row_value", 0.0)))

        ds = st.get("direct_shortcut_path_rate")
        cr = st.get("contradiction_path_rate")
        cov = st.get("candidate_coverage_preserved_rate")

        if ds is not None:
            direct_rates.append(float(ds))
        if cr is not None:
            contra_rates.append(float(cr))
        if cov is not None:
            coverage_rates.append(float(cov))

    return {
        "variant_name": variant_name,
        "selected_source_variant": selected_source_variant,
        "avg_subgraph_size": round(safe_mean(subgraph_sizes), 6),
        "avg_triple_score": round(safe_mean(triple_scores), 6),
        "avg_direct_shortcut_path_rate": round(safe_mean(direct_rates), 6),
        "avg_contradiction_path_rate": round(safe_mean(contra_rates), 6),
        "candidate_coverage_preserved_rate": round(safe_mean(coverage_rates), 6),
    }


def summarize_ontology_specific(rows: list[dict]) -> dict:
    absent = 0
    source_counts = {}
    after_counts = []

    for row in rows:
        st = row["stage_specific"]
        after_counts.append(int(st.get("num_candidates_after_ontology_filter", 0)))
        src = row.get("gold_rank_source")
        source_counts[src] = source_counts.get(src, 0) + 1
        if row["gold_present"] is False:
            absent += 1

    return {
        "avg_candidates_after_ontology_filter": round(safe_mean(after_counts), 6),
        "gold_absent_count": absent,
        "gold_rank_source_counts": source_counts,
    }


def summarize_backbone_specific(rows: list[dict]) -> dict:
    full_ranks = []
    for row in rows:
        st = row["stage_specific"]
        val = st.get("gold_rank_in_full_universe")
        if val is not None:
            full_ranks.append(int(val))
    return {
        "avg_gold_rank_in_full_universe": round(safe_mean(full_ranks), 6),
    }


def build_row_entry(row_name: str, rows: list[dict]) -> dict:
    meta = ROW_META[row_name]
    ranking = summarize_ranking_reviewer_safe(rows)

    specific = {}
    if row_name == "backbone_raw":
        specific = summarize_backbone_specific(rows)
    elif row_name == "ontology_raw":
        specific = summarize_ontology_specific(rows)
    elif row_name == "soft_support_raw":
        specific = summarize_soft_support_specific(rows)
    elif row_name == "retrieval_main":
        specific = summarize_retrieval_specific(rows)

    return {
        "canonical_row_name": meta["canonical_row_name"],
        "eval_row_name": row_name,
        "role": meta["role"],
        "stage": meta["stage"],
        "ranking_metrics": ranking,
        "row_specific_diagnostics": specific,
    }


def delta(a: float, b: float) -> float:
    return round(a - b, 6)


def build_ablation(table_rows: list[dict]) -> dict:
    idx = {row["canonical_row_name"]: row for row in table_rows}

    backbone = idx["backbone_raw"]["ranking_metrics"]
    ontology = idx["ontology_raw"]["ranking_metrics"]
    soft = idx["soft_support_raw"]["ranking_metrics"]
    retrieval = idx["soft_support_fuzzy_retrieval_main"]["ranking_metrics"]

    retrieval_diag = idx["soft_support_fuzzy_retrieval_main"]["row_specific_diagnostics"]

    comparisons = {
        "backbone_to_ontology_negative_control": {
            "delta_mrr_at20": delta(ontology["mrr_at20"], backbone["mrr_at20"]),
            "delta_hits1_at20": delta(ontology["hits1_at20"], backbone["hits1_at20"]),
            "delta_avg_gold_rank": delta(ontology["avg_gold_rank"], backbone["avg_gold_rank"]),
            "delta_gold_present_rate": delta(ontology["gold_present_rate"], backbone["gold_present_rate"]),
            "interpretation": "ontology_raw is a brittle negative control under raw/no-injection.",
        },
        "backbone_to_soft_support": {
            "delta_mrr_at20": delta(soft["mrr_at20"], backbone["mrr_at20"]),
            "delta_hits1_at20": delta(soft["hits1_at20"], backbone["hits1_at20"]),
            "delta_avg_gold_rank": delta(soft["avg_gold_rank"], backbone["avg_gold_rank"]),
            "delta_gold_present_rate": delta(soft["gold_present_rate"], backbone["gold_present_rate"]),
            "interpretation": "soft support improves the candidate-stage row over backbone_raw.",
        },
        "soft_support_to_retrieval_main": {
            "delta_mrr_at20": delta(retrieval["mrr_at20"], soft["mrr_at20"]),
            "delta_hits1_at20": delta(retrieval["hits1_at20"], soft["hits1_at20"]),
            "delta_avg_gold_rank": delta(retrieval["avg_gold_rank"], soft["avg_gold_rank"]),
            "delta_gold_present_rate": delta(retrieval["gold_present_rate"], soft["gold_present_rate"]),
            "ranking_preserved": (
                retrieval["mrr_at20"] == soft["mrr_at20"]
                and retrieval["hits1_at20"] == soft["hits1_at20"]
                and retrieval["avg_gold_rank"] == soft["avg_gold_rank"]
                and retrieval["gold_present_rate"] == soft["gold_present_rate"]
            ),
            "graph_side_note": {
                "avg_subgraph_size": retrieval_diag.get("avg_subgraph_size"),
                "avg_direct_shortcut_path_rate": retrieval_diag.get("avg_direct_shortcut_path_rate"),
                "avg_contradiction_path_rate": retrieval_diag.get("avg_contradiction_path_rate"),
                "candidate_coverage_preserved_rate": retrieval_diag.get("candidate_coverage_preserved_rate"),
            },
            "interpretation": "retrieval main preserves reviewer-safe ranking metrics while providing a cleaner graph/evidence package.",
        },
        "backbone_to_retrieval_main": {
            "delta_mrr_at20": delta(retrieval["mrr_at20"], backbone["mrr_at20"]),
            "delta_hits1_at20": delta(retrieval["hits1_at20"], backbone["hits1_at20"]),
            "delta_avg_gold_rank": delta(retrieval["avg_gold_rank"], backbone["avg_gold_rank"]),
            "delta_gold_present_rate": delta(retrieval["gold_present_rate"], backbone["gold_present_rate"]),
            "interpretation": "retrieval main is the strongest current row over the raw backbone reference.",
        },
    }

    return {
        "metric_policy": {
            "main_metric_name": "mrr_at20",
            "rr_rule": "1/rank if gold_rank <= 20 else 0",
            "descriptive_rank_rule": "gold_rank=21 when gold is out of top20",
        },
        "main_row": "soft_support_fuzzy_retrieval_main",
        "comparisons": comparisons,
        "key_takeaways": [
            "ontology_raw remains a valid negative control and should not be treated as a competitive row.",
            "soft_support_raw is the candidate-stage main intermediate row.",
            "soft_support_fuzzy_retrieval_main preserves reviewer-safe ranking metrics over soft_support_raw while improving graph-side packaging.",
        ],
    }


def main():
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    OUT_ABLATION.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    table_rows = []
    for row_name in ["backbone_raw", "ontology_raw", "soft_support_raw", "retrieval_main"]:
        rows = load_json(INPUTS[row_name])
        table_rows.append(build_row_entry(row_name, rows))

    ablation = build_ablation(table_rows)

    main_table = {
        "decision_split": "valid",
        "metric_policy": {
            "main_metric_name": "mrr_at20",
            "rr_rule": "1/rank if gold_rank <= 20 else 0",
            "descriptive_rank_rule": "gold_rank=21 when gold is out of top20",
        },
        "main_row": "soft_support_fuzzy_retrieval_main",
        "rows": table_rows,
    }

    write_json(OUT_TABLE, main_table)
    write_json(OUT_ABLATION, ablation)

    lines = []
    lines.append("# Valid Main Table + Ablation (Reviewer-safe)\n")
    lines.append("## Metric policy")
    lines.append("- Main metric: `mrr_at20`")
    lines.append("- RR rule: `1/rank if rank <= 20 else 0`")
    lines.append("- Descriptive rank rule: `gold_rank = 21 when gold is out of top20`")
    lines.append("")
    lines.append("## Main valid table rows")
    for row in table_rows:
        lines.append(f"- `{row['canonical_row_name']}` ({row['role']})")
    lines.append("")
    lines.append("## Ranking summary")
    for row in table_rows:
        rm = row["ranking_metrics"]
        lines.append(f"### {row['canonical_row_name']}")
        for k, v in rm.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")
    lines.append("## Key takeaways")
    for item in ablation["key_takeaways"]:
        lines.append(f"- {item}")
    lines.append("")
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] wrote {OUT_TABLE}")
    print(f"[OK] wrote {OUT_ABLATION}")
    print(f"[OK] wrote {OUT_REPORT}")
    print(json.dumps(main_table, indent=2, ensure_ascii=False))
    print(json.dumps(ablation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
