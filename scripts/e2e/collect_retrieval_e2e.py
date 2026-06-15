from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(".").resolve()

BACKBONE_SOFT_SUMMARY = ROOT / "outputs/e2e/e2e_backbone_soft_summary.json"

RETR_RAW_METRICS = ROOT / "outputs/e2e/ranking_metrics_test_retrieval_main_e2e.json"
RETR_PRED = ROOT / "outputs/e2e/prediction_test_retrieval_main_e2e.json"
RETR_INFER_READY = ROOT / "dataset/setting_a/e2e_infer_ready/retrieval_main/test.json"

OUT_JSON = ROOT / "outputs/e2e/e2e_retrieval_summary.json"
OUT_MD = ROOT / "outputs/e2e/reports/retrieval_e2e.md"


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(x: Any) -> str:
    return str(x).strip()


def get_target(row: Dict[str, Any]) -> str:
    if "target" in row:
        return normalize_text(row["target"])
    if "output" in row:
        return normalize_text(row["output"])
    raise KeyError("Prediction row has neither 'target' nor 'output'.")


def get_candidates(row: Dict[str, Any]) -> List[str]:
    if "rank_entities" not in row:
        raise KeyError("Prediction row missing 'rank_entities'.")
    return [normalize_text(x) for x in row["rank_entities"]]


def compute_gold_rank_from_candidates(row: Dict[str, Any], k: int = 20) -> Tuple[int, bool]:
    target = get_target(row)
    candidates = get_candidates(row)[:k]
    if target in candidates:
        return candidates.index(target) + 1, True
    return k + 1, False


def compute_adjusted_e2e_rank(row: Dict[str, Any], k: int = 20) -> Tuple[int, bool]:
    target = get_target(row)
    pred = normalize_text(row.get("pred", ""))
    candidates = get_candidates(row)[:k]

    gold_rank, gold_present = compute_gold_rank_from_candidates(row, k=k)
    if not gold_present:
        return k + 1, False

    if pred == target:
        return 1, True

    adjusted_rank = gold_rank
    if pred not in set(candidates):
        adjusted_rank += 1
    else:
        pred_pos = candidates.index(pred) + 1
        if pred_pos >= gold_rank:
            adjusted_rank += 1

    return min(adjusted_rank, k + 1), True


def summarize_candidate_ceiling(rows: List[Dict[str, Any]], k: int = 20) -> Dict[str, Any]:
    n = len(rows)
    ranks = []
    present = []

    for row in rows:
        rank, is_present = compute_gold_rank_from_candidates(row, k=k)
        ranks.append(rank)
        present.append(is_present)

    rr = [(1.0 / r) if r <= k else 0.0 for r in ranks]

    return {
        "num_examples": n,
        "k": k,
        "gold_present_rate": round(sum(present) / n, 8),
        "mrr_at20": round(sum(rr) / n, 8),
        "hits1_at20": round(sum(1 for r in ranks if r <= 1) / n, 8),
        "hits3_at20": round(sum(1 for r in ranks if r <= 3) / n, 8),
        "hits10_at20": round(sum(1 for r in ranks if r <= 10) / n, 8),
        "hits20_at20": round(sum(1 for r in ranks if r <= k) / n, 8),
        "avg_gold_rank_with_absent_as_21": round(sum(ranks) / n, 8),
        "gold_rank_21_count": int(sum(1 for r in ranks if r == k + 1)),
        "rr_rule": "1/rank if rank <= 20 else 0",
    }


def summarize_e2e_generation(rows: List[Dict[str, Any]], k: int = 20) -> Dict[str, Any]:
    n = len(rows)
    adjusted_ranks = []
    gold_present_flags = []
    exact_generated = 0
    pred_in_candidates = 0

    for row in rows:
        target = get_target(row)
        pred = normalize_text(row.get("pred", ""))
        candidates = get_candidates(row)[:k]

        adjusted_rank, gold_present = compute_adjusted_e2e_rank(row, k=k)
        adjusted_ranks.append(adjusted_rank)
        gold_present_flags.append(gold_present)

        if pred == target:
            exact_generated += 1
        if pred in set(candidates):
            pred_in_candidates += 1

    rr = [(1.0 / r) if r <= k else 0.0 for r in adjusted_ranks]

    return {
        "num_examples": n,
        "k": k,
        "gold_present_rate": round(sum(gold_present_flags) / n, 8),
        "mrr_at20": round(sum(rr) / n, 8),
        "hits1_at20": round(sum(1 for r in adjusted_ranks if r <= 1) / n, 8),
        "hits3_at20": round(sum(1 for r in adjusted_ranks if r <= 3) / n, 8),
        "hits10_at20": round(sum(1 for r in adjusted_ranks if r <= 10) / n, 8),
        "hits20_at20": round(sum(1 for r in adjusted_ranks if r <= k) / n, 8),
        "avg_adjusted_rank_with_absent_as_21": round(sum(adjusted_ranks) / n, 8),
        "rank_21_count": int(sum(1 for r in adjusted_ranks if r == k + 1)),
        "exact_generated_rate": round(exact_generated / n, 8),
        "pred_in_candidate_rate": round(pred_in_candidates / n, 8),
        "rr_rule": "1/adjusted_rank if adjusted_rank <= 20 else 0",
        "absent_gold_policy": "RR@20 = 0 no matter what the LLM generates",
    }


def delta(base: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, float]:
    return {
        "delta_mrr_at20": round(float(target["mrr_at20"]) - float(base["mrr_at20"]), 8),
        "delta_hits1_at20": round(float(target["hits1_at20"]) - float(base["hits1_at20"]), 8),
        "delta_hits3_at20": round(float(target["hits3_at20"]) - float(base["hits3_at20"]), 8),
        "delta_hits10_at20": round(float(target["hits10_at20"]) - float(base["hits10_at20"]), 8),
        "delta_hits20_at20": round(float(target["hits20_at20"]) - float(base["hits20_at20"]), 8),
    }


def summarize_retrieval_graph_package(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    subgraph_sizes = [len(x.get("subgraph", [])) for x in rows]
    selected_source_variants = sorted(set(x.get("selected_source_variant") for x in rows))

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

    return {
        "num_rows": n,
        "selected_source_variant_set": selected_source_variants,
        "avg_subgraph_size": round(sum(subgraph_sizes) / n, 8),
        "min_subgraph_size": min(subgraph_sizes),
        "max_subgraph_size": max(subgraph_sizes),
        "avg_candidate_coverage_preserved_rate": round(sum(coverage_vals) / max(len(coverage_vals), 1), 8) if coverage_vals else None,
        "avg_direct_shortcut_path_rate": round(sum(shortcut_rates) / max(len(shortcut_rates), 1), 8) if shortcut_rates else None,
    }


def main() -> None:
    backbone_soft_summary = load_json(BACKBONE_SOFT_SUMMARY)

    retr_raw_infer_metrics = load_json(RETR_RAW_METRICS)
    retr_pred_obj = load_json(RETR_PRED)
    retr_rows = retr_pred_obj["prediction"]
    retr_infer_ready_rows = load_json(RETR_INFER_READY)

    retr_candidate = summarize_candidate_ceiling(retr_rows, k=20)
    retr_e2e = summarize_e2e_generation(retr_rows, k=20)
    graph_pkg = summarize_retrieval_graph_package(retr_infer_ready_rows)

    backbone_candidate = backbone_soft_summary["backbone_raw"]["candidate_ceiling_reviewer_safe"]
    backbone_e2e = backbone_soft_summary["backbone_raw"]["e2e_generation_reviewer_safe"]
    soft_candidate = backbone_soft_summary["soft_support_raw"]["candidate_ceiling_reviewer_safe"]
    soft_e2e = backbone_soft_summary["soft_support_raw"]["e2e_generation_reviewer_safe"]

    summary = {
        "stage": "run_e2e_retrieval_main",
        "status": "BUILT_REVIEWER_SAFE",
        "metric_policy": {
            "main_metric": "reviewer_safe_mrr_at20",
            "rr_rule": "1/rank if rank <= 20 else 0",
            "gold_rank_out_of_top20": 21,
            "important_note": (
                "Raw infer.py metrics are retained only for audit. "
                "Downstream tables use reviewer-safe recomputation from prediction rows."
            ),
        },
        "retrieval_main": {
            "raw_infer_metrics_for_audit_only": retr_raw_infer_metrics,
            "candidate_ceiling_reviewer_safe": retr_candidate,
            "e2e_generation_reviewer_safe": retr_e2e,
            "graph_package": graph_pkg,
            "num_prediction_rows": len(retr_rows),
            "sample0": {
                "target": get_target(retr_rows[0]),
                "pred": normalize_text(retr_rows[0].get("pred", "")),
                "original_pred_rank": retr_rows[0].get("pred_rank"),
                "rank": retr_rows[0].get("rank"),
            },
        },
        "delta_retrieval_minus_soft": {
            "candidate_ceiling": delta(soft_candidate, retr_candidate),
            "e2e_generation": delta(soft_e2e, retr_e2e),
        },
        "delta_retrieval_minus_backbone": {
            "candidate_ceiling": delta(backbone_candidate, retr_candidate),
            "e2e_generation": delta(backbone_e2e, retr_e2e),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = []
    md.append("# E2E Infer for retrieval_main")
    md.append("")
    md.append(f"- status: **{summary['status']}**")
    md.append("- main metric: `reviewer_safe_mrr_at20`")
    md.append("- RR rule: `1/rank if rank <= 20 else 0`")
    md.append("- gold absent from top-20: `RR@20 = 0`")
    md.append("")
    md.append("## 1. retrieval_main — candidate ceiling reviewer-safe")
    for k, v in retr_candidate.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 2. retrieval_main — E2E generation reviewer-safe")
    for k, v in retr_e2e.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 3. retrieval_main — graph package")
    for k, v in graph_pkg.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 4. delta_retrieval_minus_soft")
    md.append("")
    md.append("### Candidate ceiling")
    for k, v in summary["delta_retrieval_minus_soft"]["candidate_ceiling"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("### E2E generation")
    for k, v in summary["delta_retrieval_minus_soft"]["e2e_generation"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 5. delta_retrieval_minus_backbone")
    md.append("")
    md.append("### Candidate ceiling")
    for k, v in summary["delta_retrieval_minus_backbone"]["candidate_ceiling"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("### E2E generation")
    for k, v in summary["delta_retrieval_minus_backbone"]["e2e_generation"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 6. Audit note")
    md.append(
        "Raw `infer.py` metrics are retained only for audit. "
        "Downstream tables should use the reviewer-safe recomputation in this report."
    )

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
