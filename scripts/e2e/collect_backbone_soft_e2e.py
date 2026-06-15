from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(".").resolve()

BACKBONE_RAW_METRICS = ROOT / "outputs/e2e/ranking_metrics_test_backbone_raw_e2e.json"
SOFT_RAW_METRICS = ROOT / "outputs/e2e/ranking_metrics_test_soft_support_raw_e2e.json"

BACKBONE_PRED = ROOT / "outputs/e2e/prediction_test_backbone_raw_e2e.json"
SOFT_PRED = ROOT / "outputs/e2e/prediction_test_soft_support_raw_e2e.json"

OUT_JSON = ROOT / "outputs/e2e/e2e_backbone_soft_summary.json"
OUT_MD = ROOT / "outputs/e2e/reports/backbone_soft_e2e.md"


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
    """
    Reviewer-safe gold rank:
    - rank = 1..k if gold appears in top-k candidates
    - rank = k+1 sentinel if gold is absent
    """
    target = get_target(row)
    candidates = get_candidates(row)[:k]

    if target in candidates:
        return candidates.index(target) + 1, True
    return k + 1, False


def compute_adjusted_e2e_rank(row: Dict[str, Any], k: int = 20) -> Tuple[int, bool]:
    """
    Reviewer-safe E2E rank.

    Policy:
    - If gold is absent from top-k candidates, RR@k = 0 no matter what the LLM generates.
    - If gold is present and the LLM generates the gold answer, adjusted rank = 1.
    - If gold is present but the LLM generates another candidate before the gold, the gold rank
      stays the same.
    - If gold is present but the LLM generates an entity not in candidates, or a candidate after
      the gold, the gold rank is worsened by 1, following the original DrKGC-style adjustment.
    - Any adjusted rank > k receives RR@k = 0.
    """
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
    """
    Candidate-only reviewer-safe metrics.
    This measures whether the raw/soft candidate list itself contains and ranks the gold.
    It does not use the generated LLM answer.
    """
    n = len(rows)
    if n == 0:
        raise ValueError("No prediction rows to summarize.")

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
    """
    E2E reviewer-safe metrics using generated answers plus constrained top-k policy.
    """
    n = len(rows)
    if n == 0:
        raise ValueError("No prediction rows to summarize.")

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


def delta(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    return {
        "delta_mrr_at20": round(float(b["mrr_at20"]) - float(a["mrr_at20"]), 8),
        "delta_hits1_at20": round(float(b["hits1_at20"]) - float(a["hits1_at20"]), 8),
        "delta_hits3_at20": round(float(b["hits3_at20"]) - float(a["hits3_at20"]), 8),
        "delta_hits10_at20": round(float(b["hits10_at20"]) - float(a["hits10_at20"]), 8),
        "delta_hits20_at20": round(float(b["hits20_at20"]) - float(a["hits20_at20"]), 8),
    }


def main() -> None:
    # Load raw infer.py metrics only for audit/comparison.
    backbone_raw_infer_metrics = load_json(BACKBONE_RAW_METRICS)
    soft_raw_infer_metrics = load_json(SOFT_RAW_METRICS)

    backbone_pred_obj = load_json(BACKBONE_PRED)
    soft_pred_obj = load_json(SOFT_PRED)

    backbone_rows = backbone_pred_obj["prediction"]
    soft_rows = soft_pred_obj["prediction"]

    if len(backbone_rows) != len(soft_rows):
        raise ValueError(
            f"Prediction row mismatch: backbone={len(backbone_rows)}, soft={len(soft_rows)}"
        )

    backbone_candidate = summarize_candidate_ceiling(backbone_rows, k=20)
    soft_candidate = summarize_candidate_ceiling(soft_rows, k=20)

    backbone_e2e = summarize_e2e_generation(backbone_rows, k=20)
    soft_e2e = summarize_e2e_generation(soft_rows, k=20)

    summary = {
        "stage": "run_e2e_backbone_and_soft",
        "status": "BUILT_REVIEWER_SAFE",
        "metric_policy": {
            "main_metric": "reviewer_safe_mrr_at20",
            "rr_rule": "1/rank if rank <= 20 else 0",
            "gold_rank_out_of_top20": 21,
            "important_note": (
                "Do not use raw infer.py mrr for reporting tables unless infer.py is patched. "
                "This collect script recomputes reviewer-safe metrics from prediction rows."
            ),
        },
        "backbone_raw": {
            "raw_infer_metrics_for_audit_only": backbone_raw_infer_metrics,
            "candidate_ceiling_reviewer_safe": backbone_candidate,
            "e2e_generation_reviewer_safe": backbone_e2e,
            "num_prediction_rows": len(backbone_rows),
            "sample0": {
                "target": get_target(backbone_rows[0]),
                "pred": normalize_text(backbone_rows[0].get("pred", "")),
                "original_pred_rank": backbone_rows[0].get("pred_rank"),
                "rank": backbone_rows[0].get("rank"),
            },
        },
        "soft_support_raw": {
            "raw_infer_metrics_for_audit_only": soft_raw_infer_metrics,
            "candidate_ceiling_reviewer_safe": soft_candidate,
            "e2e_generation_reviewer_safe": soft_e2e,
            "num_prediction_rows": len(soft_rows),
            "sample0": {
                "target": get_target(soft_rows[0]),
                "pred": normalize_text(soft_rows[0].get("pred", "")),
                "original_pred_rank": soft_rows[0].get("pred_rank"),
                "rank": soft_rows[0].get("rank"),
            },
        },
        "delta_soft_minus_backbone": {
            "candidate_ceiling": delta(backbone_candidate, soft_candidate),
            "e2e_generation": delta(backbone_e2e, soft_e2e),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = []
    md.append("# E2E Infer for backbone_raw and soft_support_raw")
    md.append("")
    md.append(f"- status: **{summary['status']}**")
    md.append("- main metric: `reviewer_safe_mrr_at20`")
    md.append("- RR rule: `1/rank if rank <= 20 else 0`")
    md.append("- gold absent from top-20: `RR@20 = 0`")
    md.append("")
    md.append("## 1. backbone_raw — candidate ceiling reviewer-safe")
    for k, v in backbone_candidate.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 2. backbone_raw — E2E generation reviewer-safe")
    for k, v in backbone_e2e.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 3. soft_support_raw — candidate ceiling reviewer-safe")
    for k, v in soft_candidate.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 4. soft_support_raw — E2E generation reviewer-safe")
    for k, v in soft_e2e.items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 5. delta_soft_minus_backbone")
    md.append("")
    md.append("### Candidate ceiling")
    for k, v in summary["delta_soft_minus_backbone"]["candidate_ceiling"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("### E2E generation")
    for k, v in summary["delta_soft_minus_backbone"]["e2e_generation"].items():
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
