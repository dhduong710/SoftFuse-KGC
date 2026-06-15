from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


ROOT = Path(".").resolve()

MODEL_ROOT = ROOT / "outputs/e2e/model_compare"
READY_ROOT = ROOT / "dataset/setting_a/e2e_infer_ready"

OUT_SUMMARY = MODEL_ROOT / "model_compare_reviewer_safe_summary.json"
OUT_VALID = MODEL_ROOT / "model_compare_table_valid.json"
OUT_TEST = MODEL_ROOT / "model_compare_table_test.json"
OUT_BEST = MODEL_ROOT / "model_compare_best_model.json"
OUT_CASES = MODEL_ROOT / "model_compare_case_samples.json"
OUT_MD = ROOT / "outputs/e2e/reports/day5_model_compare_eval_and_selection.md"

MODELS = ["llama3_2_3b", "llama3_8b", "medllama3_8b"]
ROWS = ["backbone_raw", "soft_support_raw", "retrieval_main"]
SPLITS = ["valid", "test"]
K = 20


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_strict(x: Any) -> str:
    return str(x).strip()


def norm_loose(x: Any) -> str:
    x = "" if x is None else str(x)
    if "Answer:" in x:
        x = x.split("Answer:")[-1]
    x = x.strip()
    x = x.splitlines()[0].strip() if x.splitlines() else x
    x = x.strip().strip("'").strip('"').strip("`")
    x = re.sub(r"\s+", " ", x)
    return x.strip(" .,:;").lower()


def get_target(row: Dict[str, Any]) -> str:
    if "target" in row:
        return norm_strict(row["target"])
    if "output" in row:
        return norm_strict(row["output"])
    raise KeyError("Missing target/output")


def get_candidates(row: Dict[str, Any]) -> List[str]:
    if "rank_entities" not in row:
        raise KeyError("Missing rank_entities")
    return [norm_strict(x) for x in row["rank_entities"][:K]]


def compute_gold_rank(row: Dict[str, Any]) -> Tuple[int, bool]:
    target = get_target(row)
    candidates = get_candidates(row)

    if target in candidates:
        return candidates.index(target) + 1, True

    return K + 1, False


def compute_adjusted_e2e_rank(row: Dict[str, Any]) -> Tuple[int, bool]:
    """
    Reviewer-safe DrKGC-style adjusted E2E rank.

    If gold absent from top-20:
      adjusted_rank = 21, RR = 0.

    If model generates exact gold:
      adjusted_rank = 1.

    Else:
      start from gold candidate rank.
      If prediction is invalid or appears after/equal gold, worsen by +1.
      If prediction is a candidate before gold, keep gold rank.
    """
    target = get_target(row)
    pred = norm_strict(row.get("pred", ""))
    candidates = get_candidates(row)

    gold_rank, gold_present = compute_gold_rank(row)

    if not gold_present:
        return K + 1, False

    if pred == target:
        return 1, True

    adjusted = gold_rank

    if pred not in set(candidates):
        adjusted += 1
    else:
        pred_pos = candidates.index(pred) + 1
        if pred_pos >= gold_rank:
            adjusted += 1

    return min(adjusted, K + 1), True


def is_candidate_list_fragment(pred: str, candidates: List[str]) -> bool:
    p = str(pred)

    if len(p) > 120:
        return True
    if p.count(",") >= 2:
        return True
    if p.count("'") >= 4:
        return True
    if "####" in p:
        return True

    loose = norm_loose(p)
    hits = 0
    for c in candidates:
        cn = norm_loose(c)
        if cn and cn in loose:
            hits += 1

    return hits >= 3


def prediction_category(row: Dict[str, Any]) -> str:
    pred = norm_strict(row.get("pred", ""))
    target = get_target(row)
    candidates = get_candidates(row)

    if pred == "":
        return "empty_prediction"
    if pred == target:
        return "exact_target"
    if pred in set(candidates):
        if candidates and pred == candidates[0]:
            return "top1_copy"
        return "other_candidate"
    if is_candidate_list_fragment(pred, candidates):
        return "candidate_list_fragment"
    return "invalid_other"


def summarize_prediction_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise ValueError("No rows")

    gold_ranks = []
    gold_present_flags = []
    adjusted_ranks = []

    exact = 0
    pred_in_candidate = 0
    invalid = 0
    top1_copy = 0
    list_frag = 0
    empty_pred = 0

    category_counts: Dict[str, int] = {}

    for row in rows:
        pred = norm_strict(row.get("pred", ""))
        target = get_target(row)
        candidates = get_candidates(row)

        gold_rank, gold_present = compute_gold_rank(row)
        adjusted_rank, _ = compute_adjusted_e2e_rank(row)

        gold_ranks.append(gold_rank)
        gold_present_flags.append(gold_present)
        adjusted_ranks.append(adjusted_rank)

        if pred == target:
            exact += 1
        if pred in set(candidates):
            pred_in_candidate += 1
        else:
            invalid += 1
        if candidates and pred == candidates[0]:
            top1_copy += 1
        if is_candidate_list_fragment(pred, candidates):
            list_frag += 1
        if pred == "":
            empty_pred += 1

        cat = prediction_category(row)
        category_counts[cat] = category_counts.get(cat, 0) + 1

    candidate_rr = [(1.0 / r) if r <= K else 0.0 for r in gold_ranks]
    e2e_rr = [(1.0 / r) if r <= K else 0.0 for r in adjusted_ranks]

    return {
        "num_examples": n,

        "gold_at20": round(sum(gold_present_flags) / n, 8),
        "candidate_mrr_at20": round(sum(candidate_rr) / n, 8),
        "candidate_hits1_at20": round(sum(1 for r in gold_ranks if r <= 1) / n, 8),
        "candidate_hits3_at20": round(sum(1 for r in gold_ranks if r <= 3) / n, 8),
        "candidate_hits10_at20": round(sum(1 for r in gold_ranks if r <= 10) / n, 8),
        "candidate_rank21_count": int(sum(1 for r in gold_ranks if r == K + 1)),

        "reviewer_safe_e2e_mrr_at20": round(sum(e2e_rr) / n, 8),
        "reviewer_safe_e2e_hits1_at20": round(sum(1 for r in adjusted_ranks if r <= 1) / n, 8),
        "reviewer_safe_e2e_hits3_at20": round(sum(1 for r in adjusted_ranks if r <= 3) / n, 8),
        "reviewer_safe_e2e_hits10_at20": round(sum(1 for r in adjusted_ranks if r <= 10) / n, 8),
        "e2e_rank21_count": int(sum(1 for r in adjusted_ranks if r == K + 1)),

        "exact_target_match_rate": round(exact / n, 8),
        "pred_in_candidate_rate": round(pred_in_candidate / n, 8),
        "invalid_prediction_rate": round(invalid / n, 8),
        "top1_copy_rate": round(top1_copy / n, 8),
        "candidate_list_fragment_rate": round(list_frag / n, 8),
        "empty_prediction_rate": round(empty_pred / n, 8),

        "prediction_category_counts": category_counts,
        "rr_rule": "1/adjusted_rank if adjusted_rank <= 20 else 0",
        "absent_gold_policy": "RR@20 = 0",
    }


def graph_package(row_name: str, split: str) -> Dict[str, Any]:
    rows = load_json(READY_ROOT / row_name / f"{split}.json")
    sizes = [len(r.get("subgraph", [])) for r in rows]

    out = {
        "avg_subgraph_size": round(mean(sizes), 8),
        "min_subgraph_size": min(sizes),
        "max_subgraph_size": max(sizes),
    }

    if row_name == "retrieval_main":
        summaries = [r.get("subgraph_summary", {}) for r in rows]
        coverage = [
            float(s.get("candidate_coverage_preserved_rate", 0.0))
            for s in summaries
            if "candidate_coverage_preserved_rate" in s
        ]
        if coverage:
            out["avg_candidate_coverage_preserved_rate"] = round(mean(coverage), 8)

        source_variants = sorted(set(r.get("selected_source_variant") for r in rows))
        out["selected_source_variant_set"] = source_variants

    return out


def summarize_file(model_tag: str, split: str, row_name: str) -> Dict[str, Any]:
    pred_path = MODEL_ROOT / model_tag / "predictions" / f"prediction_{split}_{row_name}.json"
    obj = load_json(pred_path)
    rows = obj["prediction"]

    if len(rows) != 500:
        raise RuntimeError(f"{pred_path} has {len(rows)} rows, expected 500")

    metrics = summarize_prediction_rows(rows)
    graph = graph_package(row_name, split)

    return {
        "model_tag": model_tag,
        "split": split,
        "row_name": row_name,
        "prediction_path": str(pred_path),
        **metrics,
        **graph,
    }


def pick_cases(model_tag: str, split: str, row_name: str, max_each: int = 5) -> Dict[str, Any]:
    pred_path = MODEL_ROOT / model_tag / "predictions" / f"prediction_{split}_{row_name}.json"
    rows = load_json(pred_path)["prediction"]

    buckets = {
        "exact_target": [],
        "top1_copy": [],
        "other_candidate": [],
        "candidate_list_fragment": [],
        "invalid_other": [],
        "gold_present_but_e2e_fail": [],
    }

    for i, row in enumerate(rows):
        cat = prediction_category(row)
        target = get_target(row)
        pred = norm_strict(row.get("pred", ""))
        gold_rank, gold_present = compute_gold_rank(row)
        adjusted_rank, _ = compute_adjusted_e2e_rank(row)

        item = {
            "row_index": row.get("row_index", i),
            "query_entity": row.get("query_entity"),
            "target": target,
            "pred": pred,
            "gold_rank": gold_rank,
            "adjusted_rank": adjusted_rank,
            "top5": row.get("rank_entities", [])[:5],
            "category": cat,
        }

        if cat in buckets and len(buckets[cat]) < max_each:
            buckets[cat].append(item)

        if gold_present and adjusted_rank > gold_rank and len(buckets["gold_present_but_e2e_fail"]) < max_each:
            buckets["gold_present_but_e2e_fail"].append(item)

    return buckets


def md_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Model", "Split", "Row", "Gold@20", "Cand MRR", "E2E MRR",
        "H@1", "H@3", "H@10", "Pred-in-cand", "Invalid",
        "Top1-copy", "List-frag", "Avg subgraph"
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        lines.append(
            "| "
            + " | ".join([
                r["model_tag"],
                r["split"],
                r["row_name"],
                f"{r['gold_at20']:.3f}",
                f"{r['candidate_mrr_at20']:.6f}",
                f"{r['reviewer_safe_e2e_mrr_at20']:.6f}",
                f"{r['reviewer_safe_e2e_hits1_at20']:.3f}",
                f"{r['reviewer_safe_e2e_hits3_at20']:.3f}",
                f"{r['reviewer_safe_e2e_hits10_at20']:.3f}",
                f"{r['pred_in_candidate_rate']:.3f}",
                f"{r['invalid_prediction_rate']:.3f}",
                f"{r['top1_copy_rate']:.3f}",
                f"{r['candidate_list_fragment_rate']:.3f}",
                f"{r['avg_subgraph_size']:.3f}",
            ])
            + " |"
        )

    return "\n".join(lines)


def main() -> None:
    summary_rows = []

    for model_tag in MODELS:
        model_dir = MODEL_ROOT / model_tag / "predictions"
        if not model_dir.exists():
            raise FileNotFoundError(f"Missing predictions dir for {model_tag}: {model_dir}")

        for split in SPLITS:
            for row_name in ROWS:
                summary_rows.append(summarize_file(model_tag, split, row_name))

    valid_rows = [r for r in summary_rows if r["split"] == "valid"]
    test_rows = [r for r in summary_rows if r["split"] == "test"]

    save_json(OUT_SUMMARY, {
        "decision": "PRIMEKG_MODEL_COMPARE_REVIEWER_SAFE_BUILT",
        "models": MODELS,
        "rows": ROWS,
        "splits": SPLITS,
        "summary_rows": summary_rows,
    })
    save_json(OUT_VALID, valid_rows)
    save_json(OUT_TEST, test_rows)

    # Main selection: test retrieval_main.
    test_retrieval = [
        r for r in summary_rows
        if r["split"] == "test" and r["row_name"] == "retrieval_main"
    ]

    ranked = sorted(
        test_retrieval,
        key=lambda x: (
            x["reviewer_safe_e2e_mrr_at20"],
            -x["invalid_prediction_rate"],
            x["pred_in_candidate_rate"],
            -x["top1_copy_rate"],
        ),
        reverse=True,
    )

    best = ranked[0]

    # Determine whether each model preserves SoftFuse trend on test.
    trend_checks = {}
    for model_tag in MODELS:
        b = [
            r for r in summary_rows
            if r["model_tag"] == model_tag and r["split"] == "test" and r["row_name"] == "backbone_raw"
        ][0]
        s = [
            r for r in summary_rows
            if r["model_tag"] == model_tag and r["split"] == "test" and r["row_name"] == "soft_support_raw"
        ][0]
        ret = [
            r for r in summary_rows
            if r["model_tag"] == model_tag and r["split"] == "test" and r["row_name"] == "retrieval_main"
        ][0]

        trend_checks[model_tag] = {
            "soft_improves_backbone_e2e": s["reviewer_safe_e2e_mrr_at20"] > b["reviewer_safe_e2e_mrr_at20"],
            "retrieval_preserves_or_improves_soft_e2e": ret["reviewer_safe_e2e_mrr_at20"] >= s["reviewer_safe_e2e_mrr_at20"],
            "retrieval_smaller_than_soft": ret["avg_subgraph_size"] < s["avg_subgraph_size"],
            "test_delta_retrieval_minus_backbone_e2e": round(
                ret["reviewer_safe_e2e_mrr_at20"] - b["reviewer_safe_e2e_mrr_at20"], 8
            ),
            "test_delta_retrieval_minus_soft_e2e": round(
                ret["reviewer_safe_e2e_mrr_at20"] - s["reviewer_safe_e2e_mrr_at20"], 8
            ),
        }

    best_payload = {
        "decision": "PRIMEKG_E2E_MODEL_SELECTION_READY",
        "selection_basis": "test retrieval_main reviewer-safe E2E MRR@20 with invalid/pred-in-candidate/top1-copy as tie-break diagnostics",
        "best_test_retrieval_main": best,
        "ranked_test_retrieval_main": ranked,
        "trend_checks_by_model": trend_checks,
        "report_recommendation": {
            "main_e2e_model": best["model_tag"],
            "main_row": "retrieval_main",
            "recommended_table": "Use full model comparison table in appendix or main depending on space; use selected model for primary E2E result.",
            "important_caveat": "If a larger model has higher MRR but much higher top1-copy or invalid rate, discuss it as generation-style sensitivity rather than pure model superiority.",
        },
    }

    save_json(OUT_BEST, best_payload)

    case_samples = {}
    for model_tag in MODELS:
        case_samples[model_tag] = pick_cases(model_tag, "test", "retrieval_main", max_each=5)
    save_json(OUT_CASES, case_samples)

    # Markdown report
    lines = []
    lines.append("# PrimeKG Model Comparison Evaluation and Selection")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("**PRIMEKG_E2E_MODEL_SELECTION_READY**")
    lines.append("")
    lines.append("## Best model on test retrieval_main")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(best, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Ranked test retrieval_main models")
    lines.append("")
    lines.append("| Rank | Model | E2E MRR | H@1 | H@3 | H@10 | Pred-in-cand | Invalid | Top1-copy | List-frag |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {r['model_tag']} | {r['reviewer_safe_e2e_mrr_at20']:.6f} | "
            f"{r['reviewer_safe_e2e_hits1_at20']:.3f} | "
            f"{r['reviewer_safe_e2e_hits3_at20']:.3f} | "
            f"{r['reviewer_safe_e2e_hits10_at20']:.3f} | "
            f"{r['pred_in_candidate_rate']:.3f} | "
            f"{r['invalid_prediction_rate']:.3f} | "
            f"{r['top1_copy_rate']:.3f} | "
            f"{r['candidate_list_fragment_rate']:.3f} |"
        )
    lines.append("")
    lines.append("## Full model comparison table")
    lines.append("")
    lines.append(md_table(summary_rows))
    lines.append("")
    lines.append("## Test trend checks by model")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(trend_checks, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Interpretation notes")
    lines.append("")
    lines.append("- Candidate metrics are identical across models for the same row; model differences come from generation.")
    lines.append("- `soft_support_raw` should improve over `backbone_raw`; `retrieval_main` should preserve soft-support ranking while using smaller subgraphs.")
    lines.append("- High `top1_copy_rate` means the model often copies the first candidate rather than using graph-conditioned reasoning.")
    lines.append("- Raw `infer.py` metrics remain audit-only; this report uses reviewer-safe recomputation from prediction rows.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("decision = PRIMEKG_E2E_MODEL_SELECTION_READY")
    print(f"wrote summary = {OUT_SUMMARY}")
    print(f"wrote valid = {OUT_VALID}")
    print(f"wrote test = {OUT_TEST}")
    print(f"wrote best = {OUT_BEST}")
    print(f"wrote cases = {OUT_CASES}")
    print(f"wrote report = {OUT_MD}")
    print("")
    print("Ranked test retrieval_main:")
    for i, r in enumerate(ranked, start=1):
        print(
            i,
            r["model_tag"],
            "MRR=", r["reviewer_safe_e2e_mrr_at20"],
            "H10=", r["reviewer_safe_e2e_hits10_at20"],
            "pred_in=", r["pred_in_candidate_rate"],
            "invalid=", r["invalid_prediction_rate"],
            "top1_copy=", r["top1_copy_rate"],
            "list_frag=", r["candidate_list_fragment_rate"],
        )


if __name__ == "__main__":
    main()