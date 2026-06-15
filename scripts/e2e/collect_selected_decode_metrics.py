from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


ROOT = Path(".").resolve()

PRED_ROOT = ROOT / "outputs/e2e/selected_decode_test/predictions"
READY_ROOT = ROOT / "dataset/setting_a/e2e_infer_ready"

OUT_VALID = ROOT / "outputs/e2e/selected_decode_test/primekg_e2e_final_table_valid.json"
OUT_TEST = ROOT / "outputs/e2e/selected_decode_test/primekg_e2e_final_table_test.json"
OUT_COMPARISON = ROOT / "outputs/e2e/selected_decode_test/e2e_reference_comparison.json"
OUT_SUMMARY = ROOT / "outputs/e2e/selected_decode_test/primekg_selected_decode_summary.json"
OUT_MD = ROOT / "outputs/e2e/reports/day3_selected_decode_test_rerun.md"

BEST_CONFIG = ROOT / "outputs/e2e/selected_decode_test/selected_decode_config.json"

ROWS = ["backbone_raw", "soft_support_raw", "retrieval_main"]
SPLITS = ["valid", "test"]
K = 20

# Week20 locked E2E reference from prior reports.
WEEK20_TEST_REFERENCE = {
    "backbone_raw": {
        "candidate_mrr_at20": 0.06456283,
        "e2e_mrr_at20": 0.04763573,
    },
    "soft_support_raw": {
        "candidate_mrr_at20": 0.12532618,
        "e2e_mrr_at20": 0.07467554,
    },
    "retrieval_main": {
        "candidate_mrr_at20": 0.12532618,
        "e2e_mrr_at20": 0.07468653,
    },
}


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
    return x.strip(" .,:;，。").lower()


def get_target(row: Dict[str, Any]) -> str:
    if "target" in row:
        return norm_strict(row["target"])
    if "output" in row:
        return norm_strict(row["output"])
    raise KeyError("Missing target/output")


def get_candidates(row: Dict[str, Any]) -> List[str]:
    return [norm_strict(x) for x in row["rank_entities"][:K]]


def compute_gold_rank(row: Dict[str, Any]) -> Tuple[int, bool]:
    target = get_target(row)
    candidates = get_candidates(row)
    if target in candidates:
        return candidates.index(target) + 1, True
    return K + 1, False


def compute_adjusted_rank(row: Dict[str, Any]) -> Tuple[int, bool]:
    target = get_target(row)
    pred = norm_strict(row.get("pred", ""))
    candidates = get_candidates(row)

    gold_rank, gold_present = compute_gold_rank(row)

    if not gold_present:
        return K + 1, False

    if pred == target:
        return 1, True

    adjusted_rank = gold_rank

    if pred not in set(candidates):
        adjusted_rank += 1
    else:
        pred_pos = candidates.index(pred) + 1
        if pred_pos >= gold_rank:
            adjusted_rank += 1

    return min(adjusted_rank, K + 1), True


def is_candidate_list_fragment(pred: str, candidates: List[str]) -> bool:
    p = str(pred)
    if len(p) > 120:
        return True
    if p.count(",") >= 2:
        return True
    if p.count("'") >= 4:
        return True

    loose = norm_loose(p)
    hits = 0
    for c in candidates:
        cn = norm_loose(c)
        if cn and cn in loose:
            hits += 1
    return hits >= 3


def summarize_candidate_ceiling(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ranks = []
    present = []

    for row in rows:
        rank, is_present = compute_gold_rank(row)
        ranks.append(rank)
        present.append(is_present)

    rr = [(1.0 / r) if r <= K else 0.0 for r in ranks]

    return {
        "gold_at20": round(sum(present) / len(rows), 8),
        "candidate_mrr_at20": round(sum(rr) / len(rows), 8),
        "candidate_hits1_at20": round(sum(1 for r in ranks if r <= 1) / len(rows), 8),
        "candidate_hits3_at20": round(sum(1 for r in ranks if r <= 3) / len(rows), 8),
        "candidate_hits10_at20": round(sum(1 for r in ranks if r <= 10) / len(rows), 8),
        "candidate_hits20_at20": round(sum(1 for r in ranks if r <= K) / len(rows), 8),
        "candidate_rank21_count": int(sum(1 for r in ranks if r == K + 1)),
    }


def summarize_e2e(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    adjusted_ranks = []
    gold_present_flags = []

    exact = 0
    pred_in_candidate = 0
    invalid = 0
    top1_copy = 0
    list_frag = 0
    empty_pred = 0

    for row in rows:
        pred = norm_strict(row.get("pred", ""))
        target = get_target(row)
        candidates = get_candidates(row)

        adjusted_rank, gold_present = compute_adjusted_rank(row)
        adjusted_ranks.append(adjusted_rank)
        gold_present_flags.append(gold_present)

        if pred == target:
            exact += 1
        if pred in set(candidates):
            pred_in_candidate += 1
        else:
            invalid += 1
        if candidates and pred == candidates[0]:
            top1_copy += 1
        if pred == "":
            empty_pred += 1
        if is_candidate_list_fragment(pred, candidates):
            list_frag += 1

    rr = [(1.0 / r) if r <= K else 0.0 for r in adjusted_ranks]

    return {
        "reviewer_safe_e2e_mrr_at20": round(sum(rr) / len(rows), 8),
        "reviewer_safe_e2e_hits1_at20": round(sum(1 for r in adjusted_ranks if r <= 1) / len(rows), 8),
        "reviewer_safe_e2e_hits3_at20": round(sum(1 for r in adjusted_ranks if r <= 3) / len(rows), 8),
        "reviewer_safe_e2e_hits10_at20": round(sum(1 for r in adjusted_ranks if r <= 10) / len(rows), 8),
        "reviewer_safe_e2e_hits20_at20": round(sum(1 for r in adjusted_ranks if r <= K) / len(rows), 8),
        "e2e_rank21_count": int(sum(1 for r in adjusted_ranks if r == K + 1)),
        "gold_present_rate": round(sum(gold_present_flags) / len(rows), 8),
        "exact_target_match_rate": round(exact / len(rows), 8),
        "pred_in_candidate_rate": round(pred_in_candidate / len(rows), 8),
        "invalid_prediction_rate": round(invalid / len(rows), 8),
        "top1_copy_rate": round(top1_copy / len(rows), 8),
        "candidate_list_fragment_rate": round(list_frag / len(rows), 8),
        "empty_prediction_rate": round(empty_pred / len(rows), 8),
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


def summarize_file(split: str, row_name: str) -> Dict[str, Any]:
    pred_path = PRED_ROOT / f"prediction_{split}_{row_name}.json"
    obj = load_json(pred_path)
    rows = obj["prediction"]

    if len(rows) != 500:
        raise RuntimeError(f"{pred_path} has {len(rows)} rows, expected 500")

    candidate = summarize_candidate_ceiling(rows)
    e2e = summarize_e2e(rows)
    graph = graph_package(row_name, split)

    return {
        "split": split,
        "row_name": row_name,
        "num_examples": len(rows),
        "prediction_path": str(pred_path),
        **candidate,
        **e2e,
        **graph,
        "rr_rule": "1/adjusted_rank if adjusted_rank <= 20 else 0",
        "absent_gold_policy": "RR@20 = 0",
    }


def md_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Split", "Row", "Gold@20", "Cand MRR", "E2E MRR", "E2E H@1",
        "E2E H@3", "E2E H@10", "Pred-in-cand", "Invalid", "Top1-copy",
        "List-frag", "Avg subgraph"
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join([
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
    config = load_json(BEST_CONFIG)

    all_rows = []
    by_split = {}

    for split in SPLITS:
        split_rows = []
        for row_name in ROWS:
            row = summarize_file(split, row_name)
            split_rows.append(row)
            all_rows.append(row)

        by_split[split] = split_rows

    save_json(OUT_VALID, by_split["valid"])
    save_json(OUT_TEST, by_split["test"])

    # Week20 comparison only for test.
    selected_test = {r["row_name"]: r for r in by_split["test"]}
    comparison = {
        "decision": "WEEK20_VS_WEEK24_COMPARISON_BUILT",
        "reference_e2e_table": WEEK20_TEST_REFERENCE,
        "selected_test": by_split["test"],
        "delta_selected_minus_reference": {},
    }

    for row_name, old in WEEK20_TEST_REFERENCE.items():
        new = selected_test[row_name]
        comparison["delta_selected_minus_reference"][row_name] = {
            "delta_candidate_mrr_at20": round(new["candidate_mrr_at20"] - old["candidate_mrr_at20"], 8),
            "delta_e2e_mrr_at20": round(new["reviewer_safe_e2e_mrr_at20"] - old["e2e_mrr_at20"], 8),
        }

    save_json(OUT_COMPARISON, comparison)

    checks = {
        "soft_improves_backbone_test_e2e": (
            selected_test["soft_support_raw"]["reviewer_safe_e2e_mrr_at20"]
            > selected_test["backbone_raw"]["reviewer_safe_e2e_mrr_at20"]
        ),
        "retrieval_preserves_or_improves_soft_test_e2e": (
            selected_test["retrieval_main"]["reviewer_safe_e2e_mrr_at20"]
            >= selected_test["soft_support_raw"]["reviewer_safe_e2e_mrr_at20"]
        ),
        "retrieval_smaller_than_soft": (
            selected_test["retrieval_main"]["avg_subgraph_size"]
            < selected_test["soft_support_raw"]["avg_subgraph_size"]
        ),
        "retrieval_candidate_coverage_preserved": (
            selected_test["retrieval_main"].get("avg_candidate_coverage_preserved_rate") == 1.0
        ),
    }

    decision = (
        "PRIMEKG_FROZEN_DECODE_E2E_READY"
        if all(checks.values())
        else "PRIMEKG_FROZEN_DECODE_E2E_NEEDS_REVIEW"
    )

    summary = {
        "decision": decision,
        "selected_config": config,
        "checks": checks,
        "valid_table": by_split["valid"],
        "test_table": by_split["test"],
        "e2e_reference_comparison": comparison,
    }

    save_json(OUT_SUMMARY, summary)

    lines = []
    lines.append("# PrimeKG Selected Decoding Valid/Test Rerun")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"**{decision}**")
    lines.append("")
    lines.append("## Frozen config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(config, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Reviewer-safe final table")
    lines.append("")
    lines.append(md_table(all_rows))
    lines.append("")
    lines.append("## Week20 vs Week24 test comparison")
    lines.append("")
    lines.append("| Row | Week20 E2E MRR | Week24 E2E MRR | Delta |")
    lines.append("|---|---:|---:|---:|")
    for row_name, old in WEEK20_TEST_REFERENCE.items():
        new = selected_test[row_name]
        delta = comparison["delta_selected_minus_reference"][row_name]["delta_e2e_mrr_at20"]
        lines.append(
            f"| {row_name} | {old['e2e_mrr_at20']:.6f} | "
            f"{new['reviewer_safe_e2e_mrr_at20']:.6f} | {delta:.6f} |"
        )
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    for k, v in checks.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Metrics are reviewer-safe recomputation from prediction rows.")
    lines.append("- Raw `infer.py` MRR is audit-only.")
    lines.append("- This config was selected on valid only in Day 2.")
    lines.append("- If Week24 equals Week20, this confirms the original decoding config was already optimal under the valid sweep.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("decision =", decision)
    print("wrote valid =", OUT_VALID)
    print("wrote test =", OUT_TEST)
    print("wrote comparison =", OUT_COMPARISON)
    print("wrote summary =", OUT_SUMMARY)
    print("wrote report =", OUT_MD)
    print("")
    print(md_table(all_rows))


if __name__ == "__main__":
    main()