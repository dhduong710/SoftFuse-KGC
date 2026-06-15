#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect rule sensitivity locked-test and E2E metrics.

Inputs:
- dataset/setting_a/rule_sensitivity/{variant}/{valid,test}.json
- outputs/sensitivity/rule_sensitivity/prediction_{split}_rule_sensitivity_{variant}.json

Outputs:
- outputs/sensitivity/rule_sensitivity/rule_sensitivity_test.json
- outputs/sensitivity/rule_sensitivity/rule_sensitivity_e2e_summary.json
- outputs/sensitivity/reports/day3_rule_sensitivity_test_e2e.md
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


ROOT = Path(__file__).resolve().parents[2]

VARIANT_ROOT = ROOT / "dataset" / "setting_a" / "rule_sensitivity"
RESULTS_DIR = ROOT / "outputs" / "sensitivity" / "rule_sensitivity"
REPORTS_DIR = ROOT / "outputs" / "sensitivity" / "reports"

VARIANTS = ["main_rules", "no_rules", "random_rules"]
SPLITS = ["valid", "test"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    """
    infer.py saves prediction files as:
    {
      "args": ...,
      "generation_config": ...,
      "prediction": [...]
    }

    Older/other collectors may save a raw list directly, so support both.
    """
    obj = load_json(path)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for key in ["prediction", "predictions", "preds", "outputs"]:
            val = obj.get(key)
            if isinstance(val, list):
                return val

    raise ValueError(
        f"Cannot find prediction rows in {path}. "
        f"Expected a list or a dict with key 'prediction'."
    )


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def norm_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \n\t\r\"'")
    return s


def norm_key(x: Any) -> str:
    return norm_text(x).lower()


def safe_len(x: Any) -> int:
    return len(x) if isinstance(x, list) else 0


def extract_subgraph(row: Dict[str, Any]) -> List[Any]:
    if isinstance(row.get("subgraph"), list):
        return row["subgraph"]
    if isinstance(row.get("selected_subgraph"), list):
        return row["selected_subgraph"]
    return []


def get_candidates(row: Dict[str, Any]) -> List[str]:
    return [norm_text(x) for x in row.get("rank_entities", [])]


def get_gold(row: Dict[str, Any]) -> str:
    return norm_text(row.get("gold_entity", row.get("output", "")))


def get_pred(pred_row: Dict[str, Any]) -> str:
    return norm_text(pred_row.get("pred", pred_row.get("prediction", "")))


def find_in_candidates(name: str, candidates: List[str]) -> Optional[int]:
    nk = norm_key(name)
    for i, c in enumerate(candidates):
        if norm_key(c) == nk:
            return i
    return None


def candidate_gold_rank(row: Dict[str, Any]) -> int:
    candidates = get_candidates(row)
    gold = get_gold(row)
    idx = find_in_candidates(gold, candidates)
    if idx is None or idx >= 20:
        return 21
    return idx + 1


def rr_at20(rank: int) -> float:
    return 1.0 / rank if rank <= 20 else 0.0


def e2e_gold_rank(row: Dict[str, Any], pred: str) -> int:
    """
    Reviewer-safe E2E rank.

    If gold is absent from top-20, RR is zero.
    If pred == gold, the LLM places gold at rank 1.
    Otherwise, use the original candidate gold rank, shifted by one if the
    generated prediction is invalid or appears at/after the gold rank according
    to the DrKGC-style evaluation behavior.
    """
    candidates = get_candidates(row)
    gold = get_gold(row)

    gold_idx = find_in_candidates(gold, candidates)
    if gold_idx is None or gold_idx >= 20:
        return 21

    gold_rank = gold_idx + 1

    if norm_key(pred) == norm_key(gold):
        return 1

    pred_idx = find_in_candidates(pred, candidates)

    # This mirrors the adapted infer.py behavior:
    # wrong prediction can occupy one rank before the gold unless it is clearly
    # already ranked below the gold.
    if pred_idx is None or pred_idx >= gold_idx:
        shifted = gold_rank + 1
    else:
        shifted = gold_rank

    return shifted if shifted <= 20 else 21


def is_pred_in_candidate(row: Dict[str, Any], pred: str) -> bool:
    return find_in_candidates(pred, get_candidates(row)) is not None


def is_top1_copy(row: Dict[str, Any], pred: str) -> bool:
    candidates = get_candidates(row)
    return bool(candidates) and norm_key(pred) == norm_key(candidates[0])


def is_list_fragment(row: Dict[str, Any], pred: str) -> bool:
    p = norm_text(pred)
    if not p:
        return True

    marker_count = 0
    for c in get_candidates(row):
        if c and norm_key(c) in norm_key(p):
            marker_count += 1

    if marker_count >= 2:
        return True

    suspicious = [
        ",",
        ";",
        "['",
        "']",
        "answer must be",
        "question:",
        "candidate",
        "entity",
    ]
    pl = p.lower()
    return any(x in pl for x in suspicious)


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def round6(x: float) -> float:
    return round(float(x), 6)


def summarize_split_variant(
    split: str,
    variant: str,
    rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if len(rows) != len(pred_rows):
        raise RuntimeError(
            f"{split}/{variant}: row count mismatch dataset={len(rows)} preds={len(pred_rows)}"
        )

    cand_ranks = []
    e2e_ranks = []
    preds = []

    pred_in_cand = []
    invalid = []
    top1_copy = []
    list_frag = []
    sub_sizes = []

    detailed = []

    for idx, (row, pred_row) in enumerate(zip(rows, pred_rows)):
        pred = get_pred(pred_row)
        gold = get_gold(row)

        c_rank = candidate_gold_rank(row)
        e_rank = e2e_gold_rank(row, pred)

        in_cand = is_pred_in_candidate(row, pred)
        is_top1 = is_top1_copy(row, pred)
        is_frag = is_list_fragment(row, pred)

        cand_ranks.append(c_rank)
        e2e_ranks.append(e_rank)
        preds.append(pred)
        pred_in_cand.append(1.0 if in_cand else 0.0)
        invalid.append(0.0 if in_cand else 1.0)
        top1_copy.append(1.0 if is_top1 else 0.0)
        list_frag.append(1.0 if is_frag else 0.0)
        sub_sizes.append(safe_len(extract_subgraph(row)))

        detailed.append({
            "row_index": idx,
            "query_entity": row.get("query_entity"),
            "gold_entity": gold,
            "pred": pred,
            "candidate_gold_rank": c_rank,
            "e2e_gold_rank": e_rank,
            "candidate_rr_at20": rr_at20(c_rank),
            "e2e_rr_at20": rr_at20(e_rank),
            "pred_in_candidate": in_cand,
            "invalid": not in_cand,
            "top1_copy": is_top1,
            "list_fragment": is_frag,
            "avg_graph_size_item": safe_len(extract_subgraph(row)),
            "top5_candidates": get_candidates(row)[:5],
        })

    summary = {
        "split": split,
        "variant": variant,
        "num_rows": len(rows),

        "gold_at20": round6(mean([1.0 if r <= 20 else 0.0 for r in cand_ranks])),
        "candidate_mrr_at20": round6(mean([rr_at20(r) for r in cand_ranks])),
        "candidate_hits1_at20": round6(mean([1.0 if r <= 1 else 0.0 for r in cand_ranks])),
        "candidate_hits3_at20": round6(mean([1.0 if r <= 3 else 0.0 for r in cand_ranks])),
        "candidate_hits10_at20": round6(mean([1.0 if r <= 10 else 0.0 for r in cand_ranks])),

        "e2e_mrr_at20": round6(mean([rr_at20(r) for r in e2e_ranks])),
        "e2e_hits1_at20": round6(mean([1.0 if r <= 1 else 0.0 for r in e2e_ranks])),
        "e2e_hits3_at20": round6(mean([1.0 if r <= 3 else 0.0 for r in e2e_ranks])),
        "e2e_hits10_at20": round6(mean([1.0 if r <= 10 else 0.0 for r in e2e_ranks])),

        "rank21_count_candidate": int(sum(1 for r in cand_ranks if r == 21)),
        "rank21_count_e2e": int(sum(1 for r in e2e_ranks if r == 21)),

        "pred_in_candidate_rate": round6(mean(pred_in_cand)),
        "invalid_rate": round6(mean(invalid)),
        "top1_copy_rate": round6(mean(top1_copy)),
        "list_fragment_rate": round6(mean(list_frag)),

        "avg_graph_size": round6(mean([float(x) for x in sub_sizes])),
        "min_graph_size": int(min(sub_sizes)) if sub_sizes else None,
        "max_graph_size": int(max(sub_sizes)) if sub_sizes else None,
    }

    return summary, detailed


def add_delta_vs_main(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_split = {}
    for s in summaries:
        by_split.setdefault(s["split"], {})[s["variant"]] = s

    out = []
    for s in summaries:
        main = by_split[s["split"]]["main_rules"]
        ss = dict(s)
        for key in [
            "candidate_mrr_at20",
            "e2e_mrr_at20",
            "e2e_hits3_at20",
            "e2e_hits10_at20",
            "pred_in_candidate_rate",
            "invalid_rate",
            "top1_copy_rate",
            "avg_graph_size",
        ]:
            ss[f"delta_{key}_vs_main"] = round6(ss[key] - main[key])
        out.append(ss)
    return out


def collect_cases(
    detailed_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]],
    max_cases: int = 10,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for split in SPLITS:
        main = detailed_by_key[(split, "main_rules")]
        for variant in ["no_rules", "random_rules"]:
            rows = detailed_by_key[(split, variant)]
            cases = []
            for m, r in zip(main, rows):
                changed = norm_key(m["pred"]) != norm_key(r["pred"])
                rank_delta = r["e2e_gold_rank"] - m["e2e_gold_rank"]
                if changed or rank_delta != 0 or r["invalid"]:
                    cases.append({
                        "split": split,
                        "variant": variant,
                        "row_index": r["row_index"],
                        "query_entity": r["query_entity"],
                        "gold_entity": r["gold_entity"],
                        "main_pred": m["pred"],
                        "variant_pred": r["pred"],
                        "main_e2e_rank": m["e2e_gold_rank"],
                        "variant_e2e_rank": r["e2e_gold_rank"],
                        "rank_delta_variant_minus_main": rank_delta,
                        "main_top1_copy": m["top1_copy"],
                        "variant_top1_copy": r["top1_copy"],
                        "variant_invalid": r["invalid"],
                        "top5_candidates": r["top5_candidates"],
                    })

            cases = sorted(
                cases,
                key=lambda x: (
                    abs(x["rank_delta_variant_minus_main"]),
                    int(x["variant_invalid"]),
                    int(x["main_pred"] != x["variant_pred"]),
                ),
                reverse=True,
            )
            out[f"{split}_{variant}"] = cases[:max_cases]

    return out


def md_escape(x: Any) -> str:
    s = "" if x is None else str(x)
    return s.replace("|", "\\|").replace("\n", " ")


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = []
    lines.append("| " + " | ".join(md_escape(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(md_escape(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(result: Dict[str, Any], path: Path) -> None:
    lines = []
    lines.append("# Rule Sensitivity Locked Test and E2E\n")
    lines.append(f"- Decision: **{result['decision']}**")
    lines.append(f"- Created at: `{result['created_at']}`")
    lines.append("- Primary model: **Llama-3.2-3B**")
    lines.append("- Selected decoding: **cfg01_mnt16_rp100_ng0**")
    lines.append("- Policy: reviewer-safe RR@20, no test tuning")
    lines.append("")

    for split in SPLITS:
        lines.append(f"## {split.upper()} summary\n")
        table_rows = []
        for s in result["summaries"]:
            if s["split"] != split:
                continue
            table_rows.append([
                s["variant"],
                s["gold_at20"],
                s["candidate_mrr_at20"],
                s["e2e_mrr_at20"],
                s["e2e_hits3_at20"],
                s["e2e_hits10_at20"],
                s["pred_in_candidate_rate"],
                s["invalid_rate"],
                s["top1_copy_rate"],
                s["avg_graph_size"],
                s["delta_e2e_mrr_at20_vs_main"],
            ])
        lines.append(md_table(
            [
                "Variant",
                "Gold@20",
                "Cand MRR@20",
                "E2E MRR@20",
                "E2E H@3",
                "E2E H@10",
                "Pred-in-cand",
                "Invalid",
                "Top1-copy",
                "Avg graph",
                "� E2E MRR vs main",
            ],
            table_rows,
        ))
        lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append(
        "- `main_rules` remains the selected E2E main graph package."
    )
    lines.append(
        "- `no_rules` uses the larger soft-support source graph. If its E2E score is close to main, "
        "SoftFuse is not overly dependent on a single hard-coded rule package."
    )
    lines.append(
        "- `random_rules` is the negative control. If it is worse or less stable, that supports the "
        "value of confidence-aware evidence selection. If it is close, report it honestly as evidence "
        "that candidate ordering dominates E2E behavior."
    )
    lines.append(
        "- These rows are appendix sensitivity evidence and do not replace the locked main result."
    )
    lines.append("")

    lines.append("## Changed / problematic cases\n")
    for key, cases in result["case_samples"].items():
        lines.append(f"### {key}")
        if not cases:
            lines.append("- No changed/problematic cases found.")
            lines.append("")
            continue
        rows = []
        for c in cases:
            rows.append([
                c["row_index"],
                c["query_entity"],
                c["gold_entity"],
                c["main_pred"],
                c["variant_pred"],
                c["main_e2e_rank"],
                c["variant_e2e_rank"],
                c["rank_delta_variant_minus_main"],
                c["variant_invalid"],
            ])
        lines.append(md_table(
            [
                "idx",
                "query",
                "gold",
                "main pred",
                "variant pred",
                "main rank",
                "variant rank",
                "rank delta",
                "variant invalid",
            ],
            rows,
        ))
        lines.append("")

    lines.append("## Final decision\n")
    lines.append(f"**{result['decision']}**")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fatal_errors = []
    summaries = []
    detailed_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for split in SPLITS:
        for variant in VARIANTS:
            data_path = VARIANT_ROOT / variant / f"{split}.json"
            pred_path = RESULTS_DIR / f"prediction_{split}_rule_sensitivity_{variant}.json"

            if not data_path.exists():
                fatal_errors.append(f"Missing dataset file: {rel(data_path)}")
                continue
            if not pred_path.exists():
                fatal_errors.append(f"Missing prediction file: {rel(pred_path)}")
                continue

            try:
                rows = load_json(data_path)
                pred_rows = load_prediction_rows(pred_path)
                summary, detailed = summarize_split_variant(split, variant, rows, pred_rows)
                summaries.append(summary)
                detailed_by_key[(split, variant)] = detailed
            except Exception as e:
                fatal_errors.append(f"{split}/{variant}: {repr(e)}")

    if fatal_errors:
        result = {
            "week": 25,
            "day": 3,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "decision": "RULE_SENSITIVITY_TEST_E2E_BLOCKED",
            "fatal_errors": fatal_errors,
        }
        write_json(result, RESULTS_DIR / "rule_sensitivity_e2e_summary.json")
        print("decision =", result["decision"])
        print("fatal_errors:")
        for e in fatal_errors:
            print("-", e)
        return

    summaries = add_delta_vs_main(summaries)
    case_samples = collect_cases(detailed_by_key)

    result = {
        "week": 25,
        "day": 3,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "RULE_SENSITIVITY_TEST_E2E_READY",
        "primary_model": "Llama-3.2-3B",
        "selected_decoding": "cfg01_mnt16_rp100_ng0",
        "reviewer_safe_policy": "RR=1/rank if rank<=20 else 0; rank21 descriptive only",
        "summaries": summaries,
        "case_samples": case_samples,
        "fatal_errors": [],
        "notes": [
            "Rule sensitivity variants are appendix evidence only.",
            "Do not use these rows to replace the Week24 main result.",
            "Candidate MRR can remain unchanged because candidate order is fixed.",
        ],
    }

    # Separate locked-test JSON for convenience.
    test_only = {
        "week": 25,
        "day": 3,
        "split": "test",
        "decision": result["decision"],
        "summaries": [s for s in summaries if s["split"] == "test"],
        "case_samples": {k: v for k, v in case_samples.items() if k.startswith("test_")},
    }

    out_summary = RESULTS_DIR / "rule_sensitivity_e2e_summary.json"
    out_test = RESULTS_DIR / "rule_sensitivity_test.json"
    out_report = REPORTS_DIR / "day3_rule_sensitivity_test_e2e.md"

    write_json(result, out_summary)
    write_json(test_only, out_test)
    write_report(result, out_report)

    print("=" * 100)
    print("decision =", result["decision"])
    print("summary_json =", rel(out_summary))
    print("test_json =", rel(out_test))
    print("report_md =", rel(out_report))
    print("=" * 100)

    for split in SPLITS:
        print(f"[{split}]")
        for s in summaries:
            if s["split"] != split:
                continue
            print(
                s["variant"],
                "Gold@20 =", s["gold_at20"],
                "CandMRR =", s["candidate_mrr_at20"],
                "E2E_MRR =", s["e2e_mrr_at20"],
                "H@10 =", s["e2e_hits10_at20"],
                "PredInCand =", s["pred_in_candidate_rate"],
                "Invalid =", s["invalid_rate"],
                "Top1Copy =", s["top1_copy_rate"],
                "AvgGraph =", s["avg_graph_size"],
                "DeltaE2E =", s["delta_e2e_mrr_at20_vs_main"],
            )
        print("-" * 100)


if __name__ == "__main__":
    main()