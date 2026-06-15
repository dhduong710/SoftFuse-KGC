#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect question-template sensitivity metrics.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


ROOT = Path(__file__).resolve().parents[2]

VARIANT_ROOT = ROOT / "dataset" / "setting_a" / "template_sensitivity"
RESULTS_DIR = ROOT / "outputs" / "sensitivity" / "template_sensitivity"
REPORTS_DIR = ROOT / "outputs" / "sensitivity" / "reports"

VARIANTS = ["T0_canonical", "T1_treatment", "T2_medication", "T3_association_neutral"]
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
    candidates = get_candidates(row)
    gold = get_gold(row)

    gold_idx = find_in_candidates(gold, candidates)
    if gold_idx is None or gold_idx >= 20:
        return 21

    gold_rank = gold_idx + 1

    if norm_key(pred) == norm_key(gold):
        return 1

    pred_idx = find_in_candidates(pred, candidates)

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
    pkey = norm_key(p)
    for c in get_candidates(row):
        ckey = norm_key(c)
        if ckey and ckey in pkey:
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


def extract_subgraph(row: Dict[str, Any]) -> List[Any]:
    if isinstance(row.get("subgraph"), list):
        return row["subgraph"]
    if isinstance(row.get("selected_subgraph"), list):
        return row["selected_subgraph"]
    return []


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
    pred_in_cand = []
    invalid = []
    top1_copy = []
    list_frag = []
    sub_sizes = []
    detailed = []

    for idx, (row, pred_row) in enumerate(zip(rows, pred_rows)):
        pred = get_pred(pred_row)

        c_rank = candidate_gold_rank(row)
        e_rank = e2e_gold_rank(row, pred)

        in_cand = is_pred_in_candidate(row, pred)
        top1 = is_top1_copy(row, pred)
        frag = is_list_fragment(row, pred)

        cand_ranks.append(c_rank)
        e2e_ranks.append(e_rank)
        pred_in_cand.append(1.0 if in_cand else 0.0)
        invalid.append(0.0 if in_cand else 1.0)
        top1_copy.append(1.0 if top1 else 0.0)
        list_frag.append(1.0 if frag else 0.0)
        sub_sizes.append(safe_len(extract_subgraph(row)))

        detailed.append({
            "row_index": idx,
            "query_entity": row.get("query_entity"),
            "question_text": row.get("question_text"),
            "gold_entity": get_gold(row),
            "pred": pred,
            "candidate_gold_rank": c_rank,
            "e2e_gold_rank": e_rank,
            "candidate_rr_at20": rr_at20(c_rank),
            "e2e_rr_at20": rr_at20(e_rank),
            "pred_in_candidate": in_cand,
            "invalid": not in_cand,
            "top1_copy": top1,
            "list_fragment": frag,
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
    }

    return summary, detailed


def add_delta_vs_t0(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_split = {}
    for s in summaries:
        by_split.setdefault(s["split"], {})[s["variant"]] = s

    out = []
    for s in summaries:
        t0 = by_split[s["split"]]["T0_canonical"]
        ss = dict(s)

        for key in [
            "candidate_mrr_at20",
            "e2e_mrr_at20",
            "e2e_hits3_at20",
            "e2e_hits10_at20",
            "pred_in_candidate_rate",
            "invalid_rate",
            "list_fragment_rate",
            "top1_copy_rate",
        ]:
            ss[f"delta_{key}_vs_T0"] = round6(ss[key] - t0[key])

        out.append(ss)

    return out


def add_prediction_change_rates(
    summaries: List[Dict[str, Any]],
    detailed_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    summary_map = {(s["split"], s["variant"]): dict(s) for s in summaries}

    for split in SPLITS:
        t0_rows = detailed_by_key[(split, "T0_canonical")]

        for variant in VARIANTS:
            rows = detailed_by_key[(split, variant)]
            changed = []
            same_rank = []

            for a, b in zip(t0_rows, rows):
                changed.append(1.0 if norm_key(a["pred"]) != norm_key(b["pred"]) else 0.0)
                same_rank.append(1.0 if a["e2e_gold_rank"] == b["e2e_gold_rank"] else 0.0)

            summary_map[(split, variant)]["prediction_change_rate_vs_T0"] = round6(mean(changed))
            summary_map[(split, variant)]["same_e2e_rank_rate_vs_T0"] = round6(mean(same_rank))

    return [summary_map[(s["split"], s["variant"])] for s in summaries]


def collect_cases(
    detailed_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]],
    max_cases: int = 10,
) -> Dict[str, Any]:
    out = {}

    for split in SPLITS:
        t0 = detailed_by_key[(split, "T0_canonical")]

        for variant in VARIANTS:
            if variant == "T0_canonical":
                continue

            rows = detailed_by_key[(split, variant)]
            cases = []

            for a, b in zip(t0, rows):
                pred_changed = norm_key(a["pred"]) != norm_key(b["pred"])
                rank_delta = b["e2e_gold_rank"] - a["e2e_gold_rank"]

                if pred_changed or rank_delta != 0 or b["invalid"]:
                    cases.append({
                        "split": split,
                        "variant": variant,
                        "row_index": b["row_index"],
                        "query_entity": b["query_entity"],
                        "gold_entity": b["gold_entity"],
                        "T0_question": a["question_text"],
                        "variant_question": b["question_text"],
                        "T0_pred": a["pred"],
                        "variant_pred": b["pred"],
                        "T0_rank": a["e2e_gold_rank"],
                        "variant_rank": b["e2e_gold_rank"],
                        "rank_delta_variant_minus_T0": rank_delta,
                        "variant_invalid": b["invalid"],
                        "variant_list_fragment": b["list_fragment"],
                        "top5_candidates": b["top5_candidates"],
                    })

            cases = sorted(
                cases,
                key=lambda x: (
                    abs(x["rank_delta_variant_minus_T0"]),
                    int(x["variant_invalid"]),
                    int(x["T0_pred"] != x["variant_pred"]),
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

    lines.append("# Question-Template Sensitivity\n")
    lines.append(f"- Decision: **{result['decision']}**")
    lines.append(f"- Created at: `{result['created_at']}`")
    lines.append("- Main row: **retrieval_main**")
    lines.append("- Primary model: **Llama-3.2-3B**")
    lines.append("- Selected decoding: **cfg01_mnt16_rp100_ng0**")
    lines.append("- Policy: reviewer-safe RR@20, no test tuning")
    lines.append("")

    lines.append("## Template variants\n")
    template_rows = []
    for k, v in result["templates"].items():
        template_rows.append([k, v])
    lines.append(md_table(["Variant", "Template"], template_rows))
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
                s["list_fragment_rate"],
                s["top1_copy_rate"],
                s["prediction_change_rate_vs_T0"],
                s["delta_e2e_mrr_at20_vs_T0"],
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
                "List-frag",
                "Top1-copy",
                "Pred change vs T0",
                "� E2E MRR vs T0",
            ],
            table_rows,
        ))
        lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append(
        "- Candidate metrics should remain identical across templates because candidate lists are fixed."
    )
    lines.append(
        "- If E2E MRR and invalid rate vary only slightly, the system is not strongly prompt-template sensitive."
    )
    lines.append(
        "- T0 remains the canonical PrimeKG indication prompt. T3 is robustness-only and should not redefine the task."
    )
    lines.append("")

    lines.append("## Changed/problematic cases\n")
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
                c["T0_pred"],
                c["variant_pred"],
                c["T0_rank"],
                c["variant_rank"],
                c["rank_delta_variant_minus_T0"],
                c["variant_invalid"],
            ])
        lines.append(md_table(
            [
                "idx",
                "query",
                "gold",
                "T0 pred",
                "variant pred",
                "T0 rank",
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

    variants_path = RESULTS_DIR / "template_variants.json"
    templates = {}
    if variants_path.exists():
        variant_manifest = load_json(variants_path)
        templates = {
            k: v["template"]
            for k, v in variant_manifest.get("template_variants", {}).items()
        }

    fatal_errors = []
    summaries = []
    detailed_by_key = {}

    for split in SPLITS:
        for variant in VARIANTS:
            data_path = VARIANT_ROOT / variant / f"{split}.json"
            pred_path = RESULTS_DIR / f"prediction_{split}_template_sensitivity_{variant}.json"

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
            "day": 4,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "decision": "QUESTION_TEMPLATE_SENSITIVITY_BLOCKED",
            "fatal_errors": fatal_errors,
        }
        write_json(result, RESULTS_DIR / "template_sensitivity_e2e_summary.json")
        print("decision =", result["decision"])
        print("fatal_errors:")
        for e in fatal_errors:
            print("-", e)
        return

    summaries = add_delta_vs_t0(summaries)
    summaries = add_prediction_change_rates(summaries, detailed_by_key)

    case_samples = collect_cases(detailed_by_key)

    result = {
        "week": 25,
        "day": 4,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "QUESTION_TEMPLATE_SENSITIVITY_READY",
        "primary_model": "Llama-3.2-3B",
        "selected_decoding": "cfg01_mnt16_rp100_ng0",
        "main_row": "retrieval_main",
        "reviewer_safe_policy": "RR=1/rank if rank<=20 else 0",
        "templates": templates,
        "summaries": summaries,
        "case_samples": case_samples,
        "fatal_errors": [],
        "notes": [
            "Only question wording is changed.",
            "Candidate ranking and subgraph are fixed.",
            "T0_canonical remains the main prompt.",
            "T3_association_neutral is robustness-only.",
        ],
    }

    valid_only = {
        "week": 25,
        "day": 4,
        "split": "valid",
        "decision": result["decision"],
        "summaries": [s for s in summaries if s["split"] == "valid"],
        "case_samples": {k: v for k, v in case_samples.items() if k.startswith("valid_")},
    }

    test_only = {
        "week": 25,
        "day": 4,
        "split": "test",
        "decision": result["decision"],
        "summaries": [s for s in summaries if s["split"] == "test"],
        "case_samples": {k: v for k, v in case_samples.items() if k.startswith("test_")},
    }

    out_summary = RESULTS_DIR / "template_sensitivity_e2e_summary.json"
    out_valid = RESULTS_DIR / "template_sensitivity_valid.json"
    out_test = RESULTS_DIR / "template_sensitivity_test.json"
    out_report = REPORTS_DIR / "day4_template_sensitivity.md"

    write_json(result, out_summary)
    write_json(valid_only, out_valid)
    write_json(test_only, out_test)
    write_report(result, out_report)

    print("=" * 100)
    print("decision =", result["decision"])
    print("summary_json =", rel(out_summary))
    print("valid_json =", rel(out_valid))
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
                "ListFrag =", s["list_fragment_rate"],
                "Top1Copy =", s["top1_copy_rate"],
                "PredChangeVsT0 =", s["prediction_change_rate_vs_T0"],
                "DeltaE2E =", s["delta_e2e_mrr_at20_vs_T0"],
            )
        print("-" * 100)


if __name__ == "__main__":
    main()