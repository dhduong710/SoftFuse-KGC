#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(".")
RESULT_ROOT = ROOT / "outputs" / "drkg" / "e2e_drkg_rgcn"
REPORT_DIR = ROOT / "outputs" / "drkg" / "reports"

SUMMARY_PATH = ROOT / "outputs" / "drkg" / "drkg_e2e_rgcn_summary.json"
REPORT_PATH = REPORT_DIR / "day8_drkg_e2e_rgcn_summary.md"

ROWS = {
    "backbone_raw": "dataset/setting_e_drkg/softfuse_ready/rgcn",
    "soft_support_sweep": "dataset/setting_e_drkg/e2e_soft_support_ready/rgcn_sweep_selected",
    "fuzzy_retrieval_main": "dataset/setting_e_drkg/e2e_fuzzy_retrieval_ready/rgcn",
}

TOP_K = 20
ABSENT_RANK = 21


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def norm(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    s = s.replace("</s>", " ")
    s = s.replace("<|end_of_text|>", " ")
    s = s.replace("<|eot_id|>", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip().strip("'\"`.,;: ")
    return s.lower()


def extract_answer(pred_raw: Any) -> str:
    s = "" if pred_raw is None else str(pred_raw)
    if "Answer:" in s:
        s = s.split("Answer:")[-1]
    s = s.strip()
    s = s.split("\n")[0].strip()
    s = s.strip().strip("'\"`.,;: ")
    return s


def match_candidate(pred_raw: Any, candidates: list[str]) -> tuple[str | None, int]:
    ans = extract_answer(pred_raw)
    ans_n = norm(ans)

    cand_norm = {norm(c): (c, i + 1) for i, c in enumerate(candidates)}
    if ans_n in cand_norm:
        return cand_norm[ans_n]

    raw_n = norm(pred_raw)
    matches = []
    for i, c in enumerate(candidates, start=1):
        cn = norm(c)
        if cn and re.search(r"(^|[^a-z0-9])" + re.escape(cn) + r"([^a-z0-9]|$)", raw_n):
            matches.append((c, i))

    if matches:
        return matches[0]

    return None, ABSENT_RANK


def compute_candidate_metrics(dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.array([int(r["rank"]) for r in dataset_rows], dtype=np.int64)
    present = ranks <= TOP_K
    rr = np.where(present, 1.0 / ranks, 0.0)

    sizes = [len(r.get("subgraph", [])) for r in dataset_rows]
    top1 = [r["rank_entities"][0] for r in dataset_rows if r.get("rank_entities")]
    top1_counter = Counter(top1)
    most_common = top1_counter.most_common(10)

    return {
        "num_rows": len(dataset_rows),
        "candidate_gold_at20": float(np.mean(present)) if len(ranks) else 0.0,
        "candidate_mrr_at20": float(np.mean(rr)) if len(ranks) else 0.0,
        "candidate_hits1": float(np.mean(ranks <= 1)) if len(ranks) else 0.0,
        "candidate_hits3": float(np.mean(ranks <= 3)) if len(ranks) else 0.0,
        "candidate_hits10": float(np.mean(ranks <= 10)) if len(ranks) else 0.0,
        "candidate_rank21_count": int(np.sum(ranks == ABSENT_RANK)),
        "avg_subgraph_size": float(np.mean(sizes)) if sizes else 0.0,
        "top1_dominance": float(most_common[0][1] / len(dataset_rows)) if dataset_rows and most_common else 0.0,
        "top1_most_common": [{"entity": k, "count": int(v)} for k, v in most_common],
    }


def compute_e2e_metrics(pred_rows: list[dict[str, Any]], dataset_rows: list[dict[str, Any]]):
    out_rows = []

    adjusted_ranks = []
    strict_rr = []
    invalid = 0
    pred_in_candidate = 0
    exact_match = 0

    for i, pred_row in enumerate(pred_rows):
        ds = dataset_rows[i]

        pred_raw = pred_row.get("pred", "")
        candidates = ds["rank_entities"]
        gold = ds["gold_entity"]
        gold_rank = int(ds["rank"])

        pred_candidate, pred_candidate_rank = match_candidate(pred_raw, candidates)

        if pred_candidate is None:
            invalid += 1
        else:
            pred_in_candidate += 1

        pred_is_gold = pred_candidate is not None and norm(pred_candidate) == norm(gold)

        if pred_is_gold and gold_rank <= TOP_K:
            exact_match += 1
            adjusted_rank = 1
            strict_item_rr = 1.0 / gold_rank
        else:
            strict_item_rr = 0.0
            if gold_rank > TOP_K:
                adjusted_rank = ABSENT_RANK
            else:
                if pred_candidate is None:
                    adjusted_rank = min(ABSENT_RANK, gold_rank + 1)
                elif pred_candidate_rank < gold_rank:
                    adjusted_rank = min(ABSENT_RANK, gold_rank + 1)
                else:
                    adjusted_rank = gold_rank

        adjusted_ranks.append(int(adjusted_rank))
        strict_rr.append(float(strict_item_rr))

        item = dict(ds)
        item.update({
            "pred_raw": pred_raw,
            "pred_extracted": extract_answer(pred_raw),
            "pred_candidate": pred_candidate,
            "pred_candidate_rank": int(pred_candidate_rank),
            "pred_in_candidate": pred_candidate is not None,
            "pred_is_gold": bool(pred_is_gold),
            "e2e_adjusted_rank": int(adjusted_rank),
            "e2e_adjusted_rr": float(1.0 / adjusted_rank) if adjusted_rank <= TOP_K else 0.0,
            "e2e_strict_rr": float(strict_item_rr),
        })
        out_rows.append(item)

    ranks = np.array(adjusted_ranks, dtype=np.int64)
    adj_rr = np.where(ranks <= TOP_K, 1.0 / ranks, 0.0)
    strict_rr_arr = np.array(strict_rr, dtype=np.float64)

    metrics = {
        "num_rows": len(pred_rows),
        "e2e_adjusted_mrr_at20": float(np.mean(adj_rr)) if len(ranks) else 0.0,
        "e2e_adjusted_hits1": float(np.mean(ranks <= 1)) if len(ranks) else 0.0,
        "e2e_adjusted_hits3": float(np.mean(ranks <= 3)) if len(ranks) else 0.0,
        "e2e_adjusted_hits10": float(np.mean(ranks <= 10)) if len(ranks) else 0.0,
        "e2e_adjusted_rank21_count": int(np.sum(ranks == ABSENT_RANK)),
        "strict_exact_match_rate": float(exact_match / len(pred_rows)) if pred_rows else 0.0,
        "strict_exact_match_mrr_at20": float(np.mean(strict_rr_arr)) if len(strict_rr_arr) else 0.0,
        "pred_in_candidate_rate": float(pred_in_candidate / len(pred_rows)) if pred_rows else 0.0,
        "invalid_rate": float(invalid / len(pred_rows)) if pred_rows else 0.0,
        "invalid_count": int(invalid),
        "exact_match_count": int(exact_match),
    }

    return metrics, out_rows


def locate_prediction_file(row_name: str, split: str) -> Path:
    candidates = [
        RESULT_ROOT / row_name / f"prediction_{split}.json",
        RESULT_ROOT / row_name / "checkpoint-final" / f"prediction_{split}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find prediction_{split}.json for {row_name}. Checked: {[str(x) for x in candidates]}")


def collect_one(row_name: str, split: str, dataset_path: Path) -> dict[str, Any]:
    pred_path = locate_prediction_file(row_name, split)
    dataset_file = dataset_path / f"{split}.json"

    if not dataset_file.exists():
        raise FileNotFoundError(dataset_file)

    pred_obj = read_json(pred_path)
    pred_rows = pred_obj["prediction"] if isinstance(pred_obj, dict) and "prediction" in pred_obj else pred_obj
    dataset_rows = read_json(dataset_file)

    if len(pred_rows) != len(dataset_rows):
        raise RuntimeError(
            f"Row count mismatch {row_name} {split}: pred={len(pred_rows)} dataset={len(dataset_rows)}"
        )

    candidate_metrics = compute_candidate_metrics(dataset_rows)
    e2e_metrics, scored_rows = compute_e2e_metrics(pred_rows, dataset_rows)

    out_dir = RESULT_ROOT / row_name
    write_json(scored_rows, out_dir / f"reviewer_safe_e2e_rows_{split}.json")
    write_json(
        {
            "row_name": row_name,
            "split": split,
            "prediction_path": str(pred_path),
            "candidate_metrics": candidate_metrics,
            "e2e_metrics": e2e_metrics,
        },
        out_dir / f"reviewer_safe_e2e_metrics_{split}.json",
    )

    return {
        "row_name": row_name,
        "split": split,
        "prediction_path": str(pred_path),
        **candidate_metrics,
        **e2e_metrics,
    }


def write_report(summary: dict[str, Any]) -> None:
    lines = []
    lines.append("# DRKG R-GCN E2E summary")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append("- Rows: `backbone_raw`, `soft_support_sweep`, `fuzzy_retrieval_main`")
    lines.append("- Source family: `R-GCN consistency source`")
    lines.append("")

    for split in ["valid", "test"]:
        lines.append(f"## {split.capitalize()} metrics")
        lines.append("")
        lines.append("| Row | Cand Gold@20 | Cand MRR | E2E adj MRR | E2E H@1 | E2E H@3 | E2E H@10 | Pred-in-cand | Invalid | Top1Dom | Avg graph |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in summary["by_split"][split]:
            lines.append(
                f"| {r['row_name']} | {r['candidate_gold_at20']:.3f} | {r['candidate_mrr_at20']:.6f} | "
                f"{r['e2e_adjusted_mrr_at20']:.6f} | {r['e2e_adjusted_hits1']:.3f} | "
                f"{r['e2e_adjusted_hits3']:.3f} | {r['e2e_adjusted_hits10']:.3f} | "
                f"{r['pred_in_candidate_rate']:.3f} | {r['invalid_rate']:.3f} | "
                f"{r['top1_dominance']:.3f} | {r['avg_subgraph_size']:.2f} |"
            )
        lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- DRKG is expected to be harder because R-GCN Gold@20 is only around 0.20–0.218.")
    lines.append("- Soft-support sweep should be evaluated as a modest R-GCN-source improvement, not as a claim over DistMult.")
    lines.append("- Fuzzy retrieval should preserve candidate metrics while reducing graph size from 100 to 55.")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    by_split = {"valid": [], "test": []}

    for row_name, dataset_dir in ROWS.items():
        for split in ["valid", "test"]:
            item = collect_one(row_name, split, Path(dataset_dir))
            by_split[split].append(item)

    for split in ["valid", "test"]:
        by_split[split] = sorted(
            by_split[split],
            key=lambda x: x["e2e_adjusted_mrr_at20"],
            reverse=True,
        )

    expected_rows = set(ROWS.keys())
    present_rows = {r["row_name"] for split in by_split.values() for r in split}
    missing_rows = sorted(expected_rows - present_rows)

    decision = "DAY8_DRKG_E2E_RGCN_ALL_ROWS_READY" if not missing_rows else "DAY8_DRKG_E2E_RGCN_PARTIAL_READY"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "dataset": "DRKG",
        "task": "(?, DRUGBANK::treats, disease)",
        "source": "R-GCN consistency source",
        "rows": list(ROWS.keys()),
        "missing_rows": missing_rows,
        "by_split": by_split,
        "metric_note": "Adjusted E2E rank is used for consistency with prior SoftFuse-KGC E2E reporting; strict exact-match metrics are saved in per-row JSON outputs.",
    }

    write_json(summary, SUMMARY_PATH)
    write_report(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved:\n  {SUMMARY_PATH}\n  {REPORT_PATH}")


if __name__ == "__main__":
    main()
