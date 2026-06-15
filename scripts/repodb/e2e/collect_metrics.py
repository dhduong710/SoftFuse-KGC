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
RESULT_ROOT = ROOT / "outputs" / "repodb" / "e2e_repodb_rgcn"
REPORT_DIR = ROOT / "outputs" / "repodb" / "reports"

SUMMARY_PATH = ROOT / "outputs" / "repodb" / "repodb_e2e_rgcn_summary.json"
REPORT_PATH = REPORT_DIR / "day8_repodb_e2e_rgcn_summary.md"

ROWS = {
    "backbone_raw": "dataset/setting_f_repodb/e2e_soft_support_ready/rgcn_raw_display_control",
    "soft_support_sweep": "dataset/setting_f_repodb/e2e_soft_support_ready/rgcn_sweep_selected",
    "fuzzy_retrieval_main": "dataset/setting_f_repodb/e2e_fuzzy_retrieval_ready/rgcn",
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


def norm(x: Any) -> str:
    s = "" if x is None else str(x)
    s = s.replace("</s>", " ")
    s = s.replace("<|end_of_text|>", " ")
    s = s.replace("<|eot_id|>", " ")
    s = re.sub(r"\s+", " ", s)
    s = s.strip().strip("'\"`.,;:()[]{} ")
    return s.lower()


def extract_answer(pred_raw: Any) -> str:
    s = "" if pred_raw is None else str(pred_raw)

    for key in ["Answer:", "answer:", "ANSWER:"]:
        if key in s:
            s = s.split(key)[-1]

    s = s.strip()
    s = s.split("\n")[0].strip()
    s = s.strip().strip("'\"`.,;:()[]{} ")
    return s


def get_prediction_text(pred_row: Any) -> str:
    if isinstance(pred_row, str):
        return pred_row

    if isinstance(pred_row, dict):
        for k in [
            "pred",
            "prediction",
            "generated",
            "generated_text",
            "pred_raw",
            "output",
            "text",
            "response",
        ]:
            if k in pred_row:
                return str(pred_row[k])

    return str(pred_row)


def load_prediction_rows(path: Path) -> list[Any]:
    obj = read_json(path)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, dict):
        for k in ["prediction", "predictions", "results", "rows", "data"]:
            if k in obj and isinstance(obj[k], list):
                return obj[k]

    raise RuntimeError(f"Unsupported prediction JSON format: {path}")


def match_candidate(pred_raw: Any, row: dict[str, Any]) -> tuple[str | None, str | None, int]:
    pred_text = get_prediction_text(pred_raw)
    ans = extract_answer(pred_text)

    displays = list(row.get("rank_entities", []))
    canonicals = list(row.get("rank_entities_canonical", []))

    # Ensure same length.
    if len(canonicals) != len(displays):
        canonicals = [None] * len(displays)

    ans_n = norm(ans)
    raw_n = norm(pred_text)

    # 1. Exact display match.
    for i, disp in enumerate(displays, start=1):
        if ans_n == norm(disp):
            canonical = canonicals[i - 1] if canonicals[i - 1] else None
            return disp, canonical, i

    # 2. Exact canonical match, just in case model emits DBID/canonical form.
    for i, can in enumerate(canonicals, start=1):
        if can and ans_n == norm(can):
            return displays[i - 1], can, i

    # 3. Containment display match.
    for i, disp in enumerate(displays, start=1):
        dn = norm(disp)
        if dn and re.search(r"(^|[^a-z0-9])" + re.escape(dn) + r"([^a-z0-9]|$)", raw_n):
            canonical = canonicals[i - 1] if canonicals[i - 1] else None
            return disp, canonical, i

    # 4. Containment canonical match.
    for i, can in enumerate(canonicals, start=1):
        if not can:
            continue
        cn = norm(can)
        if cn and re.search(r"(^|[^a-z0-9])" + re.escape(cn) + r"([^a-z0-9]|$)", raw_n):
            return displays[i - 1], can, i

    return None, None, ABSENT_RANK


def compute_candidate_metrics(dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.array([int(r["rank"]) for r in dataset_rows], dtype=np.int64)
    present = ranks <= TOP_K
    rr = np.where(present, 1.0 / ranks, 0.0)

    top1 = []
    for r in dataset_rows:
        if r.get("rank_entities_canonical"):
            top1.append(r["rank_entities_canonical"][0])
        elif r.get("rank_entities"):
            top1.append(r["rank_entities"][0])

    c = Counter(top1)
    mc = c.most_common(10)

    sizes = [len(r.get("subgraph", [])) for r in dataset_rows]

    return {
        "num_rows": len(dataset_rows),
        "candidate_gold_at20": float(np.mean(present)) if len(ranks) else 0.0,
        "candidate_mrr_at20": float(np.mean(rr)) if len(ranks) else 0.0,
        "candidate_hits1": float(np.mean(ranks <= 1)) if len(ranks) else 0.0,
        "candidate_hits3": float(np.mean(ranks <= 3)) if len(ranks) else 0.0,
        "candidate_hits10": float(np.mean(ranks <= 10)) if len(ranks) else 0.0,
        "candidate_hits20": float(np.mean(ranks <= 20)) if len(ranks) else 0.0,
        "candidate_rank21_count": int(np.sum(ranks == ABSENT_RANK)),
        "avg_subgraph_size": float(np.mean(sizes)) if sizes else 0.0,
        "top1_dominance": float(mc[0][1] / len(dataset_rows)) if dataset_rows and mc else 0.0,
        "top1_most_common": [{"entity": k, "count": int(v)} for k, v in mc],
    }


def adjusted_rank_from_prediction(
    pred_is_gold: bool,
    pred_candidate_rank: int,
    gold_rank: int,
) -> int:
    gold_rank = int(gold_rank)

    if pred_is_gold and gold_rank <= TOP_K:
        return 1

    if gold_rank > TOP_K:
        return ABSENT_RANK

    if pred_candidate_rank == ABSENT_RANK:
        return min(ABSENT_RANK, gold_rank + 1)

    if pred_candidate_rank < gold_rank:
        return min(ABSENT_RANK, gold_rank + 1)

    return gold_rank


def compute_e2e_metrics(pred_rows: list[Any], dataset_rows: list[dict[str, Any]]):
    scored_rows = []

    adjusted_ranks = []
    pred_in_candidate = 0
    invalid = 0
    exact_match = 0

    for pred_obj, row in zip(pred_rows, dataset_rows):
        pred_raw = get_prediction_text(pred_obj)
        pred_extracted = extract_answer(pred_raw)

        pred_display, pred_canonical, pred_rank = match_candidate(pred_raw, row)

        if pred_display is None:
            invalid += 1
        else:
            pred_in_candidate += 1

        gold_display = row.get("gold_display") or row.get("output") or row.get("gold_name") or row["gold_entity"]
        gold_canonical = row.get("gold_entity_canonical") or row.get("gold_entity")

        pred_is_gold = False
        if pred_display is not None:
            if norm(pred_display) == norm(gold_display):
                pred_is_gold = True
            if pred_canonical is not None and norm(pred_canonical) == norm(gold_canonical):
                pred_is_gold = True

        if pred_is_gold:
            exact_match += 1

        gold_rank = int(row["rank"])
        adjusted_rank = adjusted_rank_from_prediction(
            pred_is_gold=pred_is_gold,
            pred_candidate_rank=int(pred_rank),
            gold_rank=gold_rank,
        )

        adjusted_ranks.append(adjusted_rank)

        out = dict(row)
        out.update({
            "pred_raw": pred_raw,
            "pred_extracted": pred_extracted,
            "pred_candidate": pred_display,
            "pred_candidate_canonical": pred_canonical,
            "pred_candidate_rank": int(pred_rank),
            "pred_in_candidate": pred_display is not None,
            "pred_is_gold": bool(pred_is_gold),
            "e2e_adjusted_rank": int(adjusted_rank),
            "e2e_adjusted_rr": float(1.0 / adjusted_rank) if adjusted_rank <= TOP_K else 0.0,
        })
        scored_rows.append(out)

    ranks = np.array(adjusted_ranks, dtype=np.int64)
    rr = np.where(ranks <= TOP_K, 1.0 / ranks, 0.0)

    metrics = {
        "num_rows": len(dataset_rows),
        "e2e_mrr_at20": float(np.mean(rr)) if len(ranks) else 0.0,
        "e2e_hits1": float(np.mean(ranks <= 1)) if len(ranks) else 0.0,
        "e2e_hits3": float(np.mean(ranks <= 3)) if len(ranks) else 0.0,
        "e2e_hits10": float(np.mean(ranks <= 10)) if len(ranks) else 0.0,
        "e2e_hits20": float(np.mean(ranks <= 20)) if len(ranks) else 0.0,
        "e2e_rank21_count": int(np.sum(ranks == ABSENT_RANK)),
        "pred_in_candidate_rate": float(pred_in_candidate / len(dataset_rows)) if dataset_rows else 0.0,
        "invalid_rate": float(invalid / len(dataset_rows)) if dataset_rows else 0.0,
        "invalid_count": int(invalid),
        "exact_match_count": int(exact_match),
        "exact_match_rate": float(exact_match / len(dataset_rows)) if dataset_rows else 0.0,
    }

    return metrics, scored_rows


def locate_prediction_file(row_name: str, split: str) -> Path:
    candidates = [
        RESULT_ROOT / row_name / f"prediction_{split}.json",
        RESULT_ROOT / row_name / "checkpoint-final" / f"prediction_{split}.json",
        RESULT_ROOT / row_name / f"predictions_{split}.json",
        RESULT_ROOT / row_name / "checkpoint-final" / f"predictions_{split}.json",
    ]

    for p in candidates:
        if p.exists():
            return p

    # Fallback recursive search.
    row_dir = RESULT_ROOT / row_name
    found = sorted(row_dir.rglob(f"*{split}*.json")) if row_dir.exists() else []
    found = [p for p in found if "prediction" in p.name or "predictions" in p.name]
    if found:
        return found[0]

    raise FileNotFoundError(f"Cannot find prediction file for row={row_name}, split={split}")


def collect_one(row_name: str, dataset_dir: Path, split: str) -> dict[str, Any]:
    dataset_file = dataset_dir / f"{split}.json"
    pred_file = locate_prediction_file(row_name, split)

    dataset_rows = read_json(dataset_file)
    pred_rows = load_prediction_rows(pred_file)

    if len(dataset_rows) != len(pred_rows):
        raise RuntimeError(
            f"Row count mismatch {row_name} {split}: dataset={len(dataset_rows)} pred={len(pred_rows)}"
        )

    candidate_metrics = compute_candidate_metrics(dataset_rows)
    e2e_metrics, scored_rows = compute_e2e_metrics(pred_rows, dataset_rows)

    out_dir = RESULT_ROOT / row_name
    write_json(scored_rows, out_dir / f"reviewer_safe_e2e_rows_{split}.json")
    write_json(
        {
            "row_name": row_name,
            "split": split,
            "dataset_path": str(dataset_dir),
            "prediction_path": str(pred_file),
            "candidate_metrics": candidate_metrics,
            "e2e_metrics": e2e_metrics,
        },
        out_dir / f"reviewer_safe_e2e_metrics_{split}.json",
    )

    return {
        "row_name": row_name,
        "split": split,
        "dataset_path": str(dataset_dir),
        "prediction_path": str(pred_file),
        **candidate_metrics,
        **e2e_metrics,
    }


def write_report(summary: dict[str, Any]) -> None:
    lines = []
    lines.append("# repoDB R-GCN E2E summary")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append("- Dataset: `repoDB`")
    lines.append("- Task: `(?, repoDB_approved_indication, disease)`")
    lines.append("- Rows: `backbone_raw`, `soft_support_sweep`, `fuzzy_retrieval_main`")
    lines.append("")

    for split in ["valid", "test"]:
        lines.append(f"## {split.capitalize()} metrics")
        lines.append("")
        lines.append("| Row | Cand Gold@20 | Cand MRR | E2E MRR | E2E H@1 | E2E H@3 | E2E H@10 | Pred-in-cand | Invalid | Exact match | Top1Dom | Avg graph |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for r in summary["by_split"][split]:
            lines.append(
                f"| {r['row_name']} | {r['candidate_gold_at20']:.3f} | "
                f"{r['candidate_mrr_at20']:.6f} | {r['e2e_mrr_at20']:.6f} | "
                f"{r['e2e_hits1']:.3f} | {r['e2e_hits3']:.3f} | {r['e2e_hits10']:.3f} | "
                f"{r['pred_in_candidate_rate']:.3f} | {r['invalid_rate']:.3f} | "
                f"{r['exact_match_rate']:.3f} | {r['top1_dominance']:.3f} | "
                f"{r['avg_subgraph_size']:.2f} |"
            )
        lines.append("")

    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- `backbone_raw` is the main R-GCN source with display-name patch.")
    lines.append("- `soft_support_sweep` is a diagnostic row; Day 6 showed it did not improve repoDB ranking.")
    lines.append("- `fuzzy_retrieval_main` is the graph-efficiency row: it preserves candidate metrics and reduces graph size from 100 to 55.")
    lines.append("- DistMult remains the strongest standalone structure-only baseline and should be reported separately.")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    by_split = {"valid": [], "test": []}

    for row_name, dataset_path in ROWS.items():
        for split in ["valid", "test"]:
            item = collect_one(row_name, Path(dataset_path), split)
            by_split[split].append(item)

    for split in ["valid", "test"]:
        by_split[split] = sorted(
            by_split[split],
            key=lambda x: x["e2e_mrr_at20"],
            reverse=True,
        )

    missing_rows = []
    expected_rows = set(ROWS.keys())
    present_rows = {x["row_name"] for split_rows in by_split.values() for x in split_rows}
    missing_rows = sorted(expected_rows - present_rows)

    decision = "DAY8_REPODB_E2E_ALL_ROWS_READY" if not missing_rows else "DAY8_REPODB_E2E_PARTIAL_READY"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_f_repodb",
        "dataset": "repoDB",
        "task": "(?, repoDB_approved_indication, disease)",
        "rows": list(ROWS.keys()),
        "missing_rows": missing_rows,
        "by_split": by_split,
        "metric_note": (
            "E2E adjusted rank uses pred_is_gold => rank 1; if predicted candidate is ahead of the gold, "
            "the gold is shifted down by one; absent-gold remains rank 21."
        ),
    }

    write_json(summary, SUMMARY_PATH)
    write_report(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved:\n  {SUMMARY_PATH}\n  {REPORT_PATH}")


if __name__ == "__main__":
    main()
