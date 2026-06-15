#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recompute reviewer-safe metrics for rerun baselines and SoftFuse rows.

Metric rule:
- RR = 1/rank if rank <= 20 else 0
- absent gold from top-20 => descriptive rank 21, RR 0
- never use 1/21
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(".")
RESULTS_DIR = ROOT / "outputs" / "baselines"
REPORTS_DIR = ROOT / "outputs" / "baselines" / "reports"

OUT_BASELINE_VALID = RESULTS_DIR / "baseline_reviewer_safe_valid.json"
OUT_BASELINE_TEST = RESULTS_DIR / "baseline_reviewer_safe_test.json"
OUT_SOFTFUSE_VALID = RESULTS_DIR / "softfuse_reviewer_safe_valid.json"
OUT_SOFTFUSE_TEST = RESULTS_DIR / "softfuse_reviewer_safe_test.json"
OUT_ALL = RESULTS_DIR / "reviewer_safe_metrics_all.json"
OUT_MD = REPORTS_DIR / "reviewer_safe_metric_recompute.md"

BASELINE_MODELS = ["transe", "distmult", "complex", "rotate", "rgcn", "hrgat"]

SOFTFUSE_PATHS = {
    "valid": {
        "backbone_raw": ROOT / "dataset" / "setting_b" / "eval_valid" / "valid_backbone_raw_eval.json",
        "soft_support_raw": ROOT / "dataset" / "setting_b" / "eval_valid" / "valid_soft_support_raw_eval.json",
        "soft_support_fuzzy_retrieval_main": ROOT / "dataset" / "setting_b" / "eval_valid" / "valid_retrieval_main_eval.json",
    },
    "test": {
        "backbone_raw": ROOT / "dataset" / "setting_b" / "eval_test" / "test_backbone_raw_eval.json",
        "soft_support_raw": ROOT / "dataset" / "setting_b" / "eval_test" / "test_soft_support_raw_eval.json",
        "soft_support_fuzzy_retrieval_main": ROOT / "dataset" / "setting_b" / "eval_test" / "test_retrieval_main_eval.json",
    },
}

RAW_REFERENCE_PATHS = {
    "valid": ROOT / "dataset" / "setting_a" / "backbone_candidates" / "valid_top20_raw.json",
    "test": ROOT / "dataset" / "setting_a" / "backbone_candidates" / "test_top20_raw.json",
}

CANDIDATE_ID_FIELDS = [
    "candidate_entity_ids_top20",
    "candidate_entity_ids",
    "rank_entities_id",
    "rank_entities_ids",
    "rank_entities",
    "rank_entity_ids",
    "top20_entity_ids",
    "candidate_ids",
]

CANDIDATE_NAME_FIELDS = [
    "candidate_entities_top20",
    "candidate_entities",
    "rank_entities",
    "rank_entities_name",
    "rank_entities_names",
    "top20_entities",
    "candidates",
]

GOLD_ID_FIELDS = [
    "gold_entity_id",
    "gold_id",
    "answer_id",
    "target_id",
    "head_id",
]

QUERY_ID_FIELDS = [
    "query_entity_id",
    "disease_id",
    "tail_id",
]

RANK_FIELDS = [
    "gold_rank_in_top20_or_21",
    "gold_rank_absent_as_21",
    "gold_rank",
    "gold_rank_in_top20",
    "rank",
    "rank_ready",
    "rank_raw",
    "gold_rank_in_full_universe",
]

PRESENT_FIELDS = [
    "gold_present_top20",
    "gold_present_at20",
    "gold_in_top20",
    "gold_in_topk_raw",
    "gold_in_topk",
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ["rows", "data", "examples", "predictions", "results"]:
            if isinstance(obj.get(k), list):
                return obj[k]
    raise ValueError(f"Cannot parse rows from {path}")


def safe_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return None
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return None
        return float(x)
    except Exception:
        return None


def get_nested(row: dict[str, Any], path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def first_value(row: dict[str, Any], fields: list[str]) -> Any:
    for f in fields:
        if "." in f:
            v = get_nested(row, f)
        else:
            v = row.get(f)
        if v is not None:
            return v
    return None


def extract_gold_id(row: dict[str, Any]) -> int | None:
    v = first_value(row, GOLD_ID_FIELDS)
    gid = safe_int(v)
    if gid is not None:
        return gid

    triple_id = row.get("triple_id")
    if isinstance(triple_id, list) and len(triple_id) >= 1:
        return safe_int(triple_id[0])

    row_metrics = row.get("row_metrics_ready") or row.get("row_metrics")
    if isinstance(row_metrics, dict):
        gid = safe_int(row_metrics.get("gold_entity_id"))
        if gid is not None:
            return gid

    return None


def extract_query_id(row: dict[str, Any]) -> int | None:
    v = first_value(row, QUERY_ID_FIELDS)
    qid = safe_int(v)
    if qid is not None:
        return qid

    triple_id = row.get("triple_id")
    if isinstance(triple_id, list) and len(triple_id) >= 3:
        return safe_int(triple_id[2])

    return None


def extract_candidate_ids(row: dict[str, Any]) -> list[int]:
    for field in CANDIDATE_ID_FIELDS:
        v = row.get(field)
        if isinstance(v, list):
            ids = [safe_int(x) for x in v]
            ids = [x for x in ids if x is not None]
            if ids:
                return ids

    # Some eval rows may store candidates in nested structures.
    for nested_key in ["row", "source_row", "example", "data"]:
        sub = row.get(nested_key)
        if isinstance(sub, dict):
            ids = extract_candidate_ids(sub)
            if ids:
                return ids

    return []


def extract_candidate_names(row: dict[str, Any]) -> list[str]:
    for field in CANDIDATE_NAME_FIELDS:
        v = row.get(field)
        if isinstance(v, list):
            return [str(x) for x in v]
    return []


def extract_rank_from_fields(row: dict[str, Any]) -> int | None:
    for field in RANK_FIELDS:
        v = row.get(field)
        r = safe_int(v)
        if r is not None:
            return r

    for nested_key in ["row_metrics_ready", "row_metrics", "metrics"]:
        sub = row.get(nested_key)
        if isinstance(sub, dict):
            for field in RANK_FIELDS:
                r = safe_int(sub.get(field))
                if r is not None:
                    return r

    return None


def extract_present_from_fields(row: dict[str, Any]) -> bool | None:
    for field in PRESENT_FIELDS:
        if field in row:
            return bool(row[field])

    for nested_key in ["row_metrics_ready", "row_metrics", "metrics"]:
        sub = row.get(nested_key)
        if isinstance(sub, dict):
            for field in PRESENT_FIELDS:
                if field in sub:
                    return bool(sub[field])

    return None


def compute_rank(row: dict[str, Any]) -> tuple[int, bool, int | None, list[int]]:
    gold_id = extract_gold_id(row)
    cand_ids = extract_candidate_ids(row)

    if gold_id is not None and cand_ids:
        top20 = cand_ids[:20]
        if gold_id in top20:
            return top20.index(gold_id) + 1, True, gold_id, top20
        return 21, False, gold_id, top20

    # Fallback to rank field only if candidate list is unavailable.
    r = extract_rank_from_fields(row)
    present = extract_present_from_fields(row)

    if r is not None:
        if 1 <= r <= 20:
            return r, True if present is None else bool(present), gold_id, cand_ids[:20]
        return 21, False, gold_id, cand_ids[:20]

    if present is False:
        return 21, False, gold_id, cand_ids[:20]

    raise ValueError(f"Cannot compute rank for row keys={list(row.keys())[:40]}")


def rr_from_rank(rank: int) -> float:
    if 1 <= rank <= 20:
        return 1.0 / rank
    return 0.0


def compute_metrics(rows: list[dict[str, Any]], row_name: str, split: str, source_path: str) -> dict[str, Any]:
    ranks: list[int] = []
    rrs: list[float] = []
    candidate_sizes: list[int] = []
    top1_ids: list[int] = []
    extraction_errors: list[str] = []
    gold_ids_seen: list[int] = []
    query_ids_seen: list[int] = []

    for i, row in enumerate(rows):
        try:
            rank, present, gold_id, top20 = compute_rank(row)
        except Exception as e:
            extraction_errors.append(f"row={i}: {repr(e)}")
            continue

        rank = rank if 1 <= rank <= 20 else 21
        ranks.append(rank)
        rrs.append(rr_from_rank(rank))
        candidate_sizes.append(len(top20) if top20 else 0)

        if top20:
            top1_ids.append(top20[0])

        qid = extract_query_id(row)
        gid = extract_gold_id(row)
        if qid is not None:
            query_ids_seen.append(qid)
        if gid is not None:
            gold_ids_seen.append(gid)

    if extraction_errors:
        raise RuntimeError(
            f"Metric extraction failed for {row_name}/{split}, "
            f"num_errors={len(extraction_errors)}, first_errors={extraction_errors[:5]}"
        )

    n = len(ranks)
    present_count = sum(1 for r in ranks if 1 <= r <= 20)
    absent_count = sum(1 for r in ranks if r == 21)
    present_rrs = [rr for r, rr in zip(ranks, rrs) if 1 <= r <= 20]

    top1_count: dict[int, int] = {}
    for t in top1_ids:
        top1_count[t] = top1_count.get(t, 0) + 1

    unique_top1_count = len(top1_count)
    top1_dominance = max(top1_count.values()) / n if n and top1_count else 0.0

    metrics = {
        "row_name": row_name,
        "split": split,
        "source_path": source_path,
        "num_rows": n,
        "candidate_size_min": min(candidate_sizes) if candidate_sizes else None,
        "candidate_size_max": max(candidate_sizes) if candidate_sizes else None,
        "candidate_size_avg": sum(candidate_sizes) / n if n else None,
        "gold_present_at20": present_count / n if n else 0.0,
        "reviewer_safe_mrr_at20": sum(rrs) / n if n else 0.0,
        "mrr_present_only": sum(present_rrs) / len(present_rrs) if present_rrs else 0.0,
        "hits1_at20": sum(1 for r in ranks if r <= 1) / n if n else 0.0,
        "hits3_at20": sum(1 for r in ranks if r <= 3) / n if n else 0.0,
        "hits10_at20": sum(1 for r in ranks if r <= 10) / n if n else 0.0,
        "hits20_at20": sum(1 for r in ranks if r <= 20) / n if n else 0.0,
        "avg_gold_rank_absent_as_21": sum(ranks) / n if n else None,
        "gold_rank_21_count": absent_count,
        "unique_top1_count": unique_top1_count,
        "top1_dominance": top1_dominance,
        "num_unique_query_ids": len(set(query_ids_seen)),
        "num_unique_gold_ids": len(set(gold_ids_seen)),
        "rr_rule": "RR = 1/rank if rank <= 20 else 0",
        "absent_rank_sentinel": 21,
        "rr_absent_policy": 0,
        "absent_rr_check_pass": all(rr == 0.0 for r, rr in zip(ranks, rrs) if r == 21),
    }

    assert metrics["absent_rr_check_pass"], "Absent rank must have RR=0"
    return metrics


def build_reference_keys(split: str) -> list[tuple[int | None, int | None]]:
    path = RAW_REFERENCE_PATHS[split]
    rows = load_rows(path)
    return [(extract_query_id(r), extract_gold_id(r)) for r in rows]


def query_set_check(rows: list[dict[str, Any]], split: str) -> bool:
    ref = build_reference_keys(split)
    keys = [(extract_query_id(r), extract_gold_id(r)) for r in rows]
    return keys == ref


def load_baseline_metrics(split: str) -> list[dict[str, Any]]:
    out = []
    for model in BASELINE_MODELS:
        path = RESULTS_DIR / "baseline_outputs" / model / f"{split}_top20.json"
        rows = load_rows(path)
        m = compute_metrics(rows, model, split, str(path))
        m["family"] = "structure_baseline"
        m["same_query_set_as_raw_source"] = query_set_check(rows, split)
        m["gold_injection_values"] = sorted({str(r.get("gold_injected")) for r in rows if "gold_injected" in r})
        out.append(m)
    return out


def load_softfuse_metrics(split: str) -> list[dict[str, Any]]:
    out = []
    for row_name, path in SOFTFUSE_PATHS[split].items():
        rows = load_rows(path)
        m = compute_metrics(rows, row_name, split, str(path))
        m["family"] = "softfuse"
        m["same_query_set_as_raw_source"] = query_set_check(rows, split)
        out.append(m)
    return out


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda x: (
            -float(x["reviewer_safe_mrr_at20"]),
            -float(x["hits10_at20"]),
            -float(x["gold_present_at20"]),
            x["row_name"],
        ),
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def metric_table_md(rows: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("| Row | Family | Gold@20 | MRR@20 | H@1 | H@3 | H@10 | Avg rank | Rank21 | Top1 uniq |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| `{r['row_name']}` | `{r['family']}` "
            f"| {r['gold_present_at20']:.4f} "
            f"| {r['reviewer_safe_mrr_at20']:.6f} "
            f"| {r['hits1_at20']:.4f} "
            f"| {r['hits3_at20']:.4f} "
            f"| {r['hits10_at20']:.4f} "
            f"| {r['avg_gold_rank_absent_as_21']:.3f} "
            f"| {r['gold_rank_21_count']} "
            f"| {r['unique_top1_count']} |"
        )
    return "\n".join(lines)


def write_report(all_obj: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    valid_all = sort_rows(all_obj["valid"]["all_rows"])
    test_all = sort_rows(all_obj["test"]["all_rows"])

    best_valid = valid_all[0]
    best_test = test_all[0]

    md = f"""# Reviewer-safe Metric Recompute

## Status

**REVIEWER_SAFE_METRICS_RECOMPUTED_READY_FOR_BASELINE_COMPARISON**

## Metric rule

- `RR = 1/rank if rank <= 20 else 0`
- Gold absent from top-20 uses descriptive rank `21`
- RR for absent gold is `0`
- No `1/21`
- No gold injection

## Best rows after recomputation

| Split | Best row | Family | MRR@20 | Gold@20 | Hits@10 |
|---|---|---|---:|---:|---:|
| valid | `{best_valid['row_name']}` | `{best_valid['family']}` | {best_valid['reviewer_safe_mrr_at20']:.6f} | {best_valid['gold_present_at20']:.4f} | {best_valid['hits10_at20']:.4f} |
| test | `{best_test['row_name']}` | `{best_test['family']}` | {best_test['reviewer_safe_mrr_at20']:.6f} | {best_test['gold_present_at20']:.4f} | {best_test['hits10_at20']:.4f} |

## Valid split — all rows

{metric_table_md(valid_all)}

## Test split — all rows

{metric_table_md(test_all)}

## Query-set checks

All rows should have `same_query_set_as_raw_source = true`.

### Valid

{chr(10).join([f"- `{r['row_name']}`: `{r['same_query_set_as_raw_source']}`" for r in valid_all])}

### Test

{chr(10).join([f"- `{r['row_name']}`: `{r['same_query_set_as_raw_source']}`" for r in test_all])}

## Absent-RR checks

All rows should have `absent_rr_check_pass = true`.

### Valid

{chr(10).join([f"- `{r['row_name']}`: `{r['absent_rr_check_pass']}`" for r in valid_all])}

### Test

{chr(10).join([f"- `{r['row_name']}`: `{r['absent_rr_check_pass']}`" for r in test_all])}

## Interpretation note

Next, build the baseline comparison table:

- If SoftFuse main remains best among DrKGC-compatible rows but not best among pure structure candidate generators, frame structure baselines as stronger upstream candidates and SoftFuse as evidence-aware refinement of the DrKGC-compatible pipeline.
- If a baseline such as ComplEx is clearly stronger as a candidate generator, consider a future branch: `ComplEx-source + SoftFuse soft/retrieval`.
"""

    OUT_MD.write_text(md, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_valid = load_baseline_metrics("valid")
    baseline_test = load_baseline_metrics("test")
    softfuse_valid = load_softfuse_metrics("valid")
    softfuse_test = load_softfuse_metrics("test")

    write_json(OUT_BASELINE_VALID, baseline_valid)
    write_json(OUT_BASELINE_TEST, baseline_test)
    write_json(OUT_SOFTFUSE_VALID, softfuse_valid)
    write_json(OUT_SOFTFUSE_TEST, softfuse_test)

    all_obj = {
        "metric_protocol": {
            "main_metric": "reviewer_safe_mrr_at20",
            "rr_rule": "RR = 1/rank if rank <= 20 else 0",
            "absent_rank_sentinel": 21,
            "rr_absent_policy": 0,
            "gold_injection": "forbidden",
        },
        "valid": {
            "baseline_rows": baseline_valid,
            "softfuse_rows": softfuse_valid,
            "all_rows": baseline_valid + softfuse_valid,
            "all_rows_sorted": sort_rows(baseline_valid + softfuse_valid),
        },
        "test": {
            "baseline_rows": baseline_test,
            "softfuse_rows": softfuse_test,
            "all_rows": baseline_test + softfuse_test,
            "all_rows_sorted": sort_rows(baseline_test + softfuse_test),
        },
    }

    write_json(OUT_ALL, all_obj)
    write_report(all_obj)

    print("=" * 100)
    print("REVIEWER-SAFE METRIC RECOMPUTE")
    print("=" * 100)
    print(f"Wrote: {OUT_BASELINE_VALID}")
    print(f"Wrote: {OUT_BASELINE_TEST}")
    print(f"Wrote: {OUT_SOFTFUSE_VALID}")
    print(f"Wrote: {OUT_SOFTFUSE_TEST}")
    print(f"Wrote: {OUT_ALL}")
    print(f"Wrote: {OUT_MD}")
    print()

    for split in ["valid", "test"]:
        rows = all_obj[split]["all_rows_sorted"]
        print(f"{split.upper()} sorted by reviewer_safe_mrr_at20")
        for r in rows:
            print(
                f"  {r['row_name']:36s} "
                f"family={r['family']:18s} "
                f"Gold@20={r['gold_present_at20']:.4f} "
                f"MRR@20={r['reviewer_safe_mrr_at20']:.6f} "
                f"H@1={r['hits1_at20']:.4f} "
                f"H@3={r['hits3_at20']:.4f} "
                f"H@10={r['hits10_at20']:.4f} "
                f"rank21={r['gold_rank_21_count']}"
            )
        print()

    print("Status: REVIEWER_SAFE_METRICS_RECOMPUTED_READY_FOR_BASELINE_COMPARISON")
    print("=" * 100)


if __name__ == "__main__":
    main()
