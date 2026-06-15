#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PharmKG transfer Day 1: Freeze PharmKG transfer protocol and evaluate backbone_raw.

Input:
- dataset/setting_c_pharmkg/backbone_raw_source/valid_top20_raw.json
- dataset/setting_c_pharmkg/backbone_raw_source/test_top20_raw.json
- dataset/setting_c_pharmkg/softfuse_ready/valid.json
- dataset/setting_c_pharmkg/softfuse_ready/test.json
- outputs/pharmkg/dataset2_baseline_main_table.json

Output:
- dataset/setting_c_pharmkg/transfer_eval_ready/backbone_raw_valid.json
- dataset/setting_c_pharmkg/transfer_eval_ready/backbone_raw_test.json
- outputs/pharmkg/dataset2_transfer_manifest.json
- outputs/pharmkg/backbone_raw_eval_valid.json
- outputs/pharmkg/backbone_raw_eval_test.json
- outputs/pharmkg/reports/day1_transfer_protocol_and_backbone.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(".")

RAW_SOURCE_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "backbone_raw_source"
SOFTFUSE_READY_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "softfuse_ready"
EVAL_READY_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "transfer_eval_ready"

RESULT_WEEK22_DIR = ROOT / "outputs" / "pharmkg"
RESULT_DIR = ROOT / "outputs" / "pharmkg"
REPORT_DIR = ROOT / "outputs" / "pharmkg" / "reports"

BASELINE_TABLE_PATH = RESULT_WEEK22_DIR / "dataset2_baseline_main_table.json"

PROTOCOL_PATH = RESULT_DIR / "dataset2_transfer_manifest.json"
VALID_EVAL_PATH = RESULT_DIR / "backbone_raw_eval_valid.json"
TEST_EVAL_PATH = RESULT_DIR / "backbone_raw_eval_test.json"
REPORT_PATH = REPORT_DIR / "day1_transfer_protocol_and_backbone.md"

TOP_K = 20
ABSENT_RANK = 21
TARGET_RELATION = "T"
TARGET_RELATION_NORMALIZED = "therapeutic_association_proxy"
SOURCE_MODEL = "rgcn"


def ensure_dirs() -> None:
    for p in [EVAL_READY_DIR, RESULT_DIR, REPORT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_protocol() -> dict[str, Any]:
    return {
        "week": 23,
        "day": 1,
        "decision": "TRANSFER_MANIFEST_READY",
        "dataset": "PharmKG-8k",
        "setting": "setting_c_pharmkg",
        "task": {
            "task_template": "(?, T, disease)",
            "prediction_type": "predicted_head",
            "target_relation_raw": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "report_label": "therapeutic association proxy",
            "do_not_call_it": [
                "clinical indication",
                "PrimeKG indication",
                "confirmed treatment label"
            ],
            "candidate_universe": "drug_only_from_train_T_heads",
            "query_universe": "disease_only_from_train_T_tails",
        },
        "source": {
            "main_source_model": SOURCE_MODEL,
            "drkgc_aligned_alternative": "hrgat",
            "source_file_valid": str(RAW_SOURCE_DIR / "valid_top20_raw.json"),
            "source_file_test": str(RAW_SOURCE_DIR / "test_top20_raw.json"),
        },
        "reviewer_safe_metric_policy": {
            "top_k": TOP_K,
            "gold_injection": False,
            "rank_absent_sentinel": ABSENT_RANK,
            "rr_policy": "RR = 1/rank if gold is present in top-20 else 0",
            "rr_absent_policy": 0.0,
            "main_metric": "reviewer_safe_mrr_at20",
            "do_not_use": [
                "1/21 reciprocal rank for absent gold",
                "classical full-ranking filtered KGC metric name",
                "valid/test gold injection"
            ],
        },
        "transfer_rows_to_build": [
            "backbone_raw",
            "hard_support_raw",
            "soft_support_raw",
            "fuzzy_retrieval_main",
        ],
        "expected_backbone_metrics_from_structure_baseline": {
            "valid": {
                "gold_present_at20": 0.070,
                "mrr_at20_approx": 0.01784647174291137,
            },
            "test": {
                "gold_present_at20": 0.092,
                "mrr_at20_approx": 0.02048076984075436,
            },
        },
    }


def compute_rank_and_rr(gold_entity: str, candidates: list[str]) -> tuple[int, bool, float]:
    if gold_entity in candidates:
        rank = candidates.index(gold_entity) + 1
        if rank <= TOP_K:
            return rank, True, 1.0 / rank
    return ABSENT_RANK, False, 0.0


def convert_raw_to_eval_rows(
    raw_rows: list[dict[str, Any]],
    ready_rows: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    ready_by_key = {
        (int(r["query_entity_id"]), int(r["gold_entity_id"])): r
        for r in ready_rows
    }

    eval_rows = []

    for i, row in enumerate(raw_rows):
        query_id = int(row["query_entity_id"])
        gold_id = int(row["gold_entity_id"])
        key = (query_id, gold_id)
        ready = ready_by_key.get(key)

        candidates = list(row["candidate_entities"])
        candidate_ids = [int(x) for x in row["candidate_entity_ids"]]

        if len(candidates) != TOP_K:
            raise ValueError(f"{split} row {i}: expected {TOP_K} candidates, got {len(candidates)}")
        if len(candidate_ids) != TOP_K:
            raise ValueError(f"{split} row {i}: expected {TOP_K} candidate IDs, got {len(candidate_ids)}")
        if row.get("gold_injection") is not False:
            raise ValueError(f"{split} row {i}: gold_injection must be false.")

        rank, present, rr = compute_rank_and_rr(row["gold_entity"], candidates)

        if int(row["gold_rank_in_top20_or_21"]) != rank:
            raise ValueError(
                f"{split} row {i}: raw rank mismatch. "
                f"raw={row['gold_rank_in_top20_or_21']}, recomputed={rank}"
            )

        eval_row = {
            "eval_row_name": "backbone_raw",
            "row_index": int(i),
            "split": split,
            "query_entity": row["query_entity"],
            "query_entity_id": query_id,
            "gold_entity": row["gold_entity"],
            "gold_entity_id": gold_id,
            "candidate_entities": candidates,
            "candidate_entity_ids": candidate_ids,
            "num_candidates": int(len(candidates)),
            "gold_present": bool(present),
            "gold_rank": int(rank),
            "gold_rank_source": "raw_rgcn_top20",
            "reciprocal_rank_item": float(rr),
            "hits1_item": int(rank <= 1),
            "hits3_item": int(rank <= 3),
            "hits10_item": int(rank <= 10),
            "hits20_item": int(rank <= 20),
            "candidate_universe": row["candidate_universe"],
            "gold_injection": False,
            "source_model": row["source_model"],
            "target_relation": row["target_relation"],
            "target_relation_normalized": row["target_relation_normalized"],
            "stage_specific": {
                "stage": "backbone_raw",
                "source": "pharmkg_rgcn_top20_raw",
                "ready_row_found": ready is not None,
                "subgraph_size_from_ready": len(ready["subgraph"]) if ready is not None else None,
                "rank_entities_match_ready": (
                    candidates == ready.get("rank_entities", [])
                    if ready is not None
                    else None
                ),
                "rr_policy": "RR = 1/rank if gold is present in top-20 else 0",
                "rank_absent_sentinel": ABSENT_RANK,
            },
        }

        eval_rows.append(eval_row)

    return eval_rows


def summarize_eval_rows(eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(eval_rows)
    if n == 0:
        raise ValueError("No eval rows.")

    ranks = [int(r["gold_rank"]) for r in eval_rows]
    rr = [float(r["reciprocal_rank_item"]) for r in eval_rows]
    present = [bool(r["gold_present"]) for r in eval_rows]

    present_ranks = [r for r, p in zip(ranks, present) if p]

    metrics = {
        "eval_row_name": "backbone_raw",
        "num_rows": int(n),
        "gold_present_at20": sum(present) / n,
        "mrr_at20": sum(rr) / n,
        "mrr_present_only": (
            sum(1.0 / r for r in present_ranks) / len(present_ranks)
            if present_ranks
            else 0.0
        ),
        "hits1_at20": sum(r <= 1 for r in ranks) / n,
        "hits3_at20": sum(r <= 3 for r in ranks) / n,
        "hits10_at20": sum(r <= 10 for r in ranks) / n,
        "hits20_at20": sum(r <= 20 for r in ranks) / n,
        "avg_gold_rank_absent_as_21": sum(ranks) / n,
        "gold_rank_21_count": sum(r == ABSENT_RANK for r in ranks),
        "top_k": TOP_K,
        "gold_injection": False,
        "rr_policy": "RR = 1/rank if gold is present in top-20 else 0",
        "rank_absent_sentinel": ABSENT_RANK,
    }

    return metrics


def find_rgcn_baseline_metric(split: str) -> dict[str, Any] | None:
    if not BASELINE_TABLE_PATH.exists():
        return None

    table = read_json(BASELINE_TABLE_PATH)
    rows = table.get(split, [])
    for row in rows:
        if row.get("model_name") == "rgcn":
            return row
    return None


def compare_with_baseline(metric: dict[str, Any], split: str) -> dict[str, Any]:
    ref = find_rgcn_baseline_metric(split)
    if ref is None:
        return {
            "available": False,
            "match": None,
            "note": "Week22 rgcn metric not found.",
        }

    diffs = {
        "gold_present_at20_diff": abs(metric["gold_present_at20"] - ref["gold_present_at20"]),
        "mrr_at20_diff": abs(metric["mrr_at20"] - ref["mrr_at20"]),
        "hits1_at20_diff": abs(metric["hits1_at20"] - ref["hits1_at20"]),
        "hits3_at20_diff": abs(metric["hits3_at20"] - ref["hits3_at20"]),
        "hits10_at20_diff": abs(metric["hits10_at20"] - ref["hits10_at20"]),
        "hits20_at20_diff": abs(metric["hits20_at20"] - ref["hits20_at20"]),
    }

    match = all(v < 1e-12 for v in diffs.values())

    return {
        "available": True,
        "match": bool(match),
        "rgcn_baseline_metric": ref,
        "transfer_backbone_metric": metric,
        "diffs": diffs,
    }


def write_report(
    protocol: dict[str, Any],
    valid_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    valid_compare: dict[str, Any],
    test_compare: dict[str, Any],
) -> None:
    md = f"""# PharmKG transfer Day 1 — Transfer Transfer Manifest and Backbone Raw Evaluation

## Decision

`TRANSFER_BACKBONE_RAW_READY`

## Frozen transfer protocol

- Dataset: `{protocol["dataset"]}`
- Setting: `{protocol["setting"]}`
- Task: `{protocol["task"]["task_template"]}`
- Target relation raw: `{protocol["task"]["target_relation_raw"]}`
- Target relation normalized: `{protocol["task"]["target_relation_normalized"]}`
- Report label: `{protocol["task"]["report_label"]}`
- Source model: `{protocol["source"]["main_source_model"]}`
- Candidate universe: `{protocol["task"]["candidate_universe"]}`
- Top-K: `{protocol["reviewer_safe_metric_policy"]["top_k"]}`
- Gold injection: `{protocol["reviewer_safe_metric_policy"]["gold_injection"]}`
- Main metric: `{protocol["reviewer_safe_metric_policy"]["main_metric"]}`
- RR policy: `{protocol["reviewer_safe_metric_policy"]["rr_policy"]}`

Important wording:

Do **not** call relation `T` clinical indication. Use `therapeutic association proxy`.

## Backbone raw metrics

| Split | Gold@20 | MRR@20 | MRR present-only | H@1 | H@3 | H@10 | H@20 | Rank21 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | {valid_metrics["gold_present_at20"]:.3f} | {valid_metrics["mrr_at20"]:.12f} | {valid_metrics["mrr_present_only"]:.12f} | {valid_metrics["hits1_at20"]:.3f} | {valid_metrics["hits3_at20"]:.3f} | {valid_metrics["hits10_at20"]:.3f} | {valid_metrics["hits20_at20"]:.3f} | {valid_metrics["gold_rank_21_count"]} |
| test | {test_metrics["gold_present_at20"]:.3f} | {test_metrics["mrr_at20"]:.12f} | {test_metrics["mrr_present_only"]:.12f} | {test_metrics["hits1_at20"]:.3f} | {test_metrics["hits3_at20"]:.3f} | {test_metrics["hits10_at20"]:.3f} | {test_metrics["hits20_at20"]:.3f} | {test_metrics["gold_rank_21_count"]} |

## Baseline consistency check

- valid match R-GCN baseline: `{valid_compare["match"]}`
- test match R-GCN baseline: `{test_compare["match"]}`

## Day 1 interpretation

The PharmKG transfer starts from a difficult R-GCN raw source:

- valid Gold@20 = `{valid_metrics["gold_present_at20"]:.3f}`
- test Gold@20 = `{test_metrics["gold_present_at20"]:.3f}`

This means PharmKG transfer should treat raw candidate bottleneck as a central diagnostic, not hide it.

## Files written

- `{PROTOCOL_PATH}`
- `{EVAL_READY_DIR / "backbone_raw_valid.json"}`
- `{EVAL_READY_DIR / "backbone_raw_test.json"}`
- `{VALID_EVAL_PATH}`
- `{TEST_EVAL_PATH}`
- `{REPORT_PATH}`

## Next step

Day 2 will build candidate support features and hard-support negative control.
"""
    REPORT_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    protocol = build_protocol()

    raw_valid = read_json(RAW_SOURCE_DIR / "valid_top20_raw.json")
    raw_test = read_json(RAW_SOURCE_DIR / "test_top20_raw.json")

    ready_valid = read_json(SOFTFUSE_READY_DIR / "valid.json")
    ready_test = read_json(SOFTFUSE_READY_DIR / "test.json")

    valid_eval_rows = convert_raw_to_eval_rows(raw_valid, ready_valid, split="valid")
    test_eval_rows = convert_raw_to_eval_rows(raw_test, ready_test, split="test")

    valid_metrics = summarize_eval_rows(valid_eval_rows)
    test_metrics = summarize_eval_rows(test_eval_rows)

    valid_compare = compare_with_baseline(valid_metrics, "valid")
    test_compare = compare_with_baseline(test_metrics, "test")

    protocol["decision"] = "TRANSFER_BACKBONE_RAW_READY"
    protocol["backbone_raw_metrics"] = {
        "valid": valid_metrics,
        "test": test_metrics,
    }
    protocol["baseline_consistency_check"] = {
        "valid": valid_compare,
        "test": test_compare,
    }

    if len(valid_eval_rows) != 500:
        raise RuntimeError(f"Expected valid rows = 500, got {len(valid_eval_rows)}")
    if len(test_eval_rows) != 500:
        raise RuntimeError(f"Expected test rows = 500, got {len(test_eval_rows)}")

    if valid_compare["available"] and not valid_compare["match"]:
        raise RuntimeError("Valid metrics do not match R-GCN baseline baseline.")
    if test_compare["available"] and not test_compare["match"]:
        raise RuntimeError("Test metrics do not match R-GCN baseline baseline.")

    write_json(protocol, PROTOCOL_PATH)
    write_json(valid_eval_rows, EVAL_READY_DIR / "backbone_raw_valid.json")
    write_json(test_eval_rows, EVAL_READY_DIR / "backbone_raw_test.json")
    write_json(valid_metrics, VALID_EVAL_PATH)
    write_json(test_metrics, TEST_EVAL_PATH)

    write_report(
        protocol=protocol,
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        valid_compare=valid_compare,
        test_compare=test_compare,
    )

    print("Saved:")
    print(f"  {PROTOCOL_PATH}")
    print(f"  {EVAL_READY_DIR / 'backbone_raw_valid.json'}")
    print(f"  {EVAL_READY_DIR / 'backbone_raw_test.json'}")
    print(f"  {VALID_EVAL_PATH}")
    print(f"  {TEST_EVAL_PATH}")
    print(f"  {REPORT_PATH}")

    print("\nDecision:", protocol["decision"])

    print("\nVALID backbone_raw metrics:")
    print(json.dumps(valid_metrics, ensure_ascii=False, indent=2))

    print("\nTEST backbone_raw metrics:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    print("\nWeek22 consistency:")
    print("valid match =", valid_compare["match"])
    print("test match =", test_compare["match"])


if __name__ == "__main__":
    main()