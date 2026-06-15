#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week 23 Day 3: Build PharmKG soft-support transfer row.

Main variant:
    soft_support_pharmkg_b050

Frozen formula from PrimeKG Novelty 2:
    support_score =
        1.0 * evidence_positive
        - 0.50 * direct_T_candidate_query_flag
        - 0.10 * contra_flag

For PharmKG:
    contra_flag = 0

No pruning:
    only reorder top-20 candidates.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(".")

SUPPORT_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "support_features"
TRANSFER_EVAL_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "transfer_eval_ready"
SOFT_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "soft_support"

RESULT_DIR = ROOT / "outputs" / "pharmkg"
REPORT_DIR = ROOT / "outputs" / "pharmkg" / "reports"

VALID_SOFT_EVAL_PATH = RESULT_DIR / "soft_support_raw_eval_valid.json"
TEST_SOFT_EVAL_PATH = RESULT_DIR / "soft_support_raw_eval_test.json"
REPORT_PATH = REPORT_DIR / "day3_soft_support_transfer.md"

TOP_K = 20
ABSENT_RANK = 21
TARGET_RELATION = "T"
TARGET_RELATION_NORMALIZED = "therapeutic_association_proxy"

VARIANT_NAME = "soft_support_pharmkg_b050"

CONFIG = {
    "variant_name": VARIANT_NAME,
    "formula": "support_score = 1.0*evidence_positive - 0.50*direct_T_candidate_query_flag - 0.10*contra_flag",
    "evidence_positive_weight": 1.0,
    "direct_T_candidate_query_penalty": 0.50,
    "contra_penalty_weight": 0.10,
    "contra_flag_policy": "fixed_zero_for_pharmkg",
    "sort_policy": [
        "support_score descending",
        "base_rank ascending"
    ],
    "pruning": False,
    "top_k": TOP_K,
    "gold_injection": False,
    "target_relation": TARGET_RELATION,
    "target_relation_normalized": TARGET_RELATION_NORMALIZED,
    "report_label": "soft support transfer on PharmKG therapeutic-association proxy task",
}


def ensure_dirs() -> None:
    for p in [SOFT_DIR, RESULT_DIR, REPORT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def evidence_positive(feature: dict[str, Any]) -> int:
    vec = feature.get("support_feature_vector", {})
    if "evidence_positive" in vec:
        return int(vec["evidence_positive"])
    return int(feature.get("evidence_edge_touch_count", 0) > 0)


def compute_support_score(feature: dict[str, Any]) -> float:
    ev = evidence_positive(feature)
    direct = int(feature.get("direct_T_candidate_query_flag", 0))
    contra = int(feature.get("contra_flag", 0))

    score = (
        CONFIG["evidence_positive_weight"] * ev
        - CONFIG["direct_T_candidate_query_penalty"] * direct
        - CONFIG["contra_penalty_weight"] * contra
    )
    return float(score)


def compute_rank_and_rr(gold_entity: str, candidates: list[str]) -> tuple[int, bool, float]:
    if gold_entity in candidates:
        rank = candidates.index(gold_entity) + 1
        if rank <= TOP_K:
            return rank, True, 1.0 / rank
    return ABSENT_RANK, False, 0.0


def get_backbone_by_row_index(split: str) -> dict[int, dict[str, Any]]:
    path = TRANSFER_EVAL_DIR / f"backbone_raw_{split}.json"
    rows = read_json(path)
    return {int(r["row_index"]): r for r in rows}


def build_soft_rows_for_split(
    split: str,
    support_rows: list[dict[str, Any]],
    backbone_by_idx: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    top20_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []

    for row in support_rows:
        row_index = int(row["row_index"])
        features = list(row["candidate_feature_rows"])

        if len(features) != TOP_K:
            raise ValueError(f"{split} row {row_index}: expected {TOP_K} candidates, got {len(features)}")

        scored = []
        for f in features:
            f2 = dict(f)
            f2["support_score"] = compute_support_score(f2)
            f2["evidence_positive"] = evidence_positive(f2)
            scored.append(f2)

        scored_sorted = sorted(
            scored,
            key=lambda x: (
                -float(x["support_score"]),
                int(x["base_rank"]),
            ),
        )

        for new_rank, f in enumerate(scored_sorted, start=1):
            f["support_rank"] = int(new_rank)

        candidate_entities = [f["candidate_entity"] for f in scored_sorted]
        candidate_ids = [int(f["candidate_entity_id"]) for f in scored_sorted]
        support_scores = [float(f["support_score"]) for f in scored_sorted]

        gold_entity = row["gold_entity"]
        gold_id = int(row["gold_entity_id"])

        soft_rank, soft_present, soft_rr = compute_rank_and_rr(gold_entity, candidate_entities)

        backbone = backbone_by_idx.get(row_index)
        if backbone is None:
            raise KeyError(f"Missing backbone row for {split} row_index={row_index}")

        raw_rank = int(backbone["gold_rank"])
        raw_present = bool(backbone["gold_present"])
        raw_rr = float(backbone["reciprocal_rank_item"])

        if raw_present != bool(row["source_ready_row"]["gold_in_topk_raw"]):
            raise ValueError(f"{split} row {row_index}: raw present mismatch.")

        rank_delta = raw_rank - soft_rank
        rr_delta = soft_rr - raw_rr

        if soft_rank < raw_rank:
            change_label = "improved"
        elif soft_rank > raw_rank:
            change_label = "worsened"
        else:
            change_label = "unchanged"

        top20_row = {
            "split": split,
            "row_index": row_index,
            "query_entity": row["query_entity"],
            "query_entity_id": int(row["query_entity_id"]),
            "gold_entity": gold_entity,
            "gold_entity_id": gold_id,
            "candidate_entities": candidate_entities,
            "candidate_entity_ids": candidate_ids,
            "support_scores": support_scores,
            "support_rank_order": [int(f["base_rank"]) for f in scored_sorted],
            "num_candidates": int(len(candidate_entities)),
            "gold_rank_in_top20_or_21": int(soft_rank),
            "gold_in_topk_soft_support": bool(soft_present),
            "gold_present_raw": bool(raw_present),
            "gold_rank_raw": int(raw_rank),
            "rank_delta_vs_backbone": int(rank_delta),
            "rr_delta_vs_backbone": float(rr_delta),
            "change_label_vs_backbone": change_label,
            "candidate_universe": "drug_only_from_train_T_heads",
            "gold_injection": False,
            "variant_name": VARIANT_NAME,
            "target_relation": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "candidate_debug_rows": scored_sorted,
        }

        eval_row = {
            "eval_row_name": "soft_support_raw",
            "row_index": row_index,
            "split": split,
            "query_entity": row["query_entity"],
            "query_entity_id": int(row["query_entity_id"]),
            "gold_entity": gold_entity,
            "gold_entity_id": gold_id,
            "candidate_entities": candidate_entities,
            "candidate_entity_ids": candidate_ids,
            "num_candidates": int(len(candidate_entities)),
            "gold_present": bool(soft_present),
            "gold_rank": int(soft_rank),
            "gold_rank_source": VARIANT_NAME,
            "reciprocal_rank_item": float(soft_rr),
            "hits1_item": int(soft_rank <= 1),
            "hits3_item": int(soft_rank <= 3),
            "hits10_item": int(soft_rank <= 10),
            "hits20_item": int(soft_rank <= 20),
            "candidate_universe": "drug_only_from_train_T_heads",
            "gold_injection": False,
            "target_relation": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "stage_specific": {
                "stage": "soft_support_raw",
                "variant_name": VARIANT_NAME,
                "formula": CONFIG["formula"],
                "raw_rank": int(raw_rank),
                "raw_present": bool(raw_present),
                "raw_rr": float(raw_rr),
                "soft_rank": int(soft_rank),
                "soft_present": bool(soft_present),
                "soft_rr": float(soft_rr),
                "rank_delta_vs_backbone": int(rank_delta),
                "rr_delta_vs_backbone": float(rr_delta),
                "change_label_vs_backbone": change_label,
                "num_direct_T_candidates_before": int(sum(f["direct_T_candidate_query_flag"] for f in scored)),
                "num_direct_T_candidates_top5_after": int(sum(f["direct_T_candidate_query_flag"] for f in scored_sorted[:5])),
                "num_evidence_positive_candidates": int(sum(f["evidence_positive"] for f in scored)),
                "contra_flag_policy": "fixed_zero_for_pharmkg",
                "pruning": False,
            },
        }

        debug_rows.append(
            {
                "split": split,
                "row_index": row_index,
                "query_entity": row["query_entity"],
                "gold_entity": gold_entity,
                "raw_rank": int(raw_rank),
                "soft_rank": int(soft_rank),
                "raw_present": bool(raw_present),
                "soft_present": bool(soft_present),
                "rank_delta_vs_backbone": int(rank_delta),
                "rr_delta_vs_backbone": float(rr_delta),
                "change_label_vs_backbone": change_label,
                "num_direct_T_candidates_before": int(sum(f["direct_T_candidate_query_flag"] for f in scored)),
                "num_direct_T_candidates_top5_after": int(sum(f["direct_T_candidate_query_flag"] for f in scored_sorted[:5])),
                "gold_direct_T_flag": int(
                    next((f["direct_T_candidate_query_flag"] for f in scored if f["gold_flag"] == 1), 0)
                ),
                "gold_support_score": float(
                    next((f["support_score"] for f in scored if f["gold_flag"] == 1), 0.0)
                ),
            }
        )

        top20_rows.append(top20_row)
        eval_rows.append(eval_row)

    summary = summarize_eval_rows(eval_rows, "soft_support_raw")
    summary["change_summary_vs_backbone"] = summarize_changes(debug_rows)
    summary["variant_name"] = VARIANT_NAME

    return top20_rows, eval_rows, summary


def summarize_eval_rows(rows: list[dict[str, Any]], row_name: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise ValueError("No eval rows.")

    ranks = [int(r["gold_rank"]) for r in rows]
    rr = [float(r["reciprocal_rank_item"]) for r in rows]
    present = [bool(r["gold_present"]) for r in rows]
    present_ranks = [rank for rank, is_present in zip(ranks, present) if is_present]
    candidate_sizes = [int(r["num_candidates"]) for r in rows]

    return {
        "eval_row_name": row_name,
        "num_rows": int(n),
        "gold_present_at20": float(sum(present) / n),
        "mrr_at20": float(sum(rr) / n),
        "mrr_present_only": (
            float(sum(1.0 / r for r in present_ranks) / len(present_ranks))
            if present_ranks
            else 0.0
        ),
        "hits1_at20": float(sum(r <= 1 for r in ranks) / n),
        "hits3_at20": float(sum(r <= 3 for r in ranks) / n),
        "hits10_at20": float(sum(r <= 10 for r in ranks) / n),
        "hits20_at20": float(sum(r <= 20 for r in ranks) / n),
        "avg_gold_rank_absent_as_21": float(sum(ranks) / n),
        "gold_rank_21_count": int(sum(r == ABSENT_RANK for r in ranks)),
        "avg_candidate_size": float(sum(candidate_sizes) / n),
        "min_candidate_size": int(min(candidate_sizes)),
        "max_candidate_size": int(max(candidate_sizes)),
        "top_k": TOP_K,
        "gold_injection": False,
        "rr_policy": "RR = 1/rank if gold is present in top-20 else 0",
        "rank_absent_sentinel": ABSENT_RANK,
    }


def summarize_changes(debug_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(debug_rows)
    counter = Counter(r["change_label_vs_backbone"] for r in debug_rows)

    raw_present_rows = [r for r in debug_rows if r["raw_present"]]
    improved_present = [r for r in raw_present_rows if r["change_label_vs_backbone"] == "improved"]
    worsened_present = [r for r in raw_present_rows if r["change_label_vs_backbone"] == "worsened"]

    rank_deltas_present = [int(r["rank_delta_vs_backbone"]) for r in raw_present_rows]
    rr_deltas = [float(r["rr_delta_vs_backbone"]) for r in debug_rows]

    return {
        "num_rows": int(n),
        "num_improved_vs_backbone": int(counter.get("improved", 0)),
        "num_worsened_vs_backbone": int(counter.get("worsened", 0)),
        "num_unchanged_vs_backbone": int(counter.get("unchanged", 0)),
        "improved_rate": float(counter.get("improved", 0) / max(1, n)),
        "worsened_rate": float(counter.get("worsened", 0) / max(1, n)),
        "num_raw_gold_present_rows": int(len(raw_present_rows)),
        "num_improved_among_gold_present": int(len(improved_present)),
        "num_worsened_among_gold_present": int(len(worsened_present)),
        "avg_rank_delta_among_gold_present": (
            float(sum(rank_deltas_present) / len(rank_deltas_present))
            if rank_deltas_present
            else 0.0
        ),
        "avg_rr_delta_all_rows": (
            float(sum(rr_deltas) / len(rr_deltas))
            if rr_deltas
            else 0.0
        ),
        "num_rows_with_direct_T_before": int(sum(r["num_direct_T_candidates_before"] > 0 for r in debug_rows)),
        "avg_direct_T_candidates_before": float(
            sum(r["num_direct_T_candidates_before"] for r in debug_rows) / max(1, n)
        ),
        "avg_direct_T_candidates_top5_after": float(
            sum(r["num_direct_T_candidates_top5_after"] for r in debug_rows) / max(1, n)
        ),
    }


def compare_to_backbone(soft_metrics: dict[str, Any], split: str) -> dict[str, Any]:
    backbone_path = RESULT_DIR / f"backbone_raw_eval_{split}.json"
    if not backbone_path.exists():
        # Day 1 result path uses same outputs/pharmkg directory.
        pass
    backbone = read_json(backbone_path)

    return {
        "split": split,
        "backbone_gold_present_at20": backbone["gold_present_at20"],
        "soft_gold_present_at20": soft_metrics["gold_present_at20"],
        "delta_gold_present_at20": soft_metrics["gold_present_at20"] - backbone["gold_present_at20"],
        "backbone_mrr_at20": backbone["mrr_at20"],
        "soft_mrr_at20": soft_metrics["mrr_at20"],
        "delta_mrr_at20": soft_metrics["mrr_at20"] - backbone["mrr_at20"],
        "backbone_hits1": backbone["hits1_at20"],
        "soft_hits1": soft_metrics["hits1_at20"],
        "delta_hits1": soft_metrics["hits1_at20"] - backbone["hits1_at20"],
        "backbone_hits3": backbone["hits3_at20"],
        "soft_hits3": soft_metrics["hits3_at20"],
        "delta_hits3": soft_metrics["hits3_at20"] - backbone["hits3_at20"],
        "backbone_hits10": backbone["hits10_at20"],
        "soft_hits10": soft_metrics["hits10_at20"],
        "delta_hits10": soft_metrics["hits10_at20"] - backbone["hits10_at20"],
    }


def decide(valid_compare: dict[str, Any], test_compare: dict[str, Any]) -> str:
    valid_mrr_delta = valid_compare["delta_mrr_at20"]
    test_mrr_delta = test_compare["delta_mrr_at20"]

    if valid_mrr_delta >= 0 and test_mrr_delta >= 0:
        return "SOFT_SUPPORT_TRANSFER_READY"

    if valid_mrr_delta > 0 or test_mrr_delta > 0:
        return "SOFT_SUPPORT_TRANSFER_MIXED_READY"

    return "SOFT_SUPPORT_TRANSFER_DIAGNOSTIC_ONLY"


def write_report(
    valid_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    valid_compare: dict[str, Any],
    test_compare: dict[str, Any],
    decision: str,
    debug_summary: dict[str, Any],
) -> None:
    md = f"""# Week 23 Day 3 — Soft Support Transfer on PharmKG

## Decision

`{decision}`

## Variant

- Variant name: `{VARIANT_NAME}`
- Formula: `{CONFIG["formula"]}`
- Pruning: `false`
- Candidate size: `20`
- Gold injection: `false`

Because PharmKG uses compact relation codes and no explicit contraindication relation is used, `contra_flag = 0`.

## Main metrics

| Split | Gold@20 | MRR@20 | MRR present-only | H@1 | H@3 | H@10 | H@20 | Rank21 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | {valid_metrics["gold_present_at20"]:.3f} | {valid_metrics["mrr_at20"]:.12f} | {valid_metrics["mrr_present_only"]:.12f} | {valid_metrics["hits1_at20"]:.3f} | {valid_metrics["hits3_at20"]:.3f} | {valid_metrics["hits10_at20"]:.3f} | {valid_metrics["hits20_at20"]:.3f} | {valid_metrics["gold_rank_21_count"]} |
| test | {test_metrics["gold_present_at20"]:.3f} | {test_metrics["mrr_at20"]:.12f} | {test_metrics["mrr_present_only"]:.12f} | {test_metrics["hits1_at20"]:.3f} | {test_metrics["hits3_at20"]:.3f} | {test_metrics["hits10_at20"]:.3f} | {test_metrics["hits20_at20"]:.3f} | {test_metrics["gold_rank_21_count"]} |

## Deltas vs backbone_raw

| Split | Delta Gold@20 | Delta MRR@20 | Delta H@1 | Delta H@3 | Delta H@10 |
|---|---:|---:|---:|---:|---:|
| valid | {valid_compare["delta_gold_present_at20"]:.6f} | {valid_compare["delta_mrr_at20"]:.12f} | {valid_compare["delta_hits1"]:.6f} | {valid_compare["delta_hits3"]:.6f} | {valid_compare["delta_hits10"]:.6f} |
| test | {test_compare["delta_gold_present_at20"]:.6f} | {test_compare["delta_mrr_at20"]:.12f} | {test_compare["delta_hits1"]:.6f} | {test_compare["delta_hits3"]:.6f} | {test_compare["delta_hits10"]:.6f} |

## Change summary vs backbone

### Valid

- Improved rows: `{debug_summary["valid"]["change_summary_vs_backbone"]["num_improved_vs_backbone"]}`
- Worsened rows: `{debug_summary["valid"]["change_summary_vs_backbone"]["num_worsened_vs_backbone"]}`
- Unchanged rows: `{debug_summary["valid"]["change_summary_vs_backbone"]["num_unchanged_vs_backbone"]}`
- Raw gold-present rows: `{debug_summary["valid"]["change_summary_vs_backbone"]["num_raw_gold_present_rows"]}`
- Improved among gold-present: `{debug_summary["valid"]["change_summary_vs_backbone"]["num_improved_among_gold_present"]}`
- Worsened among gold-present: `{debug_summary["valid"]["change_summary_vs_backbone"]["num_worsened_among_gold_present"]}`

### Test

- Improved rows: `{debug_summary["test"]["change_summary_vs_backbone"]["num_improved_vs_backbone"]}`
- Worsened rows: `{debug_summary["test"]["change_summary_vs_backbone"]["num_worsened_vs_backbone"]}`
- Unchanged rows: `{debug_summary["test"]["change_summary_vs_backbone"]["num_unchanged_vs_backbone"]}`
- Raw gold-present rows: `{debug_summary["test"]["change_summary_vs_backbone"]["num_raw_gold_present_rows"]}`
- Improved among gold-present: `{debug_summary["test"]["change_summary_vs_backbone"]["num_improved_among_gold_present"]}`
- Worsened among gold-present: `{debug_summary["test"]["change_summary_vs_backbone"]["num_worsened_among_gold_present"]}`

## Interpretation

Soft support is a no-pruning candidate reordering step. Therefore, Gold@20 should stay identical to backbone_raw. Any metric change comes from rank movement among already-present gold candidates.

On PharmKG, the main available signal is the direct-T shortcut penalty, because evidence and shortest-path features are saturated after Day 2.

## Files written

- `dataset/setting_c_pharmkg/soft_support/valid_top20_soft_support_main.json`
- `dataset/setting_c_pharmkg/soft_support/test_top20_soft_support_main.json`
- `dataset/setting_c_pharmkg/soft_support/soft_support_config.json`
- `dataset/setting_c_pharmkg/soft_support/soft_support_debug_summary.json`
- `outputs/pharmkg/soft_support_raw_eval_valid.json`
- `outputs/pharmkg/soft_support_raw_eval_test.json`

## Next step

Day 4 will build fuzzy retrieval / confidence-aware subgraph selection.
"""
    REPORT_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    write_json(CONFIG, SOFT_DIR / "soft_support_config.json")

    valid_support = read_json(SUPPORT_DIR / "valid_support_features.json")
    test_support = read_json(SUPPORT_DIR / "test_support_features.json")

    valid_backbone = get_backbone_by_row_index("valid")
    test_backbone = get_backbone_by_row_index("test")

    valid_top20, valid_eval_rows, valid_metrics = build_soft_rows_for_split(
        split="valid",
        support_rows=valid_support,
        backbone_by_idx=valid_backbone,
    )

    test_top20, test_eval_rows, test_metrics = build_soft_rows_for_split(
        split="test",
        support_rows=test_support,
        backbone_by_idx=test_backbone,
    )

    valid_compare = compare_to_backbone(valid_metrics, "valid")
    test_compare = compare_to_backbone(test_metrics, "test")

    decision = decide(valid_compare, test_compare)

    debug_summary = {
        "week": 23,
        "day": 3,
        "variant_name": VARIANT_NAME,
        "config": CONFIG,
        "valid": valid_metrics,
        "test": test_metrics,
        "valid_compare_to_backbone": valid_compare,
        "test_compare_to_backbone": test_compare,
    }

    write_json(valid_top20, SOFT_DIR / "valid_top20_soft_support_main.json")
    write_json(test_top20, SOFT_DIR / "test_top20_soft_support_main.json")
    write_json(debug_summary, SOFT_DIR / "soft_support_debug_summary.json")

    write_json(valid_metrics, VALID_SOFT_EVAL_PATH)
    write_json(test_metrics, TEST_SOFT_EVAL_PATH)

    write_report(
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        valid_compare=valid_compare,
        test_compare=test_compare,
        decision=decision,
        debug_summary=debug_summary,
    )

    if len(valid_top20) != 500 or len(test_top20) != 500:
        raise RuntimeError("Expected valid/test rows = 500.")

    if abs(valid_compare["delta_gold_present_at20"]) > 1e-12:
        raise RuntimeError("Valid Gold@20 changed, but soft support should not prune.")
    if abs(test_compare["delta_gold_present_at20"]) > 1e-12:
        raise RuntimeError("Test Gold@20 changed, but soft support should not prune.")

    print("Saved:")
    print(f"  {SOFT_DIR / 'valid_top20_soft_support_main.json'}")
    print(f"  {SOFT_DIR / 'test_top20_soft_support_main.json'}")
    print(f"  {SOFT_DIR / 'soft_support_config.json'}")
    print(f"  {SOFT_DIR / 'soft_support_debug_summary.json'}")
    print(f"  {VALID_SOFT_EVAL_PATH}")
    print(f"  {TEST_SOFT_EVAL_PATH}")
    print(f"  {REPORT_PATH}")

    print("\nDecision:", decision)

    print("\nVALID soft metrics:")
    print(json.dumps(valid_metrics, ensure_ascii=False, indent=2))

    print("\nTEST soft metrics:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    print("\nCompare to backbone:")
    print(json.dumps(
        {
            "valid": valid_compare,
            "test": test_compare,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()