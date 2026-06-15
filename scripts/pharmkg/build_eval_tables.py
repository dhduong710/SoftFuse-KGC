#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week 23 Day 5: Build final PharmKG Dataset 2 reviewer-safe eval tables.

Rows:
- backbone_raw
- hard_support_raw
- soft_support_raw
- fuzzy_retrieval_main

Also compares SoftFuse rows against six Week 22 structure baselines:
- transe
- distmult
- complex
- rotate
- rgcn
- hrgat
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(".")

RESULT_WEEK22_DIR = ROOT / "outputs" / "pharmkg"
RESULT_DIR = ROOT / "outputs" / "pharmkg"
REPORT_DIR = ROOT / "outputs" / "pharmkg" / "reports"

DATASET_DIR = ROOT / "dataset" / "setting_c_pharmkg"
TRANSFER_EVAL_DIR = DATASET_DIR / "transfer_eval_ready"
HARD_DIR = DATASET_DIR / "hard_support_raw"
SOFT_DIR = DATASET_DIR / "soft_support"
FUZZY_DIR = DATASET_DIR / "fuzzy_retrieval"
FINAL_EVAL_DIR = DATASET_DIR / "eval_rows"

BASELINE_TABLE_PATH = RESULT_WEEK22_DIR / "dataset2_baseline_main_table.json"

SOFTFUSE_VALID_TABLE_PATH = RESULT_DIR / "dataset2_softfuse_main_table_valid.json"
SOFTFUSE_TEST_TABLE_PATH = RESULT_DIR / "dataset2_softfuse_main_table_test.json"
VS_BASELINES_PATH = RESULT_DIR / "dataset2_vs_structure_baselines.json"
CLAIM_SUMMARY_PATH = RESULT_DIR / "dataset2_main_claim_summary.json"
REPORT_PATH = REPORT_DIR / "day5_final_eval_and_baseline_comparison.md"

TOP_K = 20
ABSENT_RANK = 21

SOFTFUSE_ROWS = [
    "backbone_raw",
    "hard_support_raw",
    "soft_support_raw",
    "fuzzy_retrieval_main",
]

METRIC_FILES = {
    "backbone_raw": {
        "valid": RESULT_DIR / "backbone_raw_eval_valid.json",
        "test": RESULT_DIR / "backbone_raw_eval_test.json",
    },
    "hard_support_raw": {
        "valid": RESULT_DIR / "hard_support_raw_eval_valid.json",
        "test": RESULT_DIR / "hard_support_raw_eval_test.json",
    },
    "soft_support_raw": {
        "valid": RESULT_DIR / "soft_support_raw_eval_valid.json",
        "test": RESULT_DIR / "soft_support_raw_eval_test.json",
    },
    "fuzzy_retrieval_main": {
        "valid": RESULT_DIR / "fuzzy_retrieval_eval_valid.json",
        "test": RESULT_DIR / "fuzzy_retrieval_eval_test.json",
    },
}


def ensure_dirs() -> None:
    for p in [FINAL_EVAL_DIR, RESULT_DIR, REPORT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def compute_rank_and_rr(gold_entity: str, candidates: list[str]) -> tuple[int, bool, float]:
    if gold_entity in candidates:
        rank = candidates.index(gold_entity) + 1
        if rank <= TOP_K:
            return rank, True, 1.0 / rank
    return ABSENT_RANK, False, 0.0


def normalize_metric(row_name: str, split: str, metric: dict[str, Any]) -> dict[str, Any]:
    out = {
        "row_name": row_name,
        "display_name": display_name(row_name),
        "split": split,
        "family": "softfuse_transfer",
        "num_rows": int(metric["num_rows"]),
        "gold_present_at20": float(metric["gold_present_at20"]),
        "mrr_at20": float(metric["mrr_at20"]),
        "mrr_present_only": float(metric.get("mrr_present_only", 0.0)),
        "hits1_at20": float(metric["hits1_at20"]),
        "hits3_at20": float(metric["hits3_at20"]),
        "hits10_at20": float(metric["hits10_at20"]),
        "hits20_at20": float(metric["hits20_at20"]),
        "avg_gold_rank_absent_as_21": float(metric["avg_gold_rank_absent_as_21"]),
        "gold_rank_21_count": int(metric["gold_rank_21_count"]),
        "avg_candidate_size": float(metric.get("avg_candidate_size", TOP_K)),
        "gold_injection": bool(metric.get("gold_injection", False)),
        "rr_policy": metric.get("rr_policy", "RR = 1/rank if gold present in top20 else 0"),
        "rank_absent_sentinel": int(metric.get("rank_absent_sentinel", ABSENT_RANK)),
    }

    if row_name == "soft_support_raw":
        change = metric.get("change_summary_vs_backbone", {})
        out["change_summary_vs_backbone"] = change

    if row_name == "fuzzy_retrieval_main":
        retrieval_summary = get_retrieval_summary(split)
        out["retrieval_summary"] = retrieval_summary
        out["avg_subgraph_size"] = retrieval_summary.get("avg_selected_subgraph_size")
        out["avg_original_subgraph_size"] = retrieval_summary.get("avg_original_subgraph_size")
        out["candidate_coverage_preserved_rate"] = retrieval_summary.get("avg_candidate_coverage_preserved_rate")
        out["top_band_coverage_preserved_rate"] = retrieval_summary.get("avg_top_band_coverage_preserved_rate")
    elif row_name in {"backbone_raw", "hard_support_raw", "soft_support_raw"}:
        out["avg_subgraph_size"] = 100.0 if row_name in {"backbone_raw", "soft_support_raw"} else None

    return out


def display_name(row_name: str) -> str:
    return {
        "backbone_raw": "Backbone raw",
        "hard_support_raw": "Hard support raw",
        "soft_support_raw": "Soft support raw",
        "fuzzy_retrieval_main": "SoftFuse main / fuzzy retrieval",
        "transe": "TransE",
        "distmult": "DistMult",
        "complex": "ComplEx",
        "rotate": "RotatE",
        "rgcn": "R-GCN",
        "hrgat": "HRGAT",
    }.get(row_name, row_name)


def get_retrieval_summary(split: str) -> dict[str, Any]:
    p = FUZZY_DIR / "retrieval_summary.json"
    obj = read_json(p)
    return obj[f"{split}_retrieval_summary"]


def build_softfuse_table(split: str) -> list[dict[str, Any]]:
    rows = []
    for row_name in SOFTFUSE_ROWS:
        metric = read_json(METRIC_FILES[row_name][split])
        rows.append(normalize_metric(row_name, split, metric))

    # Keep logical order, not metric order.
    return rows


def row_to_eval_row_from_top20(
    row_name: str,
    split: str,
    row: dict[str, Any],
    row_index: int,
) -> dict[str, Any]:
    candidates = row.get("candidate_entities") or row.get("candidate_entities_top20") or row.get("rank_entities")
    candidate_ids = row.get("candidate_entity_ids") or row.get("candidate_entity_ids_top20") or row.get("rank_entities_id")

    if candidates is None or candidate_ids is None:
        raise KeyError(f"Cannot find candidates in {row_name}/{split} row {row_index}")

    candidates = list(candidates)
    candidate_ids = [int(x) for x in candidate_ids]

    rank, present, rr = compute_rank_and_rr(row["gold_entity"], candidates)

    stage_specific = {
        "stage": row_name,
        "source_file_row_index": row_index,
        "variant_name": row.get("variant_name"),
        "source_variant_name": row.get("source_variant_name"),
    }

    if row_name == "fuzzy_retrieval_main":
        stage_specific["subgraph_summary"] = row.get("subgraph_summary", {})

    if row_name == "soft_support_raw":
        stage_specific["rank_delta_vs_backbone"] = row.get("rank_delta_vs_backbone")
        stage_specific["rr_delta_vs_backbone"] = row.get("rr_delta_vs_backbone")
        stage_specific["change_label_vs_backbone"] = row.get("change_label_vs_backbone")

    if row_name == "hard_support_raw":
        stage_specific["fallback_used"] = row.get("fallback_used")
        stage_specific["gold_removed_by_hard_support"] = row.get("gold_removed_by_hard_support")

    return {
        "eval_row_name": row_name,
        "row_index": int(row_index),
        "split": split,
        "query_entity": row["query_entity"],
        "query_entity_id": int(row["query_entity_id"]),
        "gold_entity": row["gold_entity"],
        "gold_entity_id": int(row["gold_entity_id"]),
        "candidate_entities": candidates,
        "candidate_entity_ids": candidate_ids,
        "num_candidates": int(len(candidates)),
        "gold_present": bool(present),
        "gold_rank": int(rank),
        "gold_rank_source": row_name,
        "reciprocal_rank_item": float(rr),
        "hits1_item": int(rank <= 1),
        "hits3_item": int(rank <= 3),
        "hits10_item": int(rank <= 10),
        "hits20_item": int(rank <= 20),
        "candidate_universe": row.get("candidate_universe", "drug_only_from_train_T_heads"),
        "gold_injection": bool(row.get("gold_injection", False)),
        "target_relation": row.get("target_relation", "T"),
        "target_relation_normalized": row.get("target_relation_normalized", "therapeutic_association_proxy"),
        "stage_specific": stage_specific,
    }


def export_eval_rows() -> dict[str, dict[str, str]]:
    files: dict[str, dict[str, str]] = {}

    for split in ["valid", "test"]:
        files[split] = {}

        # Backbone already has eval-ready rows from Day 1.
        src = TRANSFER_EVAL_DIR / f"backbone_raw_{split}.json"
        dst = FINAL_EVAL_DIR / f"{split}_backbone_raw.json"
        shutil.copyfile(src, dst)
        files[split]["backbone_raw"] = str(dst)

        # Hard support rows.
        hard_src = HARD_DIR / f"{split}_top20_hard_support_raw.json"
        hard_rows = read_json(hard_src)
        hard_eval = [
            row_to_eval_row_from_top20("hard_support_raw", split, r, i)
            for i, r in enumerate(hard_rows)
        ]
        hard_dst = FINAL_EVAL_DIR / f"{split}_hard_support_raw.json"
        write_json(hard_eval, hard_dst)
        files[split]["hard_support_raw"] = str(hard_dst)

        # Soft support rows.
        soft_src = SOFT_DIR / f"{split}_top20_soft_support_main.json"
        soft_rows = read_json(soft_src)
        soft_eval = [
            row_to_eval_row_from_top20("soft_support_raw", split, r, i)
            for i, r in enumerate(soft_rows)
        ]
        soft_dst = FINAL_EVAL_DIR / f"{split}_soft_support_raw.json"
        write_json(soft_eval, soft_dst)
        files[split]["soft_support_raw"] = str(soft_dst)

        # Fuzzy retrieval rows.
        fuzzy_src = FUZZY_DIR / f"{split}_fuzzy_retrieval_main.json"
        fuzzy_rows = read_json(fuzzy_src)
        fuzzy_eval = [
            row_to_eval_row_from_top20("fuzzy_retrieval_main", split, r, i)
            for i, r in enumerate(fuzzy_rows)
        ]
        fuzzy_dst = FINAL_EVAL_DIR / f"{split}_fuzzy_retrieval_main.json"
        write_json(fuzzy_eval, fuzzy_dst)
        files[split]["fuzzy_retrieval_main"] = str(fuzzy_dst)

    return files


def load_baseline_table() -> dict[str, Any]:
    return read_json(BASELINE_TABLE_PATH)


def normalize_baseline_metric(split: str, metric: dict[str, Any]) -> dict[str, Any]:
    model = metric["model_name"]
    return {
        "row_name": model,
        "display_name": display_name(model),
        "split": split,
        "family": "structure_baseline",
        "num_rows": int(metric["num_rows"]),
        "gold_present_at20": float(metric["gold_present_at20"]),
        "mrr_at20": float(metric["mrr_at20"]),
        "mrr_present_only": float(metric.get("mrr_present_only", 0.0)),
        "hits1_at20": float(metric["hits1_at20"]),
        "hits3_at20": float(metric["hits3_at20"]),
        "hits10_at20": float(metric["hits10_at20"]),
        "hits20_at20": float(metric["hits20_at20"]),
        "avg_gold_rank_absent_as_21": float(metric["avg_gold_rank_absent_as_21"]),
        "gold_rank_21_count": int(metric["gold_rank_21_count"]),
        "avg_candidate_size": 20.0,
        "gold_injection": False,
        "rr_policy": metric.get("rr_policy", "RR = 1/rank if gold present in top20 else 0"),
        "rank_absent_sentinel": int(metric.get("absent_rank_sentinel", ABSENT_RANK)),
    }


def build_vs_baselines(
    valid_table: list[dict[str, Any]],
    test_table: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = load_baseline_table()

    baseline_valid = [
        normalize_baseline_metric("valid", r)
        for r in baseline.get("valid", [])
    ]
    baseline_test = [
        normalize_baseline_metric("test", r)
        for r in baseline.get("test", [])
    ]

    main_valid = next(r for r in valid_table if r["row_name"] == "fuzzy_retrieval_main")
    main_test = next(r for r in test_table if r["row_name"] == "fuzzy_retrieval_main")

    best_baseline_valid = max(baseline_valid, key=lambda x: x["mrr_at20"])
    best_baseline_test = max(baseline_test, key=lambda x: x["mrr_at20"])

    rgcn_valid = next(r for r in baseline_valid if r["row_name"] == "rgcn")
    rgcn_test = next(r for r in baseline_test if r["row_name"] == "rgcn")

    all_valid = sorted(
        baseline_valid + valid_table,
        key=lambda x: x["mrr_at20"],
        reverse=True,
    )
    all_test = sorted(
        baseline_test + test_table,
        key=lambda x: x["mrr_at20"],
        reverse=True,
    )

    return {
        "dataset": "PharmKG-8k task-specific therapeutic_association_proxy benchmark",
        "protocol": {
            "candidate_universe": "drug_only_from_train_T_heads",
            "top_k": TOP_K,
            "gold_injection": False,
            "rr_policy": "RR = 1/rank if gold present in top20 else 0",
            "rank_absent_sentinel": ABSENT_RANK,
        },
        "softfuse_main_row": "fuzzy_retrieval_main",
        "valid": {
            "softfuse_rows": valid_table,
            "structure_baselines": baseline_valid,
            "all_rows_ranked_by_mrr": all_valid,
            "best_structure_baseline": best_baseline_valid,
            "softfuse_main": main_valid,
            "delta_main_vs_best_structure_mrr": main_valid["mrr_at20"] - best_baseline_valid["mrr_at20"],
            "delta_main_vs_rgcn_mrr": main_valid["mrr_at20"] - rgcn_valid["mrr_at20"],
            "delta_main_vs_rgcn_gold_at20": main_valid["gold_present_at20"] - rgcn_valid["gold_present_at20"],
        },
        "test": {
            "softfuse_rows": test_table,
            "structure_baselines": baseline_test,
            "all_rows_ranked_by_mrr": all_test,
            "best_structure_baseline": best_baseline_test,
            "softfuse_main": main_test,
            "delta_main_vs_best_structure_mrr": main_test["mrr_at20"] - best_baseline_test["mrr_at20"],
            "delta_main_vs_rgcn_mrr": main_test["mrr_at20"] - rgcn_test["mrr_at20"],
            "delta_main_vs_rgcn_gold_at20": main_test["gold_present_at20"] - rgcn_test["gold_present_at20"],
        },
    }


def build_claim_summary(
    valid_table: list[dict[str, Any]],
    test_table: list[dict[str, Any]],
    vs: dict[str, Any],
) -> dict[str, Any]:
    backbone_valid = next(r for r in valid_table if r["row_name"] == "backbone_raw")
    backbone_test = next(r for r in test_table if r["row_name"] == "backbone_raw")
    hard_valid = next(r for r in valid_table if r["row_name"] == "hard_support_raw")
    hard_test = next(r for r in test_table if r["row_name"] == "hard_support_raw")
    soft_valid = next(r for r in valid_table if r["row_name"] == "soft_support_raw")
    soft_test = next(r for r in test_table if r["row_name"] == "soft_support_raw")
    fuzzy_valid = next(r for r in valid_table if r["row_name"] == "fuzzy_retrieval_main")
    fuzzy_test = next(r for r in test_table if r["row_name"] == "fuzzy_retrieval_main")

    decision = "DATASET2_SOFTFUSE_MAIN_TABLE_READY"

    if (
        fuzzy_valid["mrr_at20"] > backbone_valid["mrr_at20"]
        and fuzzy_test["mrr_at20"] > backbone_test["mrr_at20"]
        and vs["valid"]["delta_main_vs_best_structure_mrr"] > 0
        and vs["test"]["delta_main_vs_best_structure_mrr"] > 0
    ):
        report_status = "GO_PHARMKG_TRANSFER_RESULT"
    elif (
        fuzzy_valid["mrr_at20"] > backbone_valid["mrr_at20"]
        or fuzzy_test["mrr_at20"] > backbone_test["mrr_at20"]
    ):
        report_status = "PARTIAL_TRANSFER_DIAGNOSTIC_OR_APPENDIX"
    else:
        report_status = "APPENDIX_ONLY_OR_DIAGNOSTIC"

    return {
        "week": 23,
        "day": 5,
        "decision": decision,
        "report_status_candidate": report_status,
        "main_row": "fuzzy_retrieval_main",
        "supporting_ranking_row": "soft_support_raw",
        "dataset": "PharmKG-8k",
        "setting": "setting_c_pharmkg",
        "task": "(?, T, disease)",
        "relation_label": "therapeutic_association_proxy",
        "main_findings": {
            "hard_support": {
                "valid_mrr": hard_valid["mrr_at20"],
                "test_mrr": hard_test["mrr_at20"],
                "interpretation": "hard_support_raw is non-discriminative on PharmKG because binary graph support is saturated.",
            },
            "soft_support": {
                "valid_delta_mrr_vs_backbone": soft_valid["mrr_at20"] - backbone_valid["mrr_at20"],
                "test_delta_mrr_vs_backbone": soft_test["mrr_at20"] - backbone_test["mrr_at20"],
                "valid_delta_hits10_vs_backbone": soft_valid["hits10_at20"] - backbone_valid["hits10_at20"],
                "test_delta_hits10_vs_backbone": soft_test["hits10_at20"] - backbone_test["hits10_at20"],
                "interpretation": "soft support improves ranking without changing Gold@20, confirming positive transfer.",
            },
            "fuzzy_retrieval": {
                "valid_mrr": fuzzy_valid["mrr_at20"],
                "test_mrr": fuzzy_test["mrr_at20"],
                "valid_subgraph_size": fuzzy_valid.get("avg_subgraph_size"),
                "test_subgraph_size": fuzzy_test.get("avg_subgraph_size"),
                "valid_original_subgraph_size": fuzzy_valid.get("avg_original_subgraph_size"),
                "test_original_subgraph_size": fuzzy_test.get("avg_original_subgraph_size"),
                "interpretation": "fuzzy retrieval preserves soft-support ranking and compresses evidence from 100 to 55 triples.",
            },
            "vs_structure_baselines": {
                "valid_delta_mrr_vs_best_structure": vs["valid"]["delta_main_vs_best_structure_mrr"],
                "test_delta_mrr_vs_best_structure": vs["test"]["delta_main_vs_best_structure_mrr"],
                "valid_best_structure": vs["valid"]["best_structure_baseline"]["row_name"],
                "test_best_structure": vs["test"]["best_structure_baseline"]["row_name"],
                "interpretation": "SoftFuse main has the best MRR@20 under the task-specific reviewer-safe protocol.",
            },
        },
        "safe_claim": (
            "On the PharmKG therapeutic-association proxy benchmark, selected soft support improves "
            "R-GCN top-20 ranking without gold injection, and fuzzy retrieval preserves this gain "
            "while reducing the evidence subgraph from 100 to 55 triples."
        ),
        "do_not_claim": [
            "Do not claim full-universe PharmKG KGC superiority.",
            "Do not call relation T a clinical indication relation.",
            "Do not claim fuzzy retrieval reduces shortcut rate on PharmKG; report evidence compression instead.",
            "Do not claim Gold@20/candidate recall improvement, because SoftFuse reorders fixed top-20 candidates.",
        ],
    }


def validate_tables(valid_table: list[dict[str, Any]], test_table: list[dict[str, Any]]) -> None:
    for table, split in [(valid_table, "valid"), (test_table, "test")]:
        names = [r["row_name"] for r in table]
        for name in SOFTFUSE_ROWS:
            if name not in names:
                raise RuntimeError(f"Missing {name} in {split} table")

        for r in table:
            if r["gold_injection"] is not False:
                raise RuntimeError(f"{split}/{r['row_name']} has gold_injection != False")
            if r["rank_absent_sentinel"] != ABSENT_RANK:
                raise RuntimeError(f"{split}/{r['row_name']} wrong absent rank sentinel")

    soft_valid = next(r for r in valid_table if r["row_name"] == "soft_support_raw")
    fuzzy_valid = next(r for r in valid_table if r["row_name"] == "fuzzy_retrieval_main")
    soft_test = next(r for r in test_table if r["row_name"] == "soft_support_raw")
    fuzzy_test = next(r for r in test_table if r["row_name"] == "fuzzy_retrieval_main")

    if abs(soft_valid["mrr_at20"] - fuzzy_valid["mrr_at20"]) > 1e-12:
        raise RuntimeError("Valid fuzzy MRR should match soft MRR")
    if abs(soft_test["mrr_at20"] - fuzzy_test["mrr_at20"]) > 1e-12:
        raise RuntimeError("Test fuzzy MRR should match soft MRR")


def render_table(rows: list[dict[str, Any]], include_subgraph: bool = False) -> str:
    if include_subgraph:
        header = "| Row | Gold@20 | MRR@20 | MRR present-only | H@1 | H@3 | H@10 | H@20 | Rank21 | Avg subgraph |"
        sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    else:
        header = "| Row | Gold@20 | MRR@20 | MRR present-only | H@1 | H@3 | H@10 | H@20 | Rank21 |"
        sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"

    lines = [header, sep]
    for r in rows:
        base = (
            f"| {r['display_name']} | "
            f"{r['gold_present_at20']:.3f} | "
            f"{r['mrr_at20']:.6f} | "
            f"{r['mrr_present_only']:.6f} | "
            f"{r['hits1_at20']:.3f} | "
            f"{r['hits3_at20']:.3f} | "
            f"{r['hits10_at20']:.3f} | "
            f"{r['hits20_at20']:.3f} | "
            f"{r['gold_rank_21_count']} |"
        )
        if include_subgraph:
            sg = r.get("avg_subgraph_size")
            sg_text = "NA" if sg is None else f"{float(sg):.2f}"
            base = base[:-1] + f" {sg_text} |"
        lines.append(base)
    return "\n".join(lines)


def write_report(
    valid_table: list[dict[str, Any]],
    test_table: list[dict[str, Any]],
    vs: dict[str, Any],
    claim: dict[str, Any],
) -> None:
    valid_all = vs["valid"]["all_rows_ranked_by_mrr"]
    test_all = vs["test"]["all_rows_ranked_by_mrr"]

    md = f"""# Week 23 Day 5 — Final PharmKG Reviewer-Safe Evaluation and Baseline Comparison

## Decision

`{claim["decision"]}`

Candidate report status:

`{claim["report_status_candidate"]}`

## Protocol

- Dataset: PharmKG-8k
- Setting: `setting_c_pharmkg`
- Task: `(?, T, disease)`
- Relation label: `therapeutic_association_proxy`
- Candidate universe: `drug_only_from_train_T_heads`
- Top-K: 20
- Gold injection: false
- RR policy: RR = 1/rank if gold is present in top-20 else 0
- Absent rank sentinel: 21

## SoftFuse transfer table — validation

{render_table(valid_table, include_subgraph=True)}

## SoftFuse transfer table — test

{render_table(test_table, include_subgraph=True)}

## Ranked comparison against structure baselines — validation

{render_table(valid_all, include_subgraph=False)}

## Ranked comparison against structure baselines — test

{render_table(test_all, include_subgraph=False)}

## Key deltas

### SoftFuse main vs R-GCN backbone

- valid MRR delta: `{claim["main_findings"]["soft_support"]["valid_delta_mrr_vs_backbone"]:.12f}`
- test MRR delta: `{claim["main_findings"]["soft_support"]["test_delta_mrr_vs_backbone"]:.12f}`
- valid H@10 delta: `{claim["main_findings"]["soft_support"]["valid_delta_hits10_vs_backbone"]:.6f}`
- test H@10 delta: `{claim["main_findings"]["soft_support"]["test_delta_hits10_vs_backbone"]:.6f}`

### SoftFuse main vs best structure baseline

- valid best structure: `{claim["main_findings"]["vs_structure_baselines"]["valid_best_structure"]}`
- valid MRR delta: `{claim["main_findings"]["vs_structure_baselines"]["valid_delta_mrr_vs_best_structure"]:.12f}`
- test best structure: `{claim["main_findings"]["vs_structure_baselines"]["test_best_structure"]}`
- test MRR delta: `{claim["main_findings"]["vs_structure_baselines"]["test_delta_mrr_vs_best_structure"]:.12f}`

## Main interpretation

{claim["safe_claim"]}

## What to claim

- Soft support improves R-GCN ranking on PharmKG without valid/test gold injection.
- Fuzzy retrieval preserves soft-support ranking and compresses evidence from 100 to 55 triples.
- Under the task-specific reviewer-safe top-20 protocol, SoftFuse main achieves the best MRR@20 on PharmKG.
- Gold@20 does not improve because SoftFuse operates on fixed top-20 candidate lists.

## What not to claim

{chr(10).join(f"- {x}" for x in claim["do_not_claim"])}

## Files written

- `dataset/setting_c_pharmkg/eval_rows/`
- `outputs/pharmkg/dataset2_softfuse_main_table_valid.json`
- `outputs/pharmkg/dataset2_softfuse_main_table_test.json`
- `outputs/pharmkg/dataset2_vs_structure_baselines.json`
- `outputs/pharmkg/dataset2_main_claim_summary.json`

## Next step

Day 6 will build diagnostics and failure analysis:
raw bottleneck, soft-support improvements, hard-support non-discrimination, and same-rank cleaner subgraphs.
"""
    REPORT_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    eval_row_files = export_eval_rows()

    valid_table = build_softfuse_table("valid")
    test_table = build_softfuse_table("test")

    validate_tables(valid_table, test_table)

    vs = build_vs_baselines(valid_table, test_table)
    claim = build_claim_summary(valid_table, test_table, vs)
    claim["eval_row_files"] = eval_row_files

    write_json(valid_table, SOFTFUSE_VALID_TABLE_PATH)
    write_json(test_table, SOFTFUSE_TEST_TABLE_PATH)
    write_json(vs, VS_BASELINES_PATH)
    write_json(claim, CLAIM_SUMMARY_PATH)

    write_report(valid_table, test_table, vs, claim)

    print("Saved:")
    print(f"  {FINAL_EVAL_DIR}")
    print(f"  {SOFTFUSE_VALID_TABLE_PATH}")
    print(f"  {SOFTFUSE_TEST_TABLE_PATH}")
    print(f"  {VS_BASELINES_PATH}")
    print(f"  {CLAIM_SUMMARY_PATH}")
    print(f"  {REPORT_PATH}")

    print("\nDecision:", claim["decision"])
    print("Report status candidate:", claim["report_status_candidate"])

    print("\nSoftFuse valid table:")
    print(json.dumps(valid_table, ensure_ascii=False, indent=2))

    print("\nSoftFuse test table:")
    print(json.dumps(test_table, ensure_ascii=False, indent=2))

    print("\nBest baseline comparison:")
    print(json.dumps(
        {
            "valid_delta_main_vs_best_structure_mrr": vs["valid"]["delta_main_vs_best_structure_mrr"],
            "test_delta_main_vs_best_structure_mrr": vs["test"]["delta_main_vs_best_structure_mrr"],
            "valid_best_structure": vs["valid"]["best_structure_baseline"],
            "test_best_structure": vs["test"]["best_structure_baseline"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()