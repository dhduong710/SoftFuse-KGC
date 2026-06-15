#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week 22 Day 4: Build PharmKG Dataset 2 splits, enriched graph, and leak checks.

Task:
    (?, T, disease)

Meaning:
    T = therapeutic_association_proxy
    missing head = drug_or_chemical
    query tail = disease

Input:
    data/raw/pharmkg/PharmKG-8k/train.tsv
    data/raw/pharmkg/PharmKG-8k/valid.tsv
    data/raw/pharmkg/PharmKG-8k/test.tsv
    outputs/pharmkg/dataset2_task_spec.json

Output:
    dataset/setting_c_pharmkg/splits/train.json
    dataset/setting_c_pharmkg/splits/valid.json
    dataset/setting_c_pharmkg/splits/test.json

    dataset/setting_c_pharmkg/splits/train_target.tsv
    dataset/setting_c_pharmkg/splits/valid_target.tsv
    dataset/setting_c_pharmkg/splits/test_target.tsv
    dataset/setting_c_pharmkg/splits/candidate_universe.json
    dataset/setting_c_pharmkg/splits/query_universe.json

    dataset/setting_c_pharmkg/graph/entity2id.json
    dataset/setting_c_pharmkg/graph/id2entity.json
    dataset/setting_c_pharmkg/graph/relation2id.json
    dataset/setting_c_pharmkg/graph/id2relation.json
    dataset/setting_c_pharmkg/graph/type_map.json
    dataset/setting_c_pharmkg/graph/train_enriched.tsv
    dataset/setting_c_pharmkg/graph/train_enriched_relation_counts.json

    outputs/pharmkg/dataset2_split_summary.json
    outputs/pharmkg/dataset2_leak_check.json
    outputs/pharmkg/reports/day4_split_and_graph_build.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
RAW_DIR = ROOT / "data" / "raw" / "pharmkg" / "PharmKG-8k"

RESULT_DIR = ROOT / "outputs" / "pharmkg"
REPORT_DIR = ROOT / "outputs" / "pharmkg" / "reports"

SPLIT_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "splits"
GRAPH_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "graph"

TASK_SPEC_PATH = RESULT_DIR / "dataset2_task_spec.json"

SPLIT_SUMMARY_PATH = RESULT_DIR / "dataset2_split_summary.json"
LEAK_CHECK_PATH = RESULT_DIR / "dataset2_leak_check.json"
REPORT_PATH = REPORT_DIR / "day4_split_and_graph_build.md"

TARGET_RELATION = "T"
TARGET_RELATION_NORMALIZED = "therapeutic_association_proxy"

DEFAULT_SEED = 2025
DEFAULT_VALID_SIZE = 500
DEFAULT_TEST_SIZE = 500


def ensure_dirs() -> None:
    for path in [RESULT_DIR, REPORT_DIR, SPLIT_DIR, GRAPH_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_raw_split(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing raw split file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
        dtype=str,
        keep_default_na=False,
    )

    expected_cols = ["head", "relation", "tail"]
    if list(df.columns) != expected_cols:
        raise RuntimeError(f"Unexpected columns for {path}: {list(df.columns)}")

    df["split"] = split
    return df


def load_raw_splits() -> dict[str, pd.DataFrame]:
    return {
        "train": read_raw_split(RAW_DIR / "train.tsv", "train"),
        "valid": read_raw_split(RAW_DIR / "valid.tsv", "valid"),
        "test": read_raw_split(RAW_DIR / "test.tsv", "test"),
    }


def target_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["relation"] == TARGET_RELATION].copy()


def triples_as_set(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    if len(df) == 0:
        return set()
    return set(map(tuple, df[["head", "relation", "tail"]].values.tolist()))


def write_target_tsv(df: pd.DataFrame, path: Path) -> None:
    df[["head", "relation", "tail"]].to_csv(
        path,
        sep="\t",
        header=False,
        index=False,
    )


def filter_eval_target_rows(
    eval_t: pd.DataFrame,
    train_t_set: set[tuple[str, str, str]],
    candidate_universe: set[str],
    query_universe: set[str],
    extra_forbidden_target_triples: set[tuple[str, str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Coverage-safe filtering for valid/test target rows.

    Required:
    - head in candidate_universe
    - tail in query_universe
    - triple not in train target positives
    - triple not in extra_forbidden_target_triples, if provided
    - exact triple deduplication
    """

    if extra_forbidden_target_triples is None:
        extra_forbidden_target_triples = set()

    before = int(len(eval_t))

    covered = eval_t[
        eval_t["head"].isin(candidate_universe)
        & eval_t["tail"].isin(query_universe)
    ].copy()

    after_coverage = int(len(covered))

    covered_triples = list(
        map(tuple, covered[["head", "relation", "tail"]].values.tolist())
    )
    keep_not_train = [tr not in train_t_set for tr in covered_triples]
    no_train_overlap = covered.loc[keep_not_train].copy()

    after_remove_train_overlap = int(len(no_train_overlap))

    no_train_triples = list(
        map(tuple, no_train_overlap[["head", "relation", "tail"]].values.tolist())
    )
    keep_not_extra = [tr not in extra_forbidden_target_triples for tr in no_train_triples]
    no_extra_overlap = no_train_overlap.loc[keep_not_extra].copy()

    after_remove_extra_forbidden = int(len(no_extra_overlap))

    dedup = no_extra_overlap.drop_duplicates(
        subset=["head", "relation", "tail"]
    ).copy()

    after_dedup = int(len(dedup))

    report = {
        "before_filter": before,
        "after_coverage_filter": after_coverage,
        "coverage_removed": int(before - after_coverage),
        "after_remove_train_overlap": after_remove_train_overlap,
        "train_overlap_removed": int(after_coverage - after_remove_train_overlap),
        "after_remove_extra_forbidden": after_remove_extra_forbidden,
        "extra_forbidden_removed": int(
            after_remove_train_overlap - after_remove_extra_forbidden
        ),
        "after_dedup": after_dedup,
        "duplicate_removed": int(after_remove_extra_forbidden - after_dedup),
        "num_unique_heads_after_filter": int(dedup["head"].nunique()),
        "num_unique_tails_after_filter": int(dedup["tail"].nunique()),
    }

    return dedup.reset_index(drop=True), report


def sample_eval_rows(
    df: pd.DataFrame,
    n: int,
    seed: int,
    split_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(df) == 0:
        raise RuntimeError(f"No rows available for {split_name} after filtering.")

    if len(df) >= n:
        sampled = df.sample(n=n, random_state=seed).copy()
        mode = "sampled_fixed_n"
    else:
        sampled = df.copy()
        mode = "used_all_available_less_than_requested"

    sampled = sampled.sort_values(["tail", "head", "relation"]).reset_index(drop=True)

    report = {
        "split": split_name,
        "requested_n": int(n),
        "available_after_filter": int(len(df)),
        "final_n": int(len(sampled)),
        "mode": mode,
        "seed": int(seed),
    }

    return sampled, report


def build_train_enriched(raw_train: pd.DataFrame) -> pd.DataFrame:
    """
    Use only official train.tsv for enriched graph.

    This includes:
    - train target T triples
    - train non-T support triples

    This excludes:
    - official valid/test target T triples
    - selected valid/test target triples
    """
    enriched = raw_train[["head", "relation", "tail"]].copy()
    enriched = enriched.drop_duplicates(subset=["head", "relation", "tail"])
    enriched = enriched.sort_values(["relation", "head", "tail"]).reset_index(drop=True)
    return enriched


def build_id_maps(
    enriched: pd.DataFrame,
    train_t: pd.DataFrame,
    valid_t: pd.DataFrame,
    test_t: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, str], dict[str, int], dict[str, str]]:
    entities = set(enriched["head"]) | set(enriched["tail"])
    entities |= set(train_t["head"]) | set(train_t["tail"])
    entities |= set(valid_t["head"]) | set(valid_t["tail"])
    entities |= set(test_t["head"]) | set(test_t["tail"])

    relations = set(enriched["relation"]) | {TARGET_RELATION}

    entity2id = {name: idx for idx, name in enumerate(sorted(entities))}
    id2entity = {str(idx): name for name, idx in entity2id.items()}

    relation2id = {name: idx for idx, name in enumerate(sorted(relations))}
    id2relation = {str(idx): name for name, idx in relation2id.items()}

    return entity2id, id2entity, relation2id, id2relation


def build_type_map(
    all_entities: set[str],
    candidate_universe: set[str],
    query_universe: set[str],
) -> dict[str, str]:
    type_map: dict[str, str] = {}
    ambiguous = candidate_universe & query_universe

    for ent in sorted(all_entities):
        if ent in ambiguous:
            type_map[ent] = "Ambiguous_DrugLike_and_DiseaseLike"
        elif ent in candidate_universe:
            type_map[ent] = "Drug_or_Chemical_from_train_T_head"
        elif ent in query_universe:
            type_map[ent] = "Disease_from_train_T_tail"
        else:
            type_map[ent] = "Unknown_or_Other"

    return type_map


def convert_target_rows_to_json(
    df: pd.DataFrame,
    split: str,
    entity2id: dict[str, int],
    relation2id: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rel_id = int(relation2id[TARGET_RELATION])

    for idx, row in df.reset_index(drop=True).iterrows():
        head = str(row["head"])
        rel = str(row["relation"])
        tail = str(row["tail"])

        if rel != TARGET_RELATION:
            raise RuntimeError(f"Non-target relation in target split {split}: {rel}")

        rows.append(
            {
                "split": split,
                "row_index": int(idx),
                "triple": [head, rel, tail],
                "triple_id": [int(entity2id[head]), rel_id, int(entity2id[tail])],
                "type": "predicted_head",
                "relation": rel,
                "relation_normalized": TARGET_RELATION_NORMALIZED,
                "query_entity": tail,
                "query_entity_id": int(entity2id[tail]),
                "gold_entity": head,
                "gold_entity_id": int(entity2id[head]),
                "candidate_universe": "drug_only_from_train_T_heads",
                "query_universe": "disease_only_from_train_T_tails",
                "gold_injection": False,
                "source_dataset": "PharmKG-8k",
                "setting": "setting_c_pharmkg",
            }
        )

    return rows


def run_leak_checks(
    train_t: pd.DataFrame,
    valid_t: pd.DataFrame,
    test_t: pd.DataFrame,
    enriched: pd.DataFrame,
    candidate_universe: set[str],
    query_universe: set[str],
    type_map: dict[str, str],
) -> dict[str, Any]:
    train_set = triples_as_set(train_t)
    valid_set = triples_as_set(valid_t)
    test_set = triples_as_set(test_t)
    enriched_set = triples_as_set(enriched)

    valid_train_overlap = valid_set & train_set
    test_train_overlap = test_set & train_set
    valid_test_overlap = valid_set & test_set

    selected_eval_targets = valid_set | test_set
    eval_target_in_enriched = selected_eval_targets & enriched_set

    valid_heads = set(valid_t["head"])
    test_heads = set(test_t["head"])
    valid_tails = set(valid_t["tail"])
    test_tails = set(test_t["tail"])

    type_counts = Counter(type_map.values())

    candidate_non_drug = [
        ent
        for ent in sorted(candidate_universe)
        if type_map.get(ent) != "Drug_or_Chemical_from_train_T_head"
    ]

    query_non_disease = [
        ent
        for ent in sorted(query_universe)
        if type_map.get(ent) != "Disease_from_train_T_tail"
    ]

    coverage_checks = {
        "valid_gold_drug_coverage_in_candidate_universe": float(
            len(valid_heads & candidate_universe) / max(1, len(valid_heads))
        ),
        "test_gold_drug_coverage_in_candidate_universe": float(
            len(test_heads & candidate_universe) / max(1, len(test_heads))
        ),
        "valid_query_disease_coverage_in_query_universe": float(
            len(valid_tails & query_universe) / max(1, len(valid_tails))
        ),
        "test_query_disease_coverage_in_query_universe": float(
            len(test_tails & query_universe) / max(1, len(test_tails))
        ),
    }

    exact_leak_checks = {
        "valid_positive_in_train_count": int(len(valid_train_overlap)),
        "test_positive_in_train_count": int(len(test_train_overlap)),
        "valid_test_positive_overlap_count": int(len(valid_test_overlap)),
        "selected_valid_or_test_target_in_train_enriched_count": int(
            len(eval_target_in_enriched)
        ),
    }

    failures: list[str] = []

    for key, value in exact_leak_checks.items():
        if value != 0:
            failures.append(f"{key}_nonzero")

    for key, value in coverage_checks.items():
        if abs(value - 1.0) > 1e-12:
            failures.append(f"{key}_not_1")

    if len(candidate_non_drug) != 0:
        failures.append("candidate_non_drug_count_nonzero")

    if len(query_non_disease) != 0:
        failures.append("query_non_disease_count_nonzero")

    leak = {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "target_relation": TARGET_RELATION,
        "target_relation_normalized": TARGET_RELATION_NORMALIZED,
        "gold_injection": False,
        "sizes": {
            "train": int(len(train_t)),
            "valid": int(len(valid_t)),
            "test": int(len(test_t)),
            "candidate_universe_size": int(len(candidate_universe)),
            "query_universe_size": int(len(query_universe)),
            "train_enriched_triples": int(len(enriched)),
        },
        "exact_leak_checks": exact_leak_checks,
        "coverage_checks": coverage_checks,
        "type_checks": {
            "type_counts": dict(type_counts),
            "candidate_non_drug_count": int(len(candidate_non_drug)),
            "query_non_disease_count": int(len(query_non_disease)),
            "candidate_non_drug_examples": candidate_non_drug[:20],
            "query_non_disease_examples": query_non_disease[:20],
        },
        "examples": {
            "valid_positive_in_train_examples": [
                list(x) for x in sorted(valid_train_overlap)[:20]
            ],
            "test_positive_in_train_examples": [
                list(x) for x in sorted(test_train_overlap)[:20]
            ],
            "valid_test_positive_overlap_examples": [
                list(x) for x in sorted(valid_test_overlap)[:20]
            ],
            "selected_eval_target_in_enriched_examples": [
                list(x) for x in sorted(eval_target_in_enriched)[:20]
            ],
        },
    }

    return leak


def build_split_summary(
    raw: dict[str, pd.DataFrame],
    train_t: pd.DataFrame,
    valid_t: pd.DataFrame,
    test_t: pd.DataFrame,
    valid_filter_report: dict[str, Any],
    test_filter_report: dict[str, Any],
    valid_sample_report: dict[str, Any],
    test_sample_report: dict[str, Any],
    enriched: pd.DataFrame,
    entity2id: dict[str, int],
    relation2id: dict[str, int],
    candidate_universe: set[str],
    query_universe: set[str],
    leak: dict[str, Any],
    valid_size: int,
    test_size: int,
) -> dict[str, Any]:
    raw_target_counts = {
        split: int((df["relation"] == TARGET_RELATION).sum())
        for split, df in raw.items()
    }

    enriched_relation_counts = Counter(enriched["relation"].tolist())

    if (
        leak["decision"] == "PASS"
        and len(valid_t) == valid_size
        and len(test_t) == test_size
    ):
        decision = "SPLIT_GRAPH_READY"
    elif leak["decision"] == "PASS":
        decision = "PARTIAL_READY_SMALL_EVAL_SPLIT"
    else:
        decision = "SPLIT_GRAPH_LEAK_CHECK_FAILED"

    summary = {
        "week": 22,
        "day": 4,
        "decision": decision,
        "dataset": "PharmKG-8k",
        "setting": "setting_c_pharmkg",
        "task": {
            "task_template": "(?, T, disease)",
            "target_relation_raw": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "prediction_type": "predicted_head",
            "missing_entity_type": "Drug_or_Chemical",
            "query_entity_type": "Disease",
            "candidate_universe": "drug_only_from_train_T_heads",
            "query_universe": "disease_only_from_train_T_tails",
            "gold_injection": False,
        },
        "raw_target_counts": raw_target_counts,
        "filtering": {
            "valid": valid_filter_report,
            "test": test_filter_report,
        },
        "sampling": {
            "valid": valid_sample_report,
            "test": test_sample_report,
        },
        "final_split_sizes": {
            "train": int(len(train_t)),
            "valid": int(len(valid_t)),
            "test": int(len(test_t)),
        },
        "entity_relation_counts": {
            "num_entities": int(len(entity2id)),
            "num_relations": int(len(relation2id)),
            "num_candidate_drugs": int(len(candidate_universe)),
            "num_query_diseases": int(len(query_universe)),
            "train_enriched_triples": int(len(enriched)),
            "train_enriched_relation_counts": {
                str(k): int(v) for k, v in enriched_relation_counts.most_common()
            },
        },
        "leak_check_decision": leak["decision"],
        "files": {
            "train_json": str(SPLIT_DIR / "train.json"),
            "valid_json": str(SPLIT_DIR / "valid.json"),
            "test_json": str(SPLIT_DIR / "test.json"),
            "train_target_tsv": str(SPLIT_DIR / "train_target.tsv"),
            "valid_target_tsv": str(SPLIT_DIR / "valid_target.tsv"),
            "test_target_tsv": str(SPLIT_DIR / "test_target.tsv"),
            "candidate_universe": str(SPLIT_DIR / "candidate_universe.json"),
            "query_universe": str(SPLIT_DIR / "query_universe.json"),
            "entity2id": str(GRAPH_DIR / "entity2id.json"),
            "id2entity": str(GRAPH_DIR / "id2entity.json"),
            "relation2id": str(GRAPH_DIR / "relation2id.json"),
            "id2relation": str(GRAPH_DIR / "id2relation.json"),
            "type_map": str(GRAPH_DIR / "type_map.json"),
            "train_enriched_tsv": str(GRAPH_DIR / "train_enriched.tsv"),
        },
    }

    return summary


def write_markdown_report(summary: dict[str, Any], leak: dict[str, Any]) -> None:
    rel_counts = summary["entity_relation_counts"]["train_enriched_relation_counts"]
    rel_lines = "\n".join(f"- `{rel}`: {cnt}" for rel, cnt in rel_counts.items())

    type_counts_json = json.dumps(
        leak["type_checks"]["type_counts"],
        ensure_ascii=False,
        indent=2,
    )
    type_counts_block = "\n".join("    " + line for line in type_counts_json.splitlines())

    md = f"""# Week 22 Day 4 — Build PharmKG Splits, Enriched Graph, and Leak Checks

## Decision

`{summary["decision"]}`

Leak-check decision:

`{leak["decision"]}`

## Task

- Dataset: PharmKG-8k
- Setting: `setting_c_pharmkg`
- Task: `(?, T, disease)`
- Target relation raw: `T`
- Target relation normalized: `therapeutic_association_proxy`
- Prediction type: `predicted_head`
- Missing entity type: `Drug_or_Chemical`
- Query entity type: `Disease`
- Candidate universe: `drug_only_from_train_T_heads`
- Query universe: `disease_only_from_train_T_tails`
- Gold injection: `false`

## Raw target relation counts

| Split | # T triples |
|---|---:|
| train | {summary["raw_target_counts"]["train"]} |
| valid | {summary["raw_target_counts"]["valid"]} |
| test | {summary["raw_target_counts"]["test"]} |

## Filtering and sampling

| Split | Before | After coverage | After train overlap removal | After extra forbidden removal | After dedup | Final |
|---|---:|---:|---:|---:|---:|---:|
| valid | {summary["filtering"]["valid"]["before_filter"]} | {summary["filtering"]["valid"]["after_coverage_filter"]} | {summary["filtering"]["valid"]["after_remove_train_overlap"]} | {summary["filtering"]["valid"]["after_remove_extra_forbidden"]} | {summary["filtering"]["valid"]["after_dedup"]} | {summary["final_split_sizes"]["valid"]} |
| test | {summary["filtering"]["test"]["before_filter"]} | {summary["filtering"]["test"]["after_coverage_filter"]} | {summary["filtering"]["test"]["after_remove_train_overlap"]} | {summary["filtering"]["test"]["after_remove_extra_forbidden"]} | {summary["filtering"]["test"]["after_dedup"]} | {summary["final_split_sizes"]["test"]} |

Train positives:

- train = **{summary["final_split_sizes"]["train"]}**

## Entity and graph stats

- Num entities: **{summary["entity_relation_counts"]["num_entities"]}**
- Num relations: **{summary["entity_relation_counts"]["num_relations"]}**
- Num candidate drugs: **{summary["entity_relation_counts"]["num_candidate_drugs"]}**
- Num query diseases: **{summary["entity_relation_counts"]["num_query_diseases"]}**
- Train enriched triples: **{summary["entity_relation_counts"]["train_enriched_triples"]}**

## Train enriched relation counts

{rel_lines}

## Exact leak checks

| Check | Value |
|---|---:|
| valid positive in train | {leak["exact_leak_checks"]["valid_positive_in_train_count"]} |
| test positive in train | {leak["exact_leak_checks"]["test_positive_in_train_count"]} |
| valid/test positive overlap | {leak["exact_leak_checks"]["valid_test_positive_overlap_count"]} |
| selected valid/test target in train_enriched | {leak["exact_leak_checks"]["selected_valid_or_test_target_in_train_enriched_count"]} |

## Coverage checks

| Check | Value |
|---|---:|
| valid gold drug coverage | {leak["coverage_checks"]["valid_gold_drug_coverage_in_candidate_universe"]:.6f} |
| test gold drug coverage | {leak["coverage_checks"]["test_gold_drug_coverage_in_candidate_universe"]:.6f} |
| valid query disease coverage | {leak["coverage_checks"]["valid_query_disease_coverage_in_query_universe"]:.6f} |
| test query disease coverage | {leak["coverage_checks"]["test_query_disease_coverage_in_query_universe"]:.6f} |

## Type checks

- Candidate non-drug count: **{leak["type_checks"]["candidate_non_drug_count"]}**
- Query non-disease count: **{leak["type_checks"]["query_non_disease_count"]}**

Type counts JSON:

{type_counts_block}

## Files written

- `{SPLIT_DIR / "train.json"}`
- `{SPLIT_DIR / "valid.json"}`
- `{SPLIT_DIR / "test.json"}`
- `{GRAPH_DIR / "entity2id.json"}`
- `{GRAPH_DIR / "relation2id.json"}`
- `{GRAPH_DIR / "type_map.json"}`
- `{GRAPH_DIR / "train_enriched.tsv"}`
- `{SPLIT_SUMMARY_PATH}`
- `{LEAK_CHECK_PATH}`
- `{REPORT_PATH}`

## Next step: Day 5

Rerun six structure baselines as top-20 candidate generators:

- TransE
- DistMult
- ComplEx
- RotatE
- R-GCN
- HRGAT

All baselines must use:

- candidate universe = `drug_only_from_train_T_heads`
- top_k = 20
- gold_injection = false
- reviewer-safe RR@20
"""
    REPORT_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-size", type=int, default=DEFAULT_VALID_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    ensure_dirs()

    if not TASK_SPEC_PATH.exists():
        raise FileNotFoundError(
            f"Missing Day 3 task spec: {TASK_SPEC_PATH}. "
            "Run scripts/pharmkg/select_task_schema.py first."
        )

    task_spec = read_json(TASK_SPEC_PATH)
    if task_spec.get("decision") not in {
        "GO_TASK_SELECTED_PROXY_SCHEMA",
        "GO_TASK_SELECTED",
    }:
        raise RuntimeError(
            f"Day 3 task spec decision is not GO: {task_spec.get('decision')}"
        )

    raw = load_raw_splits()

    train_t = target_only(raw["train"])
    valid_t_raw = target_only(raw["valid"])
    test_t_raw = target_only(raw["test"])

    train_t = train_t.drop_duplicates(
        subset=["head", "relation", "tail"]
    ).reset_index(drop=True)

    train_t_set = triples_as_set(train_t)

    candidate_universe = set(train_t["head"])
    query_universe = set(train_t["tail"])

    valid_filtered, valid_filter_report = filter_eval_target_rows(
        eval_t=valid_t_raw,
        train_t_set=train_t_set,
        candidate_universe=candidate_universe,
        query_universe=query_universe,
        extra_forbidden_target_triples=set(),
    )

    valid_t, valid_sample_report = sample_eval_rows(
        valid_filtered,
        n=args.valid_size,
        seed=args.seed,
        split_name="valid",
    )

    selected_valid_set = triples_as_set(valid_t)

    test_filtered, test_filter_report = filter_eval_target_rows(
        eval_t=test_t_raw,
        train_t_set=train_t_set,
        candidate_universe=candidate_universe,
        query_universe=query_universe,
        extra_forbidden_target_triples=selected_valid_set,
    )

    test_t, test_sample_report = sample_eval_rows(
        test_filtered,
        n=args.test_size,
        seed=args.seed + 1,
        split_name="test",
    )

    enriched = build_train_enriched(raw["train"])

    entity2id, id2entity, relation2id, id2relation = build_id_maps(
        enriched=enriched,
        train_t=train_t,
        valid_t=valid_t,
        test_t=test_t,
    )

    all_entities = set(entity2id.keys())
    type_map = build_type_map(
        all_entities=all_entities,
        candidate_universe=candidate_universe,
        query_universe=query_universe,
    )

    train_rows = convert_target_rows_to_json(
        train_t,
        "train",
        entity2id,
        relation2id,
    )
    valid_rows = convert_target_rows_to_json(
        valid_t,
        "valid",
        entity2id,
        relation2id,
    )
    test_rows = convert_target_rows_to_json(
        test_t,
        "test",
        entity2id,
        relation2id,
    )

    write_json(train_rows, SPLIT_DIR / "train.json")
    write_json(valid_rows, SPLIT_DIR / "valid.json")
    write_json(test_rows, SPLIT_DIR / "test.json")

    write_target_tsv(train_t, SPLIT_DIR / "train_target.tsv")
    write_target_tsv(valid_t, SPLIT_DIR / "valid_target.tsv")
    write_target_tsv(test_t, SPLIT_DIR / "test_target.tsv")

    write_json(sorted(candidate_universe), SPLIT_DIR / "candidate_universe.json")
    write_json(sorted(query_universe), SPLIT_DIR / "query_universe.json")

    write_json(entity2id, GRAPH_DIR / "entity2id.json")
    write_json(id2entity, GRAPH_DIR / "id2entity.json")
    write_json(relation2id, GRAPH_DIR / "relation2id.json")
    write_json(id2relation, GRAPH_DIR / "id2relation.json")
    write_json(type_map, GRAPH_DIR / "type_map.json")

    enriched.to_csv(
        GRAPH_DIR / "train_enriched.tsv",
        sep="\t",
        header=False,
        index=False,
    )

    enriched_counts = Counter(enriched["relation"].tolist())
    write_json(
        {str(k): int(v) for k, v in enriched_counts.most_common()},
        GRAPH_DIR / "train_enriched_relation_counts.json",
    )

    leak = run_leak_checks(
        train_t=train_t,
        valid_t=valid_t,
        test_t=test_t,
        enriched=enriched,
        candidate_universe=candidate_universe,
        query_universe=query_universe,
        type_map=type_map,
    )

    summary = build_split_summary(
        raw=raw,
        train_t=train_t,
        valid_t=valid_t,
        test_t=test_t,
        valid_filter_report=valid_filter_report,
        test_filter_report=test_filter_report,
        valid_sample_report=valid_sample_report,
        test_sample_report=test_sample_report,
        enriched=enriched,
        entity2id=entity2id,
        relation2id=relation2id,
        candidate_universe=candidate_universe,
        query_universe=query_universe,
        leak=leak,
        valid_size=args.valid_size,
        test_size=args.test_size,
    )

    write_json(summary, SPLIT_SUMMARY_PATH)
    write_json(leak, LEAK_CHECK_PATH)
    write_markdown_report(summary, leak)

    print("Saved:")
    print(f"  {SPLIT_DIR / 'train.json'}")
    print(f"  {SPLIT_DIR / 'valid.json'}")
    print(f"  {SPLIT_DIR / 'test.json'}")
    print(f"  {GRAPH_DIR / 'entity2id.json'}")
    print(f"  {GRAPH_DIR / 'relation2id.json'}")
    print(f"  {GRAPH_DIR / 'type_map.json'}")
    print(f"  {GRAPH_DIR / 'train_enriched.tsv'}")
    print(f"  {SPLIT_SUMMARY_PATH}")
    print(f"  {LEAK_CHECK_PATH}")
    print(f"  {REPORT_PATH}")

    print("\nDecision:", summary["decision"])
    print("Leak decision:", leak["decision"])
    print("Split sizes:", summary["final_split_sizes"])
    print("Num entities:", summary["entity_relation_counts"]["num_entities"])
    print("Num relations:", summary["entity_relation_counts"]["num_relations"])
    print("Num candidate drugs:", summary["entity_relation_counts"]["num_candidate_drugs"])
    print("Num query diseases:", summary["entity_relation_counts"]["num_query_diseases"])
    print("Train enriched triples:", summary["entity_relation_counts"]["train_enriched_triples"])

    print("\nFiltering:")
    print(json.dumps(summary["filtering"], ensure_ascii=False, indent=2))

    print("\nLeak checks:")
    print(json.dumps(leak["exact_leak_checks"], ensure_ascii=False, indent=2))

    print("\nCoverage checks:")
    print(json.dumps(leak["coverage_checks"], ensure_ascii=False, indent=2))

    if summary["decision"] != "SPLIT_GRAPH_READY":
        raise RuntimeError(
            "Day 4 did not reach SPLIT_GRAPH_READY. "
            f"decision={summary['decision']}, "
            f"leak={leak['decision']}, "
            f"failures={leak['failures']}"
        )


if __name__ == "__main__":
    main()