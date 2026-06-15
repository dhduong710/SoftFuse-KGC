#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week 22 Day 3: Select task, relation, schema, and entity types for PharmKG.

Input:
- data/raw/pharmkg/PharmKG-8k/train.tsv
- data/raw/pharmkg/PharmKG-8k/valid.tsv
- data/raw/pharmkg/PharmKG-8k/test.tsv

Output:
- outputs/pharmkg/dataset2_task_spec.json
- dataset/setting_c_pharmkg/task_spec/task_spec.json
- dataset/setting_c_pharmkg/task_spec/relation_role_analysis.json
- dataset/setting_c_pharmkg/task_spec/entity_type_summary.json
- dataset/setting_c_pharmkg/task_spec/type_map_preliminary.json
- dataset/setting_c_pharmkg/task_spec/target_relation_rows_summary.json
- outputs/pharmkg/reports/day3_task_and_schema_selection.md

Important:
- We do NOT call relation T "clinical indication".
- We use it as treatment / therapeutic association proxy unless a relation label map confirms otherwise.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
RAW_DIR = ROOT / "data" / "raw" / "pharmkg" / "PharmKG-8k"

RESULT_DIR = ROOT / "outputs" / "pharmkg"
REPORT_DIR = ROOT / "outputs" / "pharmkg" / "reports"
TASK_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "task_spec"

TASK_SPEC_RESULT_PATH = RESULT_DIR / "dataset2_task_spec.json"
TASK_SPEC_PATH = TASK_DIR / "task_spec.json"
RELATION_ROLE_PATH = TASK_DIR / "relation_role_analysis.json"
ENTITY_TYPE_SUMMARY_PATH = TASK_DIR / "entity_type_summary.json"
TYPE_MAP_PRELIM_PATH = TASK_DIR / "type_map_preliminary.json"
TARGET_ROWS_SUMMARY_PATH = TASK_DIR / "target_relation_rows_summary.json"
REPORT_PATH = REPORT_DIR / "day3_task_and_schema_selection.md"


SPLIT_FILES = {
    "train": RAW_DIR / "train.tsv",
    "valid": RAW_DIR / "valid.tsv",
    "test": RAW_DIR / "test.tsv",
}

TARGET_RELATION = "T"
SEED = 2025


def ensure_dirs() -> None:
    for p in [RESULT_DIR, REPORT_DIR, TASK_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_split(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
        dtype=str,
        keep_default_na=False,
    )
    df["split"] = split
    return df


def load_all_splits() -> dict[str, pd.DataFrame]:
    return {split: read_split(path, split) for split, path in SPLIT_FILES.items()}


def sample_rows(df: pd.DataFrame, n: int = 10) -> list[dict]:
    if len(df) == 0:
        return []
    return df.head(n).to_dict(orient="records")


def summarize_target_relation(dfs: dict[str, pd.DataFrame]) -> dict:
    out = {
        "target_relation_raw": TARGET_RELATION,
        "per_split": {},
        "all": {},
    }

    all_t = []

    for split, df in dfs.items():
        tdf = df[df["relation"] == TARGET_RELATION].copy()
        all_t.append(tdf)

        heads = set(tdf["head"])
        tails = set(tdf["tail"])
        overlap = heads & tails

        out["per_split"][split] = {
            "num_triples": int(len(tdf)),
            "num_unique_heads": int(len(heads)),
            "num_unique_tails": int(len(tails)),
            "num_head_tail_overlap": int(len(overlap)),
            "head_tail_overlap_rate_over_heads": float(len(overlap) / max(1, len(heads))),
            "sample_rows": sample_rows(tdf, 15),
        }

    tall = pd.concat(all_t, ignore_index=True)
    heads_all = set(tall["head"])
    tails_all = set(tall["tail"])
    overlap_all = heads_all & tails_all

    train_t = dfs["train"][dfs["train"]["relation"] == TARGET_RELATION]
    train_heads = set(train_t["head"])
    train_tails = set(train_t["tail"])

    coverage = {}
    for split in ["valid", "test"]:
        tdf = dfs[split][dfs[split]["relation"] == TARGET_RELATION]
        split_heads = set(tdf["head"])
        split_tails = set(tdf["tail"])
        coverage[split] = {
            "drug_head_coverage_in_train_T_heads": float(len(split_heads & train_heads) / max(1, len(split_heads))),
            "disease_tail_coverage_in_train_T_tails": float(len(split_tails & train_tails) / max(1, len(split_tails))),
            "num_unseen_heads_vs_train_T": int(len(split_heads - train_heads)),
            "num_unseen_tails_vs_train_T": int(len(split_tails - train_tails)),
        }

    out["all"] = {
        "num_triples": int(len(tall)),
        "num_unique_heads_as_drug_candidates": int(len(heads_all)),
        "num_unique_tails_as_disease_queries": int(len(tails_all)),
        "num_head_tail_overlap": int(len(overlap_all)),
        "head_tail_overlap_rate_over_heads": float(len(overlap_all) / max(1, len(heads_all))),
        "head_tail_overlap_rate_over_tails": float(len(overlap_all) / max(1, len(tails_all))),
        "sample_rows": sample_rows(tall, 30),
        "coverage_vs_train_T": coverage,
    }

    return out


def build_preliminary_type_map(dfs: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    all_df = pd.concat(dfs.values(), ignore_index=True)
    tdf = all_df[all_df["relation"] == TARGET_RELATION]

    drug_like = set(tdf["head"])
    disease_like = set(tdf["tail"])
    all_entities = set(all_df["head"]) | set(all_df["tail"])

    ambiguous = drug_like & disease_like

    type_map = {}
    for e in sorted(all_entities):
        if e in ambiguous:
            # Keep this explicit; later Day 4 can decide whether to exclude ambiguous nodes.
            etype = "Ambiguous_DrugLike_and_DiseaseLike"
        elif e in drug_like:
            etype = "Drug_or_Chemical_from_T_head"
        elif e in disease_like:
            etype = "Disease_from_T_tail"
        else:
            etype = "Unknown_or_Other"
        type_map[e] = etype

    summary = {
        "construction_rule": {
            "drug_or_chemical": "entities appearing as head in relation T",
            "disease": "entities appearing as tail in relation T",
            "ambiguous": "entities appearing as both head and tail in relation T",
            "unknown_or_other": "all remaining entities in PharmKG-8k splits",
        },
        "num_total_entities_seen": int(len(all_entities)),
        "num_drug_or_chemical_from_T_head": int(len(drug_like - ambiguous)),
        "num_disease_from_T_tail": int(len(disease_like - ambiguous)),
        "num_ambiguous_drug_and_disease": int(len(ambiguous)),
        "num_unknown_or_other": int(len(all_entities - drug_like - disease_like)),
        "ambiguous_examples": sorted(list(ambiguous))[:50],
        "warning": (
            "This is a preliminary type map inferred from relation T direction. "
            "It should be reported as a task-specific candidate/query type map, "
            "not as the original PharmKG full entity ontology."
        ),
    }

    return type_map, summary


def relation_role_analysis(dfs: dict[str, pd.DataFrame], type_map: dict) -> list[dict]:
    all_df = pd.concat(dfs.values(), ignore_index=True)

    drug_types = {"Drug_or_Chemical_from_T_head", "Ambiguous_DrugLike_and_DiseaseLike"}
    disease_types = {"Disease_from_T_tail", "Ambiguous_DrugLike_and_DiseaseLike"}

    rows = []
    for rel, rdf in all_df.groupby("relation"):
        heads = list(rdf["head"])
        tails = list(rdf["tail"])
        h_types = [type_map.get(h, "Unknown_or_Other") for h in heads]
        t_types = [type_map.get(t, "Unknown_or_Other") for t in tails]

        h_counter = Counter(h_types)
        t_counter = Counter(t_types)

        n = len(rdf)
        head_drug_rate = sum(h_counter[t] for t in drug_types) / max(1, n)
        tail_drug_rate = sum(t_counter[t] for t in drug_types) / max(1, n)
        head_disease_rate = sum(h_counter[t] for t in disease_types) / max(1, n)
        tail_disease_rate = sum(t_counter[t] for t in disease_types) / max(1, n)

        if rel == TARGET_RELATION:
            proposed_role = "target_relation"
            notes = "Candidate therapeutic/treatment proxy relation; used for drug-only head prediction."
        elif head_drug_rate > 0.10 or tail_drug_rate > 0.10 or head_disease_rate > 0.10 or tail_disease_rate > 0.10:
            proposed_role = "support_relation_candidate"
            notes = "Touches inferred Drug/Disease task entities; can be retained in enriched graph pending Day 4 graph build."
        else:
            proposed_role = "unknown_or_other_support_candidate"
            notes = "Does not strongly touch inferred T-based drug/disease sets; keep as optional support only if needed."

        rows.append({
            "relation": rel,
            "num_triples_all_splits": int(n),
            "num_unique_heads": int(rdf["head"].nunique()),
            "num_unique_tails": int(rdf["tail"].nunique()),
            "head_type_counts": dict(h_counter),
            "tail_type_counts": dict(t_counter),
            "head_drug_like_rate": float(head_drug_rate),
            "tail_drug_like_rate": float(tail_drug_rate),
            "head_disease_like_rate": float(head_disease_rate),
            "tail_disease_like_rate": float(tail_disease_rate),
            "proposed_role": proposed_role,
            "notes": notes,
            "sample_rows": sample_rows(rdf, 8),
        })

    rows.sort(key=lambda x: (x["proposed_role"] != "target_relation", -x["num_triples_all_splits"], x["relation"]))
    return rows


def decide_task(target_summary: dict, type_summary: dict) -> tuple[str, list[str]]:
    notes = []

    train_n = target_summary["per_split"]["train"]["num_triples"]
    valid_n = target_summary["per_split"]["valid"]["num_triples"]
    test_n = target_summary["per_split"]["test"]["num_triples"]

    num_drugs = target_summary["all"]["num_unique_heads_as_drug_candidates"]
    num_diseases = target_summary["all"]["num_unique_tails_as_disease_queries"]

    valid_cov_d = target_summary["all"]["coverage_vs_train_T"]["valid"]["drug_head_coverage_in_train_T_heads"]
    valid_cov_q = target_summary["all"]["coverage_vs_train_T"]["valid"]["disease_tail_coverage_in_train_T_tails"]
    test_cov_d = target_summary["all"]["coverage_vs_train_T"]["test"]["drug_head_coverage_in_train_T_heads"]
    test_cov_q = target_summary["all"]["coverage_vs_train_T"]["test"]["disease_tail_coverage_in_train_T_tails"]

    if train_n == 0:
        return "NO_GO_TARGET_RELATION_T_NOT_FOUND", ["Relation T does not appear in train."]

    if valid_n < 500 or test_n < 500:
        notes.append("T relation has fewer than 500 valid/test triples; Day 4 may need smaller evaluation sets.")

    if num_drugs < 100:
        notes.append("Inferred drug universe from T heads is small.")
    if num_diseases < 100:
        notes.append("Inferred disease query set from T tails is small.")

    if min(valid_cov_d, valid_cov_q, test_cov_d, test_cov_q) < 0.95:
        notes.append(
            "Coverage of valid/test T heads/tails in train T is below 0.95; Day 4 should filter or resplit with coverage rule."
        )

    if type_summary["num_ambiguous_drug_and_disease"] > 0:
        notes.append(
            "Some entities appear as both T-head and T-tail; Day 4 should mark or exclude ambiguous entities if needed."
        )

    notes.append(
        "No explicit relation-label map was available in PharmKG-8k files; relation T is selected as a therapeutic association proxy, not clinical indication."
    )
    notes.append(
        "Preliminary type map is task-specific: T heads are Drug/Chemical candidates and T tails are Disease queries."
    )

    if train_n >= 1000 and valid_n >= 500 and test_n >= 500 and num_drugs >= 100 and num_diseases >= 100:
        return "GO_TASK_SELECTED_PROXY_SCHEMA", notes

    return "PARTIAL_READY_NEEDS_MANUAL_SCHEMA_CHECK", notes


def build_task_spec(
    decision: str,
    notes: list[str],
    target_summary: dict,
    type_summary: dict,
    relation_roles: list[dict],
) -> dict:
    support_candidates = [
        r["relation"] for r in relation_roles
        if r["proposed_role"] == "support_relation_candidate"
    ]
    unknown_support = [
        r["relation"] for r in relation_roles
        if r["proposed_role"] == "unknown_or_other_support_candidate"
    ]

    task_spec = {
        "week": 22,
        "day": 3,
        "decision": decision,
        "decision_notes": notes,

        "dataset": {
            "name": "PharmKG-8k",
            "setting_name": "setting_c_pharmkg",
            "source_split": "official PharmKG-8k train/valid/test TSV files",
            "raw_dir": str(RAW_DIR),
        },

        "task": {
            "task_template": "(?, T, disease)",
            "prediction_type": "predicted_head",
            "missing_entity_type": "Drug_or_Chemical",
            "query_entity_type": "Disease",
            "candidate_universe": "drug_only_from_T_heads",
            "target_relation_raw": TARGET_RELATION,
            "target_relation_normalized": "therapeutic_association_proxy",
            "target_relation_report_label": "therapeutic association proxy",
            "direction": "drug_or_chemical -> disease",
            "do_not_call_it": [
                "clinical indication",
                "PrimeKG indication",
                "confirmed treatment label without explicit relation-label map"
            ],
        },

        "counts": {
            "target_relation_per_split": {
                split: target_summary["per_split"][split]["num_triples"]
                for split in ["train", "valid", "test"]
            },
            "target_relation_all": target_summary["all"]["num_triples"],
            "num_candidate_drugs_from_T_heads": target_summary["all"]["num_unique_heads_as_drug_candidates"],
            "num_query_diseases_from_T_tails": target_summary["all"]["num_unique_tails_as_disease_queries"],
            "num_ambiguous_T_head_tail_entities": type_summary["num_ambiguous_drug_and_disease"],
        },

        "day4_split_policy_recommendation": {
            "preferred": (
                "Use official PharmKG-8k split as base, select relation T triples, "
                "then apply Setting-A-style coverage filtering/subsampling with seed 2025."
            ),
            "valid_test_size": "500/500 if coverage permits; otherwise 10%/10%",
            "coverage_rule": [
                "valid/test T-head drug must appear in train T heads",
                "valid/test T-tail disease must appear in train T tails",
                "no valid/test target triples included in train_enriched.tsv"
            ],
        },

        "relations": {
            "target_relation": TARGET_RELATION,
            "support_relation_candidates": support_candidates,
            "unknown_or_other_support_candidates": unknown_support,
            "excluded_relations_day3": [],
            "support_relation_policy": (
                "For Day 4, retain target train triples plus support relation candidates in train_enriched.tsv. "
                "Do not include valid/test T target triples."
            ),
        },

        "prompt_template_recommendation": {
            "head_prediction_question": "What drug is therapeutically associated with {disease}?",
            "biomedical_instruction": (
                "You are a biomedical scientist. The task is to predict the answer based on the given question, "
                "and you only need to answer one entity. The answer must be in (candidate list)."
            ),
        },

        "reviewer_safe_protocol": {
            "top_k": 20,
            "gold_injection": False,
            "main_metric": "reviewer_safe_mrr_at20",
            "rr_policy": "RR = 1/rank if rank <= 20 else 0",
            "absent_rank_sentinel": 21,
        },
    }

    return task_spec


def write_report(task_spec: dict, target_summary: dict, type_summary: dict, relation_roles: list[dict]) -> None:
    target_samples = target_summary["all"]["sample_rows"][:12]
    sample_md = "\n".join(
        f"- `{x['head']}    {x['relation']}    {x['tail']}`"
        for x in target_samples
    )

    role_rows = []
    for r in relation_roles[:28]:
        role_rows.append(
            f"| {r['relation']} | {r['num_triples_all_splits']} | "
            f"{r['head_drug_like_rate']:.3f} | {r['tail_disease_like_rate']:.3f} | "
            f"{r['proposed_role']} |"
        )
    role_table = "\n".join(role_rows)

    md = f"""# Week 22 Day 3 — Task, Relation, Schema, and Entity-Type Selection

## Decision

`{task_spec["decision"]}`

## Decision notes

{chr(10).join(f"- {n}" for n in task_spec["decision_notes"])}

## Selected task

- Task template: `{task_spec["task"]["task_template"]}`
- Prediction type: `{task_spec["task"]["prediction_type"]}`
- Missing entity type: `{task_spec["task"]["missing_entity_type"]}`
- Query entity type: `{task_spec["task"]["query_entity_type"]}`
- Candidate universe: `{task_spec["task"]["candidate_universe"]}`
- Raw relation code: `{task_spec["task"]["target_relation_raw"]}`
- Normalized relation name: `{task_spec["task"]["target_relation_normalized"]}`
- Direction: `{task_spec["task"]["direction"]}`

## Important report wording

Use:

`therapeutic association proxy`

Do **not** call this relation:

- clinical indication
- PrimeKG indication
- confirmed treatment label

unless an explicit PharmKG relation-label map is later found.

## Target relation T counts

| Split | # T triples |
|---|---:|
| train | {target_summary["per_split"]["train"]["num_triples"]} |
| valid | {target_summary["per_split"]["valid"]["num_triples"]} |
| test | {target_summary["per_split"]["test"]["num_triples"]} |
| all | {target_summary["all"]["num_triples"]} |

## Inferred entity types from T direction

- Candidate drug/chemical entities from T heads: **{target_summary["all"]["num_unique_heads_as_drug_candidates"]}**
- Disease query entities from T tails: **{target_summary["all"]["num_unique_tails_as_disease_queries"]}**
- Ambiguous T head/tail entities: **{type_summary["num_ambiguous_drug_and_disease"]}**
- Unknown/other entities: **{type_summary["num_unknown_or_other"]}**

## Coverage of valid/test T entities in train T

| Split | Drug-head coverage | Disease-tail coverage | Unseen heads | Unseen tails |
|---|---:|---:|---:|---:|
| valid | {target_summary["all"]["coverage_vs_train_T"]["valid"]["drug_head_coverage_in_train_T_heads"]:.4f} | {target_summary["all"]["coverage_vs_train_T"]["valid"]["disease_tail_coverage_in_train_T_tails"]:.4f} | {target_summary["all"]["coverage_vs_train_T"]["valid"]["num_unseen_heads_vs_train_T"]} | {target_summary["all"]["coverage_vs_train_T"]["valid"]["num_unseen_tails_vs_train_T"]} |
| test | {target_summary["all"]["coverage_vs_train_T"]["test"]["drug_head_coverage_in_train_T_heads"]:.4f} | {target_summary["all"]["coverage_vs_train_T"]["test"]["disease_tail_coverage_in_train_T_tails"]:.4f} | {target_summary["all"]["coverage_vs_train_T"]["test"]["num_unseen_heads_vs_train_T"]} | {target_summary["all"]["coverage_vs_train_T"]["test"]["num_unseen_tails_vs_train_T"]} |

## Sample T rows

{sample_md}

## Relation role analysis

| Relation | # triples | Head drug-like rate | Tail disease-like rate | Proposed role |
|---|---:|---:|---:|---|
{role_table}

## Day 4 recommendation

Use official PharmKG-8k split as base, select T triples, then apply Setting-A-style coverage filtering/subsampling with seed 2025.

Recommended Day 4 policy:

- valid/test size = 500/500 if coverage permits
- candidate universe = T-head drug/chemical entities
- query universe = T-tail disease entities
- train_enriched.tsv = train T target triples + support relation candidates
- do not include valid/test T target triples in train_enriched.tsv

## Files written

- `{TASK_SPEC_RESULT_PATH}`
- `{TASK_SPEC_PATH}`
- `{RELATION_ROLE_PATH}`
- `{ENTITY_TYPE_SUMMARY_PATH}`
- `{TYPE_MAP_PRELIM_PATH}`
- `{TARGET_ROWS_SUMMARY_PATH}`
- `{REPORT_PATH}`

## Next step

Day 4 should build:

- train/valid/test task rows
- entity2id/relation2id
- type_map
- train_enriched.tsv
- split_summary
- leak_check
"""
    REPORT_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    dfs = load_all_splits()
    target_summary = summarize_target_relation(dfs)
    type_map, type_summary = build_preliminary_type_map(dfs)
    relation_roles = relation_role_analysis(dfs, type_map)

    decision, notes = decide_task(target_summary, type_summary)

    task_spec = build_task_spec(
        decision=decision,
        notes=notes,
        target_summary=target_summary,
        type_summary=type_summary,
        relation_roles=relation_roles,
    )

    write_json(task_spec, TASK_SPEC_RESULT_PATH)
    write_json(task_spec, TASK_SPEC_PATH)
    write_json(relation_roles, RELATION_ROLE_PATH)
    write_json(type_summary, ENTITY_TYPE_SUMMARY_PATH)
    write_json(type_map, TYPE_MAP_PRELIM_PATH)
    write_json(target_summary, TARGET_ROWS_SUMMARY_PATH)
    write_report(task_spec, target_summary, type_summary, relation_roles)

    print("Saved:")
    print(f"  {TASK_SPEC_RESULT_PATH}")
    print(f"  {TASK_SPEC_PATH}")
    print(f"  {RELATION_ROLE_PATH}")
    print(f"  {ENTITY_TYPE_SUMMARY_PATH}")
    print(f"  {TYPE_MAP_PRELIM_PATH}")
    print(f"  {TARGET_ROWS_SUMMARY_PATH}")
    print(f"  {REPORT_PATH}")

    print("\nDecision:", decision)
    print("Target relation:", TARGET_RELATION)
    print("Normalized:", task_spec["task"]["target_relation_normalized"])
    print("Direction:", task_spec["task"]["direction"])
    print("T counts:", task_spec["counts"]["target_relation_per_split"])
    print("Num candidate drugs:", task_spec["counts"]["num_candidate_drugs_from_T_heads"])
    print("Num query diseases:", task_spec["counts"]["num_query_diseases_from_T_tails"])
    print("Num ambiguous:", task_spec["counts"]["num_ambiguous_T_head_tail_entities"])
    print("\nNotes:")
    for n in notes:
        print(" -", n)


if __name__ == "__main__":
    main()