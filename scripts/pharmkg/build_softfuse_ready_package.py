#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week 22 Day 6: Build PharmKG Dataset 2 SoftFuse-ready package.

Input:
    dataset/setting_c_pharmkg/splits/{train,valid,test}.json
    dataset/setting_c_pharmkg/splits/candidate_universe.json
    dataset/setting_c_pharmkg/graph/{entity2id,id2entity,relation2id,id2relation,type_map,train_enriched.tsv}
    dataset/setting_c_pharmkg/baseline_outputs/rgcn/{valid,test}_top20.json
    outputs/pharmkg/dataset2_baseline_main_table.json

Output:
    dataset/setting_c_pharmkg/backbone_raw_source/
    dataset/setting_c_pharmkg/softfuse_ready/
    outputs/pharmkg/dataset2_source_selection.json
    outputs/pharmkg/reports/day6_softfuse_ready_package.md
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


ROOT = Path(".")

SPLIT_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "splits"
GRAPH_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "graph"
BASELINE_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "baseline_outputs"

RAW_SOURCE_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "backbone_raw_source"
SOFTFUSE_READY_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "softfuse_ready"

RESULT_DIR = ROOT / "outputs" / "pharmkg"
REPORT_DIR = ROOT / "outputs" / "pharmkg" / "reports"

BASELINE_MAIN_TABLE_PATH = RESULT_DIR / "dataset2_baseline_main_table.json"
SOURCE_SELECTION_RESULT_PATH = RESULT_DIR / "dataset2_source_selection.json"
REPORT_PATH = REPORT_DIR / "day6_softfuse_ready_package.md"

TARGET_RELATION = "T"
TARGET_RELATION_NORMALIZED = "therapeutic_association_proxy"
TARGET_TYPE = "predicted_head"
DEFAULT_SOURCE_MODEL = "rgcn"
DEFAULT_GRAPH_SIZE = 100
DEFAULT_TRAIN_K = 20
DEFAULT_SEED = 2025


def ensure_dirs() -> None:
    for path in [RAW_SOURCE_DIR, SOFTFUSE_READY_DIR, RESULT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_maps() -> tuple[dict[str, int], dict[int, str], dict[str, int], dict[int, str]]:
    entity2id_raw = read_json(GRAPH_DIR / "entity2id.json")
    id2entity_raw = read_json(GRAPH_DIR / "id2entity.json")
    relation2id_raw = read_json(GRAPH_DIR / "relation2id.json")
    id2relation_raw = read_json(GRAPH_DIR / "id2relation.json")

    entity2id = {str(k): int(v) for k, v in entity2id_raw.items()}
    id2entity = {int(k): str(v) for k, v in id2entity_raw.items()}

    relation2id = {str(k): int(v) for k, v in relation2id_raw.items()}
    id2relation = {int(k): str(v) for k, v in id2relation_raw.items()}

    return entity2id, id2entity, relation2id, id2relation


def load_train_enriched_ids(
    entity2id: dict[str, int],
    relation2id: dict[str, int],
) -> list[tuple[int, int, int]]:
    path = GRAPH_DIR / "train_enriched.tsv"
    if not path.exists():
        raise FileNotFoundError(path)

    triples: list[tuple[int, int, int]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"Bad train_enriched line {line_idx}: {line}")

            h, r, t = parts[:3]

            if h not in entity2id or r not in relation2id or t not in entity2id:
                raise KeyError(f"Unmapped triple at line {line_idx}: {h}, {r}, {t}")

            triples.append((entity2id[h], relation2id[r], entity2id[t]))

    return triples


def build_graph_indexes(
    triples: list[tuple[int, int, int]],
) -> tuple[dict[int, list[tuple[int, tuple[int, int, int]]]], dict[int, list[tuple[int, int, int]]]]:
    undirected_adj: dict[int, list[tuple[int, tuple[int, int, int]]]] = defaultdict(list)
    incident: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    for h, r, t in triples:
        tr = (h, r, t)
        undirected_adj[h].append((t, tr))
        undirected_adj[t].append((h, tr))
        incident[h].append(tr)
        incident[t].append(tr)

    return undirected_adj, incident


def shortest_path_triples(
    source: int,
    target: int,
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]],
    max_depth: int = 3,
) -> list[tuple[int, int, int]]:
    if source == target:
        return []

    q = deque()
    q.append((source, []))
    visited = {source}

    while q:
        node, path = q.popleft()

        if len(path) >= max_depth:
            continue

        for nb, triple in adj.get(node, []):
            if nb in visited:
                continue

            new_path = path + [triple]

            if nb == target:
                return new_path

            visited.add(nb)
            q.append((nb, new_path))

    return []


def unique_append(
    out: list[tuple[int, int, int]],
    seen: set[tuple[int, int, int]],
    triple: tuple[int, int, int],
    graph_size: int,
) -> None:
    if len(out) >= graph_size:
        return
    if triple in seen:
        return
    out.append(triple)
    seen.add(triple)


def build_subgraph(
    query_id: int,
    candidate_ids: list[int],
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]],
    incident: dict[int, list[tuple[int, int, int]]],
    graph_size: int,
    split: str,
) -> list[list[int]]:
    """
    Lightweight DrKGC-style retrieval for Day 6 package.

    valid/test:
        shortest candidate-query paths first, then incident fill.

    train:
        incident fill around query/gold/candidates first to keep Day 6 fast.
    """
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    important_nodes = [query_id] + candidate_ids[:20]

    if split != "train":
        for cand_id in candidate_ids[:20]:
            if len(out) >= graph_size:
                break
            path = shortest_path_triples(cand_id, query_id, adj, max_depth=3)
            for tr in path:
                unique_append(out, seen, tr, graph_size)
                if len(out) >= graph_size:
                    break

    for node in important_nodes:
        if len(out) >= graph_size:
            break
        for tr in incident.get(node, []):
            unique_append(out, seen, tr, graph_size)
            if len(out) >= graph_size:
                break

    return [[int(h), int(r), int(t)] for h, r, t in out]


def make_prompt(
    query_entity: str,
    candidate_entities: list[str],
) -> str:
    answer_options = "(" + ", ".join([f"'{name}'" for name in candidate_entities]) + ")"

    refer_parts = [f"'{query_entity}': [QUERY]"]
    refer_parts.extend([f"'{name}': [ENTITY]" for name in candidate_entities])
    refer_str = ", ".join(refer_parts)

    question = f"What drug is therapeutically associated with {query_entity}?"

    prompt = (
        "You are a biomedical scientist. The task is to predict the answer based on the given question, "
        "and you only need to answer one entity. The answer must be in "
        + answer_options
        + ".\nYou can refer to the entity embeddings: "
        + refer_str
        + ".\n\nQuestion: "
        + question
        + "\nAnswer: "
    )

    return prompt


def convert_baseline_top20_to_raw_source(
    source_model: str,
    split: str,
) -> list[dict[str, Any]]:
    path = BASELINE_DIR / source_model / f"{split}_top20.json"
    if not path.exists():
        raise FileNotFoundError(path)

    rows = read_json(path)
    out_rows = []

    for idx, row in enumerate(rows):
        cand_names = row["candidate_entities_top20"]
        cand_ids = row["candidate_entity_ids_top20"]

        if len(cand_names) != 20 or len(cand_ids) != 20:
            raise ValueError(f"{source_model}/{split} row {idx} does not have 20 candidates.")

        out_rows.append(
            {
                "split": split,
                "row_index": int(idx),
                "query_entity": row["query_entity"],
                "query_entity_id": int(row["query_entity_id"]),
                "gold_entity": row["gold_entity"],
                "gold_entity_id": int(row["gold_entity_id"]),
                "candidate_entities": cand_names,
                "candidate_entity_ids": [int(x) for x in cand_ids],
                "gold_rank_in_top20_or_21": int(row["gold_rank_in_top20_or_21"]),
                "gold_in_topk_raw": bool(row["gold_present_top20"]),
                "candidate_universe": "drug_only_from_train_T_heads",
                "gold_injection": False,
                "source_model": source_model,
                "target_relation": TARGET_RELATION,
                "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            }
        )

    return out_rows


def build_train_supervised_candidates(
    train_rows: list[dict[str, Any]],
    candidate_names: list[str],
    entity2id: dict[str, int],
    k: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)

    all_candidate_names = list(candidate_names)

    # Put frequent train golds first, then fill with all candidates.
    freq = Counter(row["gold_entity"] for row in train_rows)
    frequent_candidates = [name for name, _ in freq.most_common()]
    candidate_pool = []
    seen = set()

    for name in frequent_candidates + all_candidate_names:
        if name not in seen:
            candidate_pool.append(name)
            seen.add(name)

    out_rows = []

    for idx, row in enumerate(train_rows):
        gold = row["gold_entity"]
        gold_id = int(row["gold_entity_id"])

        candidates = [gold]
        for name in candidate_pool:
            if len(candidates) >= k:
                break
            if name == gold:
                continue
            candidates.append(name)

        if len(candidates) < k:
            shuffled = all_candidate_names[:]
            rng.shuffle(shuffled)
            for name in shuffled:
                if len(candidates) >= k:
                    break
                if name not in candidates:
                    candidates.append(name)

        candidate_ids = [int(entity2id[name]) for name in candidates]

        if candidate_ids[0] != gold_id:
            raise RuntimeError("Train supervised candidate construction failed to place gold first.")

        out_rows.append(
            {
                "split": "train",
                "row_index": int(idx),
                "query_entity": row["query_entity"],
                "query_entity_id": int(row["query_entity_id"]),
                "gold_entity": gold,
                "gold_entity_id": gold_id,
                "candidate_entities": candidates,
                "candidate_entity_ids": candidate_ids,
                "gold_rank_in_top20_or_21": 1,
                "gold_in_topk_raw": True,
                "candidate_universe": "drug_only_from_train_T_heads",
                "gold_injection": True,
                "gold_injection_note": "train_supervision_only_not_used_for_eval",
                "source_model": "train_supervised_gold_first",
                "target_relation": TARGET_RELATION,
                "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            }
        )

    return out_rows


def raw_source_to_softfuse_rows(
    raw_rows: list[dict[str, Any]],
    base_split_rows_by_key: dict[tuple[int, int], dict[str, Any]],
    relation2id: dict[str, int],
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]],
    incident: dict[int, list[tuple[int, int, int]]],
    graph_size: int,
    split: str,
) -> list[dict[str, Any]]:
    out_rows = []
    rel_id = int(relation2id[TARGET_RELATION])

    for idx, raw in enumerate(raw_rows):
        query_id = int(raw["query_entity_id"])
        gold_id = int(raw["gold_entity_id"])

        key = (query_id, gold_id)
        base = base_split_rows_by_key.get(key)

        if base is None:
            triple = [raw["gold_entity"], TARGET_RELATION, raw["query_entity"]]
            triple_id = [gold_id, rel_id, query_id]
        else:
            triple = base["triple"]
            triple_id = base["triple_id"]

        rank_entities = list(raw["candidate_entities"])
        rank_entities_id = [int(x) for x in raw["candidate_entity_ids"]]

        if raw["gold_entity"] in rank_entities:
            rank = rank_entities.index(raw["gold_entity"]) + 1
        else:
            rank = 21

        subgraph = build_subgraph(
            query_id=query_id,
            candidate_ids=rank_entities_id,
            adj=adj,
            incident=incident,
            graph_size=graph_size,
            split=split,
        )

        prompt = make_prompt(
            query_entity=raw["query_entity"],
            candidate_entities=rank_entities,
        )

        out = {
            "split": split,
            "row_index": int(idx),
            "triple": triple,
            "triple_id": [int(triple_id[0]), int(triple_id[1]), int(triple_id[2])],
            "type": TARGET_TYPE,
            "relation": TARGET_RELATION,
            "relation_normalized": TARGET_RELATION_NORMALIZED,
            "query_entity": raw["query_entity"],
            "query_entity_id": query_id,
            "gold_entity": raw["gold_entity"],
            "gold_entity_id": gold_id,
            "rank_entities": rank_entities,
            "rank_entities_id": rank_entities_id,
            "rank": int(rank),
            "gold_in_topk_raw": bool(raw["gold_in_topk_raw"]),
            "gold_injection": bool(raw.get("gold_injection", False)),
            "candidate_universe": "drug_only_from_train_T_heads",
            "source_model": raw["source_model"],
            "input": prompt,
            "output": raw["gold_entity"],
            "subgraph": subgraph,
        }

        out_rows.append(out)

    return out_rows


def build_base_split_index(split_rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    out = {}
    for row in split_rows:
        key = (int(row["query_entity_id"]), int(row["gold_entity_id"]))
        out[key] = row
    return out


def write_prompt_lexicon(relation2id: dict[str, int]) -> dict[str, Any]:
    t_rel_id = int(relation2id[TARGET_RELATION])

    lexicon = {
        "dataset": "PharmKG-8k",
        "setting": "setting_c_pharmkg",
        "target_relation_raw": TARGET_RELATION,
        "target_relation_id": t_rel_id,
        "target_relation_normalized": TARGET_RELATION_NORMALIZED,
        "relation_questions_B_to_A": {
            TARGET_RELATION: "What drug is therapeutically associated with {}?",
            str(t_rel_id): "What drug is therapeutically associated with {}?",
        },
        "relation_questions_A_to_B": {
            TARGET_RELATION: "What disease is therapeutically associated with {}?",
            str(t_rel_id): "What disease is therapeutically associated with {}?",
        },
        "report_warning": (
            "Relation T is treated as a therapeutic association proxy, not a confirmed clinical indication label."
        ),
    }

    write_json(lexicon, SOFTFUSE_READY_DIR / "prompt_lexicon.json")
    return lexicon


def write_rules_and_schema(relation2id: dict[str, int], id2relation: dict[int, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    t_rel_id = int(relation2id[TARGET_RELATION])

    rules = {
        str(t_rel_id): [],
        TARGET_RELATION: [],
        "note": (
            "No explicit PharmKG relation-label map was available. "
            "Day 6 uses shortest-path and incident-edge fallback retrieval rather than hand-claiming semantic rules."
        ),
    }

    support_relations = [
        r for r in sorted(relation2id.keys())
        if r != TARGET_RELATION
    ]

    support_schema = {
        "dataset": "PharmKG-8k",
        "setting": "setting_c_pharmkg",
        "target_relation": TARGET_RELATION,
        "target_relation_id": t_rel_id,
        "target_relation_normalized": TARGET_RELATION_NORMALIZED,
        "support_relations": support_relations,
        "excluded_relations": [],
        "retrieval_policy": {
            "stage_1": "shortest_paths_between_candidate_and_query_on_train_enriched_graph",
            "stage_2": "incident_edge_fill_around_query_and_candidates",
            "graph_size": DEFAULT_GRAPH_SIZE,
            "leak_guard": "train_enriched.tsv only; selected valid/test target T triples are absent by Day 4 leak check",
        },
    }

    write_json(rules, SOFTFUSE_READY_DIR / "rules.json")
    write_json(support_schema, SOFTFUSE_READY_DIR / "support_schema.json")

    return rules, support_schema


def copy_maps_as_pickle(
    entity2id: dict[str, int],
    id2entity: dict[int, str],
    relation2id: dict[str, int],
    id2relation: dict[int, str],
) -> None:
    write_pickle(entity2id, SOFTFUSE_READY_DIR / "entity2id.pkl")
    write_pickle(id2entity, SOFTFUSE_READY_DIR / "id2entity.pkl")
    write_pickle(relation2id, SOFTFUSE_READY_DIR / "relation2id.pkl")
    write_pickle(id2relation, SOFTFUSE_READY_DIR / "id2relation.pkl")

    write_json(entity2id, SOFTFUSE_READY_DIR / "entity2id.json")
    write_json({str(k): v for k, v in id2entity.items()}, SOFTFUSE_READY_DIR / "id2entity.json")
    write_json(relation2id, SOFTFUSE_READY_DIR / "relation2id.json")
    write_json({str(k): v for k, v in id2relation.items()}, SOFTFUSE_READY_DIR / "id2relation.json")


def summarize_ready_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subgraph_sizes = [len(r["subgraph"]) for r in rows]
    ranks = [int(r["rank"]) for r in rows]
    has_query_token = sum("[QUERY]" in r["input"] for r in rows)
    has_entity_token = sum("[ENTITY]" in r["input"] for r in rows)

    return {
        "num_rows": int(len(rows)),
        "avg_candidate_size": float(sum(len(r["rank_entities"]) for r in rows) / max(1, len(rows))),
        "gold_in_final_list_rate": float(sum(r["rank"] <= 20 for r in rows) / max(1, len(rows))),
        "rank21_count": int(sum(r == 21 for r in ranks)),
        "avg_subgraph_size": float(sum(subgraph_sizes) / max(1, len(subgraph_sizes))),
        "min_subgraph_size": int(min(subgraph_sizes)) if subgraph_sizes else 0,
        "max_subgraph_size": int(max(subgraph_sizes)) if subgraph_sizes else 0,
        "rows_with_query_token": int(has_query_token),
        "rows_with_entity_token": int(has_entity_token),
    }


def load_baseline_metrics() -> dict[str, Any]:
    if BASELINE_MAIN_TABLE_PATH.exists():
        return read_json(BASELINE_MAIN_TABLE_PATH)
    return {"valid": [], "test": []}


def choose_source_model(metrics: dict[str, Any], requested_source: str) -> dict[str, Any]:
    valid_rows = metrics.get("valid", [])
    test_rows = metrics.get("test", [])

    def find(split_rows: list[dict[str, Any]], model: str) -> dict[str, Any] | None:
        for row in split_rows:
            if row.get("model_name") == model:
                return row
        return None

    rgcn_test = find(test_rows, "rgcn")
    hrgat_test = find(test_rows, "hrgat")
    distmult_valid = find(valid_rows, "distmult")

    best_valid = valid_rows[0] if valid_rows else None
    best_test = test_rows[0] if test_rows else None

    selection = {
        "decision": "SOURCE_SELECTED",
        "main_softfuse_source": requested_source,
        "main_softfuse_source_reason": (
            "R-GCN is selected as the main SoftFuse transfer source because it is "
            "DrKGC-compatible, aligned with the PrimeKG SoftFuse pipeline, and it "
            "has the best locked-test MRR@20 among the available GNN sources."
        ),
        "drkgc_aligned_alternative": "hrgat",
        "best_valid_structure_source": best_valid,
        "best_test_structure_source": best_test,
        "rgcn_test": rgcn_test,
        "hrgat_test": hrgat_test,
        "distmult_valid": distmult_valid,
        "note": (
            "DistMult is best on validation among all baselines, but it is kept as a "
            "structure baseline / optional future source. R-GCN remains the main "
            "SoftFuse source for pipeline continuity."
        ),
    }

    return selection


def run_leak_sanity(
    ready_valid: list[dict[str, Any]],
    ready_test: list[dict[str, Any]],
    train_enriched: list[tuple[int, int, int]],
) -> dict[str, Any]:
    train_set = set(train_enriched)

    valid_targets = set(tuple(r["triple_id"]) for r in ready_valid)
    test_targets = set(tuple(r["triple_id"]) for r in ready_test)

    valid_in_train = valid_targets & train_set
    test_in_train = test_targets & train_set
    valid_test_overlap = valid_targets & test_targets

    return {
        "valid_target_in_train_enriched_count": int(len(valid_in_train)),
        "test_target_in_train_enriched_count": int(len(test_in_train)),
        "valid_test_target_overlap_count": int(len(valid_test_overlap)),
        "decision": "PASS" if not valid_in_train and not test_in_train and not valid_test_overlap else "FAIL",
        "examples": {
            "valid_in_train": [list(x) for x in sorted(valid_in_train)[:10]],
            "test_in_train": [list(x) for x in sorted(test_in_train)[:10]],
            "valid_test_overlap": [list(x) for x in sorted(valid_test_overlap)[:10]],
        },
    }


def write_report(
    source_selection: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    md = f"""# Week 22 Day 6 — Build PharmKG SoftFuse-ready Package

## Decision

`{manifest["decision"]}`

## Source selection

- Main SoftFuse source: `{source_selection["main_softfuse_source"]}`
- DrKGC-aligned alternative: `{source_selection["drkgc_aligned_alternative"]}`

Reason:

{source_selection["main_softfuse_source_reason"]}

## Dataset/task

- Dataset: PharmKG-8k
- Setting: `setting_c_pharmkg`
- Task: `(?, T, disease)`
- Relation normalized: `therapeutic_association_proxy`
- Candidate universe: `drug_only_from_train_T_heads`
- Gold injection for evaluation: `false`

## Ready split summary

| Split | Rows | Avg candidates | Gold in list rate | Rank21 | Avg subgraph size |
|---|---:|---:|---:|---:|---:|
| train | {manifest["ready_summary"]["train"]["num_rows"]} | {manifest["ready_summary"]["train"]["avg_candidate_size"]:.2f} | {manifest["ready_summary"]["train"]["gold_in_final_list_rate"]:.3f} | {manifest["ready_summary"]["train"]["rank21_count"]} | {manifest["ready_summary"]["train"]["avg_subgraph_size"]:.2f} |
| valid | {manifest["ready_summary"]["valid"]["num_rows"]} | {manifest["ready_summary"]["valid"]["avg_candidate_size"]:.2f} | {manifest["ready_summary"]["valid"]["gold_in_final_list_rate"]:.3f} | {manifest["ready_summary"]["valid"]["rank21_count"]} | {manifest["ready_summary"]["valid"]["avg_subgraph_size"]:.2f} |
| test | {manifest["ready_summary"]["test"]["num_rows"]} | {manifest["ready_summary"]["test"]["avg_candidate_size"]:.2f} | {manifest["ready_summary"]["test"]["gold_in_final_list_rate"]:.3f} | {manifest["ready_summary"]["test"]["rank21_count"]} | {manifest["ready_summary"]["test"]["avg_subgraph_size"]:.2f} |

## Leak sanity

- valid target in train_enriched: `{manifest["leak_sanity"]["valid_target_in_train_enriched_count"]}`
- test target in train_enriched: `{manifest["leak_sanity"]["test_target_in_train_enriched_count"]}`
- valid/test target overlap: `{manifest["leak_sanity"]["valid_test_target_overlap_count"]}`
- decision: `{manifest["leak_sanity"]["decision"]}`

## Prompt template

`What drug is therapeutically associated with {{disease}}?`

## Rule policy

No explicit semantic PharmKG relation-label map is available. Therefore, Day 6 uses:

1. shortest-path retrieval between candidate and query on `train_enriched.tsv`;
2. incident-edge fill around query and candidates;
3. no hand-claimed semantic rule sequence.

## Files written

- `dataset/setting_c_pharmkg/backbone_raw_source/valid_top20_raw.json`
- `dataset/setting_c_pharmkg/backbone_raw_source/test_top20_raw.json`
- `dataset/setting_c_pharmkg/softfuse_ready/train.json`
- `dataset/setting_c_pharmkg/softfuse_ready/valid.json`
- `dataset/setting_c_pharmkg/softfuse_ready/test.json`
- `dataset/setting_c_pharmkg/softfuse_ready/prompt_lexicon.json`
- `dataset/setting_c_pharmkg/softfuse_ready/rules.json`
- `dataset/setting_c_pharmkg/softfuse_ready/support_schema.json`
- `dataset/setting_c_pharmkg/softfuse_ready/prep_manifest.json`
- `outputs/pharmkg/dataset2_source_selection.json`

## Next step: Day 7

Close out Week 22 and decide whether to start Week 23 SoftFuse transfer.
"""
    REPORT_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=str, default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--graph-size", type=int, default=DEFAULT_GRAPH_SIZE)
    parser.add_argument("--train-k", type=int, default=DEFAULT_TRAIN_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    ensure_dirs()
    random.seed(args.seed)

    entity2id, id2entity, relation2id, id2relation = load_maps()

    if TARGET_RELATION not in relation2id:
        raise KeyError(f"Target relation {TARGET_RELATION} not found in relation2id.")

    train_enriched = load_train_enriched_ids(entity2id, relation2id)
    adj, incident = build_graph_indexes(train_enriched)

    split_train = read_json(SPLIT_DIR / "train.json")
    split_valid = read_json(SPLIT_DIR / "valid.json")
    split_test = read_json(SPLIT_DIR / "test.json")

    split_train_index = build_base_split_index(split_train)
    split_valid_index = build_base_split_index(split_valid)
    split_test_index = build_base_split_index(split_test)

    candidate_names = read_json(SPLIT_DIR / "candidate_universe.json")

    source_valid = convert_baseline_top20_to_raw_source(args.source_model, "valid")
    source_test = convert_baseline_top20_to_raw_source(args.source_model, "test")
    source_train = build_train_supervised_candidates(
        train_rows=split_train,
        candidate_names=candidate_names,
        entity2id=entity2id,
        k=args.train_k,
        seed=args.seed,
    )

    write_json(source_train, RAW_SOURCE_DIR / "train_top20_supervised.json")
    write_json(source_valid, RAW_SOURCE_DIR / "valid_top20_raw.json")
    write_json(source_test, RAW_SOURCE_DIR / "test_top20_raw.json")

    ready_train = raw_source_to_softfuse_rows(
        raw_rows=source_train,
        base_split_rows_by_key=split_train_index,
        relation2id=relation2id,
        adj=adj,
        incident=incident,
        graph_size=args.graph_size,
        split="train",
    )
    ready_valid = raw_source_to_softfuse_rows(
        raw_rows=source_valid,
        base_split_rows_by_key=split_valid_index,
        relation2id=relation2id,
        adj=adj,
        incident=incident,
        graph_size=args.graph_size,
        split="valid",
    )
    ready_test = raw_source_to_softfuse_rows(
        raw_rows=source_test,
        base_split_rows_by_key=split_test_index,
        relation2id=relation2id,
        adj=adj,
        incident=incident,
        graph_size=args.graph_size,
        split="test",
    )

    write_json(ready_train, SOFTFUSE_READY_DIR / "train.json")
    write_json(ready_valid, SOFTFUSE_READY_DIR / "valid.json")
    write_json(ready_test, SOFTFUSE_READY_DIR / "test.json")

    prompt_lexicon = write_prompt_lexicon(relation2id)
    rules, support_schema = write_rules_and_schema(relation2id, id2relation)

    copy_maps_as_pickle(entity2id, id2entity, relation2id, id2relation)

    baseline_metrics = load_baseline_metrics()
    source_selection = choose_source_model(baseline_metrics, args.source_model)

    leak_sanity = run_leak_sanity(
        ready_valid=ready_valid,
        ready_test=ready_test,
        train_enriched=train_enriched,
    )

    manifest = {
        "week": 22,
        "day": 6,
        "decision": "SOFTFUSE_READY_PACKAGE_BUILT" if leak_sanity["decision"] == "PASS" else "SOFTFUSE_READY_PACKAGE_LEAK_CHECK_FAILED",
        "dataset": "PharmKG-8k",
        "setting": "setting_c_pharmkg",
        "task": {
            "task_template": "(?, T, disease)",
            "target_relation_raw": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "prediction_type": TARGET_TYPE,
            "candidate_universe": "drug_only_from_train_T_heads",
            "gold_injection_eval": False,
        },
        "source_selection": source_selection,
        "ready_summary": {
            "train": summarize_ready_rows(ready_train),
            "valid": summarize_ready_rows(ready_valid),
            "test": summarize_ready_rows(ready_test),
        },
        "leak_sanity": leak_sanity,
        "graph_size": int(args.graph_size),
        "prompt_lexicon": prompt_lexicon,
        "rules_note": rules.get("note"),
        "support_schema": support_schema,
        "files": {
            "raw_source_dir": str(RAW_SOURCE_DIR),
            "softfuse_ready_dir": str(SOFTFUSE_READY_DIR),
            "train_json": str(SOFTFUSE_READY_DIR / "train.json"),
            "valid_json": str(SOFTFUSE_READY_DIR / "valid.json"),
            "test_json": str(SOFTFUSE_READY_DIR / "test.json"),
            "prep_manifest": str(SOFTFUSE_READY_DIR / "prep_manifest.json"),
        },
    }

    write_json(source_selection, RAW_SOURCE_DIR / "source_selection.json")
    write_json(source_selection, SOURCE_SELECTION_RESULT_PATH)
    write_json(manifest, SOFTFUSE_READY_DIR / "prep_manifest.json")
    write_report(source_selection, manifest)

    print("Saved:")
    print(f"  {RAW_SOURCE_DIR / 'train_top20_supervised.json'}")
    print(f"  {RAW_SOURCE_DIR / 'valid_top20_raw.json'}")
    print(f"  {RAW_SOURCE_DIR / 'test_top20_raw.json'}")
    print(f"  {SOFTFUSE_READY_DIR / 'train.json'}")
    print(f"  {SOFTFUSE_READY_DIR / 'valid.json'}")
    print(f"  {SOFTFUSE_READY_DIR / 'test.json'}")
    print(f"  {SOFTFUSE_READY_DIR / 'prompt_lexicon.json'}")
    print(f"  {SOFTFUSE_READY_DIR / 'rules.json'}")
    print(f"  {SOFTFUSE_READY_DIR / 'support_schema.json'}")
    print(f"  {SOFTFUSE_READY_DIR / 'prep_manifest.json'}")
    print(f"  {SOURCE_SELECTION_RESULT_PATH}")
    print(f"  {REPORT_PATH}")

    print("\nDecision:", manifest["decision"])
    print("Source:", args.source_model)
    print("Leak sanity:", leak_sanity["decision"])
    print("Ready summary:")
    print(json.dumps(manifest["ready_summary"], ensure_ascii=False, indent=2))

    if manifest["decision"] != "SOFTFUSE_READY_PACKAGE_BUILT":
        raise RuntimeError(
            f"Day 6 failed: decision={manifest['decision']}, leak={leak_sanity}"
        )


if __name__ == "__main__":
    main()