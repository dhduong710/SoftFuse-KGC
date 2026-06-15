#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week 23 Day 2: Build PharmKG support features and hard-support negative control.

Input:
- dataset/setting_c_pharmkg/softfuse_ready/{valid,test}.json
- dataset/setting_c_pharmkg/transfer_eval_ready/backbone_raw_{valid,test}.json
- dataset/setting_c_pharmkg/graph/{entity2id,relation2id,type_map,train_enriched.tsv}

Output:
- dataset/setting_c_pharmkg/support_features/valid_support_features.json
- dataset/setting_c_pharmkg/support_features/test_support_features.json
- dataset/setting_c_pharmkg/support_features/support_feature_summary.json

- dataset/setting_c_pharmkg/hard_support_raw/valid_top20_hard_support_raw.json
- dataset/setting_c_pharmkg/hard_support_raw/test_top20_hard_support_raw.json
- dataset/setting_c_pharmkg/hard_support_raw/hard_support_audit.json

- outputs/pharmkg/hard_support_raw_eval_valid.json
- outputs/pharmkg/hard_support_raw_eval_test.json

- outputs/pharmkg/reports/day2_support_features_and_hard_control.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any


ROOT = Path(".")

SOFTFUSE_READY_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "softfuse_ready"
TRANSFER_EVAL_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "transfer_eval_ready"
SUPPORT_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "support_features"
HARD_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "hard_support_raw"
GRAPH_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "graph"

RESULT_DIR = ROOT / "outputs" / "pharmkg"
REPORT_DIR = ROOT / "outputs" / "pharmkg" / "reports"

VALID_HARD_EVAL_PATH = RESULT_DIR / "hard_support_raw_eval_valid.json"
TEST_HARD_EVAL_PATH = RESULT_DIR / "hard_support_raw_eval_test.json"
REPORT_PATH = REPORT_DIR / "day2_support_features_and_hard_control.md"

TOP_K = 20
ABSENT_RANK = 21
TARGET_RELATION = "T"
TARGET_RELATION_NORMALIZED = "therapeutic_association_proxy"


def ensure_dirs() -> None:
    for p in [SUPPORT_DIR, HARD_DIR, RESULT_DIR, REPORT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_maps() -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    entity2id_raw = read_json(GRAPH_DIR / "entity2id.json")
    relation2id_raw = read_json(GRAPH_DIR / "relation2id.json")
    type_map = read_json(GRAPH_DIR / "type_map.json")

    entity2id = {str(k): int(v) for k, v in entity2id_raw.items()}
    relation2id = {str(k): int(v) for k, v in relation2id_raw.items()}

    return entity2id, relation2id, type_map


def load_train_enriched_id_triples(
    entity2id: dict[str, int],
    relation2id: dict[str, int],
) -> list[tuple[int, int, int]]:
    path = GRAPH_DIR / "train_enriched.tsv"
    triples: list[tuple[int, int, int]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"Bad line {line_idx}: {line}")

            h, r, t = parts[:3]
            triples.append((int(entity2id[h]), int(relation2id[r]), int(entity2id[t])))

    return triples


def build_global_indexes(
    train_triples: list[tuple[int, int, int]],
) -> tuple[
    set[tuple[int, int, int]],
    dict[int, list[tuple[int, tuple[int, int, int]]]],
    dict[int, list[tuple[int, int, int]]],
]:
    triple_set = set(train_triples)
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]] = defaultdict(list)
    incident: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    for h, r, t in train_triples:
        tr = (h, r, t)
        adj[h].append((t, tr))
        adj[t].append((h, tr))
        incident[h].append(tr)
        incident[t].append(tr)

    return triple_set, adj, incident


def subgraph_indexes(
    subgraph: list[list[int]],
) -> tuple[
    list[tuple[int, int, int]],
    dict[int, list[tuple[int, tuple[int, int, int]]]],
    dict[int, list[tuple[int, int, int]]],
]:
    triples = [(int(h), int(r), int(t)) for h, r, t in subgraph]
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]] = defaultdict(list)
    incident: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    for h, r, t in triples:
        tr = (h, r, t)
        adj[h].append((t, tr))
        adj[t].append((h, tr))
        incident[h].append(tr)
        incident[t].append(tr)

    return triples, adj, incident


def shortest_path_len_in_adj(
    source: int,
    target: int,
    adj: dict[int, list[tuple[int, tuple[int, int, int]]]],
    max_depth: int = 4,
) -> int | None:
    if source == target:
        return 0

    q = deque([(source, 0)])
    visited = {source}

    while q:
        node, depth = q.popleft()

        if depth >= max_depth:
            continue

        for nb, _triple in adj.get(node, []):
            if nb in visited:
                continue

            next_depth = depth + 1
            if nb == target:
                return next_depth

            visited.add(nb)
            q.append((nb, next_depth))

    return None


def count_direct_edges_between(
    a: int,
    b: int,
    triples: list[tuple[int, int, int]],
) -> int:
    count = 0
    for h, _r, t in triples:
        if (h == a and t == b) or (h == b and t == a):
            count += 1
    return count


def direct_target_flag(
    cand_id: int,
    query_id: int,
    target_relation_id: int,
    train_triple_set: set[tuple[int, int, int]],
) -> int:
    return int((cand_id, target_relation_id, query_id) in train_triple_set)


def candidate_type_valid(candidate_type: str) -> int:
    return int(candidate_type == "Drug_or_Chemical_from_train_T_head")


def query_type_valid(query_type: str) -> int:
    return int(query_type == "Disease_from_train_T_tail")


def build_candidate_features_for_row(
    ready_row: dict[str, Any],
    type_map_by_id: dict[int, str],
    train_triple_set: set[tuple[int, int, int]],
    target_relation_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query_id = int(ready_row["query_entity_id"])
    gold_id = int(ready_row["gold_entity_id"])

    candidate_entities = list(ready_row["rank_entities"])
    candidate_ids = [int(x) for x in ready_row["rank_entities_id"]]

    subgraph_triples, sub_adj, sub_incident = subgraph_indexes(ready_row["subgraph"])

    query_touch_count = len(sub_incident.get(query_id, []))

    feature_rows: list[dict[str, Any]] = []

    for pos, (cand_name, cand_id) in enumerate(zip(candidate_entities, candidate_ids), start=1):
        candidate_touch_triples = sub_incident.get(cand_id, [])
        candidate_edge_touch_count = len(candidate_touch_triples)
        evidence_edge_touch_count = candidate_edge_touch_count

        candidate_query_edge_count = count_direct_edges_between(
            cand_id,
            query_id,
            subgraph_triples,
        )

        sp_len = shortest_path_len_in_adj(
            cand_id,
            query_id,
            sub_adj,
            max_depth=4,
        )

        support_relation_ids = sorted({int(r) for _h, r, _t in candidate_touch_triples})
        support_relation_diversity = len(support_relation_ids)

        direct_t_flag = direct_target_flag(
            cand_id=cand_id,
            query_id=query_id,
            target_relation_id=target_relation_id,
            train_triple_set=train_triple_set,
        )

        cand_type = type_map_by_id.get(cand_id, "Unknown_or_Other")
        q_type = type_map_by_id.get(query_id, "Unknown_or_Other")

        type_valid = candidate_type_valid(cand_type) * query_type_valid(q_type)

        evidence_positive = int(evidence_edge_touch_count > 0)
        shortest_path_exists = int(sp_len is not None)

        hard_support_keep_flag = int(shortest_path_exists == 1)

        row = {
            "candidate_entity": cand_name,
            "candidate_entity_id": int(cand_id),
            "base_rank": int(pos),
            "gold_flag": int(cand_id == gold_id),
            "candidate_in_subgraph": int(candidate_edge_touch_count > 0),
            "evidence_edge_touch_count": int(evidence_edge_touch_count),
            "query_edge_touch_count": int(query_touch_count),
            "candidate_edge_touch_count": int(candidate_edge_touch_count),
            "candidate_query_edge_count": int(candidate_query_edge_count),
            "shortest_path_exists": int(shortest_path_exists),
            "shortest_path_len": int(sp_len) if sp_len is not None else None,
            "support_relation_diversity": int(support_relation_diversity),
            "support_relation_ids": support_relation_ids,
            "direct_T_candidate_query_flag": int(direct_t_flag),
            "type_valid_flag": int(type_valid),
            "candidate_type": cand_type,
            "query_type": q_type,
            "contra_flag": 0,
            "conflict_flag": 0,
            "contra_penalty": 0.0,
            "hard_support_keep_flag": int(hard_support_keep_flag),
            "support_feature_vector": {
                "evidence_positive": int(evidence_positive),
                "direct_T_candidate_query_flag": int(direct_t_flag),
                "shortest_path_exists": int(shortest_path_exists),
                "shortest_path_len_capped": int(sp_len) if sp_len is not None else 5,
                "support_relation_diversity": int(support_relation_diversity),
                "candidate_edge_touch_count": int(candidate_edge_touch_count),
                "query_edge_touch_count": int(query_touch_count),
                "contra_flag": 0,
            },
        }

        feature_rows.append(row)

    row_summary = {
        "num_candidates": int(len(feature_rows)),
        "num_candidates_with_evidence": int(sum(x["evidence_edge_touch_count"] > 0 for x in feature_rows)),
        "num_candidates_with_shortest_path": int(sum(x["shortest_path_exists"] for x in feature_rows)),
        "num_candidates_with_direct_T": int(sum(x["direct_T_candidate_query_flag"] for x in feature_rows)),
        "num_hard_keep": int(sum(x["hard_support_keep_flag"] for x in feature_rows)),
        "gold_present_raw": int(any(x["gold_flag"] for x in feature_rows)),
        "gold_has_evidence": int(any(x["gold_flag"] and x["evidence_edge_touch_count"] > 0 for x in feature_rows)),
        "gold_has_shortest_path": int(any(x["gold_flag"] and x["shortest_path_exists"] for x in feature_rows)),
        "gold_has_direct_T": int(any(x["gold_flag"] and x["direct_T_candidate_query_flag"] for x in feature_rows)),
    }

    return feature_rows, row_summary


def build_support_features(
    split: str,
    ready_rows: list[dict[str, Any]],
    type_map_by_id: dict[int, str],
    train_triple_set: set[tuple[int, int, int]],
    target_relation_id: int,
) -> list[dict[str, Any]]:
    out = []

    for idx, ready in enumerate(ready_rows):
        feature_rows, row_summary = build_candidate_features_for_row(
            ready_row=ready,
            type_map_by_id=type_map_by_id,
            train_triple_set=train_triple_set,
            target_relation_id=target_relation_id,
        )

        out.append(
            {
                "split": split,
                "row_index": int(idx),
                "row_uid": f"{split}_{idx}",
                "query_entity": ready["query_entity"],
                "query_entity_id": int(ready["query_entity_id"]),
                "gold_entity": ready["gold_entity"],
                "gold_entity_id": int(ready["gold_entity_id"]),
                "candidate_feature_rows": feature_rows,
                "row_summary": row_summary,
                "source_ready_row": {
                    "source_model": ready.get("source_model"),
                    "rank": int(ready["rank"]),
                    "gold_in_topk_raw": bool(ready["gold_in_topk_raw"]),
                    "subgraph_size": int(len(ready["subgraph"])),
                },
            }
        )

    return out


def compute_rank_and_rr(gold_entity: str, candidates: list[str]) -> tuple[int, bool, float]:
    if gold_entity in candidates:
        rank = candidates.index(gold_entity) + 1
        if rank <= TOP_K:
            return rank, True, 1.0 / rank
    return ABSENT_RANK, False, 0.0


def summarize_eval_rows(rows: list[dict[str, Any]], row_name: str) -> dict[str, Any]:
    n = len(rows)
    ranks = [int(r["gold_rank"]) for r in rows]
    rr = [float(r["reciprocal_rank_item"]) for r in rows]
    present = [bool(r["gold_present"]) for r in rows]
    present_ranks = [rank for rank, is_present in zip(ranks, present) if is_present]
    candidate_sizes = [int(r["num_candidates"]) for r in rows]

    return {
        "eval_row_name": row_name,
        "num_rows": int(n),
        "gold_present_at20": float(sum(present) / max(1, n)),
        "mrr_at20": float(sum(rr) / max(1, n)),
        "mrr_present_only": (
            float(sum(1.0 / r for r in present_ranks) / len(present_ranks))
            if present_ranks
            else 0.0
        ),
        "hits1_at20": float(sum(r <= 1 for r in ranks) / max(1, n)),
        "hits3_at20": float(sum(r <= 3 for r in ranks) / max(1, n)),
        "hits10_at20": float(sum(r <= 10 for r in ranks) / max(1, n)),
        "hits20_at20": float(sum(r <= 20 for r in ranks) / max(1, n)),
        "avg_gold_rank_absent_as_21": float(sum(ranks) / max(1, n)),
        "gold_rank_21_count": int(sum(r == ABSENT_RANK for r in ranks)),
        "avg_candidate_size": float(sum(candidate_sizes) / max(1, n)),
        "min_candidate_size": int(min(candidate_sizes)) if candidate_sizes else 0,
        "max_candidate_size": int(max(candidate_sizes)) if candidate_sizes else 0,
        "top_k": TOP_K,
        "gold_injection": False,
        "rr_policy": "RR = 1/rank if gold is present in top-20 else 0",
        "rank_absent_sentinel": ABSENT_RANK,
    }


def build_hard_support_rows(
    split: str,
    support_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    top20_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    audit_rows: list[dict[str, Any]] = []

    for row in support_rows:
        features = row["candidate_feature_rows"]
        kept = [f for f in features if int(f["hard_support_keep_flag"]) == 1]

        fallback_used = False
        if len(kept) == 0:
            kept = features
            fallback_used = True

        kept = kept[:TOP_K]

        candidate_entities = [f["candidate_entity"] for f in kept]
        candidate_ids = [int(f["candidate_entity_id"]) for f in kept]

        gold_entity = row["gold_entity"]
        gold_id = int(row["gold_entity_id"])

        raw_candidates = [f["candidate_entity"] for f in features]
        raw_rank, raw_present, _raw_rr = compute_rank_and_rr(gold_entity, raw_candidates)

        rank, present, rr = compute_rank_and_rr(gold_entity, candidate_entities)

        gold_removed = bool(raw_present and not present)

        top20_row = {
            "split": split,
            "row_index": int(row["row_index"]),
            "query_entity": row["query_entity"],
            "query_entity_id": int(row["query_entity_id"]),
            "gold_entity": gold_entity,
            "gold_entity_id": gold_id,
            "candidate_entities": candidate_entities,
            "candidate_entity_ids": candidate_ids,
            "num_candidates": int(len(candidate_entities)),
            "gold_rank_in_top20_or_21": int(rank),
            "gold_in_topk_hard_support": bool(present),
            "gold_present_raw": bool(raw_present),
            "gold_rank_raw": int(raw_rank),
            "gold_removed_by_hard_support": bool(gold_removed),
            "fallback_used": bool(fallback_used),
            "candidate_universe": "drug_only_from_train_T_heads",
            "gold_injection": False,
            "variant_name": "hard_support_raw",
            "target_relation": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "candidate_debug_rows": kept,
        }

        eval_row = {
            "eval_row_name": "hard_support_raw",
            "row_index": int(row["row_index"]),
            "split": split,
            "query_entity": row["query_entity"],
            "query_entity_id": int(row["query_entity_id"]),
            "gold_entity": gold_entity,
            "gold_entity_id": gold_id,
            "candidate_entities": candidate_entities,
            "candidate_entity_ids": candidate_ids,
            "num_candidates": int(len(candidate_entities)),
            "gold_present": bool(present),
            "gold_rank": int(rank),
            "gold_rank_source": "hard_support_raw",
            "reciprocal_rank_item": float(rr),
            "hits1_item": int(rank <= 1),
            "hits3_item": int(rank <= 3),
            "hits10_item": int(rank <= 10),
            "hits20_item": int(rank <= 20),
            "candidate_universe": "drug_only_from_train_T_heads",
            "gold_injection": False,
            "target_relation": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "stage_specific": {
                "stage": "hard_support_raw",
                "negative_control": True,
                "report_label": "hard graph-support control, not ontology filtering",
                "raw_rank": int(raw_rank),
                "raw_present": bool(raw_present),
                "fallback_used": bool(fallback_used),
                "num_candidates_before": int(len(features)),
                "num_candidates_after": int(len(candidate_entities)),
                "gold_removed_by_hard_support": bool(gold_removed),
                "keep_rule": "shortest_path_exists == 1; fallback to raw if empty",
            },
        }

        audit_rows.append(
            {
                "split": split,
                "row_index": int(row["row_index"]),
                "query_entity": row["query_entity"],
                "gold_entity": gold_entity,
                "raw_present": bool(raw_present),
                "raw_rank": int(raw_rank),
                "hard_present": bool(present),
                "hard_rank": int(rank),
                "num_candidates_before": int(len(features)),
                "num_candidates_after": int(len(candidate_entities)),
                "fallback_used": bool(fallback_used),
                "gold_removed_by_hard_support": bool(gold_removed),
                "num_hard_keep_before_fallback": int(sum(f["hard_support_keep_flag"] for f in features)),
            }
        )

        top20_rows.append(top20_row)
        eval_rows.append(eval_row)

    audit_summary = {
        "split": split,
        "num_rows": int(len(audit_rows)),
        "fallback_rows": int(sum(r["fallback_used"] for r in audit_rows)),
        "fallback_rate": float(sum(r["fallback_used"] for r in audit_rows) / max(1, len(audit_rows))),
        "raw_gold_present_rows": int(sum(r["raw_present"] for r in audit_rows)),
        "hard_gold_present_rows": int(sum(r["hard_present"] for r in audit_rows)),
        "gold_removed_rows": int(sum(r["gold_removed_by_hard_support"] for r in audit_rows)),
        "gold_removed_rate_given_raw_present": (
            float(
                sum(r["gold_removed_by_hard_support"] for r in audit_rows)
                / max(1, sum(r["raw_present"] for r in audit_rows))
            )
        ),
        "avg_candidate_size_after": float(
            sum(r["num_candidates_after"] for r in audit_rows) / max(1, len(audit_rows))
        ),
    }

    return top20_rows, eval_rows, {"summary": audit_summary, "rows": audit_rows}


def summarize_support_features(support_rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    all_cands = []
    for row in support_rows:
        all_cands.extend(row["candidate_feature_rows"])

    n_rows = len(support_rows)
    n_cands = len(all_cands)

    gold_rows = [c for c in all_cands if c["gold_flag"] == 1]

    def avg(key: str, rows: list[dict[str, Any]]) -> float:
        if not rows:
            return 0.0
        vals = [x[key] for x in rows if x[key] is not None]
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    return {
        "split": split,
        "num_rows": int(n_rows),
        "num_candidate_rows": int(n_cands),
        "avg_evidence_edge_touch_count": avg("evidence_edge_touch_count", all_cands),
        "avg_query_edge_touch_count": avg("query_edge_touch_count", all_cands),
        "avg_candidate_edge_touch_count": avg("candidate_edge_touch_count", all_cands),
        "candidate_with_evidence_rate": float(
            sum(c["evidence_edge_touch_count"] > 0 for c in all_cands) / max(1, n_cands)
        ),
        "candidate_with_shortest_path_rate": float(
            sum(c["shortest_path_exists"] for c in all_cands) / max(1, n_cands)
        ),
        "candidate_with_direct_T_rate": float(
            sum(c["direct_T_candidate_query_flag"] for c in all_cands) / max(1, n_cands)
        ),
        "avg_support_relation_diversity": avg("support_relation_diversity", all_cands),
        "num_gold_candidate_rows": int(len(gold_rows)),
        "gold_with_evidence_rate": float(
            sum(c["evidence_edge_touch_count"] > 0 for c in gold_rows) / max(1, len(gold_rows))
        ),
        "gold_with_shortest_path_rate": float(
            sum(c["shortest_path_exists"] for c in gold_rows) / max(1, len(gold_rows))
        ),
        "gold_with_direct_T_rate": float(
            sum(c["direct_T_candidate_query_flag"] for c in gold_rows) / max(1, len(gold_rows))
        ),
        "row_summary_counts": {
            "rows_with_any_hard_keep": int(
                sum(r["row_summary"]["num_hard_keep"] > 0 for r in support_rows)
            ),
            "rows_with_gold_present_raw": int(
                sum(r["row_summary"]["gold_present_raw"] for r in support_rows)
            ),
            "rows_where_gold_has_shortest_path": int(
                sum(r["row_summary"]["gold_has_shortest_path"] for r in support_rows)
            ),
        },
    }


def write_report(
    valid_support_summary: dict[str, Any],
    test_support_summary: dict[str, Any],
    valid_hard_metrics: dict[str, Any],
    test_hard_metrics: dict[str, Any],
    hard_audit: dict[str, Any],
    decision: str,
) -> None:
    md = f"""# Week 23 Day 2 — Support Features and Hard-Support Negative Control

## Decision

`{decision}`

## Important wording

This is **not** PharmKG ontology filtering.

Use report-safe wording:

`hard graph-support control on PharmKG therapeutic-association proxy task`

Relation `T` is still called `therapeutic_association_proxy`, not clinical indication.

## Support feature summary

| Split | Rows | Candidate rows | Evidence rate | Shortest-path rate | Direct-T rate | Gold with path rate |
|---|---:|---:|---:|---:|---:|---:|
| valid | {valid_support_summary["num_rows"]} | {valid_support_summary["num_candidate_rows"]} | {valid_support_summary["candidate_with_evidence_rate"]:.4f} | {valid_support_summary["candidate_with_shortest_path_rate"]:.4f} | {valid_support_summary["candidate_with_direct_T_rate"]:.4f} | {valid_support_summary["gold_with_shortest_path_rate"]:.4f} |
| test | {test_support_summary["num_rows"]} | {test_support_summary["num_candidate_rows"]} | {test_support_summary["candidate_with_evidence_rate"]:.4f} | {test_support_summary["candidate_with_shortest_path_rate"]:.4f} | {test_support_summary["candidate_with_direct_T_rate"]:.4f} | {test_support_summary["gold_with_shortest_path_rate"]:.4f} |

## Hard-support eval metrics

| Split | Gold@20 | MRR@20 | MRR present-only | H@1 | H@3 | H@10 | H@20 | Rank21 | Avg cand size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | {valid_hard_metrics["gold_present_at20"]:.3f} | {valid_hard_metrics["mrr_at20"]:.12f} | {valid_hard_metrics["mrr_present_only"]:.12f} | {valid_hard_metrics["hits1_at20"]:.3f} | {valid_hard_metrics["hits3_at20"]:.3f} | {valid_hard_metrics["hits10_at20"]:.3f} | {valid_hard_metrics["hits20_at20"]:.3f} | {valid_hard_metrics["gold_rank_21_count"]} | {valid_hard_metrics["avg_candidate_size"]:.3f} |
| test | {test_hard_metrics["gold_present_at20"]:.3f} | {test_hard_metrics["mrr_at20"]:.12f} | {test_hard_metrics["mrr_present_only"]:.12f} | {test_hard_metrics["hits1_at20"]:.3f} | {test_hard_metrics["hits3_at20"]:.3f} | {test_hard_metrics["hits10_at20"]:.3f} | {test_hard_metrics["hits20_at20"]:.3f} | {test_hard_metrics["gold_rank_21_count"]} | {test_hard_metrics["avg_candidate_size"]:.3f} |

## Hard-support audit

| Split | Fallback rows | Raw gold present | Hard gold present | Gold removed | Gold removed rate | Avg cand size |
|---|---:|---:|---:|---:|---:|---:|
| valid | {hard_audit["valid"]["summary"]["fallback_rows"]} | {hard_audit["valid"]["summary"]["raw_gold_present_rows"]} | {hard_audit["valid"]["summary"]["hard_gold_present_rows"]} | {hard_audit["valid"]["summary"]["gold_removed_rows"]} | {hard_audit["valid"]["summary"]["gold_removed_rate_given_raw_present"]:.4f} | {hard_audit["valid"]["summary"]["avg_candidate_size_after"]:.3f} |
| test | {hard_audit["test"]["summary"]["fallback_rows"]} | {hard_audit["test"]["summary"]["raw_gold_present_rows"]} | {hard_audit["test"]["summary"]["hard_gold_present_rows"]} | {hard_audit["test"]["summary"]["gold_removed_rows"]} | {hard_audit["test"]["summary"]["gold_removed_rate_given_raw_present"]:.4f} | {hard_audit["test"]["summary"]["avg_candidate_size_after"]:.3f} |

## Interpretation

Day 2 prepares the candidate-level evidence layer for soft support.

The hard-support row is a negative control. If it removes gold or reduces MRR, that supports the same scientific motivation as PrimeKG Novelty 2: hard support is brittle under raw no-injection retrieval.

## Files written

- `dataset/setting_c_pharmkg/support_features/valid_support_features.json`
- `dataset/setting_c_pharmkg/support_features/test_support_features.json`
- `dataset/setting_c_pharmkg/support_features/support_feature_summary.json`
- `dataset/setting_c_pharmkg/hard_support_raw/valid_top20_hard_support_raw.json`
- `dataset/setting_c_pharmkg/hard_support_raw/test_top20_hard_support_raw.json`
- `dataset/setting_c_pharmkg/hard_support_raw/hard_support_audit.json`
- `outputs/pharmkg/hard_support_raw_eval_valid.json`
- `outputs/pharmkg/hard_support_raw_eval_test.json`

## Next step

Day 3 will run selected soft support:

`support_score = 1.0 * evidence_positive - 0.50 * direct_T_candidate_query_flag`
"""
    REPORT_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    entity2id, relation2id, type_map = load_maps()
    id2entity = {v: k for k, v in entity2id.items()}
    type_map_by_id = {entity2id[name]: typ for name, typ in type_map.items() if name in entity2id}

    target_relation_id = int(relation2id[TARGET_RELATION])

    train_triples = load_train_enriched_id_triples(entity2id, relation2id)
    train_triple_set, _global_adj, _global_incident = build_global_indexes(train_triples)

    valid_ready = read_json(SOFTFUSE_READY_DIR / "valid.json")
    test_ready = read_json(SOFTFUSE_READY_DIR / "test.json")

    valid_support = build_support_features(
        split="valid",
        ready_rows=valid_ready,
        type_map_by_id=type_map_by_id,
        train_triple_set=train_triple_set,
        target_relation_id=target_relation_id,
    )

    test_support = build_support_features(
        split="test",
        ready_rows=test_ready,
        type_map_by_id=type_map_by_id,
        train_triple_set=train_triple_set,
        target_relation_id=target_relation_id,
    )

    write_json(valid_support, SUPPORT_DIR / "valid_support_features.json")
    write_json(test_support, SUPPORT_DIR / "test_support_features.json")

    valid_support_summary = summarize_support_features(valid_support, "valid")
    test_support_summary = summarize_support_features(test_support, "test")

    support_summary = {
        "week": 23,
        "day": 2,
        "dataset": "PharmKG-8k",
        "setting": "setting_c_pharmkg",
        "target_relation": TARGET_RELATION,
        "target_relation_normalized": TARGET_RELATION_NORMALIZED,
        "valid": valid_support_summary,
        "test": test_support_summary,
        "notes": [
            "contra_flag/conflict_flag are fixed to zero because PharmKG-8k relation labels are compact codes and no explicit contraindication relation is used.",
            "hard_support_raw is a negative control, not an ontology filter.",
        ],
    }

    write_json(support_summary, SUPPORT_DIR / "support_feature_summary.json")

    valid_hard_top20, valid_hard_eval_rows, valid_hard_audit = build_hard_support_rows(
        split="valid",
        support_rows=valid_support,
    )

    test_hard_top20, test_hard_eval_rows, test_hard_audit = build_hard_support_rows(
        split="test",
        support_rows=test_support,
    )

    write_json(valid_hard_top20, HARD_DIR / "valid_top20_hard_support_raw.json")
    write_json(test_hard_top20, HARD_DIR / "test_top20_hard_support_raw.json")

    valid_hard_metrics = summarize_eval_rows(valid_hard_eval_rows, "hard_support_raw")
    test_hard_metrics = summarize_eval_rows(test_hard_eval_rows, "hard_support_raw")

    write_json(valid_hard_metrics, VALID_HARD_EVAL_PATH)
    write_json(test_hard_metrics, TEST_HARD_EVAL_PATH)

    hard_audit = {
        "week": 23,
        "day": 2,
        "variant_name": "hard_support_raw",
        "report_label": "hard graph-support control, not ontology filtering",
        "keep_rule": "shortest_path_exists == 1; fallback to raw if empty",
        "valid": valid_hard_audit,
        "test": test_hard_audit,
    }

    write_json(hard_audit, HARD_DIR / "hard_support_audit.json")

    if len(valid_support) != 500 or len(test_support) != 500:
        raise RuntimeError("Expected 500 support rows for valid/test.")

    decision = "SUPPORT_FEATURES_READY_HARD_CONTROL_BUILT"

    if (
        valid_hard_audit["summary"]["gold_removed_rows"] > 0
        or test_hard_audit["summary"]["gold_removed_rows"] > 0
    ):
        decision = "SUPPORT_FEATURES_READY_HARD_CONTROL_BUILT_WITH_PRUNING"

    write_report(
        valid_support_summary=valid_support_summary,
        test_support_summary=test_support_summary,
        valid_hard_metrics=valid_hard_metrics,
        test_hard_metrics=test_hard_metrics,
        hard_audit=hard_audit,
        decision=decision,
    )

    print("Saved:")
    print(f"  {SUPPORT_DIR / 'valid_support_features.json'}")
    print(f"  {SUPPORT_DIR / 'test_support_features.json'}")
    print(f"  {SUPPORT_DIR / 'support_feature_summary.json'}")
    print(f"  {HARD_DIR / 'valid_top20_hard_support_raw.json'}")
    print(f"  {HARD_DIR / 'test_top20_hard_support_raw.json'}")
    print(f"  {HARD_DIR / 'hard_support_audit.json'}")
    print(f"  {VALID_HARD_EVAL_PATH}")
    print(f"  {TEST_HARD_EVAL_PATH}")
    print(f"  {REPORT_PATH}")

    print("\nDecision:", decision)

    print("\nSupport feature summary:")
    print(json.dumps(support_summary, ensure_ascii=False, indent=2))

    print("\nHard-support valid metrics:")
    print(json.dumps(valid_hard_metrics, ensure_ascii=False, indent=2))

    print("\nHard-support test metrics:")
    print(json.dumps(test_hard_metrics, ensure_ascii=False, indent=2))

    print("\nHard-support audit summary:")
    print(json.dumps(
        {
            "valid": valid_hard_audit["summary"],
            "test": test_hard_audit["summary"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()