#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(".")
SETTING_DIR = ROOT / "dataset" / "setting_e_drkg"

SPLIT_DIR = SETTING_DIR / "splits"
GRAPH_DIR = SETTING_DIR / "graph"
BASELINE_DIR = SETTING_DIR / "baseline_outputs"

SOURCE_ROOT = SETTING_DIR / "backbone_raw_source"
READY_ROOT = SETTING_DIR / "softfuse_ready"

RESULT_DIR = ROOT / "outputs" / "drkg"
REPORT_DIR = ROOT / "outputs" / "drkg" / "reports"

TARGET_RELATION = "DRUGBANK::treats::Compound:Disease"
TARGET_RELATION_NORMALIZED = "drugbank_treats"
TOP_K = 20
ABSENT_RANK = 21


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def write_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_maps():
    entity2id = read_json(GRAPH_DIR / "entity2id.json")
    id2entity = {int(k): v for k, v in read_json(GRAPH_DIR / "id2entity.json").items()}
    relation2id = read_json(GRAPH_DIR / "relation2id.json")
    id2relation = {int(k): v for k, v in read_json(GRAPH_DIR / "id2relation.json").items()}
    type_map = read_json(GRAPH_DIR / "type_map.json")
    return entity2id, id2entity, relation2id, id2relation, type_map


def load_target_rows(split: str) -> list[dict[str, Any]]:
    return read_json(SPLIT_DIR / f"{split}_target_rows.json")


def load_candidate_universe() -> tuple[list[str], list[int]]:
    obj = read_json(SPLIT_DIR / "candidate_universe_compound.json")
    return obj["candidate_entities"], [int(x) for x in obj["candidate_entity_ids"]]


def get_candidates_from_baseline_row(row: dict[str, Any], entity2id: dict[str, int]):
    names = (
        row.get("candidate_entities_top20")
        or row.get("candidate_entities")
        or row.get("rank_entities")
    )
    ids = (
        row.get("candidate_entity_ids_top20")
        or row.get("candidate_entity_ids")
        or row.get("rank_entities_id")
    )
    scores = row.get("scores_top20") or row.get("scores") or row.get("score_top20") or []

    if names is None:
        raise KeyError(f"Cannot find candidate names in baseline row keys={sorted(row.keys())}")

    names = list(names)

    if ids is None:
        ids = [int(entity2id[x]) for x in names]
    else:
        ids = [int(x) for x in ids]

    if not scores:
        scores = [float(TOP_K - i) for i in range(len(ids))]
    else:
        scores = [float(x) for x in scores]

    if len(names) != len(ids):
        raise RuntimeError(f"candidate name/id mismatch: {len(names)} vs {len(ids)}")

    if len(names) < TOP_K:
        raise RuntimeError(f"candidate list shorter than TOP_K={TOP_K}: {len(names)}")

    return names[:TOP_K], ids[:TOP_K], scores[:TOP_K]


def rank_gold(gold_id: int, candidate_ids: list[int]) -> tuple[int, bool]:
    gold_id = int(gold_id)
    if gold_id in candidate_ids:
        return candidate_ids.index(gold_id) + 1, True
    return ABSENT_RANK, False


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.array([int(r["rank"]) for r in rows], dtype=np.int64)
    present = np.array([bool(r["gold_in_topk_raw"]) for r in rows], dtype=bool)

    rr = np.zeros(len(rows), dtype=np.float64)
    for i, rank in enumerate(ranks):
        if present[i] and rank <= TOP_K:
            rr[i] = 1.0 / rank

    top1 = [r["rank_entities"][0] for r in rows if r.get("rank_entities")]
    top1_counter = Counter(top1)
    most_common = top1_counter.most_common(10)

    return {
        "num_rows": int(len(rows)),
        "gold_present_at20": float(np.mean(present)) if rows else 0.0,
        "mrr_at20": float(np.mean(rr)) if rows else 0.0,
        "hits1_at20": float(np.mean(ranks <= 1)) if rows else 0.0,
        "hits3_at20": float(np.mean(ranks <= 3)) if rows else 0.0,
        "hits10_at20": float(np.mean(ranks <= 10)) if rows else 0.0,
        "hits20_at20": float(np.mean(ranks <= 20)) if rows else 0.0,
        "rank21_count": int(np.sum(ranks == ABSENT_RANK)),
        "avg_rank_absent_as_21": float(np.mean(ranks)) if rows else 0.0,
        "unique_top1_count": int(len(top1_counter)),
        "top1_dominance": float(most_common[0][1] / len(rows)) if rows and most_common else 0.0,
        "top1_most_common": [{"entity": k, "count": int(v)} for k, v in most_common],
    }


def make_eval_source_rows(
    source_model: str,
    split: str,
    target_rows: list[dict[str, Any]],
    entity2id: dict[str, int],
    id2entity: dict[int, str],
):
    baseline_path = BASELINE_DIR / source_model / f"{split}_top20.json"
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)

    baseline_rows = read_json(baseline_path)
    if len(baseline_rows) != len(target_rows):
        raise RuntimeError(
            f"{source_model} {split} row count mismatch: "
            f"baseline={len(baseline_rows)} target={len(target_rows)}"
        )

    out = []

    for i, (target, base) in enumerate(zip(target_rows, baseline_rows)):
        cand_names, cand_ids, scores = get_candidates_from_baseline_row(base, entity2id)
        rank, present = rank_gold(target["gold_entity_id"], cand_ids)

        row = dict(target)
        row.update({
            "source_model": source_model,
            "target_relation": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "candidate_entities": cand_names,
            "candidate_entity_ids": cand_ids,
            "scores": scores,
            "rank_entities": cand_names,
            "rank_entities_id": cand_ids,
            "rank": int(rank),
            "gold_rank_in_top20_or_21": int(rank),
            "gold_in_topk_raw": bool(present),
            "gold_present_top20": bool(present),
            "gold_injected": False,
            "train_gold_forced_into_candidates": False,
            "reviewer_safe_rr_item": float(1.0 / rank) if present and rank <= TOP_K else 0.0,
        })
        out.append(row)

    return out


def build_global_candidate_order(eval_source_rows: list[dict[str, Any]]) -> list[int]:
    score = Counter()
    for row in eval_source_rows:
        for pos, cid in enumerate(row["rank_entities_id"], start=1):
            score[int(cid)] += TOP_K + 1 - pos

    ranked = [cid for cid, _ in score.most_common()]
    return ranked


def make_train_source_rows(
    source_model: str,
    train_rows: list[dict[str, Any]],
    eval_source_rows: list[dict[str, Any]],
    candidate_names: list[str],
    candidate_ids: list[int],
    id2entity: dict[int, str],
    train_ensure_gold: bool,
):
    train_baseline_path = BASELINE_DIR / source_model / "train_top20.json"

    if train_baseline_path.exists():
        baseline_rows = read_json(train_baseline_path)
        if len(baseline_rows) != len(train_rows):
            raise RuntimeError(
                f"{source_model} train baseline row mismatch: "
                f"baseline={len(baseline_rows)} train={len(train_rows)}"
            )
        # Reuse eval converter style by writing a small local conversion.
        out = []
        dummy_entity2id = {v: k for k, v in id2entity.items()}
        for target, base in zip(train_rows, baseline_rows):
            cand_names, cand_ids, scores = get_candidates_from_baseline_row(base, dummy_entity2id)
            rank, present = rank_gold(target["gold_entity_id"], cand_ids)
            row = dict(target)
            row.update({
                "source_model": source_model,
                "target_relation": TARGET_RELATION,
                "target_relation_normalized": TARGET_RELATION_NORMALIZED,
                "candidate_entities": cand_names,
                "candidate_entity_ids": cand_ids,
                "scores": scores,
                "rank_entities": cand_names,
                "rank_entities_id": cand_ids,
                "rank": int(rank),
                "gold_rank_in_top20_or_21": int(rank),
                "gold_in_topk_raw": bool(present),
                "gold_present_top20": bool(present),
                "gold_injected": False,
                "train_gold_forced_into_candidates": False,
                "reviewer_safe_rr_item": float(1.0 / rank) if present and rank <= TOP_K else 0.0,
            })
            out.append(row)
        return out

    global_order = build_global_candidate_order(eval_source_rows)
    all_ids = list(global_order)
    for cid in candidate_ids:
        cid = int(cid)
        if cid not in all_ids:
            all_ids.append(cid)

    out = []
    forced_count = 0

    for target in train_rows:
        gold_id = int(target["gold_entity_id"])

        chosen = []
        for cid in all_ids:
            cid = int(cid)
            if cid not in chosen:
                chosen.append(cid)
            if len(chosen) >= TOP_K:
                break

        forced = False
        if train_ensure_gold and gold_id not in chosen:
            chosen[-1] = gold_id
            forced = True
            forced_count += 1

        # Deduplicate after forced replacement.
        dedup = []
        for cid in chosen:
            if cid not in dedup:
                dedup.append(cid)

        if len(dedup) < TOP_K:
            for cid in all_ids:
                if cid not in dedup:
                    dedup.append(cid)
                if len(dedup) >= TOP_K:
                    break

        chosen = dedup[:TOP_K]
        names = [id2entity[int(cid)] for cid in chosen]
        scores = [float(TOP_K - i) for i in range(TOP_K)]

        rank, present = rank_gold(gold_id, chosen)

        row = dict(target)
        row.update({
            "source_model": source_model,
            "target_relation": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "candidate_entities": names,
            "candidate_entity_ids": [int(x) for x in chosen],
            "scores": scores,
            "rank_entities": names,
            "rank_entities_id": [int(x) for x in chosen],
            "rank": int(rank),
            "gold_rank_in_top20_or_21": int(rank),
            "gold_in_topk_raw": bool(present),
            "gold_present_top20": bool(present),
            "gold_injected": False,
            "train_gold_forced_into_candidates": bool(forced),
            "reviewer_safe_rr_item": float(1.0 / rank) if present and rank <= TOP_K else 0.0,
        })
        out.append(row)

    return out


def kind_by_entity_id(id2entity: dict[int, str], type_map: dict[str, Any]) -> dict[int, str]:
    out = {}
    for eid, name in id2entity.items():
        meta = type_map.get(name, {})
        out[int(eid)] = meta.get("kind", "UNKNOWN")
    return out


def read_train_edges_ids(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 3:
                yield int(row[0]), int(row[1]), int(row[2])


def build_adjacency(train_ids_path: Path):
    adj = defaultdict(list)
    edge_set = set()

    for h, r, t in read_train_edges_ids(train_ids_path):
        edge = (int(h), int(r), int(t))
        edge_set.add(edge)
        adj[int(h)].append((int(r), int(t), edge))
        adj[int(t)].append((int(r), int(h), edge))

    return adj, edge_set


def relation_family_name(rel: str) -> str:
    if rel == TARGET_RELATION:
        return "target_treats"
    if "Compound:Disease" in rel:
        return "aux_compound_disease"
    if "Compound:Gene" in rel or "Gene:Compound" in rel:
        return "compound_gene"
    if "Disease:Gene" in rel or "Gene:Disease" in rel:
        return "disease_gene"
    if "Gene:Gene" in rel:
        return "gene_gene"
    return "other"


def rel_ids_by_family(id2relation: dict[int, str], family: str) -> set[int]:
    return {int(rid) for rid, name in id2relation.items() if relation_family_name(name) == family}


def add_edge(subg: list[list[int]], seen: set[tuple[int, int, int]], edge: tuple[int, int, int]) -> bool:
    if edge in seen:
        return False
    seen.add(edge)
    subg.append([int(edge[0]), int(edge[1]), int(edge[2])])
    return True


def collect_edges(
    adj,
    node_id: int,
    relation_ids: set[int] | None,
    kind_filter: str | None,
    kind_by_id: dict[int, str],
    limit: int,
):
    out = []
    for r, nb, edge in adj.get(int(node_id), []):
        if relation_ids is not None and int(r) not in relation_ids:
            continue
        if kind_filter is not None and kind_by_id.get(int(nb)) != kind_filter:
            continue
        out.append((int(r), int(nb), edge))
        if len(out) >= limit:
            break
    return out


def retrieve_subgraph(row, adj, id2relation, kind_by_id, args):
    q = int(row["query_entity_id"])
    candidates = [int(x) for x in row["rank_entities_id"]]

    direct_rels = rel_ids_by_family(id2relation, "target_treats") | rel_ids_by_family(id2relation, "aux_compound_disease")
    compound_gene_rels = rel_ids_by_family(id2relation, "compound_gene")
    disease_gene_rels = rel_ids_by_family(id2relation, "disease_gene")
    gene_gene_rels = rel_ids_by_family(id2relation, "gene_gene")

    subg = []
    seen = set()

    # 1. Direct candidate-query evidence.
    for c in candidates:
        for r, nb, edge in adj.get(c, []):
            if nb == q and r in direct_rels:
                add_edge(subg, seen, edge)
                if len(subg) >= args.graph_size:
                    return subg

    # 2. Disease-gene context.
    disease_gene_edges = collect_edges(
        adj=adj,
        node_id=q,
        relation_ids=disease_gene_rels,
        kind_filter="Gene",
        kind_by_id=kind_by_id,
        limit=args.disease_gene_cap,
    )
    disease_genes = {nb for _, nb, _ in disease_gene_edges}

    # 3. Candidate-gene shared evidence.
    candidate_gene_cache = {}
    for c in candidates:
        c_edges = collect_edges(
            adj=adj,
            node_id=c,
            relation_ids=compound_gene_rels,
            kind_filter="Gene",
            kind_by_id=kind_by_id,
            limit=args.candidate_gene_cap,
        )
        candidate_gene_cache[c] = c_edges
        c_genes = {nb for _, nb, _ in c_edges}
        shared = c_genes & disease_genes

        if shared:
            for _, nb, edge in c_edges:
                if nb in shared:
                    add_edge(subg, seen, edge)
                    if len(subg) >= args.graph_size:
                        return subg
            for _, nb, edge in disease_gene_edges:
                if nb in shared:
                    add_edge(subg, seen, edge)
                    if len(subg) >= args.graph_size:
                        return subg

    # 4. Disease-gene fill.
    for _, _, edge in disease_gene_edges:
        add_edge(subg, seen, edge)
        if len(subg) >= args.graph_size:
            return subg

    # 5. Candidate-gene fill.
    for c in candidates:
        for _, _, edge in candidate_gene_cache.get(c, []):
            add_edge(subg, seen, edge)
            if len(subg) >= args.graph_size:
                return subg

    # 6. Gene-gene bridges.
    for c in candidates:
        for _, g, _ in candidate_gene_cache.get(c, []):
            for r, nb, edge in adj.get(g, []):
                if r in gene_gene_rels and nb in disease_genes:
                    add_edge(subg, seen, edge)
                    if len(subg) >= args.graph_size:
                        return subg

    # 7. Local fill around query/candidates.
    for node in [q] + candidates:
        for _, _, edge in adj.get(node, []):
            add_edge(subg, seen, edge)
            if len(subg) >= args.graph_size:
                return subg

    return subg


def add_prompt(row: dict[str, Any]) -> None:
    query = row["query_entity"]
    candidates = row["rank_entities"]

    answer_options = "(" + ", ".join([f"'{x}'" for x in candidates]) + ")"
    refer_parts = [f"'{query}': [QUERY]"]
    refer_parts.extend([f"'{x}': [ENTITY]" for x in candidates])
    refer_str = ", ".join(refer_parts)

    question = f"What compound treats {query}?"

    row["input"] = (
        "You are a biomedical scientist. The task is to predict the answer based on the given question, "
        "and you only need to answer one entity. The answer must be in "
        + answer_options
        + ".\nYou can refer to the entity embeddings: "
        + refer_str
        + ".\n\nQuestion: "
        + question
        + "\nAnswer: "
    )
    row["output"] = row["gold_entity"]


def choose_embedding_for_source(source_model: str):
    model_dir = BASELINE_DIR / source_model
    candidates = [
        model_dir / f"entity_embeddings_{source_model}.pt",
        model_dir / "entity_embeddings.pt",
        model_dir / "entity_embedding.pt",
        model_dir / "embeddings.pt",
    ]

    if source_model == "rgcn":
        candidates.insert(0, model_dir / "entity_embeddings_rgcn.pt")

    # Fallback to R-GCN embedding because GraphEnhancer needs a graph-compatible KGE tensor.
    candidates.append(BASELINE_DIR / "rgcn" / "entity_embeddings_rgcn.pt")

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"No usable embedding found for source={source_model}; checked: {[str(x) for x in candidates]}"
    )


def copy_static_files(ready_dir: Path, embedding_path: Path):
    for name in [
        "entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl",
        "entity2id.json", "id2entity.json", "relation2id.json", "id2relation.json",
        "type_map.json", "graph_summary.json", "leak_check.json",
    ]:
        src = GRAPH_DIR / name
        if src.exists():
            shutil.copy2(src, ready_dir / name)

    shutil.copy2(embedding_path, ready_dir / "entity_embeddings_rgcn.pt")


def audit_rows(rows: list[dict[str, Any]], split: str, edge_set: set[tuple[int, int, int]]):
    bad_k = 0
    bad_prompt = 0
    bad_subgraph = 0
    sizes = []
    rank21 = 0
    leaks = 0
    train_forced = 0

    for row in rows:
        if len(row.get("rank_entities_id", [])) != TOP_K:
            bad_k += 1

        if row.get("input", "").count("[QUERY]") != 1 or row.get("input", "").count("[ENTITY]") != TOP_K:
            bad_prompt += 1

        sg = row.get("subgraph", [])
        if not isinstance(sg, list) or any((not isinstance(x, list) or len(x) != 3) for x in sg):
            bad_subgraph += 1

        sizes.append(len(sg))

        if int(row.get("rank", ABSENT_RANK)) == ABSENT_RANK:
            rank21 += 1

        if row.get("train_gold_forced_into_candidates"):
            train_forced += 1

        if split in {"valid", "test"}:
            gold = tuple(int(x) for x in row["triple_id"])
            if gold in edge_set:
                leaks += 1
            if any(tuple(int(y) for y in edge) == gold for edge in sg):
                leaks += 1

    return {
        "split": split,
        "num_rows": len(rows),
        "bad_candidate_len": bad_k,
        "bad_prompt_placeholders": bad_prompt,
        "bad_subgraph": bad_subgraph,
        "avg_subgraph_size": float(np.mean(sizes)) if sizes else 0.0,
        "min_subgraph_size": int(np.min(sizes)) if sizes else 0,
        "max_subgraph_size": int(np.max(sizes)) if sizes else 0,
        "rank21_count": int(rank21),
        "train_gold_forced_count": int(train_forced),
        "valid_test_exact_leak_count": int(leaks),
        "schema_pass": bad_k == 0 and bad_prompt == 0 and bad_subgraph == 0 and leaks == 0,
    }


def build_one_source(
    source_model: str,
    entity2id,
    id2entity,
    relation2id,
    id2relation,
    type_map,
    candidate_names,
    candidate_ids,
    adj,
    edge_set,
    kind_by_id,
    args,
):
    print("=" * 100)
    print(f"[source] {source_model}")
    print("=" * 100)

    source_dir = mkdir(SOURCE_ROOT / source_model)
    ready_dir = mkdir(READY_ROOT / source_model)

    train_targets = load_target_rows("train")
    valid_targets = load_target_rows("valid")
    test_targets = load_target_rows("test")

    valid_source = make_eval_source_rows(source_model, "valid", valid_targets, entity2id, id2entity)
    test_source = make_eval_source_rows(source_model, "test", test_targets, entity2id, id2entity)
    eval_for_train = valid_source + test_source

    train_source = make_train_source_rows(
        source_model=source_model,
        train_rows=train_targets,
        eval_source_rows=eval_for_train,
        candidate_names=candidate_names,
        candidate_ids=candidate_ids,
        id2entity=id2entity,
        train_ensure_gold=args.train_ensure_gold,
    )

    source_rows = {
        "train": train_source,
        "valid": valid_source,
        "test": test_source,
    }

    for split, rows in source_rows.items():
        write_json(rows, source_dir / f"{split}_top20_raw.json")

    candidate_metrics = {
        split: compute_metrics(rows)
        for split, rows in source_rows.items()
    }

    ready_rows = {}
    for split, rows in source_rows.items():
        ready = []
        for row in rows:
            out = dict(row)
            add_prompt(out)
            out["subgraph"] = retrieve_subgraph(out, adj, id2relation, kind_by_id, args)
            ready.append(out)

        ready_rows[split] = ready
        write_json(ready, ready_dir / f"{split}.json")
        print(f"[ready] {source_model} {split}: rows={len(ready)}")

    embedding_path = choose_embedding_for_source(source_model)
    copy_static_files(ready_dir, embedding_path)

    prompt_lexicon = {
        "head_prediction": {
            TARGET_RELATION: "What compound treats {}?"
        },
        "tail_prediction": {},
    }
    rules = {
        TARGET_RELATION: [
            ["Compound:Gene", "Gene:Disease"],
            ["Compound:Disease"],
            ["Gene:Gene"]
        ]
    }
    support_schema = {
        "setting": "setting_e_drkg",
        "target_relation": TARGET_RELATION,
        "source_model": source_model,
        "positive_evidence_families": [
            "target_treats",
            "aux_compound_disease",
            "compound_gene",
            "disease_gene",
            "gene_gene_bridge"
        ],
        "shortcut_relations": [
            "DRUGBANK::treats::Compound:Disease",
            "GNBR::T::Compound:Disease",
            "GNBR::Sa::Compound:Disease",
            "GNBR::Pa::Compound:Disease",
            "GNBR::C::Compound:Disease",
            "GNBR::J::Compound:Disease",
            "GNBR::Pr::Compound:Disease",
            "GNBR::Mp::Compound:Disease"
        ],
        "contradiction_relations": [],
        "note": "DRKG Day5 SoftFuse-ready package. DistMult is the main source; R-GCN is diagnostic."
    }

    write_json(prompt_lexicon, ready_dir / "prompt_lexicon.json")
    write_json(rules, ready_dir / "rules.json")
    write_json(support_schema, ready_dir / "support_schema.json")

    audit = {
        split: audit_rows(rows, split, edge_set)
        for split, rows in ready_rows.items()
    }

    decision = "DAY5_DRKG_SOFTFUSE_READY"
    if not all(a["schema_pass"] for a in audit.values()):
        decision = "DAY5_DRKG_SOFTFUSE_NEEDS_FIX"

    manifest = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_e_drkg",
        "dataset": "DRKG",
        "source_model": source_model,
        "task": "(?, DRUGBANK::treats, disease)",
        "prediction_type": "predicted_head",
        "target_relation": TARGET_RELATION,
        "target_relation_id": int(relation2id[TARGET_RELATION]),
        "target_relation_normalized": TARGET_RELATION_NORMALIZED,
        "candidate_universe": "train_target_relation_compound_heads",
        "top_k": TOP_K,
        "valid_test_gold_injection": False,
        "train_ensure_gold": bool(args.train_ensure_gold),
        "graph_size": int(args.graph_size),
        "graph_num_rels": int(len(relation2id)),
        "embedding_source_path": str(embedding_path),
        "ready_embedding_path": "entity_embeddings_rgcn.pt",
        "candidate_metrics": candidate_metrics,
        "ready_audit": audit,
        "notes": [
            "Valid/test rows preserve reviewer-safe no-gold-injection protocol.",
            "Train candidate gold forcing is training-only compatibility.",
            "Source model controls candidate order; embedding may fall back to R-GCN tensor for GraphEnhancer compatibility."
        ],
    }

    write_json(manifest, source_dir / "source_manifest.json")
    write_json(manifest, ready_dir / "prep_manifest.json")

    return manifest


def write_report(summary: dict[str, Any]) -> None:
    path = REPORT_DIR / "day5_drkg_softfuse_ready.md"
    lines = []
    lines.append("# DRKG SoftFuse-ready packages")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append(f"- Main source: `{summary['main_source']}`")
    lines.append(f"- Diagnostic source: `{summary['diagnostic_source']}`")
    lines.append("")

    for source_model, m in summary["sources"].items():
        lines.append(f"## Source: `{source_model}`")
        lines.append("")
        lines.append(f"- Decision: `{m['decision']}`")
        lines.append(f"- graph_num_rels: `{m['graph_num_rels']}`")
        lines.append(f"- embedding source: `{m['embedding_source_path']}`")
        lines.append("")
        lines.append("| Split | Gold@20 | MRR@20 | H@1 | H@3 | H@10 | Rank21 | Top1Dom | Avg graph | Schema pass |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for split in ["train", "valid", "test"]:
            cm = m["candidate_metrics"][split]
            au = m["ready_audit"][split]
            lines.append(
                f"| {split} | {cm['gold_present_at20']:.3f} | {cm['mrr_at20']:.6f} | "
                f"{cm['hits1_at20']:.3f} | {cm['hits3_at20']:.3f} | {cm['hits10_at20']:.3f} | "
                f"{cm['rank21_count']} | {cm['top1_dominance']:.3f} | "
                f"{au['avg_subgraph_size']:.2f} | {au['schema_pass']} |"
            )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- DistMult is the main DRKG source because it has the best validation/test MRR and low top-1 dominance.")
    lines.append("- R-GCN is kept as a graph-compatible diagnostic source but should not be the main DRKG source due to low Gold@20 and top-1 collapse.")
    lines.append("- Day 6 should apply soft support first to DistMult. R-GCN can be processed after DistMult as a diagnostic.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["distmult", "rgcn"])
    parser.add_argument("--main-source", default="distmult")
    parser.add_argument("--diagnostic-source", default="rgcn")
    parser.add_argument("--graph-size", type=int, default=100)
    parser.add_argument("--candidate-gene-cap", type=int, default=12)
    parser.add_argument("--disease-gene-cap", type=int, default=60)
    parser.add_argument("--train-ensure-gold", action="store_true")
    args = parser.parse_args()

    for p in [SOURCE_ROOT, READY_ROOT, RESULT_DIR, REPORT_DIR]:
        mkdir(p)

    entity2id, id2entity, relation2id, id2relation, type_map = load_maps()
    candidate_names, candidate_ids = load_candidate_universe()

    print("[graph] loading adjacency")
    adj, edge_set = build_adjacency(GRAPH_DIR / "train_enriched_ids.tsv")
    kind_by_id = kind_by_entity_id(id2entity, type_map)

    print("[maps] entities =", len(entity2id), "relations =", len(relation2id))
    print("[task] target_relation_id =", relation2id[TARGET_RELATION])
    print("[candidate universe] size =", len(candidate_ids))

    manifests = {}
    for source_model in args.sources:
        manifests[source_model] = build_one_source(
            source_model=source_model,
            entity2id=entity2id,
            id2entity=id2entity,
            relation2id=relation2id,
            id2relation=id2relation,
            type_map=type_map,
            candidate_names=candidate_names,
            candidate_ids=candidate_ids,
            adj=adj,
            edge_set=edge_set,
            kind_by_id=kind_by_id,
            args=args,
        )

    decision = "DAY5_DRKG_SOFTFUSE_READY"
    if not all(m["decision"] == "DAY5_DRKG_SOFTFUSE_READY" for m in manifests.values()):
        decision = "DAY5_DRKG_SOFTFUSE_PARTIAL_OR_NEEDS_FIX"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_e_drkg",
        "dataset": "DRKG",
        "task": "(?, DRUGBANK::treats, disease)",
        "target_relation": TARGET_RELATION,
        "main_source": args.main_source,
        "diagnostic_source": args.diagnostic_source,
        "sources": manifests,
        "next_step": "Day 6 should run support + soft-support on distmult main source first; rgcn remains diagnostic.",
    }

    write_json(summary, RESULT_DIR / "day5_drkg_softfuse_ready_summary.json")
    write_report(summary)

    print("\n[DONE] Day 5 DRKG SoftFuse-ready")
    print(json.dumps({
        "decision": decision,
        "main_source": args.main_source,
        "diagnostic_source": args.diagnostic_source,
        "source_decisions": {k: v["decision"] for k, v in manifests.items()},
        "candidate_metrics": {k: v["candidate_metrics"] for k, v in manifests.items()},
        "audit": {k: v["ready_audit"] for k, v in manifests.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
