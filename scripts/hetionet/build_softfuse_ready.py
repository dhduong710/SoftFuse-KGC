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
SETTING_DIR = ROOT / "dataset" / "setting_d_hetionet"
SPLIT_DIR = SETTING_DIR / "splits"
GRAPH_DIR = SETTING_DIR / "graph"
BASELINE_DIR = SETTING_DIR / "baseline_outputs"
SOURCE_DIR = SETTING_DIR / "backbone_raw_source"
READY_DIR = SETTING_DIR / "softfuse_ready"
RESULT_DIR = ROOT / "outputs" / "hetionet"
REPORT_DIR = ROOT / "outputs" / "hetionet" / "reports"

TARGET_RELATION = "CtD"
TARGET_RELATION_NORMALIZED = "compound_treats_disease"
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


def normalize_id2entity(obj: dict) -> dict[int, str]:
    return {int(k): v for k, v in obj.items()}


def load_maps():
    entity2id = read_json(GRAPH_DIR / "entity2id.json")
    id2entity = normalize_id2entity(read_json(GRAPH_DIR / "id2entity.json"))
    relation2id = read_json(GRAPH_DIR / "relation2id.json")
    id2relation = {int(k): v for k, v in read_json(GRAPH_DIR / "id2relation.json").items()}
    type_map = read_json(GRAPH_DIR / "type_map.json")
    return entity2id, id2entity, relation2id, id2relation, type_map


def load_rows(split: str) -> list[dict[str, Any]]:
    return read_json(SPLIT_DIR / f"{split}_target_rows.json")


def load_candidate_universe() -> tuple[list[str], list[int]]:
    obj = read_json(SPLIT_DIR / "candidate_universe_compound.json")
    return obj["candidate_entities"], [int(x) for x in obj["candidate_entity_ids"]]


def load_rgcn_scoring():
    rgcn_dir = BASELINE_DIR / "rgcn"
    emb_path = rgcn_dir / "entity_embeddings_rgcn.pt"
    state_path = rgcn_dir / "model_state.pt"

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing R-GCN embedding: {emb_path}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing R-GCN model state: {state_path}")

    z = torch.load(emb_path, map_location="cpu").float()
    state = torch.load(state_path, map_location="cpu")

    if "score_rel.weight" not in state:
        raise KeyError("model_state.pt does not contain score_rel.weight")

    score_rel = state["score_rel.weight"].float()

    return z, score_rel, emb_path, state_path


def score_topk(z, score_rel, candidate_ids, relation_id: int, query_id: int, k: int = TOP_K):
    cand = torch.LongTensor(candidate_ids)
    q = int(query_id)

    h = z[cand]
    r = score_rel[int(relation_id)].view(1, -1)
    t = z[q].view(1, -1)

    scores = torch.sum(h * r * t, dim=-1)
    top_scores, top_idx = torch.topk(scores, k=min(k, len(candidate_ids)), largest=True)

    top_idx = top_idx.cpu().numpy().tolist()
    top_scores = top_scores.cpu().numpy().tolist()

    out_ids = [int(candidate_ids[i]) for i in top_idx]
    out_scores = [float(x) for x in top_scores]
    return out_ids, out_scores


def ensure_train_gold(row, top_ids, top_scores):
    gold_id = int(row["gold_entity_id"])
    if gold_id in top_ids:
        return top_ids, top_scores, False

    # Training compatibility only: replace last candidate with gold.
    new_ids = list(top_ids)
    new_scores = list(top_scores)
    if len(new_ids) >= TOP_K:
        new_ids[-1] = gold_id
        new_scores[-1] = float(min(new_scores) - 1e-6) if new_scores else 0.0
    else:
        new_ids.append(gold_id)
        new_scores.append(float(min(new_scores) - 1e-6) if new_scores else 0.0)

    return new_ids, new_scores, True


def rank_gold(gold_id: int, candidate_ids: list[int]) -> tuple[int, bool]:
    if int(gold_id) in candidate_ids:
        return candidate_ids.index(int(gold_id)) + 1, True
    return ABSENT_RANK, False


def compute_candidate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.array([r["rank"] for r in rows], dtype=np.int64)
    present = np.array([r["gold_in_topk_raw"] for r in rows], dtype=bool)

    rr = np.zeros(len(rows), dtype=np.float64)
    for i, rank in enumerate(ranks):
        if present[i] and rank <= TOP_K:
            rr[i] = 1.0 / rank

    top1 = [r["candidate_entities"][0] for r in rows if r["candidate_entities"]]
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


def export_source_rows(split: str, split_rows, z, score_rel, candidate_names, candidate_ids, id2entity, relation2id, train_ensure_gold: bool):
    relation_id = int(relation2id[TARGET_RELATION])
    out = []
    num_train_forced = 0

    for row in split_rows:
        top_ids, top_scores = score_topk(
            z=z,
            score_rel=score_rel,
            candidate_ids=candidate_ids,
            relation_id=relation_id,
            query_id=int(row["query_entity_id"]),
            k=TOP_K,
        )

        forced = False
        if split == "train" and train_ensure_gold:
            top_ids, top_scores, forced = ensure_train_gold(row, top_ids, top_scores)
            if forced:
                num_train_forced += 1

        top_names = [id2entity[int(x)] for x in top_ids]
        rank, present = rank_gold(int(row["gold_entity_id"]), top_ids)

        out_row = dict(row)
        out_row.update({
            "source_model": "rgcn",
            "target_relation": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "candidate_entities": top_names,
            "candidate_entity_ids": [int(x) for x in top_ids],
            "scores": [float(x) for x in top_scores],
            "rank_entities": top_names,
            "rank_entities_id": [int(x) for x in top_ids],
            "rank": int(rank),
            "gold_rank_in_top20_or_21": int(rank),
            "gold_in_topk_raw": bool(present),
            "gold_present_top20": bool(present),
            "gold_injected": False,
            "train_gold_forced_into_candidates": bool(forced),
            "reviewer_safe_rr_item": float(1.0 / rank) if present and rank <= TOP_K else 0.0,
        })
        out.append(out_row)

    return out, num_train_forced


def add_prompt(row: dict[str, Any]) -> None:
    query = row["query_entity"]
    candidates = row["rank_entities"]

    answer_options = "(" + ", ".join([f"'{x}'" for x in candidates]) + ")"
    refer_parts = [f"'{query}': [QUERY]"]
    refer_parts.extend([f"'{x}': [ENTITY]" for x in candidates])
    refer_str = ", ".join(refer_parts)

    question = f"What compound treats {query}?"

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

    row["input"] = prompt
    row["output"] = row["gold_entity"]


def build_kind_by_id(id2entity: dict[int, str], type_map: dict[str, Any]) -> dict[int, str]:
    out = {}
    for eid, name in id2entity.items():
        meta = type_map.get(name, {})
        out[int(eid)] = meta.get("kind", "UNKNOWN")
    return out


def read_train_edges_ids(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            yield int(row[0]), int(row[1]), int(row[2])


def build_adjacency(train_ids_path: Path):
    adj = defaultdict(list)
    edge_set = set()
    rel_counts = Counter()
    n = 0

    for h, r, t in read_train_edges_ids(train_ids_path):
        tup = (h, r, t)
        edge_set.add(tup)
        adj[h].append((r, t, tup))
        adj[t].append((r, h, tup))
        rel_counts[r] += 1
        n += 1

    return adj, edge_set, rel_counts, n


def add_edge(subg, seen, edge):
    if edge in seen:
        return False
    seen.add(edge)
    subg.append([int(edge[0]), int(edge[1]), int(edge[2])])
    return True


def collect_node_edges(adj, node_id: int, relation_ids: set[int] | None, neighbor_kind: str | None, kind_by_id, limit: int):
    out = []
    for r, nb, edge in adj.get(int(node_id), []):
        if relation_ids is not None and int(r) not in relation_ids:
            continue
        if neighbor_kind is not None and kind_by_id.get(int(nb)) != neighbor_kind:
            continue
        out.append((int(r), int(nb), edge))
        if len(out) >= limit:
            break
    return out


def retrieve_subgraph(row, adj, kind_by_id, relation2id, graph_size: int, cand_gene_cap: int, disease_gene_cap: int):
    q = int(row["query_entity_id"])
    candidates = [int(x) for x in row["rank_entities_id"]]

    rel_code_to_id = {k: int(v) for k, v in relation2id.items()}

    compound_gene_rels = {rel_code_to_id[x] for x in ["CbG", "CdG", "CuG"] if x in rel_code_to_id}
    disease_gene_rels = {rel_code_to_id[x] for x in ["DaG", "DdG", "DuG"] if x in rel_code_to_id}
    gene_gene_rels = {rel_code_to_id[x] for x in ["GiG", "Gr>G", "GcG"] if x in rel_code_to_id}
    direct_rels = {rel_code_to_id[x] for x in ["CtD", "CpD"] if x in rel_code_to_id}

    subg = []
    seen = set()

    # 1. Direct candidate-query edges, including train CtD/CpD if present.
    for c in candidates:
        for r, nb, edge in adj.get(c, []):
            if nb == q and r in direct_rels:
                add_edge(subg, seen, edge)
                if len(subg) >= graph_size:
                    return subg

    # Precollect disease genes.
    disease_gene_edges = collect_node_edges(
        adj=adj,
        node_id=q,
        relation_ids=disease_gene_rels,
        neighbor_kind="Gene",
        kind_by_id=kind_by_id,
        limit=disease_gene_cap,
    )
    disease_genes = {nb for _, nb, _ in disease_gene_edges}

    # 2. Candidate-gene + disease-gene shared evidence.
    for c in candidates:
        cand_gene_edges = collect_node_edges(
            adj=adj,
            node_id=c,
            relation_ids=compound_gene_rels,
            neighbor_kind="Gene",
            kind_by_id=kind_by_id,
            limit=cand_gene_cap,
        )

        cand_genes = {nb for _, nb, _ in cand_gene_edges}
        shared = cand_genes & disease_genes

        if shared:
            for _, nb, edge in cand_gene_edges:
                if nb in shared:
                    add_edge(subg, seen, edge)
                    if len(subg) >= graph_size:
                        return subg
            for _, nb, edge in disease_gene_edges:
                if nb in shared:
                    add_edge(subg, seen, edge)
                    if len(subg) >= graph_size:
                        return subg

    # 3. Add disease gene context.
    for _, _, edge in disease_gene_edges:
        add_edge(subg, seen, edge)
        if len(subg) >= graph_size:
            return subg

    # 4. Add candidate gene context.
    candidate_gene_cache = {}
    for c in candidates:
        cand_gene_edges = collect_node_edges(
            adj=adj,
            node_id=c,
            relation_ids=compound_gene_rels,
            neighbor_kind="Gene",
            kind_by_id=kind_by_id,
            limit=cand_gene_cap,
        )
        candidate_gene_cache[c] = cand_gene_edges
        for _, _, edge in cand_gene_edges:
            add_edge(subg, seen, edge)
            if len(subg) >= graph_size:
                return subg

    # 5. Add simple gene-gene bridges if available.
    disease_gene_set = set(disease_genes)
    for c in candidates:
        for _, g, _ in candidate_gene_cache.get(c, []):
            for r, nb, edge in adj.get(g, []):
                if r in gene_gene_rels and nb in disease_gene_set:
                    add_edge(subg, seen, edge)
                    if len(subg) >= graph_size:
                        return subg

    # 6. Fill with top local edges around query and candidates.
    fill_nodes = [q] + candidates
    for node in fill_nodes:
        for _, _, edge in adj.get(node, []):
            add_edge(subg, seen, edge)
            if len(subg) >= graph_size:
                return subg

    return subg


def count_placeholders(text: str):
    return text.count("[QUERY]"), text.count("[ENTITY]")


def audit_ready_rows(rows: list[dict[str, Any]], split: str, edge_set: set[tuple[int, int, int]]):
    bad_k = 0
    bad_prompt = 0
    bad_subgraph_type = 0
    subgraph_sizes = []
    exact_leaks = 0
    rank21 = 0
    train_forced = 0

    for row in rows:
        if len(row.get("rank_entities_id", [])) != TOP_K:
            bad_k += 1

        qn, en = count_placeholders(row.get("input", ""))
        if qn != 1 or en != TOP_K:
            bad_prompt += 1

        sg = row.get("subgraph", [])
        if not isinstance(sg, list) or any((not isinstance(x, list) or len(x) != 3) for x in sg):
            bad_subgraph_type += 1

        subgraph_sizes.append(len(sg))

        if row.get("rank") == ABSENT_RANK:
            rank21 += 1

        if row.get("train_gold_forced_into_candidates"):
            train_forced += 1

        gold = tuple(int(x) for x in row["triple_id"])
        if split in {"valid", "test"}:
            if any(tuple(int(y) for y in e) == gold for e in sg):
                exact_leaks += 1
            if gold in edge_set:
                exact_leaks += 1

    return {
        "split": split,
        "num_rows": len(rows),
        "bad_candidate_len": bad_k,
        "bad_prompt_placeholders": bad_prompt,
        "bad_subgraph_type": bad_subgraph_type,
        "avg_subgraph_size": float(np.mean(subgraph_sizes)) if subgraph_sizes else 0.0,
        "min_subgraph_size": int(np.min(subgraph_sizes)) if subgraph_sizes else 0,
        "max_subgraph_size": int(np.max(subgraph_sizes)) if subgraph_sizes else 0,
        "rank21_count": int(rank21),
        "train_gold_forced_count": int(train_forced),
        "valid_test_exact_leak_count": int(exact_leaks),
        "schema_pass": bad_k == 0 and bad_prompt == 0 and bad_subgraph_type == 0 and exact_leaks == 0,
    }


def write_report(path: Path, summary: dict[str, Any]):
    lines = []
    lines.append("# Hetionet SoftFuse-ready package")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append(f"- Source model: `{summary['source_model']}`")
    lines.append(f"- Target task: `(? , CtD, disease)`")
    lines.append(f"- graph_num_rels: `{summary['graph_num_rels']}`")
    lines.append(f"- R-GCN embedding shape: `{summary['rgcn_embedding_shape']}`")
    lines.append("")
    lines.append("## Candidate metrics")
    lines.append("")
    lines.append("| Split | Gold@20 | MRR@20 | H@1 | H@3 | H@10 | Rank21 | Top1 dominance |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split, m in summary["candidate_metrics"].items():
        lines.append(
            f"| {split} | {m['gold_present_at20']:.3f} | {m['mrr_at20']:.6f} | "
            f"{m['hits1_at20']:.3f} | {m['hits3_at20']:.3f} | {m['hits10_at20']:.3f} | "
            f"{m['rank21_count']} | {m['top1_dominance']:.3f} |"
        )
    lines.append("")
    lines.append("## Ready package audit")
    lines.append("")
    lines.append("| Split | Rows | Bad K | Bad prompt | Avg graph | Min graph | Max graph | Leaks | Schema pass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split, a in summary["ready_audit"].items():
        lines.append(
            f"| {split} | {a['num_rows']} | {a['bad_candidate_len']} | {a['bad_prompt_placeholders']} | "
            f"{a['avg_subgraph_size']:.2f} | {a['min_subgraph_size']} | {a['max_subgraph_size']} | "
            f"{a['valid_test_exact_leak_count']} | {a['schema_pass']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- R-GCN is retained as the main graph-compatible source despite top-1 collapse.")
    lines.append("- Valid/test rows remain no-gold-injection.")
    lines.append("- `train_gold_forced_into_candidates` only affects supervised train prompts when the observed train answer is absent from top-20.")
    lines.append("- Day 5 should test whether soft-support can reduce top-1 collapse and improve early ranking.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--graph-size", type=int, default=80)
    parser.add_argument("--cand-gene-cap", type=int, default=5)
    parser.add_argument("--disease-gene-cap", type=int, default=40)
    parser.add_argument("--train-ensure-gold", action="store_true")
    args = parser.parse_args()

    mkdir(SOURCE_DIR)
    mkdir(READY_DIR)
    mkdir(RESULT_DIR)
    mkdir(REPORT_DIR)

    entity2id, id2entity, relation2id, id2relation, type_map = load_maps()
    candidate_names, candidate_ids = load_candidate_universe()

    z, score_rel, emb_path, state_path = load_rgcn_scoring()
    print("[load] R-GCN embedding:", emb_path, tuple(z.shape))
    print("[load] R-GCN state:", state_path)
    print("[maps] entities =", len(entity2id), "relations =", len(relation2id), "CtD id =", relation2id[TARGET_RELATION])

    source_rows = {}
    train_forced_by_split = {}

    for split in ["train", "valid", "test"]:
        rows = load_rows(split)
        out_rows, forced = export_source_rows(
            split=split,
            split_rows=rows,
            z=z,
            score_rel=score_rel,
            candidate_names=candidate_names,
            candidate_ids=candidate_ids,
            id2entity=id2entity,
            relation2id=relation2id,
            train_ensure_gold=args.train_ensure_gold,
        )
        source_rows[split] = out_rows
        train_forced_by_split[split] = forced

        write_json(out_rows, SOURCE_DIR / f"{split}_top20_raw.json")
        print(f"[source] {split}: rows={len(out_rows)} train_forced={forced}")

    candidate_metrics = {
        split: compute_candidate_metrics(rows)
        for split, rows in source_rows.items()
    }

    source_manifest = {
        "created_at": now_iso(),
        "setting": "setting_d_hetionet",
        "source_model": "rgcn",
        "source_model_status": "usable_with_top1_collapse_caution",
        "task": "(?, CtD, disease)",
        "target_relation": TARGET_RELATION,
        "target_relation_id": int(relation2id[TARGET_RELATION]),
        "candidate_universe": "Compound",
        "top_k": TOP_K,
        "valid_test_gold_injection": False,
        "train_ensure_gold": bool(args.train_ensure_gold),
        "train_forced_counts": train_forced_by_split,
        "candidate_metrics": candidate_metrics,
        "rgcn_embedding_path": str(emb_path),
        "rgcn_embedding_shape": list(z.shape),
    }
    write_json(source_manifest, SOURCE_DIR / "source_manifest.json")

    print("[graph] building adjacency from train_enriched_ids.tsv")
    adj, edge_set, rel_counts, n_edges = build_adjacency(GRAPH_DIR / "train_enriched_ids.tsv")
    print("[graph] train edges =", n_edges, "adj nodes =", len(adj))

    kind_by_id = build_kind_by_id(id2entity, type_map)

    prompt_lexicon = {
        "head_prediction": {
            TARGET_RELATION: "What compound treats {}?"
        },
        "tail_prediction": {},
    }
    rules = {
        TARGET_RELATION: [
            ["CbG", "DaG"],
            ["CdG", "DdG"],
            ["CuG", "DuG"],
            ["CbG", "GiG", "DaG"],
            ["CdG", "GiG", "DdG"],
            ["CuG", "GiG", "DuG"],
            ["CpD"]
        ]
    }
    support_schema = {
        "setting": "setting_d_hetionet",
        "target_relation": TARGET_RELATION,
        "positive_evidence_families": [
            "direct_train_CtD_or_CpD",
            "compound_gene_and_disease_gene_shared_support",
            "compound_gene_context",
            "disease_gene_context",
            "gene_gene_bridge"
        ],
        "shortcut_relations": ["CtD", "CpD"],
        "contradiction_relations": [],
        "note": "Hetionet Day4 package creates compact source evidence subgraphs. Day5 will compute candidate support features."
    }

    write_json(prompt_lexicon, READY_DIR / "prompt_lexicon.json")
    write_json(rules, READY_DIR / "rules.json")
    write_json(support_schema, READY_DIR / "support_schema.json")

    ready_rows = {}
    for split in ["train", "valid", "test"]:
        ready = []
        for row in source_rows[split]:
            new_row = dict(row)
            add_prompt(new_row)
            new_row["subgraph"] = retrieve_subgraph(
                row=new_row,
                adj=adj,
                kind_by_id=kind_by_id,
                relation2id=relation2id,
                graph_size=args.graph_size,
                cand_gene_cap=args.cand_gene_cap,
                disease_gene_cap=args.disease_gene_cap,
            )
            ready.append(new_row)

        ready_rows[split] = ready
        write_json(ready, READY_DIR / f"{split}.json")
        print(f"[ready] {split}: rows={len(ready)}")

    # Copy mappings and embedding to ready package.
    for name in [
        "entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl",
        "entity2id.json", "id2entity.json", "relation2id.json", "id2relation.json",
        "type_map.json", "graph_summary.json", "leak_check.json"
    ]:
        src = GRAPH_DIR / name
        if src.exists():
            shutil.copy2(src, READY_DIR / name)

    shutil.copy2(emb_path, READY_DIR / "entity_embeddings_rgcn.pt")

    ready_audit = {
        split: audit_ready_rows(rows, split, edge_set)
        for split, rows in ready_rows.items()
    }

    decision = "DAY4_HETIONET_SOFTFUSE_READY"
    if not all(x["schema_pass"] for x in ready_audit.values()):
        decision = "DAY4_HETIONET_SOFTFUSE_NEEDS_FIX"

    prep_manifest = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_d_hetionet",
        "dataset": "Hetionet v1.0",
        "source_model": "rgcn",
        "source_model_status": "usable_with_top1_collapse_caution",
        "task": "(?, CtD, disease)",
        "prediction_type": "predicted_head",
        "target_relation": TARGET_RELATION,
        "target_relation_id": int(relation2id[TARGET_RELATION]),
        "candidate_universe": "Compound",
        "top_k": TOP_K,
        "graph_size": int(args.graph_size),
        "graph_num_rels": int(len(relation2id)),
        "rgcn_embedding_path": "entity_embeddings_rgcn.pt",
        "rgcn_embedding_shape": list(z.shape),
        "valid_test_gold_injection": False,
        "train_ensure_gold": bool(args.train_ensure_gold),
        "required_fields": ["input", "output", "query_entity_id", "rank_entities_id", "subgraph", "rank", "rank_entities"],
        "candidate_metrics": candidate_metrics,
        "ready_audit": ready_audit,
        "source_manifest_path": str(SOURCE_DIR / "source_manifest.json"),
    }
    write_json(prep_manifest, READY_DIR / "prep_manifest.json")

    summary = dict(prep_manifest)
    write_json(summary, RESULT_DIR / "day4_hetionet_softfuse_ready_audit.json")
    write_report(REPORT_DIR / "day4_hetionet_softfuse_ready.md", summary)

    print("\n[DONE] Day 4")
    print(json.dumps({
        "decision": decision,
        "graph_num_rels": len(relation2id),
        "embedding_shape": list(z.shape),
        "candidate_metrics": candidate_metrics,
        "ready_audit": ready_audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
