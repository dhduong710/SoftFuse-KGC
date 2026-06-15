#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week 22 Day 5b: Rerun Dataset 2 GNN baselines.

Models:
- R-GCN-lite
- HRGAT-lite

Task:
    PharmKG Dataset 2
    (?, T, disease)
    candidate_universe = drug_only_from_train_T_heads
    top_k = 20
    gold_injection = false
    reviewer-safe RR@20

This script:
1. Reads Day 4 split/graph package.
2. Trains R-GCN-lite and/or HRGAT-lite on train_enriched.tsv.
3. Exports valid/test top-20 rows using the same schema as Day 5 KGE outputs.
4. Computes reviewer-safe metrics.
5. Updates:
   - outputs/pharmkg/dataset2_baseline_reviewer_safe_valid.json
   - outputs/pharmkg/dataset2_baseline_reviewer_safe_test.json
   - outputs/pharmkg/dataset2_baseline_main_table.json
   - outputs/pharmkg/reports/day5_baseline_rerun_dataset2.md
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(".")

SPLIT_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "splits"
GRAPH_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "graph"
BASELINE_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "baseline_outputs"

RESULT_DIR = ROOT / "outputs" / "pharmkg"
REPORT_DIR = ROOT / "outputs" / "pharmkg" / "reports"

VALID_METRICS_PATH = RESULT_DIR / "dataset2_baseline_reviewer_safe_valid.json"
TEST_METRICS_PATH = RESULT_DIR / "dataset2_baseline_reviewer_safe_test.json"
MAIN_TABLE_PATH = RESULT_DIR / "dataset2_baseline_main_table.json"
REPORT_PATH = REPORT_DIR / "day5_baseline_rerun_dataset2.md"

TARGET_RELATION = "T"
TARGET_RELATION_NORMALIZED = "therapeutic_association_proxy"
TOP_K = 20
ABSENT_RANK = 21


def ensure_dirs() -> None:
    for p in [BASELINE_DIR, RESULT_DIR, REPORT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_maps() -> tuple[dict[str, int], dict[str, str], dict[str, int], dict[str, str]]:
    entity2id = read_json(GRAPH_DIR / "entity2id.json")
    id2entity = read_json(GRAPH_DIR / "id2entity.json")
    relation2id = read_json(GRAPH_DIR / "relation2id.json")
    id2relation = read_json(GRAPH_DIR / "id2relation.json")
    return entity2id, id2entity, relation2id, id2relation


def load_train_enriched_ids(
    entity2id: dict[str, int],
    relation2id: dict[str, int],
) -> torch.LongTensor:
    path = GRAPH_DIR / "train_enriched.tsv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["head", "relation", "tail"],
        dtype=str,
        keep_default_na=False,
    )

    bad_h = df[~df["head"].isin(entity2id)]
    bad_t = df[~df["tail"].isin(entity2id)]
    bad_r = df[~df["relation"].isin(relation2id)]

    if len(bad_h) or len(bad_t) or len(bad_r):
        raise RuntimeError(
            f"Unmapped rows in train_enriched.tsv: "
            f"bad_h={len(bad_h)}, bad_r={len(bad_r)}, bad_t={len(bad_t)}"
        )

    h = df["head"].map(entity2id).astype(int).values
    r = df["relation"].map(relation2id).astype(int).values
    t = df["tail"].map(entity2id).astype(int).values

    arr = np.stack([h, r, t], axis=1)
    return torch.LongTensor(arr)


def load_candidate_universe(entity2id: dict[str, int]) -> tuple[list[str], torch.LongTensor]:
    names = read_json(SPLIT_DIR / "candidate_universe.json")
    ids = [int(entity2id[name]) for name in names]
    return names, torch.LongTensor(ids)


def load_eval_rows(split: str) -> list[dict[str, Any]]:
    return read_json(SPLIT_DIR / f"{split}.json")


def build_graph_edges(
    triples: torch.LongTensor,
    num_relations: int,
    device: torch.device,
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    h = triples[:, 0].to(device)
    r = triples[:, 1].to(device)
    t = triples[:, 2].to(device)

    src = torch.cat([h, t], dim=0)
    dst = torch.cat([t, h], dim=0)
    rel = torch.cat([r, r + num_relations], dim=0)

    return src, rel, dst


class RGCNLiteModel(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_score_relations: int,
        num_graph_relations: int,
        dim: int,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base = nn.Embedding(num_entities, dim)
        self.rel_gate = nn.ModuleList(
            [nn.Embedding(num_graph_relations, dim) for _ in range(layers)]
        )
        self.self_loop = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.score_rel = nn.Embedding(num_score_relations, dim)
        self.layers = layers
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.base.weight)
        nn.init.xavier_uniform_(self.score_rel.weight)
        for emb in self.rel_gate:
            nn.init.xavier_uniform_(emb.weight)

    def all_embeddings(
        self,
        edge_src: torch.LongTensor,
        edge_rel: torch.LongTensor,
        edge_dst: torch.LongTensor,
    ) -> torch.Tensor:
        h = self.base.weight
        n = h.size(0)

        for layer in range(self.layers):
            msg = h[edge_src] * self.rel_gate[layer](edge_rel)

            agg = torch.zeros_like(h)
            agg.index_add_(0, edge_dst, msg)

            deg = torch.zeros(n, device=h.device)
            deg.index_add_(0, edge_dst, torch.ones_like(edge_dst, dtype=torch.float))

            agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
            h = F.relu(self.self_loop[layer](h) + agg)
            h = self.dropout(h)

        return h

    def score_with_z(
        self,
        z: torch.Tensor,
        h: torch.LongTensor,
        r: torch.LongTensor,
        t: torch.LongTensor,
    ) -> torch.Tensor:
        return torch.sum(z[h] * self.score_rel(r) * z[t], dim=-1)

    def score_heads_with_z(
        self,
        z: torch.Tensor,
        heads: torch.LongTensor,
        rel_id: int,
        tail_id: int,
    ) -> torch.Tensor:
        r = torch.full_like(heads, fill_value=rel_id)
        t = torch.full_like(heads, fill_value=tail_id)
        return self.score_with_z(z, heads, r, t)


class HRGATLiteModel(nn.Module):
    def __init__(
        self,
        num_entities: int,
        num_score_relations: int,
        num_graph_relations: int,
        dim: int,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.base = nn.Embedding(num_entities, dim)
        self.rel_gate = nn.ModuleList(
            [nn.Embedding(num_graph_relations, dim) for _ in range(layers)]
        )
        self.self_loop = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.att_src = nn.ParameterList(
            [nn.Parameter(torch.empty(dim)) for _ in range(layers)]
        )
        self.att_dst = nn.ParameterList(
            [nn.Parameter(torch.empty(dim)) for _ in range(layers)]
        )
        self.att_rel = nn.ModuleList(
            [nn.Embedding(num_graph_relations, dim) for _ in range(layers)]
        )
        self.score_rel = nn.Embedding(num_score_relations, dim)
        self.layers = layers
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.base.weight)
        nn.init.xavier_uniform_(self.score_rel.weight)

        for emb in self.rel_gate:
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.att_rel:
            nn.init.xavier_uniform_(emb.weight)
        for p in self.att_src:
            nn.init.normal_(p, std=0.02)
        for p in self.att_dst:
            nn.init.normal_(p, std=0.02)

    def all_embeddings(
        self,
        edge_src: torch.LongTensor,
        edge_rel: torch.LongTensor,
        edge_dst: torch.LongTensor,
    ) -> torch.Tensor:
        h = self.base.weight
        n = h.size(0)

        for layer in range(self.layers):
            rel_vec = self.rel_gate[layer](edge_rel)
            msg = h[edge_src] * rel_vec

            logits = (
                (h[edge_src] * self.att_src[layer]).sum(dim=-1)
                + (h[edge_dst] * self.att_dst[layer]).sum(dim=-1)
                + (self.att_rel[layer](edge_rel) * rel_vec).sum(dim=-1)
            )

            alpha = torch.sigmoid(F.leaky_relu(logits, negative_slope=0.2))
            weighted = msg * alpha.unsqueeze(-1)

            agg = torch.zeros_like(h)
            agg.index_add_(0, edge_dst, weighted)

            norm = torch.zeros(n, device=h.device)
            norm.index_add_(0, edge_dst, alpha)

            agg = agg / norm.clamp_min(1e-6).unsqueeze(-1)
            h = F.elu(self.self_loop[layer](h) + agg)
            h = self.dropout(h)

        return h

    def score_with_z(
        self,
        z: torch.Tensor,
        h: torch.LongTensor,
        r: torch.LongTensor,
        t: torch.LongTensor,
    ) -> torch.Tensor:
        return torch.sum(z[h] * self.score_rel(r) * z[t], dim=-1)

    def score_heads_with_z(
        self,
        z: torch.Tensor,
        heads: torch.LongTensor,
        rel_id: int,
        tail_id: int,
    ) -> torch.Tensor:
        r = torch.full_like(heads, fill_value=rel_id)
        t = torch.full_like(heads, fill_value=tail_id)
        return self.score_with_z(z, heads, r, t)


def build_model(
    model_name: str,
    num_entities: int,
    num_relations: int,
    dim: int,
    layers: int,
    dropout: float,
    device: torch.device,
) -> nn.Module:
    num_graph_relations = num_relations * 2

    if model_name == "rgcn":
        return RGCNLiteModel(
            num_entities=num_entities,
            num_score_relations=num_relations,
            num_graph_relations=num_graph_relations,
            dim=dim,
            layers=layers,
            dropout=dropout,
        ).to(device)

    if model_name == "hrgat":
        return HRGATLiteModel(
            num_entities=num_entities,
            num_score_relations=num_relations,
            num_graph_relations=num_graph_relations,
            dim=dim,
            layers=layers,
            dropout=dropout,
        ).to(device)

    raise ValueError(f"Unsupported model_name={model_name}")


def make_negative_batch(
    pos: torch.LongTensor,
    num_entities: int,
    candidate_ids: torch.LongTensor,
    target_relation_id: int,
    device: torch.device,
) -> torch.LongTensor:
    neg = pos.clone()
    n = neg.size(0)

    corrupt_head = torch.rand(n, device=device) < 0.5
    target_mask = neg[:, 1] == target_relation_id

    head_mask = corrupt_head | target_mask
    tail_mask = ~head_mask

    if head_mask.any():
        idx = torch.randint(
            low=0,
            high=candidate_ids.numel(),
            size=(int(head_mask.sum().item()),),
            device=device,
        )
        neg[head_mask, 0] = candidate_ids.to(device)[idx]

    if tail_mask.any():
        neg[tail_mask, 2] = torch.randint(
            low=0,
            high=num_entities,
            size=(int(tail_mask.sum().item()),),
            device=device,
        )

    return neg


def train_gnn(
    model: nn.Module,
    triples: torch.LongTensor,
    edge_src: torch.LongTensor,
    edge_rel: torch.LongTensor,
    edge_dst: torch.LongTensor,
    num_entities: int,
    candidate_ids: torch.LongTensor,
    target_relation_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.train()
    triples = triples.to(device)
    candidate_ids = candidate_ids.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    log = []
    n = triples.size(0)

    for epoch in range(1, args.gnn_epochs + 1):
        model.train()

        sample_n = min(args.gnn_sample_per_epoch, n)
        idx = torch.randint(0, n, (sample_n,), device=device)

        pos = triples[idx]
        neg = make_negative_batch(
            pos=pos,
            num_entities=num_entities,
            candidate_ids=candidate_ids,
            target_relation_id=target_relation_id,
            device=device,
        )

        z = model.all_embeddings(edge_src, edge_rel, edge_dst)

        pos_score = model.score_with_z(z, pos[:, 0], pos[:, 1], pos[:, 2])
        neg_score = model.score_with_z(z, neg[:, 0], neg[:, 1], neg[:, 2])

        loss = F.softplus(-pos_score).mean() + F.softplus(neg_score).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        item = {
            "epoch": int(epoch),
            "loss": float(loss.detach().cpu()),
        }
        log.append(item)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.gnn_epochs:
            print(f"[GNN] epoch={epoch:03d} loss={item['loss']:.6f}")

    return log


@torch.no_grad()
def score_row_top20(
    model: nn.Module,
    z: torch.Tensor,
    row: dict[str, Any],
    candidate_names: list[str],
    candidate_ids: torch.LongTensor,
    target_relation_id: int,
    device: torch.device,
) -> tuple[list[str], list[int], list[float]]:
    model.eval()

    heads = candidate_ids.to(device)
    tail_id = int(row["query_entity_id"])

    scores = model.score_heads_with_z(
        z=z,
        heads=heads,
        rel_id=target_relation_id,
        tail_id=tail_id,
    )

    top_scores, top_idx = torch.topk(scores, k=min(TOP_K, len(candidate_names)), largest=True)

    top_idx_cpu = top_idx.detach().cpu().numpy().tolist()
    top_scores_cpu = top_scores.detach().cpu().numpy().tolist()

    top_names = [candidate_names[i] for i in top_idx_cpu]
    top_ids = [int(candidate_ids[i].item()) for i in top_idx_cpu]
    top_scores_float = [float(x) for x in top_scores_cpu]

    return top_names, top_ids, top_scores_float


def gold_rank(gold: str, top_names: list[str]) -> tuple[int, bool]:
    if gold in top_names:
        return top_names.index(gold) + 1, True
    return ABSENT_RANK, False


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.array([r["gold_rank_in_top20_or_21"] for r in rows], dtype=np.int64)
    present = np.array([r["gold_present_top20"] for r in rows], dtype=bool)

    rr = np.zeros(len(rows), dtype=np.float64)
    for i, rank in enumerate(ranks):
        if present[i] and rank <= TOP_K:
            rr[i] = 1.0 / rank

    top1 = [r["candidate_entities_top20"][0] for r in rows if r["candidate_entities_top20"]]
    top1_counter = Counter(top1)
    most_common = top1_counter.most_common(10)

    return {
        "num_rows": int(len(rows)),
        "gold_present_at20": float(np.mean(present)) if len(rows) else 0.0,
        "mrr_at20": float(np.mean(rr)) if len(rows) else 0.0,
        "hits1_at20": float(np.mean(ranks <= 1)) if len(rows) else 0.0,
        "hits3_at20": float(np.mean(ranks <= 3)) if len(rows) else 0.0,
        "hits10_at20": float(np.mean(ranks <= 10)) if len(rows) else 0.0,
        "hits20_at20": float(np.mean(ranks <= 20)) if len(rows) else 0.0,
        "avg_gold_rank_absent_as_21": float(np.mean(ranks)) if len(rows) else 0.0,
        "gold_rank_21_count": int(np.sum(ranks == ABSENT_RANK)),
        "unique_top1_count": int(len(top1_counter)),
        "top1_dominance": float(most_common[0][1] / len(rows)) if rows and most_common else 0.0,
        "top1_most_common": [{"entity": k, "count": int(v)} for k, v in most_common],
        "rr_policy": "RR = 1/rank if gold present in top20 else 0",
        "absent_rank_sentinel": ABSENT_RANK,
    }


@torch.no_grad()
def export_top20(
    model_name: str,
    model: nn.Module,
    z: torch.Tensor,
    split: str,
    eval_rows: list[dict[str, Any]],
    candidate_names: list[str],
    candidate_ids: torch.LongTensor,
    target_relation_id: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows = []

    for row in eval_rows:
        top_names, top_ids, top_scores = score_row_top20(
            model=model,
            z=z,
            row=row,
            candidate_names=candidate_names,
            candidate_ids=candidate_ids,
            target_relation_id=target_relation_id,
            device=device,
        )

        rank, present = gold_rank(row["gold_entity"], top_names)

        out_rows.append(
            {
                "split": split,
                "model_name": model_name,
                "query_entity": row["query_entity"],
                "query_entity_id": row["query_entity_id"],
                "gold_entity": row["gold_entity"],
                "gold_entity_id": row["gold_entity_id"],
                "candidate_entities_top20": top_names,
                "candidate_entity_ids_top20": top_ids,
                "scores_top20": top_scores,
                "gold_rank_in_top20_or_21": int(rank),
                "gold_present_top20": bool(present),
                "candidate_universe": "drug_only_from_train_T_heads",
                "candidate_universe_size": int(len(candidate_names)),
                "gold_injected": False,
                "target_relation": TARGET_RELATION,
                "target_relation_normalized": TARGET_RELATION_NORMALIZED,
                "reviewer_safe_rr_item": float(1.0 / rank) if present and rank <= TOP_K else 0.0,
            }
        )

    metrics = compute_metrics(out_rows)
    metrics["model_name"] = model_name
    metrics["split"] = split

    return out_rows, metrics


def update_metric_file(path: Path, new_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if path.exists():
        old = read_json(path)
        if not isinstance(old, list):
            old = []
    else:
        old = []

    new_names = {m["model_name"] for m in new_metrics}
    merged = [m for m in old if m.get("model_name") not in new_names]
    merged.extend(new_metrics)
    merged = sorted(merged, key=lambda x: x["mrr_at20"], reverse=True)

    write_json(merged, path)
    return merged


def write_main_table(valid_metrics: list[dict[str, Any]], test_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    table = {
        "dataset": "PharmKG-8k task-specific therapeutic_association_proxy benchmark",
        "protocol": {
            "task": "(?, T, disease)",
            "candidate_universe": "drug_only_from_train_T_heads",
            "top_k": TOP_K,
            "gold_injection": False,
            "rr_policy": "RR = 1/rank if gold present in top20 else 0",
            "absent_rank_sentinel": ABSENT_RANK,
        },
        "valid": sorted(valid_metrics, key=lambda x: x["mrr_at20"], reverse=True),
        "test": sorted(test_metrics, key=lambda x: x["mrr_at20"], reverse=True),
    }
    write_json(table, MAIN_TABLE_PATH)
    return table


def write_report(main_table: dict[str, Any]) -> None:
    def rows(metrics: list[dict[str, Any]]) -> str:
        out = []
        for m in metrics:
            out.append(
                f"| {m['model_name']} | {m['gold_present_at20']:.3f} | "
                f"{m['mrr_at20']:.6f} | {m['hits1_at20']:.3f} | "
                f"{m['hits3_at20']:.3f} | {m['hits10_at20']:.3f} | "
                f"{m['hits20_at20']:.3f} | "
                f"{m['avg_gold_rank_absent_as_21']:.3f} | "
                f"{m['gold_rank_21_count']} | {m['unique_top1_count']} | "
                f"{m['top1_dominance']:.3f} |"
            )
        return "\n".join(out)

    md = f"""# Week 22 Day 5  PharmKG Dataset 2 Structure Baselines

## Protocol

- Dataset: PharmKG-8k task-specific benchmark
- Task: `(?, T, disease)`
- Relation normalized: `therapeutic_association_proxy`
- Candidate universe: `drug_only_from_train_T_heads`
- Top-K: 20
- Gold injection: false
- RR policy: RR = 1/rank if gold appears in top-20, else 0
- Absent rank sentinel: 21

## Validation metrics

| Model | Gold@20 | MRR@20 | H@1 | H@3 | H@10 | H@20 | Avg rank | Rank21 | Unique top1 | Top1 dominance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows(main_table["valid"])}

## Test metrics

| Model | Gold@20 | MRR@20 | H@1 | H@3 | H@10 | H@20 | Avg rank | Rank21 | Unique top1 | Top1 dominance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows(main_table["test"])}

## Decision

If all six models are present:

`DATASET2_BASELINES_READY`

Otherwise:

`DATASET2_BASELINES_PARTIAL_READY`
"""
    REPORT_PATH.write_text(md, encoding="utf-8")


def run_model(
    model_name: str,
    args: argparse.Namespace,
    device: torch.device,
    entity2id: dict[str, int],
    id2entity: dict[str, str],
    relation2id: dict[str, int],
    triples: torch.LongTensor,
    edge_src: torch.LongTensor,
    edge_rel: torch.LongTensor,
    edge_dst: torch.LongTensor,
    candidate_names: list[str],
    candidate_ids: torch.LongTensor,
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    out_dir = BASELINE_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    num_entities = len(entity2id)
    num_relations = len(relation2id)
    target_relation_id = int(relation2id[TARGET_RELATION])

    model = build_model(
        model_name=model_name,
        num_entities=num_entities,
        num_relations=num_relations,
        dim=args.dim,
        layers=args.gnn_layers,
        dropout=args.dropout,
        device=device,
    )

    start = time.time()

    train_log = train_gnn(
        model=model,
        triples=triples,
        edge_src=edge_src,
        edge_rel=edge_rel,
        edge_dst=edge_dst,
        num_entities=num_entities,
        candidate_ids=candidate_ids,
        target_relation_id=target_relation_id,
        args=args,
        device=device,
    )

    model.eval()
    with torch.no_grad():
        z = model.all_embeddings(edge_src, edge_rel, edge_dst)

    valid_top20, valid_metrics = export_top20(
        model_name=model_name,
        model=model,
        z=z,
        split="valid",
        eval_rows=valid_rows,
        candidate_names=candidate_names,
        candidate_ids=candidate_ids,
        target_relation_id=target_relation_id,
        device=device,
    )

    test_top20, test_metrics = export_top20(
        model_name=model_name,
        model=model,
        z=z,
        split="test",
        eval_rows=test_rows,
        candidate_names=candidate_names,
        candidate_ids=candidate_ids,
        target_relation_id=target_relation_id,
        device=device,
    )

    config = {
        "model_name": model_name,
        "task": "(?, T, disease)",
        "candidate_universe": "drug_only_from_train_T_heads",
        "top_k": TOP_K,
        "gold_injection": False,
        "dim": args.dim,
        "gnn_layers": args.gnn_layers,
        "gnn_epochs": args.gnn_epochs,
        "gnn_sample_per_epoch": args.gnn_sample_per_epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "seed": args.seed,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "num_train_enriched_triples": int(triples.size(0)),
        "runtime_sec": round(time.time() - start, 3),
    }

    write_json(config, out_dir / "config.json")
    write_json(train_log, out_dir / "train_log.json")
    write_json(valid_top20, out_dir / "valid_top20.json")
    write_json(test_top20, out_dir / "test_top20.json")
    write_json(
        {
            "config": config,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        },
        out_dir / "summary.json",
    )

    torch.save(model.state_dict(), out_dir / "model_state.pt")

    print(f"\nFinished {model_name}")
    print("Valid:", json.dumps(valid_metrics, ensure_ascii=False, indent=2))
    print("Test :", json.dumps(test_metrics, ensure_ascii=False, indent=2))

    return valid_metrics, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["rgcn", "hrgat"], choices=["rgcn", "hrgat"])
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--gnn-epochs", type=int, default=40)
    parser.add_argument("--gnn-sample-per-epoch", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)

    device = torch.device(args.device)
    print("Device:", device)
    print("Models:", args.models)

    entity2id, id2entity, relation2id, id2relation = load_maps()

    triples = load_train_enriched_ids(entity2id, relation2id)
    triples = triples.to(device)

    edge_src, edge_rel, edge_dst = build_graph_edges(
        triples=triples,
        num_relations=len(relation2id),
        device=device,
    )

    candidate_names, candidate_ids = load_candidate_universe(entity2id)

    valid_rows = load_eval_rows("valid")
    test_rows = load_eval_rows("test")

    print("num_entities =", len(entity2id))
    print("num_relations =", len(relation2id))
    print("num_train_enriched_triples =", int(triples.size(0)))
    print("num_graph_edges_with_inverse =", int(edge_src.size(0)))
    print("num_candidate_drugs =", len(candidate_names))
    print("valid_rows =", len(valid_rows))
    print("test_rows =", len(test_rows))

    new_valid_metrics = []
    new_test_metrics = []

    for model_name in args.models:
        print("=" * 100)
        print("Running", model_name)
        print("=" * 100)

        valid_metrics, test_metrics = run_model(
            model_name=model_name,
            args=args,
            device=device,
            entity2id=entity2id,
            id2entity=id2entity,
            relation2id=relation2id,
            triples=triples,
            edge_src=edge_src,
            edge_rel=edge_rel,
            edge_dst=edge_dst,
            candidate_names=candidate_names,
            candidate_ids=candidate_ids,
            valid_rows=valid_rows,
            test_rows=test_rows,
        )

        new_valid_metrics.append(valid_metrics)
        new_test_metrics.append(test_metrics)

    merged_valid = update_metric_file(VALID_METRICS_PATH, new_valid_metrics)
    merged_test = update_metric_file(TEST_METRICS_PATH, new_test_metrics)

    main_table = write_main_table(merged_valid, merged_test)
    write_report(main_table)

    print("\nSaved:")
    print(f"  {VALID_METRICS_PATH}")
    print(f"  {TEST_METRICS_PATH}")
    print(f"  {MAIN_TABLE_PATH}")
    print(f"  {REPORT_PATH}")

    print("\nVALID")
    for r in main_table["valid"]:
        print(r["model_name"], "Gold@20=", r["gold_present_at20"], "MRR@20=", r["mrr_at20"])

    print("\nTEST")
    for r in main_table["test"]:
        print(r["model_name"], "Gold@20=", r["gold_present_at20"], "MRR@20=", r["mrr_at20"])


if __name__ == "__main__":
    main()