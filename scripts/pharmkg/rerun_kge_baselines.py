#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week 22 Day 5: Rerun Dataset 2 structure baselines as top-20 candidate generators.

This script trains and evaluates four KGE baselines:
- TransE
- DistMult
- ComplEx
- RotatE

Task:
    PharmKG Dataset 2
    (?, T, disease)
    candidate_universe = drug_only_from_train_T_heads
    top_k = 20
    gold_injection = false
    reviewer-safe RR@20

Important:
    R-GCN and HRGAT should be added using the same output schema.
    If exact Week21 R-GCN/HRGAT code exists, use it and export to the same folder format.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
TOP_K = 20
RANK_ABSENT_SENTINEL = 21


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


def load_id_maps() -> tuple[dict[str, int], dict[str, str], dict[str, int], dict[str, str]]:
    entity2id = read_json(GRAPH_DIR / "entity2id.json")
    id2entity = read_json(GRAPH_DIR / "id2entity.json")
    relation2id = read_json(GRAPH_DIR / "relation2id.json")
    id2relation = read_json(GRAPH_DIR / "id2relation.json")
    return entity2id, id2entity, relation2id, id2relation


def load_train_enriched_ids(entity2id: dict[str, int], relation2id: dict[str, int]) -> torch.LongTensor:
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

    h = df["head"].map(entity2id).astype(int).values
    r = df["relation"].map(relation2id).astype(int).values
    t = df["tail"].map(entity2id).astype(int).values

    arr = np.stack([h, r, t], axis=1)
    return torch.LongTensor(arr)


def load_eval_rows(split: str) -> list[dict[str, Any]]:
    return read_json(SPLIT_DIR / f"{split}.json")


def load_candidate_ids(entity2id: dict[str, int]) -> tuple[list[str], torch.LongTensor]:
    names = read_json(SPLIT_DIR / "candidate_universe.json")
    ids = [entity2id[x] for x in names]
    return names, torch.LongTensor(ids)


class KGEModel(nn.Module):
    def __init__(self, model_name: str, num_entities: int, num_relations: int, dim: int, margin: float = 6.0):
        super().__init__()
        self.model_name = model_name.lower()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.dim = dim
        self.margin = margin

        if self.model_name == "complex":
            self.ent_re = nn.Embedding(num_entities, dim)
            self.ent_im = nn.Embedding(num_entities, dim)
            self.rel_re = nn.Embedding(num_relations, dim)
            self.rel_im = nn.Embedding(num_relations, dim)
            self._init_emb(self.ent_re)
            self._init_emb(self.ent_im)
            self._init_emb(self.rel_re)
            self._init_emb(self.rel_im)

        elif self.model_name == "rotate":
            self.ent_re = nn.Embedding(num_entities, dim)
            self.ent_im = nn.Embedding(num_entities, dim)
            self.rel_phase = nn.Embedding(num_relations, dim)
            self._init_emb(self.ent_re)
            self._init_emb(self.ent_im)
            nn.init.uniform_(self.rel_phase.weight, a=0.0, b=2.0 * math.pi)

        else:
            self.ent = nn.Embedding(num_entities, dim)
            self.rel = nn.Embedding(num_relations, dim)
            self._init_emb(self.ent)
            self._init_emb(self.rel)

    @staticmethod
    def _init_emb(emb: nn.Embedding) -> None:
        nn.init.xavier_uniform_(emb.weight)

    def score(self, h: torch.LongTensor, r: torch.LongTensor, t: torch.LongTensor) -> torch.Tensor:
        name = self.model_name

        if name == "transe":
            h_e = self.ent(h)
            r_e = self.rel(r)
            t_e = self.ent(t)
            return -torch.linalg.norm(h_e + r_e - t_e, ord=1, dim=-1)

        if name == "distmult":
            h_e = self.ent(h)
            r_e = self.rel(r)
            t_e = self.ent(t)
            return torch.sum(h_e * r_e * t_e, dim=-1)

        if name == "complex":
            h_re = self.ent_re(h)
            h_im = self.ent_im(h)
            r_re = self.rel_re(r)
            r_im = self.rel_im(r)
            t_re = self.ent_re(t)
            t_im = self.ent_im(t)

            return torch.sum(
                h_re * r_re * t_re
                + h_im * r_re * t_im
                + h_re * r_im * t_im
                - h_im * r_im * t_re,
                dim=-1,
            )

        if name == "rotate":
            h_re = self.ent_re(h)
            h_im = self.ent_im(h)
            t_re = self.ent_re(t)
            t_im = self.ent_im(t)

            phase = self.rel_phase(r)
            r_re = torch.cos(phase)
            r_im = torch.sin(phase)

            rot_re = h_re * r_re - h_im * r_im
            rot_im = h_re * r_im + h_im * r_re

            return -torch.linalg.norm(
                torch.cat([rot_re - t_re, rot_im - t_im], dim=-1),
                ord=1,
                dim=-1,
            )

        raise ValueError(f"Unsupported model: {self.model_name}")


def make_negative_batch(
    pos_batch: torch.LongTensor,
    num_entities: int,
    candidate_ids: torch.LongTensor,
    target_relation_id: int,
    device: torch.device,
) -> torch.LongTensor:
    """
    Corrupt heads for target relation T using drug-only candidate universe.
    Corrupt heads/tails uniformly for other support relations.

    This keeps the target task aligned with drug-only head prediction while still using support graph triples.
    """
    neg = pos_batch.clone()
    batch_size = neg.size(0)

    corrupt_head_mask = torch.rand(batch_size, device=device) < 0.5
    target_mask = neg[:, 1] == target_relation_id
    head_mask = corrupt_head_mask | target_mask
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


def train_one_model(
    model_name: str,
    triples: torch.LongTensor,
    num_entities: int,
    num_relations: int,
    candidate_ids: torch.LongTensor,
    target_relation_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[KGEModel, dict[str, Any]]:
    model = KGEModel(
        model_name=model_name,
        num_entities=num_entities,
        num_relations=num_relations,
        dim=args.dim,
    ).to(device)

    triples = triples.to(device)
    dataset = TensorDataset(triples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MarginRankingLoss(margin=args.margin)

    train_log = []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for (pos_batch,) in loader:
            pos_batch = pos_batch.to(device)
            neg_batch = make_negative_batch(
                pos_batch=pos_batch,
                num_entities=num_entities,
                candidate_ids=candidate_ids,
                target_relation_id=target_relation_id,
                device=device,
            )

            pos_score = model.score(pos_batch[:, 0], pos_batch[:, 1], pos_batch[:, 2])
            neg_score = model.score(neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2])

            y = torch.ones_like(pos_score)
            loss = loss_fn(pos_score, neg_score, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        epoch_loss = float(np.mean(losses)) if losses else 0.0
        train_log.append({"epoch": epoch, "loss": epoch_loss})
        print(f"[{model_name}] epoch={epoch:03d} loss={epoch_loss:.6f}")

    summary = {
        "model_name": model_name,
        "epochs": args.epochs,
        "dim": args.dim,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "margin": args.margin,
        "runtime_sec": round(time.time() - start, 3),
        "train_log": train_log,
    }

    return model, summary


@torch.no_grad()
def score_candidates_for_row(
    model: KGEModel,
    row: dict[str, Any],
    candidate_ids: torch.LongTensor,
    candidate_names: list[str],
    relation2id: dict[str, int],
    device: torch.device,
) -> tuple[list[str], list[int], list[float]]:
    model.eval()

    r_id = relation2id[TARGET_RELATION]
    t_id = int(row["query_entity_id"])

    h = candidate_ids.to(device)
    r = torch.full_like(h, fill_value=int(r_id), device=device)
    t = torch.full_like(h, fill_value=int(t_id), device=device)

    scores = model.score(h, r, t)
    top_scores, top_idx = torch.topk(scores, k=min(TOP_K, len(candidate_names)), largest=True)

    top_idx_cpu = top_idx.cpu().numpy().tolist()
    top_scores_cpu = top_scores.cpu().numpy().tolist()

    top_names = [candidate_names[i] for i in top_idx_cpu]
    top_ids = [int(candidate_ids[i].item()) for i in top_idx_cpu]
    top_scores_float = [float(x) for x in top_scores_cpu]

    return top_names, top_ids, top_scores_float


def gold_rank_in_topk(gold: str, top_names: list[str]) -> tuple[int, bool]:
    if gold in top_names:
        return top_names.index(gold) + 1, True
    return RANK_ABSENT_SENTINEL, False


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.array([r["gold_rank_in_top20_or_21"] for r in rows], dtype=np.int64)
    present = np.array([r["gold_present_top20"] for r in rows], dtype=bool)

    rr = np.zeros(len(rows), dtype=np.float64)
    for i, rank in enumerate(ranks):
        if present[i] and rank <= TOP_K:
            rr[i] = 1.0 / rank
        else:
            rr[i] = 0.0

    top1 = [r["candidate_entities_top20"][0] for r in rows if r["candidate_entities_top20"]]
    top1_counter = Counter(top1)
    most_common_top1 = top1_counter.most_common(10)

    metrics = {
        "num_rows": int(len(rows)),
        "gold_present_at20": float(np.mean(present)) if len(rows) else 0.0,
        "mrr_at20": float(np.mean(rr)) if len(rows) else 0.0,
        "hits1_at20": float(np.mean(ranks <= 1)) if len(rows) else 0.0,
        "hits3_at20": float(np.mean(ranks <= 3)) if len(rows) else 0.0,
        "hits10_at20": float(np.mean(ranks <= 10)) if len(rows) else 0.0,
        "hits20_at20": float(np.mean(ranks <= 20)) if len(rows) else 0.0,
        "avg_gold_rank_absent_as_21": float(np.mean(ranks)) if len(rows) else 0.0,
        "gold_rank_21_count": int(np.sum(ranks == RANK_ABSENT_SENTINEL)),
        "unique_top1_count": int(len(top1_counter)),
        "top1_dominance": float(most_common_top1[0][1] / len(rows)) if rows and most_common_top1 else 0.0,
        "top1_most_common": [{"entity": k, "count": int(v)} for k, v in most_common_top1],
        "rr_policy": "RR = 1/rank if gold present in top20 else 0",
        "absent_rank_sentinel": RANK_ABSENT_SENTINEL,
    }

    return metrics


def export_top20(
    model_name: str,
    split: str,
    model: KGEModel,
    eval_rows: list[dict[str, Any]],
    candidate_ids: torch.LongTensor,
    candidate_names: list[str],
    relation2id: dict[str, int],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows = []

    for row in eval_rows:
        top_names, top_ids, top_scores = score_candidates_for_row(
            model=model,
            row=row,
            candidate_ids=candidate_ids,
            candidate_names=candidate_names,
            relation2id=relation2id,
            device=device,
        )

        rank, present = gold_rank_in_topk(row["gold_entity"], top_names)

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
                "gold_injected": False,
                "target_relation": TARGET_RELATION,
                "target_relation_normalized": "therapeutic_association_proxy",
                "reviewer_safe_rr_item": float(1.0 / rank) if present and rank <= TOP_K else 0.0,
            }
        )

    metrics = compute_metrics(out_rows)
    metrics["model_name"] = model_name
    metrics["split"] = split

    return out_rows, metrics


def write_report(valid_metrics: list[dict[str, Any]], test_metrics: list[dict[str, Any]]) -> None:
    def table_rows(metrics_list: list[dict[str, Any]]) -> str:
        lines = []
        for m in sorted(metrics_list, key=lambda x: x["mrr_at20"], reverse=True):
            lines.append(
                f"| {m['model_name']} | {m['gold_present_at20']:.3f} | "
                f"{m['mrr_at20']:.6f} | {m['hits1_at20']:.3f} | "
                f"{m['hits3_at20']:.3f} | {m['hits10_at20']:.3f} | "
                f"{m['hits20_at20']:.3f} | {m['avg_gold_rank_absent_as_21']:.3f} | "
                f"{m['gold_rank_21_count']} | {m['unique_top1_count']} | "
                f"{m['top1_dominance']:.3f} |"
            )
        return "\n".join(lines)

    md = f"""# Week 22 Day 5 — PharmKG Dataset 2 Structure Baselines

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
{table_rows(valid_metrics)}

## Test metrics

| Model | Gold@20 | MRR@20 | H@1 | H@3 | H@10 | H@20 | Avg rank | Rank21 | Unique top1 | Top1 dominance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows(test_metrics)}

## Notes

This script currently trains TransE, DistMult, ComplEx, and RotatE.
R-GCN and HRGAT should be exported with the same JSON schema if using the exact Week21 implementations.

## Files written

- `{VALID_METRICS_PATH}`
- `{TEST_METRICS_PATH}`
- `{MAIN_TABLE_PATH}`
- `{REPORT_PATH}`
"""
    REPORT_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["transe", "distmult", "complex", "rotate"],
        choices=["transe", "distmult", "complex", "rotate"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ensure_dirs()
    set_seed(args.seed)

    device = torch.device(args.device)
    print("Device:", device)

    entity2id, id2entity, relation2id, id2relation = load_id_maps()
    triples = load_train_enriched_ids(entity2id, relation2id)

    candidate_names, candidate_ids = load_candidate_ids(entity2id)

    num_entities = len(entity2id)
    num_relations = len(relation2id)
    target_relation_id = int(relation2id[TARGET_RELATION])

    valid_rows = load_eval_rows("valid")
    test_rows = load_eval_rows("test")

    print("num_entities =", num_entities)
    print("num_relations =", num_relations)
    print("num_train_enriched_triples =", len(triples))
    print("num_candidate_drugs =", len(candidate_names))
    print("valid rows =", len(valid_rows))
    print("test rows =", len(test_rows))

    all_valid_metrics = []
    all_test_metrics = []

    for model_name in args.models:
        print("=" * 100)
        print("Training model:", model_name)

        out_dir = BASELINE_DIR / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "model_name": model_name,
            "task": "(?, T, disease)",
            "candidate_universe": "drug_only_from_train_T_heads",
            "top_k": TOP_K,
            "gold_injection": False,
            "epochs": args.epochs,
            "dim": args.dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "margin": args.margin,
            "seed": args.seed,
        }
        write_json(config, out_dir / "config.json")

        model, train_summary = train_one_model(
            model_name=model_name,
            triples=triples,
            num_entities=num_entities,
            num_relations=num_relations,
            candidate_ids=candidate_ids,
            target_relation_id=target_relation_id,
            args=args,
            device=device,
        )

        valid_top20, valid_metrics = export_top20(
            model_name=model_name,
            split="valid",
            model=model,
            eval_rows=valid_rows,
            candidate_ids=candidate_ids,
            candidate_names=candidate_names,
            relation2id=relation2id,
            device=device,
        )

        test_top20, test_metrics = export_top20(
            model_name=model_name,
            split="test",
            model=model,
            eval_rows=test_rows,
            candidate_ids=candidate_ids,
            candidate_names=candidate_names,
            relation2id=relation2id,
            device=device,
        )

        write_json(valid_top20, out_dir / "valid_top20.json")
        write_json(test_top20, out_dir / "test_top20.json")

        model_summary = {
            "config": config,
            "train_summary": train_summary,
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        }
        write_json(model_summary, out_dir / "summary.json")

        all_valid_metrics.append(valid_metrics)
        all_test_metrics.append(test_metrics)

        print("Valid metrics:", json.dumps(valid_metrics, indent=2))
        print("Test metrics:", json.dumps(test_metrics, indent=2))

    main_table = {
        "dataset": "PharmKG-8k task-specific therapeutic_association_proxy benchmark",
        "protocol": {
            "candidate_universe": "drug_only_from_train_T_heads",
            "top_k": TOP_K,
            "gold_injection": False,
            "rr_policy": "RR = 1/rank if gold present in top20 else 0",
            "absent_rank_sentinel": RANK_ABSENT_SENTINEL,
        },
        "valid": sorted(all_valid_metrics, key=lambda x: x["mrr_at20"], reverse=True),
        "test": sorted(all_test_metrics, key=lambda x: x["mrr_at20"], reverse=True),
    }

    write_json(all_valid_metrics, VALID_METRICS_PATH)
    write_json(all_test_metrics, TEST_METRICS_PATH)
    write_json(main_table, MAIN_TABLE_PATH)
    write_report(all_valid_metrics, all_test_metrics)

    print("\nSaved:")
    print(f"  {VALID_METRICS_PATH}")
    print(f"  {TEST_METRICS_PATH}")
    print(f"  {MAIN_TABLE_PATH}")
    print(f"  {REPORT_PATH}")

    print("\nBest valid model:", main_table["valid"][0]["model_name"], main_table["valid"][0]["mrr_at20"])
    print("Best test model:", main_table["test"][0]["model_name"], main_table["test"][0]["mrr_at20"])


if __name__ == "__main__":
    main()