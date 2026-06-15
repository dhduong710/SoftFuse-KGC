#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rerun all structure baselines as top-20 candidate generators.

Baselines:
- TransE
- DistMult
- ComplEx
- RotatE
- R-GCN-lite
- HRGAT-lite

Protocol:
- task: (?, indication, disease)
- missing entity: drug
- candidate universe: drug_only
- top_k = 20
- gold injection forbidden
- outputs are top-20 candidate rows for reviewer-safe metric recomputation
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(".")
OUT_ROOT = ROOT / "outputs" / "baselines" / "baseline_outputs"
REPORT_DIR = ROOT / "outputs" / "baselines" / "reports"
REPORT_PATH = REPORT_DIR / "baseline_rerun.md"

VALID_QUERY_PATH = ROOT / "dataset" / "setting_a" / "backbone_candidates" / "valid_top20_raw.json"
TEST_QUERY_PATH = ROOT / "dataset" / "setting_a" / "backbone_candidates" / "test_top20_raw.json"

TRAIN_ROW_CANDIDATES = [
    # Preferred train package. If it is missing, run:
    #   python3 scripts/e2e/build_infer_ready.py
    ROOT / "dataset" / "setting_a" / "e2e_infer_ready" / "backbone_raw" / "train.json",
    # Fallbacks produced by earlier data-processing stages.
    ROOT / "dataset" / "setting_a" / "aligned_evidence" / "train_aligned_evidence.json",
    ROOT / "dataset" / "setting_a" / "backbone_ready" / "train.json",
]

KNOWN_RELATIONS = {
    "indication",
    "target",
    "associated_with",
    "associated-with",
    "associated with",
    "ppi",
    "contraindication",
}

MODEL_NAMES = ["transe", "distmult", "complex", "rotate", "rgcn", "hrgat"]


@dataclass
class DataPack:
    train_rows_path: str
    enriched_kg_path: str | None
    train_triples: list[tuple[int, int, int]]
    train_indication_pairs: list[tuple[int, int]]
    valid_rows: list[dict[str, Any]]
    test_rows: list[dict[str, Any]]
    drug_ids: list[int]
    entity_id_to_name: dict[int, str]
    relation_to_id: dict[str, int]
    num_entities: int
    indication_rel_id: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["rows", "data", "examples", "predictions", "results"]:
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError(f"Cannot parse rows from {path}")


def safe_int(x: Any) -> int | None:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def normalize_rel(r: Any) -> str:
    if r is None:
        return "unknown"
    s = str(r).strip()
    s = s.replace(" ", "_").replace("-", "_")
    return s


def get_name(row: dict[str, Any], keys: list[str]) -> str | None:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def get_id(row: dict[str, Any], keys: list[str]) -> int | None:
    for k in keys:
        v = safe_int(row.get(k))
        if v is not None:
            return v
    return None


def extract_query_gold(row: dict[str, Any]) -> tuple[int | None, str | None, int | None, str | None]:
    qid = get_id(row, ["query_entity_id", "disease_id", "tail_id"])
    qname = get_name(row, ["query_entity", "disease", "tail"])
    gid = get_id(row, ["gold_entity_id", "answer_id", "target_id", "gold_id", "head_id"])
    gname = get_name(row, ["gold_entity", "answer", "target", "gold", "head"])

    triple_id = row.get("triple_id")
    triple = row.get("triple")

    if gid is None and isinstance(triple_id, list) and len(triple_id) >= 3:
        gid = safe_int(triple_id[0])
    if qid is None and isinstance(triple_id, list) and len(triple_id) >= 3:
        qid = safe_int(triple_id[2])

    if gname is None and isinstance(triple, list) and len(triple) >= 3:
        gname = str(triple[0])
    if qname is None and isinstance(triple, list) and len(triple) >= 3:
        qname = str(triple[2])

    return qid, qname, gid, gname


def collect_names_from_rows(rows: list[dict[str, Any]], id2name: dict[int, str]) -> None:
    for row in rows:
        qid, qname, gid, gname = extract_query_gold(row)
        if qid is not None and qname:
            id2name.setdefault(qid, qname)
        if gid is not None and gname:
            id2name.setdefault(gid, gname)

        cand_names = (
            row.get("candidate_entities")
            or row.get("candidate_entities_top20")
            or row.get("rank_entities")
            or row.get("rank_entities_name")
            or []
        )
        cand_ids = (
            row.get("candidate_entity_ids")
            or row.get("candidate_entity_ids_top20")
            or row.get("rank_entities_id")
            or row.get("rank_entities_ids")
            or []
        )
        if isinstance(cand_names, list) and isinstance(cand_ids, list):
            for cid, cname in zip(cand_ids, cand_names):
                ci = safe_int(cid)
                if ci is not None and isinstance(cname, str):
                    id2name.setdefault(ci, cname)


def find_train_rows() -> tuple[Path, list[dict[str, Any]]]:
    for path in TRAIN_ROW_CANDIDATES:
        if path.exists():
            rows = load_json_rows(path)
            pairs = 0
            for row in rows[:200]:
                qid, _, gid, _ = extract_query_gold(row)
                if qid is not None and gid is not None:
                    pairs += 1
            if pairs > 0:
                return path, rows
    raise FileNotFoundError("Could not find train rows with query/gold ids.")


def find_enriched_kg() -> Path | None:
    patterns = [
        "*train_enriched*deg1000*final*.tsv",
        "*train_enriched*.tsv",
        "*enriched*graph*.tsv",
        "*enriched*.tsv",
    ]
    roots = [ROOT / "dataset", ROOT / "outputs"]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            for pat in patterns:
                candidates.extend(root.rglob(pat))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    def score(p: Path) -> tuple[int, int]:
        s = str(p).lower()
        bonus = 0
        if "deg1000" in s:
            bonus += 10
        if "final" in s:
            bonus += 10
        if "train" in s:
            bonus += 5
        return bonus, p.stat().st_size

    candidates = sorted(set(candidates), key=score, reverse=True)
    return candidates[0]


def parse_kg_line(parts: list[str]) -> tuple[str, str, str] | None:
    if len(parts) < 3:
        return None

    cleaned = [p.strip() for p in parts[:5]]
    low = [p.lower().replace("-", "_").replace(" ", "_") for p in cleaned]

    # Common h r t
    if len(cleaned) >= 3 and low[1] in KNOWN_RELATIONS:
        return cleaned[0], normalize_rel(cleaned[1]), cleaned[2]

    # Common h t r
    if len(cleaned) >= 3 and low[2] in KNOWN_RELATIONS:
        return cleaned[0], normalize_rel(cleaned[2]), cleaned[1]

    # Try any column as relation
    for i, val in enumerate(low[:3]):
        if val in KNOWN_RELATIONS:
            others = [cleaned[j] for j in range(3) if j != i]
            if len(others) == 2:
                return others[0], normalize_rel(cleaned[i]), others[1]

    return None


def entity_token_to_id(
    token: str,
    name_to_id: dict[str, int],
    dynamic_name_to_id: dict[str, int],
    next_id_ref: list[int],
) -> int:
    token = str(token).strip()
    if token in name_to_id:
        return name_to_id[token]
    if token in dynamic_name_to_id:
        return dynamic_name_to_id[token]
    # Numeric id support
    maybe_id = safe_int(token)
    if maybe_id is not None:
        return maybe_id
    new_id = next_id_ref[0]
    dynamic_name_to_id[token] = new_id
    next_id_ref[0] += 1
    return new_id


def maybe_parse_type_map_for_drugs(id2name: dict[int, str]) -> set[int]:
    drug_ids: set[int] = set()
    paths = []
    for root in [ROOT / "dataset", ROOT / "outputs"]:
        if root.exists():
            paths.extend(root.rglob("*type*map*.json"))
            paths.extend(root.rglob("*entity*type*.json"))

    for path in paths:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        local: set[int] = set()

        def is_drug_type(v: Any) -> bool:
            return str(v).lower() in {"drug", "drugs", "compound", "medication"}

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and is_drug_type(v):
                    kid = safe_int(k)
                    if kid is not None:
                        local.add(kid)
                elif isinstance(v, dict):
                    typ = v.get("type") or v.get("entity_type") or v.get("label")
                    if is_drug_type(typ):
                        kid = safe_int(k) or safe_int(v.get("id")) or safe_int(v.get("entity_id"))
                        if kid is not None:
                            local.add(kid)
                            nm = v.get("name") or v.get("entity")
                            if isinstance(nm, str):
                                id2name.setdefault(kid, nm)
        elif isinstance(obj, list):
            for item in obj:
                if not isinstance(item, dict):
                    continue
                typ = item.get("type") or item.get("entity_type") or item.get("label")
                if is_drug_type(typ):
                    kid = safe_int(item.get("id")) or safe_int(item.get("entity_id"))
                    if kid is not None:
                        local.add(kid)
                        nm = item.get("name") or item.get("entity")
                        if isinstance(nm, str):
                            id2name.setdefault(kid, nm)

        if len(local) >= 1000:
            drug_ids |= local

    return drug_ids


def build_data_pack(args: argparse.Namespace) -> DataPack:
    valid_rows = load_json_rows(VALID_QUERY_PATH)
    test_rows = load_json_rows(TEST_QUERY_PATH)
    train_path, train_rows = find_train_rows()

    id2name: dict[int, str] = {}
    collect_names_from_rows(valid_rows, id2name)
    collect_names_from_rows(test_rows, id2name)
    collect_names_from_rows(train_rows, id2name)

    name_to_id = {v: k for k, v in id2name.items()}

    relation_to_id = {"indication": 0}
    indication_rel_id = 0

    train_indication_pairs: list[tuple[int, int]] = []
    train_triples: list[tuple[int, int, int]] = []

    drug_ids: set[int] = set()

    for row in train_rows:
        qid, qname, gid, gname = extract_query_gold(row)
        if qid is None or gid is None:
            continue
        if qname:
            id2name.setdefault(qid, qname)
        if gname:
            id2name.setdefault(gid, gname)
        train_indication_pairs.append((gid, qid))
        train_triples.append((gid, indication_rel_id, qid))
        drug_ids.add(gid)

    for rows in [valid_rows, test_rows]:
        for row in rows:
            qid, qname, gid, gname = extract_query_gold(row)
            if qid is not None and qname:
                id2name.setdefault(qid, qname)
            if gid is not None:
                drug_ids.add(gid)
                if gname:
                    id2name.setdefault(gid, gname)

            cand_ids = (
                row.get("candidate_entity_ids")
                or row.get("candidate_entity_ids_top20")
                or row.get("rank_entities_id")
                or row.get("rank_entities_ids")
                or []
            )
            cand_names = (
                row.get("candidate_entities")
                or row.get("candidate_entities_top20")
                or row.get("rank_entities")
                or []
            )
            if isinstance(cand_ids, list):
                for i, cid in enumerate(cand_ids):
                    ci = safe_int(cid)
                    if ci is not None:
                        drug_ids.add(ci)
                        if isinstance(cand_names, list) and i < len(cand_names) and isinstance(cand_names[i], str):
                            id2name.setdefault(ci, cand_names[i])

    type_map_drugs = maybe_parse_type_map_for_drugs(id2name)
    if type_map_drugs:
        drug_ids |= type_map_drugs

    max_known_id = max(list(id2name.keys()) + list(drug_ids) + [0])
    next_id_ref = [max_known_id + 1]
    dynamic_name_to_id: dict[str, int] = {}

    kg_path = find_enriched_kg()
    if args.no_support_kg:
        kg_path = None

    if kg_path is not None:
        with kg_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 3:
                    parts = line.split(",")
                parsed = parse_kg_line(parts)
                if parsed is None:
                    # Skip probable header
                    continue
                h_tok, r_name, t_tok = parsed
                h_id = entity_token_to_id(h_tok, name_to_id, dynamic_name_to_id, next_id_ref)
                t_id = entity_token_to_id(t_tok, name_to_id, dynamic_name_to_id, next_id_ref)
                if h_id not in id2name and not str(h_tok).isdigit():
                    id2name[h_id] = h_tok
                if t_id not in id2name and not str(t_tok).isdigit():
                    id2name[t_id] = t_tok

                r_name = normalize_rel(r_name)
                if r_name not in relation_to_id:
                    relation_to_id[r_name] = len(relation_to_id)
                train_triples.append((h_id, relation_to_id[r_name], t_id))

    # Deduplicate triples
    train_triples = sorted(set(train_triples))
    train_indication_pairs = sorted(set(train_indication_pairs))
    drug_ids = sorted(set(drug_ids))

    all_entity_ids = set()
    for h, _, t in train_triples:
        all_entity_ids.add(h)
        all_entity_ids.add(t)
    all_entity_ids |= set(drug_ids)
    for rows in [valid_rows, test_rows]:
        for row in rows:
            qid, _, gid, _ = extract_query_gold(row)
            if qid is not None:
                all_entity_ids.add(qid)
            if gid is not None:
                all_entity_ids.add(gid)

    num_entities = max(all_entity_ids) + 1

    if len(drug_ids) < 1000:
        print(f"WARNING: drug universe size seems small: {len(drug_ids)}")

    return DataPack(
        train_rows_path=str(train_path),
        enriched_kg_path=str(kg_path) if kg_path else None,
        train_triples=train_triples,
        train_indication_pairs=train_indication_pairs,
        valid_rows=valid_rows,
        test_rows=test_rows,
        drug_ids=drug_ids,
        entity_id_to_name=id2name,
        relation_to_id=relation_to_id,
        num_entities=num_entities,
        indication_rel_id=indication_rel_id,
    )


class TransEModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.ent = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -torch.linalg.vector_norm(self.ent(h) + self.rel(r) - self.ent(t), ord=2, dim=-1)

    def score_heads(self, heads: torch.Tensor, rel_id: int, tail_id: int) -> torch.Tensor:
        r = torch.full_like(heads, rel_id)
        t = torch.full_like(heads, tail_id)
        return self.score(heads, r, t)


class DistMultModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.ent = nn.Embedding(num_entities, dim)
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.ent(h) * self.rel(r) * self.ent(t), dim=-1)

    def score_heads(self, heads: torch.Tensor, rel_id: int, tail_id: int) -> torch.Tensor:
        r = torch.full_like(heads, rel_id)
        t = torch.full_like(heads, tail_id)
        return self.score(heads, r, t)


class ComplExModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.er = nn.Embedding(num_entities, dim)
        self.ei = nn.Embedding(num_entities, dim)
        self.rr = nn.Embedding(num_relations, dim)
        self.ri = nn.Embedding(num_relations, dim)
        for emb in [self.er, self.ei, self.rr, self.ri]:
            nn.init.xavier_uniform_(emb.weight)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        hr, hi = self.er(h), self.ei(h)
        rr, ri = self.rr(r), self.ri(r)
        tr, ti = self.er(t), self.ei(t)
        return torch.sum(hr * rr * tr + hr * ri * ti + hi * rr * ti - hi * ri * tr, dim=-1)

    def score_heads(self, heads: torch.Tensor, rel_id: int, tail_id: int) -> torch.Tensor:
        r = torch.full_like(heads, rel_id)
        t = torch.full_like(heads, tail_id)
        return self.score(heads, r, t)


class RotatEModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int):
        super().__init__()
        self.er = nn.Embedding(num_entities, dim)
        self.ei = nn.Embedding(num_entities, dim)
        self.phase = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.er.weight)
        nn.init.xavier_uniform_(self.ei.weight)
        nn.init.uniform_(self.phase.weight, -math.pi, math.pi)

    def score(self, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        hr, hi = self.er(h), self.ei(h)
        tr, ti = self.er(t), self.ei(t)
        phase = self.phase(r)
        rr, ri = torch.cos(phase), torch.sin(phase)
        rot_r = hr * rr - hi * ri
        rot_i = hr * ri + hi * rr
        return -torch.linalg.vector_norm(torch.cat([rot_r - tr, rot_i - ti], dim=-1), ord=2, dim=-1)

    def score_heads(self, heads: torch.Tensor, rel_id: int, tail_id: int) -> torch.Tensor:
        r = torch.full_like(heads, rel_id)
        t = torch.full_like(heads, tail_id)
        return self.score(heads, r, t)


class RGCNLiteModel(nn.Module):
    def __init__(self, num_entities: int, num_score_relations: int, num_graph_relations: int, dim: int, layers: int = 2):
        super().__init__()
        self.base = nn.Embedding(num_entities, dim)
        self.rel_gate = nn.ModuleList([nn.Embedding(num_graph_relations, dim) for _ in range(layers)])
        self.self_loop = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.score_rel = nn.Embedding(num_score_relations, dim)
        self.layers = layers
        nn.init.xavier_uniform_(self.base.weight)
        nn.init.xavier_uniform_(self.score_rel.weight)
        for emb in self.rel_gate:
            nn.init.xavier_uniform_(emb.weight)

    def all_embeddings(self, edge_src: torch.Tensor, edge_rel: torch.Tensor, edge_dst: torch.Tensor) -> torch.Tensor:
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
        return h

    def score_with_z(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sum(z[h] * self.score_rel(r) * z[t], dim=-1)

    def score_heads_with_z(self, z: torch.Tensor, heads: torch.Tensor, rel_id: int, tail_id: int) -> torch.Tensor:
        r = torch.full_like(heads, rel_id)
        t = torch.full_like(heads, tail_id)
        return self.score_with_z(z, heads, r, t)


class HRGATLiteModel(nn.Module):
    def __init__(self, num_entities: int, num_score_relations: int, num_graph_relations: int, dim: int, layers: int = 2):
        super().__init__()
        self.base = nn.Embedding(num_entities, dim)
        self.rel_gate = nn.ModuleList([nn.Embedding(num_graph_relations, dim) for _ in range(layers)])
        self.self_loop = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        self.att_src = nn.ParameterList([nn.Parameter(torch.empty(dim)) for _ in range(layers)])
        self.att_dst = nn.ParameterList([nn.Parameter(torch.empty(dim)) for _ in range(layers)])
        self.att_rel = nn.ModuleList([nn.Embedding(num_graph_relations, dim) for _ in range(layers)])
        self.score_rel = nn.Embedding(num_score_relations, dim)
        self.layers = layers

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

    def all_embeddings(self, edge_src: torch.Tensor, edge_rel: torch.Tensor, edge_dst: torch.Tensor) -> torch.Tensor:
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
        return h

    def score_with_z(self, z: torch.Tensor, h: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.sum(z[h] * self.score_rel(r) * z[t], dim=-1)

    def score_heads_with_z(self, z: torch.Tensor, heads: torch.Tensor, rel_id: int, tail_id: int) -> torch.Tensor:
        r = torch.full_like(heads, rel_id)
        t = torch.full_like(heads, tail_id)
        return self.score_with_z(z, heads, r, t)


def build_model(name: str, data: DataPack, args: argparse.Namespace, device: torch.device):
    num_score_rel = len(data.relation_to_id)
    if name == "transe":
        return TransEModel(data.num_entities, num_score_rel, args.dim).to(device)
    if name == "distmult":
        return DistMultModel(data.num_entities, num_score_rel, args.dim).to(device)
    if name == "complex":
        return ComplExModel(data.num_entities, num_score_rel, args.dim).to(device)
    if name == "rotate":
        return RotatEModel(data.num_entities, num_score_rel, args.dim).to(device)
    if name == "rgcn":
        return RGCNLiteModel(data.num_entities, num_score_rel, num_score_rel * 2, args.dim, args.gnn_layers).to(device)
    if name == "hrgat":
        return HRGATLiteModel(data.num_entities, num_score_rel, num_score_rel * 2, args.dim, args.gnn_layers).to(device)
    raise ValueError(name)


def tensorize_triples(triples: list[tuple[int, int, int]], device: torch.device) -> torch.Tensor:
    return torch.tensor(triples, dtype=torch.long, device=device)


def build_graph_edges(triples_tensor: torch.Tensor, num_score_rel: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h = triples_tensor[:, 0]
    r = triples_tensor[:, 1]
    t = triples_tensor[:, 2]
    src = torch.cat([h, t], dim=0)
    dst = torch.cat([t, h], dim=0)
    rel = torch.cat([r, r + num_score_rel], dim=0)
    return src, rel, dst


def train_kge(model: nn.Module, triples: torch.Tensor, args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = triples.size(0)
    log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        steps = 0

        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            pos = triples[idx]

            neg = pos.clone()
            corrupt_head = torch.rand(neg.size(0), device=device) < 0.5
            random_entities = torch.randint(0, args.num_entities_runtime, (neg.size(0),), device=device)
            neg[corrupt_head, 0] = random_entities[corrupt_head]
            neg[~corrupt_head, 2] = random_entities[~corrupt_head]

            pos_score = model.score(pos[:, 0], pos[:, 1], pos[:, 2])
            neg_score = model.score(neg[:, 0], neg[:, 1], neg[:, 2])

            loss = F.softplus(-pos_score).mean() + F.softplus(neg_score).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total_loss += float(loss.item())
            steps += 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            item = {"epoch": epoch, "loss": total_loss / max(steps, 1)}
            log.append(item)
            print(f"[KGE] epoch={epoch:03d} loss={item['loss']:.6f}")

    return log


def train_gnn(
    model: nn.Module,
    triples: torch.Tensor,
    graph_edges: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = triples.size(0)
    edge_src, edge_rel, edge_dst = graph_edges
    log = []

    for epoch in range(1, args.gnn_epochs + 1):
        model.train()
        sample_n = min(args.gnn_sample_per_epoch, n)
        idx = torch.randint(0, n, (sample_n,), device=device)
        pos = triples[idx]

        neg = pos.clone()
        corrupt_head = torch.rand(neg.size(0), device=device) < 0.5
        random_entities = torch.randint(0, args.num_entities_runtime, (neg.size(0),), device=device)
        neg[corrupt_head, 0] = random_entities[corrupt_head]
        neg[~corrupt_head, 2] = random_entities[~corrupt_head]

        z = model.all_embeddings(edge_src, edge_rel, edge_dst)

        pos_score = model.score_with_z(z, pos[:, 0], pos[:, 1], pos[:, 2])
        neg_score = model.score_with_z(z, neg[:, 0], neg[:, 1], neg[:, 2])
        loss = F.softplus(-pos_score).mean() + F.softplus(neg_score).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.gnn_epochs:
            item = {"epoch": epoch, "loss": float(loss.item())}
            log.append(item)
            print(f"[GNN] epoch={epoch:03d} loss={item['loss']:.6f}")

    return log


@torch.no_grad()
def score_all_drugs(
    model_name: str,
    model: nn.Module,
    data: DataPack,
    query_id: int,
    drug_tensor: torch.Tensor,
    device: torch.device,
    graph_z: torch.Tensor | None = None,
) -> torch.Tensor:
    if model_name in {"rgcn", "hrgat"}:
        assert graph_z is not None
        return model.score_heads_with_z(graph_z, drug_tensor, data.indication_rel_id, query_id)
    return model.score_heads(drug_tensor, data.indication_rel_id, query_id)


def export_top20(
    model_name: str,
    model: nn.Module,
    data: DataPack,
    split: str,
    rows: list[dict[str, Any]],
    out_path: Path,
    args: argparse.Namespace,
    device: torch.device,
    graph_z: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    drug_ids = data.drug_ids
    drug_tensor = torch.tensor(drug_ids, dtype=torch.long, device=device)

    output_rows = []
    gold_present_count = 0
    ranks = []

    for idx, row in enumerate(rows):
        qid, qname, gid, gname = extract_query_gold(row)
        if qid is None or gid is None:
            raise ValueError(f"Missing qid/gid in {split} row {idx}")

        scores = score_all_drugs(model_name, model, data, qid, drug_tensor, device, graph_z=graph_z)
        top_scores, top_pos = torch.topk(scores, k=min(args.top_k, len(drug_ids)), largest=True)
        top_ids = [int(drug_ids[int(p)]) for p in top_pos.detach().cpu().tolist()]
        top_scores_list = [float(x) for x in top_scores.detach().cpu().tolist()]
        top_names = [data.entity_id_to_name.get(i, str(i)) for i in top_ids]

        if gid in top_ids:
            rank = top_ids.index(gid) + 1
            present = True
            gold_present_count += 1
        else:
            rank = 21
            present = False

        ranks.append(rank)

        output_rows.append({
            "split": split,
            "model_name": model_name,
            "query_entity": qname or data.entity_id_to_name.get(qid, str(qid)),
            "query_entity_id": qid,
            "gold_entity": gname or data.entity_id_to_name.get(gid, str(gid)),
            "gold_entity_id": gid,
            "candidate_entities_top20": top_names,
            "candidate_entity_ids_top20": top_ids,
            "scores_top20": top_scores_list,
            "gold_rank_in_top20_or_21": rank,
            "gold_present_top20": present,
            "candidate_size": len(top_ids),
            "candidate_universe": "drug_only",
            "candidate_universe_size": len(drug_ids),
            "gold_injected": False,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "split": split,
        "model_name": model_name,
        "num_rows": len(output_rows),
        "candidate_size_min": min(len(r["candidate_entity_ids_top20"]) for r in output_rows) if output_rows else None,
        "candidate_size_max": max(len(r["candidate_entity_ids_top20"]) for r in output_rows) if output_rows else None,
        "gold_present_at20_raw": gold_present_count / max(len(output_rows), 1),
        "gold_rank_21_count": sum(1 for r in ranks if r == 21),
        "avg_gold_rank_absent_as_21": sum(ranks) / max(len(ranks), 1),
        "output_path": str(out_path),
    }
    return summary


def run_one_model(model_name: str, data: DataPack, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    print("=" * 100)
    print(f"RUNNING MODEL: {model_name}")
    print("=" * 100)

    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    triples = tensorize_triples(data.train_triples, device)
    args.num_entities_runtime = data.num_entities

    model = build_model(model_name, data, args, device)

    start = time.time()

    graph_z = None
    if model_name in {"rgcn", "hrgat"}:
        graph_edges = build_graph_edges(triples, len(data.relation_to_id))
        train_log = train_gnn(model, triples, graph_edges, args, device)
        model.eval()
        with torch.no_grad():
            graph_z = model.all_embeddings(*graph_edges)
    else:
        train_log = train_kge(model, triples, args, device)

    valid_summary = export_top20(
        model_name=model_name,
        model=model,
        data=data,
        split="valid",
        rows=data.valid_rows,
        out_path=out_dir / "valid_top20.json",
        args=args,
        device=device,
        graph_z=graph_z,
    )
    test_summary = export_top20(
        model_name=model_name,
        model=model,
        data=data,
        split="test",
        rows=data.test_rows,
        out_path=out_dir / "test_top20.json",
        args=args,
        device=device,
        graph_z=graph_z,
    )

    elapsed = time.time() - start

    config = {
        "model_name": model_name,
        "seed": args.seed,
        "dim": args.dim,
        "epochs": args.epochs if model_name not in {"rgcn", "hrgat"} else args.gnn_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "top_k": args.top_k,
        "gold_injection": False,
        "candidate_universe": "drug_only",
        "candidate_universe_size": len(data.drug_ids),
        "num_entities": data.num_entities,
        "num_relations": len(data.relation_to_id),
        "num_train_triples": len(data.train_triples),
        "num_train_indication_pairs": len(data.train_indication_pairs),
        "train_rows_path": data.train_rows_path,
        "enriched_kg_path": data.enriched_kg_path,
        "elapsed_seconds": elapsed,
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "train_log.json").write_text(json.dumps(train_log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    torch.save(model.state_dict(), out_dir / "model_state.pt")

    summary = {
        "model_name": model_name,
        "config": config,
        "valid_summary": valid_summary,
        "test_summary": test_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Finished {model_name} in {elapsed/60:.2f} minutes")
    print("valid:", valid_summary)
    print("test :", test_summary)
    return summary


def write_baseline_rerun_report(all_summaries: list[dict[str, Any]], data: DataPack, args: argparse.Namespace) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Rerun All Structure Baselines\n")
    lines.append("## Status\n")
    lines.append("**ALL_BASELINES_RERUN_EXPORTED_TOP20_READY_FOR_DAY4_METRIC_RECOMPUTE**\n")
    lines.append("## Protocol reminder\n")
    lines.append("- Baselines are evaluated as upstream top-20 candidate generators.")
    lines.append("- This is not a classical full-universe filtered KGC metric.")
    lines.append("- Gold injection is forbidden.")
    lines.append("- Reviewer-safe metrics can be recomputed from the exported top-20 rows.\n")

    lines.append("## Data summary\n")
    lines.append(f"- Train rows path: `{data.train_rows_path}`")
    lines.append(f"- Enriched KG path: `{data.enriched_kg_path}`")
    lines.append(f"- Number of train triples: `{len(data.train_triples)}`")
    lines.append(f"- Number of train indication pairs: `{len(data.train_indication_pairs)}`")
    lines.append(f"- Number of entities: `{data.num_entities}`")
    lines.append(f"- Number of relations: `{len(data.relation_to_id)}`")
    lines.append(f"- Drug-only candidate universe size: `{len(data.drug_ids)}`")
    lines.append(f"- Valid rows: `{len(data.valid_rows)}`")
    lines.append(f"- Test rows: `{len(data.test_rows)}`\n")

    lines.append("## Model output summary\n")
    lines.append("| Model | Valid rows | Valid Gold@20 raw | Test rows | Test Gold@20 raw | Output dir |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for s in all_summaries:
        m = s["model_name"]
        vs = s["valid_summary"]
        ts = s["test_summary"]
        lines.append(
            f"| `{m}` | {vs['num_rows']} | {vs['gold_present_at20_raw']:.4f} "
            f"| {ts['num_rows']} | {ts['gold_present_at20_raw']:.4f} "
            f"| `outputs/baselines/baseline_outputs/{m}` |"
        )

    lines.append("\n## Files produced\n")
    for s in all_summaries:
        m = s["model_name"]
        lines.append(f"### {m}")
        lines.append(f"- `outputs/baselines/baseline_outputs/{m}/valid_top20.json`")
        lines.append(f"- `outputs/baselines/baseline_outputs/{m}/test_top20.json`")
        lines.append(f"- `outputs/baselines/baseline_outputs/{m}/config.json`")
        lines.append(f"- `outputs/baselines/baseline_outputs/{m}/train_log.json`")
        lines.append(f"- `outputs/baselines/baseline_outputs/{m}/summary.json`")

    lines.append("\n## Next step\n")
    lines.append(
        "Next, recompute reviewer-safe metrics for all baseline outputs and SoftFuse rows "
        "with the fixed rule: RR = 1/rank if rank <= 20 else 0; absent gold has rank 21 and RR 0."
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--gnn-epochs", type=int, default=40)
    p.add_argument("--gnn-layers", type=int, default=2)
    p.add_argument("--gnn-sample-per-epoch", type=int, default=32768)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--no-support-kg", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    RESULTS_DIR = ROOT / "outputs" / "baselines"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print("=" * 100)
    print("RERUN STRUCTURE BASELINES")
    print("=" * 100)
    print("device:", device)
    print("models:", args.models)

    data = build_data_pack(args)

    print("\nData summary:")
    print("  train_rows_path:", data.train_rows_path)
    print("  enriched_kg_path:", data.enriched_kg_path)
    print("  num_train_triples:", len(data.train_triples))
    print("  num_train_indication_pairs:", len(data.train_indication_pairs))
    print("  num_entities:", data.num_entities)
    print("  num_relations:", len(data.relation_to_id), data.relation_to_id)
    print("  drug_universe_size:", len(data.drug_ids))
    print("  valid_rows:", len(data.valid_rows))
    print("  test_rows:", len(data.test_rows))

    (RESULTS_DIR / "baseline_rerun_data_pack_summary.json").write_text(
        json.dumps({
            "train_rows_path": data.train_rows_path,
            "enriched_kg_path": data.enriched_kg_path,
            "num_train_triples": len(data.train_triples),
            "num_train_indication_pairs": len(data.train_indication_pairs),
            "num_entities": data.num_entities,
            "num_relations": len(data.relation_to_id),
            "relation_to_id": data.relation_to_id,
            "drug_universe_size": len(data.drug_ids),
            "valid_rows": len(data.valid_rows),
            "test_rows": len(data.test_rows),
            "models": args.models,
            "gold_injection": False,
            "top_k": args.top_k,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    all_summaries = []
    for model_name in args.models:
        summary = run_one_model(model_name, data, args, device)
        all_summaries.append(summary)

    (ROOT / "outputs" / "baselines" / "baseline_rerun_summary.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_baseline_rerun_report(all_summaries, data, args)

    print("=" * 100)
    print("BASELINE RERUN DONE")
    print("Summary:", ROOT / "outputs" / "baselines" / "baseline_rerun_summary.json")
    print("Report: ", REPORT_PATH)
    print("=" * 100)


if __name__ == "__main__":
    main()
