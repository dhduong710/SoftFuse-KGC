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


ROOT = Path(".")
SETTING_DIR = ROOT / "dataset" / "setting_d_hetionet"
GRAPH_DIR = SETTING_DIR / "graph"
SOURCE_DIR = SETTING_DIR / "backbone_raw_source"
BACKBONE_READY_DIR = SETTING_DIR / "softfuse_ready"

SUPPORT_DIR = SETTING_DIR / "support_features"
SOFT_DIR = SETTING_DIR / "soft_support"
E2E_SOFT_DIR = SETTING_DIR / "e2e_soft_support_ready"

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


def load_maps():
    entity2id = read_json(GRAPH_DIR / "entity2id.json")
    id2entity = {int(k): v for k, v in read_json(GRAPH_DIR / "id2entity.json").items()}
    relation2id = read_json(GRAPH_DIR / "relation2id.json")
    id2relation = {int(k): v for k, v in read_json(GRAPH_DIR / "id2relation.json").items()}
    type_map = read_json(GRAPH_DIR / "type_map.json")
    return entity2id, id2entity, relation2id, id2relation, type_map


def read_train_edges_ids(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 3:
                yield int(row[0]), int(row[1]), int(row[2])


def build_adjacency(train_ids_path: Path):
    adj = defaultdict(list)
    edge_set = set()
    rel_counts = Counter()

    for h, r, t in read_train_edges_ids(train_ids_path):
        edge = (h, r, t)
        edge_set.add(edge)

        # undirected traversal, original edge tuple preserved
        adj[h].append((r, t, edge))
        adj[t].append((r, h, edge))
        rel_counts[r] += 1

    return adj, edge_set, rel_counts


def build_kind_by_id(id2entity: dict[int, str], type_map: dict[str, Any]) -> dict[int, str]:
    out = {}
    for eid, name in id2entity.items():
        meta = type_map.get(name, {})
        out[int(eid)] = meta.get("kind", "UNKNOWN")
    return out


def rel_ids(relation2id: dict[str, int], names: list[str]) -> set[int]:
    return {int(relation2id[x]) for x in names if x in relation2id}


def neighbor_set(adj, node_id: int, relation_ids: set[int], kind_by_id: dict[int, str] | None = None, kind: str | None = None, cap: int | None = None):
    out = []
    seen = set()

    for r, nb, edge in adj.get(int(node_id), []):
        if int(r) not in relation_ids:
            continue
        if kind is not None and kind_by_id is not None and kind_by_id.get(int(nb)) != kind:
            continue
        if int(nb) in seen:
            continue
        seen.add(int(nb))
        out.append(int(nb))
        if cap is not None and len(out) >= cap:
            break

    return set(out)


def has_direct_edge(adj, c_id: int, q_id: int, relation_ids: set[int]) -> bool:
    for r, nb, edge in adj.get(int(c_id), []):
        if int(nb) == int(q_id) and int(r) in relation_ids:
            return True
    return False


def direct_degree(adj, c_id: int, relation_ids: set[int], kind_by_id: dict[int, str]) -> int:
    count = 0
    seen = set()
    for r, nb, edge in adj.get(int(c_id), []):
        if int(r) in relation_ids and kind_by_id.get(int(nb)) == "Disease":
            if int(nb) not in seen:
                seen.add(int(nb))
                count += 1
    return count


def count_gene_bridges(adj, cand_genes: set[int], disease_genes: set[int], gene_gene_rel_ids: set[int], bridge_cap: int) -> int:
    if not cand_genes or not disease_genes:
        return 0

    disease_genes = set(int(x) for x in disease_genes)
    count = 0

    for g in list(cand_genes)[:80]:
        for r, nb, edge in adj.get(int(g), []):
            if int(r) in gene_gene_rel_ids and int(nb) in disease_genes:
                count += 1
                if count >= bridge_cap:
                    return count

    return count


def build_feature_computer(adj, relation2id, kind_by_id, args):
    compound_gene_rel_ids = rel_ids(relation2id, ["CbG", "CdG", "CuG"])
    disease_gene_rel_ids = rel_ids(relation2id, ["DaG", "DdG", "DuG"])
    gene_gene_rel_ids = rel_ids(relation2id, ["GiG", "Gr>G", "GcG"])
    direct_rel_ids = rel_ids(relation2id, ["CtD", "CpD"])

    candidate_gene_cache = {}
    disease_gene_cache = {}
    direct_degree_cache = {}

    def get_candidate_genes(c_id: int):
        c_id = int(c_id)
        if c_id not in candidate_gene_cache:
            candidate_gene_cache[c_id] = neighbor_set(
                adj=adj,
                node_id=c_id,
                relation_ids=compound_gene_rel_ids,
                kind_by_id=kind_by_id,
                kind="Gene",
                cap=args.candidate_gene_cap,
            )
        return candidate_gene_cache[c_id]

    def get_disease_genes(q_id: int):
        q_id = int(q_id)
        if q_id not in disease_gene_cache:
            disease_gene_cache[q_id] = neighbor_set(
                adj=adj,
                node_id=q_id,
                relation_ids=disease_gene_rel_ids,
                kind_by_id=kind_by_id,
                kind="Gene",
                cap=args.disease_gene_cap,
            )
        return disease_gene_cache[q_id]

    def get_direct_degree(c_id: int):
        c_id = int(c_id)
        if c_id not in direct_degree_cache:
            direct_degree_cache[c_id] = direct_degree(
                adj=adj,
                c_id=c_id,
                relation_ids=direct_rel_ids,
                kind_by_id=kind_by_id,
            )
        return direct_degree_cache[c_id]

    # For direct popularity normalization.
    all_direct_degrees = []
    compound_ids = [nid for nid, kind in kind_by_id.items() if kind == "Compound"]
    for c_id in compound_ids:
        d = get_direct_degree(c_id)
        if d > 0:
            all_direct_degrees.append(d)

    if all_direct_degrees:
        p95 = float(np.percentile(all_direct_degrees, 95))
        maxd = float(max(all_direct_degrees))
    else:
        p95 = 1.0
        maxd = 1.0

    p95 = max(p95, 1.0)

    def compute(candidate_id: int, query_id: int, original_rank: int):
        c_id = int(candidate_id)
        q_id = int(query_id)

        cand_genes = get_candidate_genes(c_id)
        disease_genes = get_disease_genes(q_id)

        shared = cand_genes & disease_genes
        bridge_count = count_gene_bridges(
            adj=adj,
            cand_genes=cand_genes,
            disease_genes=disease_genes,
            gene_gene_rel_ids=gene_gene_rel_ids,
            bridge_cap=args.bridge_cap,
        )

        direct_flag = has_direct_edge(adj, c_id, q_id, direct_rel_ids)
        c_direct_degree = get_direct_degree(c_id)
        degree_norm = min(1.0, c_direct_degree / p95)

        shared_score = min(1.0, len(shared) / max(1.0, args.shared_norm))
        bridge_score = min(1.0, bridge_count / max(1.0, args.bridge_norm))
        candidate_touch = 1.0 if len(cand_genes) > 0 else 0.0
        query_touch = 1.0 if len(disease_genes) > 0 else 0.0

        evidence_positive = (
            args.w_shared * shared_score
            + args.w_bridge * bridge_score
            + args.w_candidate_touch * candidate_touch
            + args.w_query_touch * query_touch
        )

        direct_shortcut_penalty = max(
            1.0 if direct_flag else 0.0,
            args.direct_degree_weight * degree_norm,
        )
        direct_shortcut_penalty = min(1.0, direct_shortcut_penalty)

        contradiction_penalty = 0.0

        support_score = (
            evidence_positive
            - args.direct_penalty * direct_shortcut_penalty
            - args.contra_penalty * contradiction_penalty
        )

        return {
            "candidate_id": c_id,
            "original_rank": int(original_rank),
            "candidate_gene_count": int(len(cand_genes)),
            "disease_gene_count": int(len(disease_genes)),
            "shared_gene_count": int(len(shared)),
            "gene_bridge_count": int(bridge_count),
            "direct_candidate_query_flag": bool(direct_flag),
            "candidate_direct_degree_ctd_cpd": int(c_direct_degree),
            "candidate_direct_degree_norm": float(degree_norm),
            "evidence_positive": float(evidence_positive),
            "direct_shortcut_penalty": float(direct_shortcut_penalty),
            "contradiction_penalty": float(contradiction_penalty),
            "support_score": float(support_score),
        }

    meta = {
        "compound_gene_rel_ids": sorted(compound_gene_rel_ids),
        "disease_gene_rel_ids": sorted(disease_gene_rel_ids),
        "gene_gene_rel_ids": sorted(gene_gene_rel_ids),
        "direct_rel_ids": sorted(direct_rel_ids),
        "direct_degree_p95": p95,
        "direct_degree_max": maxd,
    }

    return compute, meta


def rank_gold(gold_id: int, candidate_ids: list[int]):
    if int(gold_id) in candidate_ids:
        return candidate_ids.index(int(gold_id)) + 1, True
    return ABSENT_RANK, False


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.array([r["rank"] for r in rows], dtype=np.int64)
    present = np.array([r["gold_in_topk_raw"] for r in rows], dtype=bool)

    rr = np.zeros(len(rows), dtype=np.float64)
    for i, rank in enumerate(ranks):
        if present[i] and rank <= TOP_K:
            rr[i] = 1.0 / rank

    top1 = [r["rank_entities"][0] for r in rows if r["rank_entities"]]
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


def apply_soft_support(split: str, rows: list[dict[str, Any]], id2entity: dict[int, str], compute_feature, variant_name: str):
    support_rows = []
    soft_rows = []

    improved = 0
    worsened = 0
    unchanged = 0

    for row in rows:
        features = []
        for i, cid in enumerate(row["candidate_entity_ids"]):
            f = compute_feature(
                candidate_id=int(cid),
                query_id=int(row["query_entity_id"]),
                original_rank=i + 1,
            )
            f["candidate_entity"] = id2entity[int(cid)]
            features.append(f)

        sorted_features = sorted(
            features,
            key=lambda x: (
                -x["support_score"],
                -x["evidence_positive"],
                x["direct_shortcut_penalty"],
                x["original_rank"],
            ),
        )

        new_ids = [int(x["candidate_id"]) for x in sorted_features]
        new_names = [id2entity[int(x)] for x in new_ids]

        old_rank = int(row["rank"])
        new_rank, present = rank_gold(int(row["gold_entity_id"]), new_ids)

        if new_rank < old_rank:
            improved += 1
            change = "improved"
        elif new_rank > old_rank:
            worsened += 1
            change = "worsened"
        else:
            unchanged += 1
            change = "unchanged"

        feature_by_id = {int(x["candidate_id"]): dict(x) for x in sorted_features}
        for new_pos, cid in enumerate(new_ids, start=1):
            feature_by_id[cid]["soft_rank"] = int(new_pos)

        support_row = {
            "split": split,
            "row_index": row.get("row_index"),
            "query_entity": row["query_entity"],
            "query_entity_id": row["query_entity_id"],
            "gold_entity": row["gold_entity"],
            "gold_entity_id": row["gold_entity_id"],
            "source_model": row.get("source_model", "rgcn"),
            "variant_name": variant_name,
            "candidate_feature_rows": [feature_by_id[int(cid)] for cid in new_ids],
            "row_summary": {
                "old_rank": old_rank,
                "new_rank": int(new_rank),
                "rank_delta_vs_backbone": int(old_rank - new_rank),
                "change_label_vs_backbone": change,
                "gold_present": bool(present),
                "old_top1": row["rank_entities"][0],
                "new_top1": new_names[0],
            },
        }
        support_rows.append(support_row)

        out = dict(row)
        out.update({
            "candidate_entities": new_names,
            "candidate_entity_ids": new_ids,
            "rank_entities": new_names,
            "rank_entities_id": new_ids,
            "rank": int(new_rank),
            "gold_rank_in_top20_or_21": int(new_rank),
            "gold_in_topk_raw": bool(present),
            "gold_present_top20": bool(present),
            "reviewer_safe_rr_item": float(1.0 / new_rank) if present and new_rank <= TOP_K else 0.0,
            "variant_name": variant_name,
            "support_scores": [float(feature_by_id[int(cid)]["support_score"]) for cid in new_ids],
            "support_rank_order": new_ids,
            "candidate_debug_rows": [feature_by_id[int(cid)] for cid in new_ids],
            "rank_delta_vs_backbone": int(old_rank - new_rank),
            "change_label_vs_backbone": change,
        })
        soft_rows.append(out)

    change_summary = {
        "num_rows": len(rows),
        "num_improved_vs_backbone": improved,
        "num_worsened_vs_backbone": worsened,
        "num_unchanged_vs_backbone": unchanged,
        "improved_rate": improved / len(rows) if rows else 0.0,
        "worsened_rate": worsened / len(rows) if rows else 0.0,
    }

    return support_rows, soft_rows, change_summary


def audit_e2e(rows: list[dict[str, Any]], split: str):
    bad_k = 0
    bad_prompt = 0
    bad_subgraph = 0
    sizes = []

    for r in rows:
        if len(r.get("rank_entities_id", [])) != TOP_K:
            bad_k += 1
        if r.get("input", "").count("[QUERY]") != 1 or r.get("input", "").count("[ENTITY]") != TOP_K:
            bad_prompt += 1
        sg = r.get("subgraph", [])
        if not isinstance(sg, list) or any((not isinstance(x, list) or len(x) != 3) for x in sg):
            bad_subgraph += 1
        sizes.append(len(sg))

    return {
        "split": split,
        "num_rows": len(rows),
        "bad_candidate_len": bad_k,
        "bad_prompt_placeholders": bad_prompt,
        "bad_subgraph": bad_subgraph,
        "avg_subgraph_size": float(np.mean(sizes)) if sizes else 0.0,
        "schema_pass": bad_k == 0 and bad_prompt == 0 and bad_subgraph == 0,
    }


def write_report(path: Path, summary: dict[str, Any]):
    lines = []
    lines.append("# Hetionet support features and soft support")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append(f"- Variant: `{summary['variant_name']}`")
    lines.append(f"- Source: `R-GCN backbone_raw`")
    lines.append("")
    lines.append("## Raw vs soft metrics")
    lines.append("")
    lines.append("| Split | Row | Gold@20 | MRR@20 | H@1 | H@3 | H@10 | Rank21 | Top1 dominance |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")

    for split in ["train", "valid", "test"]:
        raw = summary["raw_metrics"][split]
        soft = summary["soft_metrics"][split]
        lines.append(f"| {split} | raw | {raw['gold_present_at20']:.3f} | {raw['mrr_at20']:.6f} | {raw['hits1_at20']:.3f} | {raw['hits3_at20']:.3f} | {raw['hits10_at20']:.3f} | {raw['rank21_count']} | {raw['top1_dominance']:.3f} |")
        lines.append(f"| {split} | soft | {soft['gold_present_at20']:.3f} | {soft['mrr_at20']:.6f} | {soft['hits1_at20']:.3f} | {soft['hits3_at20']:.3f} | {soft['hits10_at20']:.3f} | {soft['rank21_count']} | {soft['top1_dominance']:.3f} |")

    lines.append("")
    lines.append("## Change summary")
    lines.append("")
    lines.append("| Split | Improved | Worsened | Unchanged | Improved rate | Worsened rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split, c in summary["change_summary"].items():
        lines.append(f"| {split} | {c['num_improved_vs_backbone']} | {c['num_worsened_vs_backbone']} | {c['num_unchanged_vs_backbone']} | {c['improved_rate']:.3f} | {c['worsened_rate']:.3f} |")

    lines.append("")
    lines.append("## E2E package audit")
    lines.append("")
    lines.append("| Split | Rows | Bad K | Bad prompt | Bad subgraph | Avg graph | Schema pass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split, a in summary["e2e_audit"].items():
        lines.append(f"| {split} | {a['num_rows']} | {a['bad_candidate_len']} | {a['bad_prompt_placeholders']} | {a['bad_subgraph']} | {a['avg_subgraph_size']:.2f} | {a['schema_pass']} |")

    lines.append("")
    lines.append("## Interpretation note")
    lines.append("")
    lines.append("- Valid/test Gold@20 should remain unchanged because soft support does not add candidates.")
    lines.append("- The main Day 5 success criterion is improved early rank and reduced top-1 dominance.")
    lines.append("- If valid MRR drops strongly or top-1 dominance remains high, run a Day 5b penalty sweep before fuzzy retrieval.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-name", default="soft_support_b050_hetionet_main")
    parser.add_argument("--direct-penalty", type=float, default=0.50)
    parser.add_argument("--contra-penalty", type=float, default=0.10)
    parser.add_argument("--direct-degree-weight", type=float, default=1.00)
    parser.add_argument("--candidate-gene-cap", type=int, default=120)
    parser.add_argument("--disease-gene-cap", type=int, default=300)
    parser.add_argument("--bridge-cap", type=int, default=30)
    parser.add_argument("--shared-norm", type=float, default=2.0)
    parser.add_argument("--bridge-norm", type=float, default=5.0)
    parser.add_argument("--w-shared", type=float, default=0.70)
    parser.add_argument("--w-bridge", type=float, default=0.20)
    parser.add_argument("--w-candidate-touch", type=float, default=0.05)
    parser.add_argument("--w-query-touch", type=float, default=0.05)
    args = parser.parse_args()

    for p in [SUPPORT_DIR, SOFT_DIR, E2E_SOFT_DIR, RESULT_DIR, REPORT_DIR]:
        mkdir(p)

    entity2id, id2entity, relation2id, id2relation, type_map = load_maps()

    print("[graph] building adjacency")
    adj, edge_set, rel_counts = build_adjacency(GRAPH_DIR / "train_enriched_ids.tsv")
    kind_by_id = build_kind_by_id(id2entity, type_map)

    compute_feature, feature_meta = build_feature_computer(
        adj=adj,
        relation2id=relation2id,
        kind_by_id=kind_by_id,
        args=args,
    )

    raw_metrics = {}
    soft_metrics = {}
    change_summary = {}
    e2e_audit = {}

    for split in ["train", "valid", "test"]:
        print("=" * 100)
        print("[split]", split)

        source_rows = read_json(SOURCE_DIR / f"{split}_top20_raw.json")
        support_rows, soft_rows, changes = apply_soft_support(
            split=split,
            rows=source_rows,
            id2entity=id2entity,
            compute_feature=compute_feature,
            variant_name=args.variant_name,
        )

        write_json(support_rows, SUPPORT_DIR / f"{split}_support_features.json")
        write_json(soft_rows, SOFT_DIR / f"{split}_top20_soft_support_main.json")

        raw_metrics[split] = compute_metrics(source_rows)
        soft_metrics[split] = compute_metrics(soft_rows)
        change_summary[split] = changes

        # Build E2E-ready rows: copy source ready row subgraph, replace candidate order/prompt.
        backbone_ready_rows = read_json(BACKBONE_READY_DIR / f"{split}.json")
        ready_by_row_index = {r.get("row_index"): r for r in backbone_ready_rows}

        e2e_rows = []
        for sr in soft_rows:
            base = ready_by_row_index.get(sr.get("row_index"))
            if base is None:
                raise KeyError(f"Missing backbone ready row for split={split} row_index={sr.get('row_index')}")

            out = dict(base)
            for key, val in sr.items():
                out[key] = val

            # Keep the same backbone source subgraph for soft_support_raw.
            out["subgraph"] = base["subgraph"]
            out["selected_source_variant"] = args.variant_name
            add_prompt(out)
            e2e_rows.append(out)

        write_json(e2e_rows, E2E_SOFT_DIR / f"{split}.json")
        e2e_audit[split] = audit_e2e(e2e_rows, split)

        print("raw :", raw_metrics[split])
        print("soft:", soft_metrics[split])
        print("changes:", changes)

    # Copy static files for DrKGC/DataModule compatibility.
    for name in [
        "entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl",
        "entity2id.json", "id2entity.json", "relation2id.json", "id2relation.json",
        "type_map.json", "entity_embeddings_rgcn.pt",
        "prompt_lexicon.json", "rules.json", "support_schema.json",
        "graph_summary.json", "leak_check.json",
    ]:
        src = BACKBONE_READY_DIR / name
        if src.exists():
            shutil.copy2(src, E2E_SOFT_DIR / name)

    decision = "DAY5_HETIONET_SOFT_SUPPORT_READY"
    if not all(a["schema_pass"] for a in e2e_audit.values()):
        decision = "DAY5_HETIONET_SOFT_SUPPORT_NEEDS_SCHEMA_FIX"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_d_hetionet",
        "dataset": "Hetionet v1.0",
        "variant_name": args.variant_name,
        "formula": "support_score = evidence_positive - direct_penalty * direct_shortcut_penalty - contra_penalty * contradiction_penalty",
        "params": vars(args),
        "feature_meta": feature_meta,
        "raw_metrics": raw_metrics,
        "soft_metrics": soft_metrics,
        "change_summary": change_summary,
        "e2e_audit": e2e_audit,
        "notes": [
            "Soft support re-ranks only the existing top-20 candidates.",
            "No valid/test gold injection is performed.",
            "Soft_support_raw E2E package keeps the Day4 backbone source subgraph and changes candidate order/prompt only."
        ],
    }

    write_json(summary, RESULT_DIR / "day5_hetionet_soft_support_summary.json")
    write_json(summary, SOFT_DIR / "soft_support_manifest.json")
    write_json(summary, E2E_SOFT_DIR / "prep_manifest.json")
    write_report(REPORT_DIR / "day5_hetionet_soft_support.md", summary)

    print("\n[DONE] Day 5")
    print(json.dumps({
        "decision": decision,
        "raw_metrics": raw_metrics,
        "soft_metrics": soft_metrics,
        "change_summary": change_summary,
        "e2e_audit": e2e_audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
