#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(".")
SETTING_DIR = ROOT / "dataset" / "setting_f_repodb"

GRAPH_DIR = SETTING_DIR / "graph"
READY_DIR = SETTING_DIR / "softfuse_ready" / "rgcn"

SUPPORT_DIR = SETTING_DIR / "support_features" / "rgcn"
SOFT_DIR = SETTING_DIR / "soft_support" / "rgcn_sweep_selected"
E2E_DIR = SETTING_DIR / "e2e_soft_support_ready" / "rgcn_sweep_selected"

RESULT_DIR = ROOT / "outputs" / "repodb"
REPORT_DIR = ROOT / "outputs" / "repodb" / "reports"

TARGET_RELATION = "repoDB::approved_indication::Compound:Disease"
FAILED_RELATION = "repoDB::failed_or_suspended::Compound:Disease"
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


def display_name(entity: str, type_map: dict[str, Any]) -> str:
    meta = type_map.get(entity, {})
    return str(meta.get("display_name") or meta.get("raw_name") or entity)


def relation_family(rel: str) -> str:
    if rel == TARGET_RELATION:
        return "target_approved"
    if rel == FAILED_RELATION:
        return "failed_diagnostic"
    if "Compound:Gene" in rel or "Gene:Compound" in rel:
        return "compound_gene"
    if "Gene:Gene" in rel:
        return "gene_gene"
    if "Compound:Disease" in rel:
        return "compound_disease_other"
    return "other"


def infer_kind(entity: str) -> str:
    if entity.startswith("Compound::"):
        return "Compound"
    if entity.startswith("Disease::"):
        return "Disease"
    if entity.startswith("Gene::"):
        return "Gene"
    if "::" in entity:
        return entity.split("::", 1)[0]
    return "UNKNOWN"


def build_kind_by_id(id2entity: dict[int, str], type_map: dict[str, Any]) -> dict[int, str]:
    out = {}
    for eid, ent in id2entity.items():
        out[eid] = type_map.get(ent, {}).get("kind", infer_kind(ent))
    return out


def read_train_edges_ids(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 3:
                yield int(row[0]), int(row[1]), int(row[2])


def build_adjacency(path: Path):
    adj = defaultdict(list)
    edges = set()
    for h, r, t in read_train_edges_ids(path):
        edge = (h, r, t)
        edges.add(edge)
        adj[h].append((r, t, edge))
        adj[t].append((r, h, edge))
    return adj, edges


def rel_ids_by_family(id2relation: dict[int, str], family: str) -> set[int]:
    return {rid for rid, name in id2relation.items() if relation_family(name) == family}


def rebuild_prompt(row: dict[str, Any], type_map: dict[str, Any]) -> None:
    query_display = row.get("query_display") or row.get("query_name") or display_name(row["query_entity"], type_map)

    canonical = row.get("rank_entities_canonical") or row.get("candidate_entities_canonical")
    if canonical is None:
        # Before patching, rank_entities may still be canonical Compound::DB IDs.
        canonical = list(row["rank_entities"])

    displays = [display_name(x, type_map) for x in canonical]

    row["rank_entities_canonical"] = canonical
    row["candidate_entities_canonical"] = canonical

    # Important for infer.py text matching.
    row["rank_entities"] = displays
    row["candidate_entities"] = displays
    row["rank_entities_display"] = displays
    row["candidate_entities_display"] = displays

    row["query_display"] = query_display
    row["gold_entity_canonical"] = row.get("gold_entity_canonical", row["gold_entity"])
    row["gold_display"] = row.get("gold_display") or row.get("gold_name") or display_name(row["gold_entity"], type_map)
    row["output"] = row["gold_display"]

    answer_options = "(" + ", ".join([f"'{x}'" for x in displays]) + ")"
    refer_parts = [f"'{query_display}': [QUERY]"]
    refer_parts.extend([f"'{x}': [ENTITY]" for x in displays])
    refer_str = ", ".join(refer_parts)

    question = f"What drug is approved for {query_display}?"

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


def rank_gold(gold_id: int, cand_ids: list[int]) -> tuple[int, bool]:
    gold_id = int(gold_id)
    if gold_id in cand_ids:
        return cand_ids.index(gold_id) + 1, True
    return ABSENT_RANK, False


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.array([int(r["rank"]) for r in rows], dtype=np.int64)
    present = ranks <= TOP_K
    rr = np.where(present, 1.0 / ranks, 0.0)

    top1 = [r["rank_entities_canonical"][0] for r in rows if r.get("rank_entities_canonical")]
    c = Counter(top1)
    mc = c.most_common(10)

    sizes = [len(r.get("subgraph", [])) for r in rows]

    return {
        "num_rows": len(rows),
        "gold_present_at20": float(np.mean(present)) if len(rows) else 0.0,
        "mrr_at20": float(np.mean(rr)) if len(rows) else 0.0,
        "hits1_at20": float(np.mean(ranks <= 1)) if len(rows) else 0.0,
        "hits3_at20": float(np.mean(ranks <= 3)) if len(rows) else 0.0,
        "hits10_at20": float(np.mean(ranks <= 10)) if len(rows) else 0.0,
        "hits20_at20": float(np.mean(ranks <= 20)) if len(rows) else 0.0,
        "rank21_count": int(np.sum(ranks == ABSENT_RANK)),
        "avg_rank_absent_as_21": float(np.mean(ranks)) if len(rows) else 0.0,
        "unique_top1_count": int(len(c)),
        "top1_dominance": float(mc[0][1] / len(rows)) if rows and mc else 0.0,
        "top1_most_common": [{"entity": k, "count": int(v)} for k, v in mc],
        "avg_subgraph_size": float(np.mean(sizes)) if sizes else 0.0,
    }


def minmax(xs: list[float]) -> list[float]:
    lo, hi = min(xs), max(xs)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def build_global_degree(adj, id2relation, kind_by_id):
    direct_ids = rel_ids_by_family(id2relation, "target_approved") | rel_ids_by_family(id2relation, "failed_diagnostic")
    deg = Counter()

    for node, edges in adj.items():
        if kind_by_id.get(node) != "Compound":
            continue
        seen_disease = set()
        for r, nb, _ in edges:
            if r in direct_ids and kind_by_id.get(nb) == "Disease":
                seen_disease.add(nb)
        deg[node] = len(seen_disease)

    vals = [v for v in deg.values() if v > 0]
    p95 = float(np.percentile(vals, 95)) if vals else 1.0
    return deg, max(p95, 1.0)


def extract_candidate_features(row, adj, id2relation, kind_by_id, direct_degree, direct_p95):
    target_ids = rel_ids_by_family(id2relation, "target_approved")
    failed_ids = rel_ids_by_family(id2relation, "failed_diagnostic")
    compound_gene_ids = rel_ids_by_family(id2relation, "compound_gene")
    gene_gene_ids = rel_ids_by_family(id2relation, "gene_gene")

    q = int(row["query_entity_id"])
    cand_ids = [int(x) for x in row["rank_entities_id"]]

    sg = [tuple(int(x) for x in e) for e in row.get("subgraph", [])]

    # Build subgraph adjacency for local evidence.
    sg_adj = defaultdict(list)
    for h, r, t in sg:
        sg_adj[h].append((r, t, (h, r, t)))
        sg_adj[t].append((r, h, (h, r, t)))

    features = []

    for pos, cid in enumerate(cand_ids, start=1):
        approved_direct = 0
        failed_direct = 0
        compound_gene_count = 0
        candidate_genes = set()

        for r, nb, _ in sg_adj.get(cid, []):
            if nb == q and r in target_ids:
                approved_direct += 1
            if nb == q and r in failed_ids:
                failed_direct += 1
            if r in compound_gene_ids and kind_by_id.get(nb) == "Gene":
                compound_gene_count += 1
                candidate_genes.add(nb)

        bridge_count = 0
        for g in list(candidate_genes)[:80]:
            for r, nb, _ in sg_adj.get(g, []):
                if r in gene_gene_ids:
                    bridge_count += 1
                    if bridge_count >= 50:
                        break
            if bridge_count >= 50:
                break

        cg_norm = min(1.0, compound_gene_count / 8.0)
        bridge_norm = min(1.0, bridge_count / 12.0)
        approved_direct_score = 1.0 if approved_direct > 0 else 0.0

        # Failed-like direct evidence is diagnostic/negative-like.
        failed_score = 1.0 if failed_direct > 0 else 0.0

        deg = direct_degree.get(cid, 0)
        degree_norm = min(1.0, deg / direct_p95)

        evidence_positive = (
            0.55 * approved_direct_score
            + 0.30 * cg_norm
            + 0.15 * bridge_norm
        )

        features.append({
            "candidate_id": int(cid),
            "candidate_canonical": row["rank_entities_canonical"][pos - 1],
            "candidate_display": row["rank_entities"][pos - 1],
            "original_rank": int(pos),
            "approved_direct_count": int(approved_direct),
            "failed_direct_count": int(failed_direct),
            "compound_gene_count": int(compound_gene_count),
            "gene_bridge_count": int(bridge_count),
            "candidate_direct_degree": int(deg),
            "candidate_direct_degree_norm": float(degree_norm),
            "evidence_positive": float(evidence_positive),
            "direct_shortcut_penalty": float(degree_norm),
            "failed_diagnostic_penalty": float(failed_score),
        })

    return features


def rerank_row(row, features, beta: float, raw_weight: float, gamma: float, type_map):
    supports = []
    raw_priors = []

    for f in features:
        support = (
            float(f["evidence_positive"])
            - beta * float(f["direct_shortcut_penalty"])
            - gamma * float(f["failed_diagnostic_penalty"])
        )
        supports.append(support)
        raw_priors.append((TOP_K - int(f["original_rank"]) + 1) / TOP_K)

    support_norm = minmax(supports)

    scored = []
    for f, sup, sn, rp in zip(features, supports, support_norm, raw_priors):
        final_score = raw_weight * rp + (1.0 - raw_weight) * sn
        item = dict(f)
        item["support_score"] = float(sup)
        item["support_norm"] = float(sn)
        item["raw_rank_prior"] = float(rp)
        item["final_score"] = float(final_score)
        scored.append(item)

    scored = sorted(
        scored,
        key=lambda x: (-x["final_score"], -x["support_score"], int(x["original_rank"]))
    )

    new_ids = [int(x["candidate_id"]) for x in scored]
    new_canonical = [x["candidate_canonical"] for x in scored]
    new_display = [x["candidate_display"] for x in scored]

    rank, present = rank_gold(int(row["gold_entity_id"]), new_ids)

    out = dict(row)
    out["rank_entities_id"] = new_ids
    out["candidate_entity_ids"] = new_ids

    out["rank_entities_canonical"] = new_canonical
    out["candidate_entities_canonical"] = new_canonical
    out["rank_entities"] = new_display
    out["candidate_entities"] = new_display
    out["rank_entities_display"] = new_display
    out["candidate_entities_display"] = new_display

    out["rank"] = int(rank)
    out["gold_rank_in_top20_or_21"] = int(rank)
    out["gold_in_topk_raw"] = bool(present)
    out["gold_present_top20"] = bool(present)
    out["reviewer_safe_rr_item"] = float(1.0 / rank) if present and rank <= TOP_K else 0.0
    out["support_scores"] = [float(x["final_score"]) for x in scored]
    out["candidate_debug_rows"] = scored
    out["variant_name"] = f"soft_support_sweep_beta{beta:.2f}_raw{raw_weight:.2f}"
    out["rank_delta_vs_backbone"] = int(int(row["rank"]) - int(rank))
    out["change_label_vs_backbone"] = (
        "improved" if int(rank) < int(row["rank"])
        else "worsened" if int(rank) > int(row["rank"])
        else "unchanged"
    )

    rebuild_prompt(out, type_map)
    return out


def run_variant(rows, features_by_idx, beta, raw_weight, gamma, type_map):
    out = []
    for i, row in enumerate(rows):
        out.append(rerank_row(row, features_by_idx[i], beta, raw_weight, gamma, type_map))
    return out, compute_metrics(out)


def audit_rows(rows, split):
    bad_k = bad_q = bad_e = bad_sg = 0
    leaks = 0
    sizes = []
    for r in rows:
        if len(r.get("rank_entities_id", [])) != TOP_K:
            bad_k += 1
        if r.get("input", "").count("[QUERY]") != 1:
            bad_q += 1
        if r.get("input", "").count("[ENTITY]") != TOP_K:
            bad_e += 1
        sg = r.get("subgraph", [])
        if not isinstance(sg, list) or any(not isinstance(x, list) or len(x) != 3 for x in sg):
            bad_sg += 1
        sizes.append(len(sg))
        if split in {"valid", "test"}:
            gold = tuple(int(x) for x in r["triple_id"])
            if any(tuple(int(y) for y in e) == gold for e in sg):
                leaks += 1
    return {
        "split": split,
        "num_rows": len(rows),
        "bad_candidate_len": bad_k,
        "bad_query_placeholder": bad_q,
        "bad_entity_placeholder": bad_e,
        "bad_subgraph": bad_sg,
        "avg_subgraph_size": float(np.mean(sizes)) if sizes else 0.0,
        "valid_test_exact_leak_count": leaks,
        "schema_pass": bad_k == 0 and bad_q == 0 and bad_e == 0 and bad_sg == 0 and leaks == 0,
    }


def copy_static_files():
    for name in [
        "entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl",
        "entity2id.json", "id2entity.json", "relation2id.json", "id2relation.json",
        "type_map.json", "entity_embeddings_rgcn.pt",
        "prompt_lexicon.json", "rules.json", "support_schema.json",
        "graph_summary.json", "leak_check.json",
    ]:
        src = READY_DIR / name
        if src.exists():
            shutil.copy2(src, E2E_DIR / name)


def write_report(summary):
    path = REPORT_DIR / "day6_repodb_soft_support_rgcn.md"
    lines = []
    lines.append("# repoDB R-GCN soft support")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append(f"- Selected variant: `{summary['selected_variant']['variant_name']}`")
    lines.append("")
    lines.append("## Raw vs selected")
    lines.append("")
    lines.append("| Split | Row | Gold@20 | MRR@20 | H@1 | H@3 | H@10 | Rank21 | Top1Dom |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for split in ["valid", "test"]:
        raw = summary["raw_metrics"][split]
        sel = summary["selected_metrics"][split]
        lines.append(f"| {split} | raw | {raw['gold_present_at20']:.3f} | {raw['mrr_at20']:.6f} | {raw['hits1_at20']:.3f} | {raw['hits3_at20']:.3f} | {raw['hits10_at20']:.3f} | {raw['rank21_count']} | {raw['top1_dominance']:.3f} |")
        lines.append(f"| {split} | soft | {sel['gold_present_at20']:.3f} | {sel['mrr_at20']:.6f} | {sel['hits1_at20']:.3f} | {sel['hits3_at20']:.3f} | {sel['hits10_at20']:.3f} | {sel['rank21_count']} | {sel['top1_dominance']:.3f} |")
    lines.append("")
    lines.append("## Top valid variants")
    lines.append("")
    lines.append("| Variant | beta | raw_weight | Valid MRR | Valid H@10 | Valid Top1Dom | Test MRR | Test H@10 | Test Top1Dom |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in summary["top_valid_variants"][:15]:
        vm = v["metrics"]["valid"]
        tm = v["metrics"]["test"]
        lines.append(f"| {v['variant_name']} | {v['beta']:.2f} | {v['raw_weight']:.2f} | {vm['mrr_at20']:.6f} | {vm['hits10_at20']:.3f} | {vm['top1_dominance']:.3f} | {tm['mrr_at20']:.6f} | {tm['hits10_at20']:.3f} | {tm['top1_dominance']:.3f} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", nargs="+", type=float, default=[0.00, 0.05, 0.10, 0.20, 0.35, 0.50])
    parser.add_argument("--raw-weights", nargs="+", type=float, default=[0.50, 0.65, 0.80, 0.90])
    parser.add_argument("--gamma", type=float, default=0.20)
    parser.add_argument("--top1-threshold", type=float, default=0.50)
    parser.add_argument("--min-mrr-ratio", type=float, default=0.98)
    args = parser.parse_args()

    for p in [SUPPORT_DIR, SOFT_DIR, E2E_DIR, RESULT_DIR, REPORT_DIR]:
        mkdir(p)

    entity2id, id2entity, relation2id, id2relation, type_map = load_maps()
    kind_by_id = build_kind_by_id(id2entity, type_map)
    adj, edge_set = build_adjacency(GRAPH_DIR / "train_enriched_ids.tsv")
    direct_degree, direct_p95 = build_global_degree(adj, id2relation, kind_by_id)

    patched_rows = {}
    features_by_split = {}
    raw_metrics = {}

    for split in ["train", "valid", "test"]:
        rows = read_json(READY_DIR / f"{split}.json")
        patched = []
        features = []

        for row in rows:
            r = dict(row)

            # Preserve canonical candidates before switching rank_entities to display names.
            canonical = r.get("rank_entities_canonical")
            if canonical is None:
                old_rank_entities = list(r.get("rank_entities", []))
                if old_rank_entities and str(old_rank_entities[0]).startswith("Compound::"):
                    canonical = old_rank_entities
                else:
                    canonical = [id2entity[int(x)] for x in r["rank_entities_id"]]
            r["rank_entities_canonical"] = canonical
            r["candidate_entities_canonical"] = canonical

            rebuild_prompt(r, type_map)
            patched.append(r)

        for row in patched:
            features.append(extract_candidate_features(
                row=row,
                adj=adj,
                id2relation=id2relation,
                kind_by_id=kind_by_id,
                direct_degree=direct_degree,
                direct_p95=direct_p95,
            ))

        patched_rows[split] = patched
        features_by_split[split] = features
        raw_metrics[split] = compute_metrics(patched)

        write_json(features, SUPPORT_DIR / f"{split}_support_features.json")

    variants = []

    for beta in args.betas:
        for raw_weight in args.raw_weights:
            item = {
                "beta": beta,
                "raw_weight": raw_weight,
                "variant_name": f"soft_support_sweep_beta{beta:.2f}_raw{raw_weight:.2f}",
                "metrics": {},
            }

            for split in ["train", "valid", "test"]:
                rows, metrics = run_variant(
                    patched_rows[split],
                    features_by_split[split],
                    beta=beta,
                    raw_weight=raw_weight,
                    gamma=args.gamma,
                    type_map=type_map,
                )
                item["metrics"][split] = metrics

            variants.append(item)

    variants_by_valid = sorted(
        variants,
        key=lambda x: (
            x["metrics"]["valid"]["mrr_at20"],
            x["metrics"]["valid"]["hits10_at20"],
            -x["metrics"]["valid"]["top1_dominance"],
        ),
        reverse=True,
    )

    raw_valid_mrr = raw_metrics["valid"]["mrr_at20"]

    balanced = [
        v for v in variants
        if v["metrics"]["valid"]["top1_dominance"] <= args.top1_threshold
        and v["metrics"]["valid"]["mrr_at20"] >= raw_valid_mrr * args.min_mrr_ratio
    ]

    if balanced:
        selected = sorted(
            balanced,
            key=lambda x: (
                x["metrics"]["valid"]["mrr_at20"],
                x["metrics"]["valid"]["hits10_at20"],
                -x["metrics"]["valid"]["top1_dominance"],
            ),
            reverse=True,
        )[0]
        reason = "best_valid_mrr_under_top1_and_min_mrr_constraints"
    else:
        selected = variants_by_valid[0]
        reason = "best_valid_mrr_no_balanced_variant_met_constraints"

    beta = selected["beta"]
    raw_weight = selected["raw_weight"]

    selected_rows = {}
    selected_metrics = {}
    audit = {}

    for split in ["train", "valid", "test"]:
        rows, metrics = run_variant(
            patched_rows[split],
            features_by_split[split],
            beta=beta,
            raw_weight=raw_weight,
            gamma=args.gamma,
            type_map=type_map,
        )
        selected_rows[split] = rows
        selected_metrics[split] = metrics

        write_json(rows, SOFT_DIR / f"{split}_top20_soft_support_main.json")
        write_json(rows, E2E_DIR / f"{split}.json")
        audit[split] = audit_rows(rows, split)

    copy_static_files()

    decision = "DAY6_REPODB_SOFT_SUPPORT_READY"
    if not all(a["schema_pass"] for a in audit.values()):
        decision = "DAY6_REPODB_SOFT_SUPPORT_SCHEMA_FIX_NEEDED"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_f_repodb",
        "dataset": "repoDB",
        "source": "rgcn",
        "raw_metrics": raw_metrics,
        "selected_variant": {
            "variant_name": selected["variant_name"],
            "beta": beta,
            "raw_weight": raw_weight,
            "gamma": args.gamma,
        },
        "selection_reason": reason,
        "selected_metrics": selected_metrics,
        "top_valid_variants": variants_by_valid[:30],
        "all_variants": variants,
        "audit": audit,
        "display_patch": {
            "rank_entities": "display drug names for infer.py text matching",
            "rank_entities_id": "numeric canonical entity IDs for graph embeddings",
            "rank_entities_canonical": "Compound::DBxxxxx IDs preserved for audit"
        },
        "output_dirs": {
            "support_features": str(SUPPORT_DIR),
            "soft_support": str(SOFT_DIR),
            "e2e_ready": str(E2E_DIR),
        },
    }

    write_json(summary, RESULT_DIR / "day6_repodb_soft_support_rgcn_summary.json")
    write_json(summary, E2E_DIR / "prep_manifest.json")
    write_report(summary)

    print(json.dumps({
        "decision": decision,
        "selection_reason": reason,
        "selected_variant": summary["selected_variant"],
        "raw_valid": raw_metrics["valid"],
        "raw_test": raw_metrics["test"],
        "selected_valid": selected_metrics["valid"],
        "selected_test": selected_metrics["test"],
        "audit": audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
