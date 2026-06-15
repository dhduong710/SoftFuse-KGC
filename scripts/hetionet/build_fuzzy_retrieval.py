#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(".")
SETTING_DIR = ROOT / "dataset" / "setting_d_hetionet"

SOFT_E2E_DIR = SETTING_DIR / "e2e_soft_support_ready"
FUZZY_DIR = SETTING_DIR / "fuzzy_retrieval"
E2E_FUZZY_DIR = SETTING_DIR / "e2e_fuzzy_retrieval_ready"

RESULT_DIR = ROOT / "outputs" / "hetionet"
REPORT_DIR = ROOT / "outputs" / "hetionet" / "reports"

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


def load_relation_maps():
    relation2id = read_json(SOFT_E2E_DIR / "relation2id.json")
    id2relation = {int(k): v for k, v in read_json(SOFT_E2E_DIR / "id2relation.json").items()}
    return relation2id, id2relation


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.array([r["rank"] for r in rows], dtype=np.int64)
    present = np.array([r["gold_present_top20"] for r in rows], dtype=bool)

    rr = np.zeros(len(rows), dtype=np.float64)
    for i, rank in enumerate(ranks):
        if present[i] and rank <= TOP_K:
            rr[i] = 1.0 / rank

    top1 = [r["rank_entities"][0] for r in rows if r["rank_entities"]]
    top1_counter = Counter(top1)
    most_common = top1_counter.most_common(10)

    sizes = [len(r.get("subgraph", [])) for r in rows]

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
        "avg_subgraph_size": float(np.mean(sizes)) if sizes else 0.0,
        "min_subgraph_size": int(np.min(sizes)) if sizes else 0,
        "max_subgraph_size": int(np.max(sizes)) if sizes else 0,
    }


def candidate_band(rank_pos: int) -> str:
    if rank_pos <= 5:
        return "top"
    if rank_pos <= 10:
        return "mid"
    return "tail"


def candidate_band_weight(rank_pos: int, args) -> float:
    band = candidate_band(rank_pos)
    if band == "top":
        return args.band_top
    if band == "mid":
        return args.band_mid
    return args.band_tail


def rel_family(rel_name: str) -> str:
    if rel_name in {"CtD", "CpD"}:
        return "direct_drug_disease"
    if rel_name in {"CbG", "CdG", "CuG"}:
        return "compound_gene"
    if rel_name in {"DaG", "DdG", "DuG"}:
        return "disease_gene"
    if rel_name in {"GiG", "Gr>G", "GcG"}:
        return "gene_gene"
    return "other"


def family_weight(rel_name: str, args) -> float:
    fam = rel_family(rel_name)
    if fam == "compound_gene":
        return args.family_compound_gene
    if fam == "disease_gene":
        return args.family_disease_gene
    if fam == "gene_gene":
        return args.family_gene_gene
    if fam == "direct_drug_disease":
        return args.family_direct
    return args.family_other


def edge_nodes(edge):
    return int(edge[0]), int(edge[2])


def score_triples(row: dict[str, Any], relation2id: dict[str, int], id2relation: dict[int, str], args):
    q = int(row["query_entity_id"])
    cand_ids = [int(x) for x in row["rank_entities_id"]]
    cand_rank = {cid: i + 1 for i, cid in enumerate(cand_ids)}

    top_candidates = set(cand_ids[:5])
    all_candidates = set(cand_ids)

    triples = row.get("subgraph", [])
    node_degree = Counter()
    for h, r, t in triples:
        node_degree[int(h)] += 1
        node_degree[int(t)] += 1

    scored = []
    for idx, e in enumerate(triples):
        h, r, t = int(e[0]), int(e[1]), int(e[2])
        rel_name = id2relation.get(r, str(r))

        touches_query = h == q or t == q
        touched_candidates = []
        if h in all_candidates:
            touched_candidates.append(h)
        if t in all_candidates:
            touched_candidates.append(t)

        if touched_candidates:
            best_rank = min(cand_rank[c] for c in touched_candidates)
            band_w = candidate_band_weight(best_rank, args)
            cand_touch_score = args.touch_candidate * band_w
        else:
            best_rank = None
            band_w = 0.0
            cand_touch_score = 0.0

        query_touch_score = args.touch_query if touches_query else 0.0
        fam_w = family_weight(rel_name, args)

        direct_penalty = args.direct_shortcut_penalty if rel_family(rel_name) == "direct_drug_disease" else 0.0

        density = (
            np.log1p(node_degree[h]) + np.log1p(node_degree[t])
        ) / 10.0
        density_score = args.density_weight * float(density)

        top_candidate_bonus = args.top_candidate_bonus if any(c in top_candidates for c in touched_candidates) else 0.0

        score = (
            fam_w
            + cand_touch_score
            + query_touch_score
            + density_score
            + top_candidate_bonus
            - direct_penalty
        )

        scored.append({
            "triple_index": idx,
            "triple": [h, r, t],
            "relation": rel_name,
            "relation_family": rel_family(rel_name),
            "score": float(score),
            "touches_query": bool(touches_query),
            "touches_candidate": bool(len(touched_candidates) > 0),
            "touched_candidate_best_rank": best_rank,
            "candidate_band_weight": float(band_w),
            "density_score": float(density_score),
            "direct_penalty": float(direct_penalty),
        })

    return scored


def select_subgraph(row: dict[str, Any], relation2id, id2relation, args):
    scored = score_triples(row, relation2id, id2relation, args)
    source_size = len(scored)

    if source_size == 0:
        return [], scored, {
            "source_subgraph_size": 0,
            "selected_subgraph_size": 0,
            "retain_ratio_actual": 0.0,
            "candidate_coverage_preserved_rate": 0.0,
        }

    target_keep = int(round(source_size * args.retain_ratio))
    target_keep = max(args.min_keep, target_keep)
    target_keep = min(args.max_keep, target_keep, source_size)

    selected_idx = set()

    # Coverage pass: keep best triple touching each candidate if available.
    by_candidate = defaultdict(list)
    cand_ids = [int(x) for x in row["rank_entities_id"]]

    for item in scored:
        h, r, t = item["triple"]
        for cid in cand_ids:
            if h == cid or t == cid:
                by_candidate[cid].append(item)

    for cid in cand_ids:
        items = by_candidate.get(cid, [])
        if not items:
            continue
        best = sorted(items, key=lambda x: (-x["score"], x["triple_index"]))[0]
        selected_idx.add(best["triple_index"])
        if len(selected_idx) >= target_keep:
            break

    # Query coverage pass.
    query_items = [x for x in scored if x["touches_query"]]
    if query_items and len(selected_idx) < target_keep:
        best_q = sorted(query_items, key=lambda x: (-x["score"], x["triple_index"]))[0]
        selected_idx.add(best_q["triple_index"])

    # Global fill.
    for item in sorted(scored, key=lambda x: (-x["score"], x["triple_index"])):
        if len(selected_idx) >= target_keep:
            break
        selected_idx.add(item["triple_index"])

    selected_scored = [x for x in scored if x["triple_index"] in selected_idx]
    selected_scored = sorted(selected_scored, key=lambda x: x["triple_index"])

    selected_subgraph = [x["triple"] for x in selected_scored]

    touched_candidates = set()
    for e in selected_subgraph:
        h, r, t = int(e[0]), int(e[1]), int(e[2])
        if h in cand_ids:
            touched_candidates.add(h)
        if t in cand_ids:
            touched_candidates.add(t)

    source_touched_candidates = set()
    for item in scored:
        h, r, t = item["triple"]
        if h in cand_ids:
            source_touched_candidates.add(h)
        if t in cand_ids:
            source_touched_candidates.add(t)

    if source_touched_candidates:
        coverage_rate = len(touched_candidates & source_touched_candidates) / len(source_touched_candidates)
    else:
        coverage_rate = 1.0

    summary = {
        "source_subgraph_size": int(source_size),
        "selected_subgraph_size": int(len(selected_subgraph)),
        "retain_ratio_actual": float(len(selected_subgraph) / source_size),
        "target_keep": int(target_keep),
        "num_source_touched_candidates": int(len(source_touched_candidates)),
        "num_selected_touched_candidates": int(len(touched_candidates)),
        "candidate_coverage_preserved_rate": float(coverage_rate),
        "avg_selected_score": float(np.mean([x["score"] for x in selected_scored])) if selected_scored else 0.0,
        "num_direct_edges_selected": int(sum(1 for x in selected_scored if x["relation_family"] == "direct_drug_disease")),
        "num_direct_edges_source": int(sum(1 for x in scored if x["relation_family"] == "direct_drug_disease")),
    }

    return selected_subgraph, scored, summary


def audit_rows(rows: list[dict[str, Any]], split: str):
    bad_k = 0
    bad_prompt = 0
    bad_subgraph = 0
    sizes = []
    source_sizes = []
    coverage = []

    for r in rows:
        if len(r.get("rank_entities_id", [])) != TOP_K:
            bad_k += 1
        if r.get("input", "").count("[QUERY]") != 1 or r.get("input", "").count("[ENTITY]") != TOP_K:
            bad_prompt += 1
        sg = r.get("subgraph", [])
        if not isinstance(sg, list) or any((not isinstance(x, list) or len(x) != 3) for x in sg):
            bad_subgraph += 1
        sizes.append(len(sg))
        source_sizes.append(r.get("subgraph_summary", {}).get("source_subgraph_size", 0))
        coverage.append(r.get("subgraph_summary", {}).get("candidate_coverage_preserved_rate", 1.0))

    return {
        "split": split,
        "num_rows": len(rows),
        "bad_candidate_len": bad_k,
        "bad_prompt_placeholders": bad_prompt,
        "bad_subgraph": bad_subgraph,
        "avg_source_subgraph_size": float(np.mean(source_sizes)) if source_sizes else 0.0,
        "avg_selected_subgraph_size": float(np.mean(sizes)) if sizes else 0.0,
        "min_selected_subgraph_size": int(np.min(sizes)) if sizes else 0,
        "max_selected_subgraph_size": int(np.max(sizes)) if sizes else 0,
        "avg_candidate_coverage_preserved_rate": float(np.mean(coverage)) if coverage else 1.0,
        "schema_pass": bad_k == 0 and bad_prompt == 0 and bad_subgraph == 0,
    }


def compare_metric_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = ["gold_present_at20", "mrr_at20", "hits1_at20", "hits3_at20", "hits10_at20", "hits20_at20", "rank21_count"]
    for k in keys:
        if abs(float(a[k]) - float(b[k])) > 1e-12:
            return False
    return True


def write_report(path: Path, summary: dict[str, Any]):
    lines = []
    lines.append("# Hetionet confidence-aware fuzzy retrieval")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append(f"- Variant: `{summary['variant_name']}`")
    lines.append("- Candidate order: preserved from soft support")
    lines.append("")
    lines.append("## Candidate metrics check")
    lines.append("")
    lines.append("| Split | Soft MRR | Fuzzy MRR | Metrics preserved | Soft graph | Fuzzy graph | Reduction | Coverage preserved |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for split in ["train", "valid", "test"]:
        s = summary["soft_metrics"][split]
        f = summary["fuzzy_metrics"][split]
        a = summary["audit"][split]
        red = 1.0 - (a["avg_selected_subgraph_size"] / max(1.0, a["avg_source_subgraph_size"]))
        lines.append(
            f"| {split} | {s['mrr_at20']:.6f} | {f['mrr_at20']:.6f} | "
            f"{summary['metrics_preserved'][split]} | "
            f"{a['avg_source_subgraph_size']:.2f} | {a['avg_selected_subgraph_size']:.2f} | "
            f"{red:.3f} | {a['avg_candidate_coverage_preserved_rate']:.3f} |"
        )

    lines.append("")
    lines.append("## Audit")
    lines.append("")
    lines.append("| Split | Rows | Bad K | Bad prompt | Bad subgraph | Min graph | Max graph | Schema pass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split, a in summary["audit"].items():
        lines.append(
            f"| {split} | {a['num_rows']} | {a['bad_candidate_len']} | {a['bad_prompt_placeholders']} | "
            f"{a['bad_subgraph']} | {a['min_selected_subgraph_size']} | {a['max_selected_subgraph_size']} | {a['schema_pass']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Day 6 should not change candidate metrics.")
    lines.append("- The main gain is evidence efficiency: smaller selected subgraphs with candidate coverage preserved.")
    lines.append("- Day 7 can train/infer Llama-3.2-3B on backbone_raw, soft_support_raw, and fuzzy_retrieval_main if compute allows.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-name", default="fuzzy_retrieval_main_hetionet")
    parser.add_argument("--retain-ratio", type=float, default=0.55)
    parser.add_argument("--min-keep", type=int, default=28)
    parser.add_argument("--max-keep", type=int, default=48)
    parser.add_argument("--band-top", type=float, default=1.0)
    parser.add_argument("--band-mid", type=float, default=0.6)
    parser.add_argument("--band-tail", type=float, default=0.25)
    parser.add_argument("--touch-candidate", type=float, default=0.35)
    parser.add_argument("--touch-query", type=float, default=0.10)
    parser.add_argument("--top-candidate-bonus", type=float, default=0.05)
    parser.add_argument("--family-compound-gene", type=float, default=0.30)
    parser.add_argument("--family-disease-gene", type=float, default=0.30)
    parser.add_argument("--family-gene-gene", type=float, default=0.20)
    parser.add_argument("--family-direct", type=float, default=0.05)
    parser.add_argument("--family-other", type=float, default=0.05)
    parser.add_argument("--direct-shortcut-penalty", type=float, default=0.50)
    parser.add_argument("--density-weight", type=float, default=0.05)
    args = parser.parse_args()

    for p in [FUZZY_DIR, E2E_FUZZY_DIR, RESULT_DIR, REPORT_DIR]:
        mkdir(p)

    relation2id, id2relation = load_relation_maps()

    soft_metrics = {}
    fuzzy_metrics = {}
    audit = {}
    metrics_preserved = {}

    for split in ["train", "valid", "test"]:
        print("=" * 100)
        print("[split]", split)

        soft_rows = read_json(SOFT_E2E_DIR / f"{split}.json")
        fuzzy_rows = []

        for row in soft_rows:
            selected_subgraph, triple_score_rows, subgraph_summary = select_subgraph(
                row=row,
                relation2id=relation2id,
                id2relation=id2relation,
                args=args,
            )

            out = dict(row)
            out["original_subgraph"] = row.get("subgraph", [])
            out["selected_subgraph"] = selected_subgraph
            out["subgraph"] = selected_subgraph
            out["triple_score_rows"] = triple_score_rows
            out["subgraph_summary"] = subgraph_summary
            out["selected_source_variant"] = args.variant_name
            fuzzy_rows.append(out)

        write_json(fuzzy_rows, FUZZY_DIR / f"{split}_fuzzy_retrieval_main.json")
        write_json(fuzzy_rows, E2E_FUZZY_DIR / f"{split}.json")

        soft_metrics[split] = compute_metrics(soft_rows)
        fuzzy_metrics[split] = compute_metrics(fuzzy_rows)
        audit[split] = audit_rows(fuzzy_rows, split)
        metrics_preserved[split] = compare_metric_equal(soft_metrics[split], fuzzy_metrics[split])

        print("soft :", soft_metrics[split])
        print("fuzzy:", fuzzy_metrics[split])
        print("audit:", audit[split])

    # Copy static files for DrKGC compatibility.
    for name in [
        "entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl",
        "entity2id.json", "id2entity.json", "relation2id.json", "id2relation.json",
        "type_map.json", "entity_embeddings_rgcn.pt",
        "prompt_lexicon.json", "rules.json", "support_schema.json",
        "graph_summary.json", "leak_check.json",
    ]:
        src = SOFT_E2E_DIR / name
        if src.exists():
            shutil.copy2(src, E2E_FUZZY_DIR / name)

    decision = "DAY6_HETIONET_FUZZY_RETRIEVAL_READY"
    if not all(a["schema_pass"] for a in audit.values()):
        decision = "DAY6_HETIONET_FUZZY_RETRIEVAL_SCHEMA_FIX_NEEDED"
    if not all(metrics_preserved.values()):
        decision = "DAY6_HETIONET_FUZZY_RETRIEVAL_METRIC_MISMATCH"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_d_hetionet",
        "dataset": "Hetionet v1.0",
        "variant_name": args.variant_name,
        "params": vars(args),
        "soft_metrics": soft_metrics,
        "fuzzy_metrics": fuzzy_metrics,
        "metrics_preserved": metrics_preserved,
        "audit": audit,
        "notes": [
            "Fuzzy retrieval preserves soft-support candidate order and ranking metrics.",
            "Only the evidence subgraph is reduced.",
            "E2E folder is ready for DrKGC-style inference/training."
        ],
    }

    write_json(summary, RESULT_DIR / "day6_hetionet_fuzzy_retrieval_summary.json")
    write_json(summary, FUZZY_DIR / "fuzzy_retrieval_manifest.json")
    write_json(summary, E2E_FUZZY_DIR / "prep_manifest.json")
    write_report(REPORT_DIR / "day6_hetionet_fuzzy_retrieval.md", summary)

    print("\n[DONE] Day 6")
    print(json.dumps({
        "decision": decision,
        "metrics_preserved": metrics_preserved,
        "audit": audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
