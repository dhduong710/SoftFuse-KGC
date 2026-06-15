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
SETTING_DIR = ROOT / "dataset" / "setting_f_repodb"

INPUT_DIR = SETTING_DIR / "e2e_soft_support_ready" / "rgcn_raw_display_control"

FUZZY_DIR = SETTING_DIR / "fuzzy_retrieval" / "rgcn"
E2E_DIR = SETTING_DIR / "e2e_fuzzy_retrieval_ready" / "rgcn"

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


def family_weight(rel_name: str, args) -> float:
    fam = relation_family(rel_name)
    if fam == "target_approved":
        return args.family_target_approved
    if fam == "failed_diagnostic":
        return args.family_failed_diagnostic
    if fam == "compound_gene":
        return args.family_compound_gene
    if fam == "gene_gene":
        return args.family_gene_gene
    if fam == "compound_disease_other":
        return args.family_compound_disease_other
    return args.family_other


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = np.array([int(r["rank"]) for r in rows], dtype=np.int64)
    present = ranks <= TOP_K
    rr = np.where(present, 1.0 / ranks, 0.0)

    top1 = []
    for r in rows:
        if r.get("rank_entities_canonical"):
            top1.append(r["rank_entities_canonical"][0])
        elif r.get("rank_entities"):
            top1.append(r["rank_entities"][0])

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


def score_triples(row: dict[str, Any], id2relation: dict[int, str], args):
    q = int(row["query_entity_id"])
    candidate_ids = [int(x) for x in row["rank_entities_id"]]
    candidate_rank = {cid: i + 1 for i, cid in enumerate(candidate_ids)}
    candidate_set = set(candidate_ids)
    top_candidate_set = set(candidate_ids[:5])

    triples = row.get("subgraph", [])

    node_degree = Counter()
    for h, r, t in triples:
        node_degree[int(h)] += 1
        node_degree[int(t)] += 1

    scored = []

    for idx, edge in enumerate(triples):
        h, r, t = int(edge[0]), int(edge[1]), int(edge[2])
        rel_name = id2relation.get(r, str(r))
        fam = relation_family(rel_name)

        touches_query = h == q or t == q

        touched_candidates = []
        if h in candidate_set:
            touched_candidates.append(h)
        if t in candidate_set:
            touched_candidates.append(t)

        if touched_candidates:
            best_rank = min(candidate_rank[c] for c in touched_candidates)
            band_w = candidate_band_weight(best_rank, args)
            candidate_touch_score = args.touch_candidate * band_w
        else:
            best_rank = None
            band_w = 0.0
            candidate_touch_score = 0.0

        query_touch_score = args.touch_query if touches_query else 0.0
        fam_w = family_weight(rel_name, args)

        direct_penalty = 0.0
        if fam in {"target_approved", "failed_diagnostic", "compound_disease_other"}:
            direct_penalty = args.direct_shortcut_penalty

        failed_penalty = args.failed_diagnostic_penalty if fam == "failed_diagnostic" else 0.0

        density = (np.log1p(node_degree[h]) + np.log1p(node_degree[t])) / 10.0
        density_score = args.density_weight * float(density)

        top_candidate_bonus = args.top_candidate_bonus if any(c in top_candidate_set for c in touched_candidates) else 0.0

        score = (
            fam_w
            + candidate_touch_score
            + query_touch_score
            + density_score
            + top_candidate_bonus
            - direct_penalty
            - failed_penalty
        )

        scored.append({
            "triple_index": int(idx),
            "triple": [h, r, t],
            "relation": rel_name,
            "relation_family": fam,
            "score": float(score),
            "touches_query": bool(touches_query),
            "touches_candidate": bool(len(touched_candidates) > 0),
            "touched_candidate_best_rank": best_rank,
            "candidate_band_weight": float(band_w),
            "density_score": float(density_score),
            "direct_penalty": float(direct_penalty),
            "failed_diagnostic_penalty": float(failed_penalty),
        })

    return scored


def select_subgraph(row: dict[str, Any], id2relation: dict[int, str], args):
    scored = score_triples(row, id2relation, args)
    source_size = len(scored)

    if source_size == 0:
        return [], scored, {
            "source_subgraph_size": 0,
            "selected_subgraph_size": 0,
            "retain_ratio_actual": 0.0,
            "candidate_coverage_preserved_rate": 1.0,
        }

    target_keep = int(round(source_size * args.retain_ratio))
    target_keep = max(args.min_keep, target_keep)
    target_keep = min(args.max_keep, target_keep, source_size)

    candidate_ids = [int(x) for x in row["rank_entities_id"]]
    selected_idx = set()

    # Coverage pass: keep best edge touching each candidate when available.
    by_candidate = defaultdict(list)
    for item in scored:
        h, r, t = item["triple"]
        if h in candidate_ids:
            by_candidate[h].append(item)
        if t in candidate_ids:
            by_candidate[t].append(item)

    for cid in candidate_ids:
        items = by_candidate.get(cid, [])
        if not items:
            continue
        best = sorted(items, key=lambda x: (-x["score"], x["triple_index"]))[0]
        selected_idx.add(best["triple_index"])
        if len(selected_idx) >= target_keep:
            break

    # Query pass: preserve at least one query-touching edge when available.
    if len(selected_idx) < target_keep:
        query_items = [x for x in scored if x["touches_query"]]
        if query_items:
            best_q = sorted(query_items, key=lambda x: (-x["score"], x["triple_index"]))[0]
            selected_idx.add(best_q["triple_index"])

    # Family pass.
    family_min = {
        "compound_gene": args.family_min_compound_gene,
        "gene_gene": args.family_min_gene_gene,
        "target_approved": args.family_min_target,
        "failed_diagnostic": args.family_min_failed,
    }

    for fam, keep_n in family_min.items():
        fam_items = [x for x in scored if x["relation_family"] == fam]
        fam_items = sorted(fam_items, key=lambda x: (-x["score"], x["triple_index"]))
        for item in fam_items[:keep_n]:
            if len(selected_idx) >= target_keep:
                break
            selected_idx.add(item["triple_index"])

    # Global fill.
    for item in sorted(scored, key=lambda x: (-x["score"], x["triple_index"])):
        if len(selected_idx) >= target_keep:
            break
        selected_idx.add(item["triple_index"])

    selected_scored = [x for x in scored if x["triple_index"] in selected_idx]
    selected_scored = sorted(selected_scored, key=lambda x: x["triple_index"])
    selected_subgraph = [x["triple"] for x in selected_scored]

    source_touched_candidates = set()
    selected_touched_candidates = set()

    for item in scored:
        h, r, t = item["triple"]
        if h in candidate_ids:
            source_touched_candidates.add(h)
        if t in candidate_ids:
            source_touched_candidates.add(t)

    for edge in selected_subgraph:
        h, r, t = edge
        if h in candidate_ids:
            selected_touched_candidates.add(h)
        if t in candidate_ids:
            selected_touched_candidates.add(t)

    coverage_rate = (
        len(source_touched_candidates & selected_touched_candidates) / len(source_touched_candidates)
        if source_touched_candidates else 1.0
    )

    source_family_counts = Counter(x["relation_family"] for x in scored)
    selected_family_counts = Counter(x["relation_family"] for x in selected_scored)

    summary = {
        "source_subgraph_size": int(source_size),
        "selected_subgraph_size": int(len(selected_subgraph)),
        "retain_ratio_actual": float(len(selected_subgraph) / source_size),
        "target_keep": int(target_keep),
        "num_source_touched_candidates": int(len(source_touched_candidates)),
        "num_selected_touched_candidates": int(len(selected_touched_candidates)),
        "candidate_coverage_preserved_rate": float(coverage_rate),
        "avg_selected_score": float(np.mean([x["score"] for x in selected_scored])) if selected_scored else 0.0,
        "source_family_counts": dict(source_family_counts),
        "selected_family_counts": dict(selected_family_counts),
        "num_direct_edges_source": int(sum(
            1 for x in scored
            if x["relation_family"] in {"target_approved", "failed_diagnostic", "compound_disease_other"}
        )),
        "num_direct_edges_selected": int(sum(
            1 for x in selected_scored
            if x["relation_family"] in {"target_approved", "failed_diagnostic", "compound_disease_other"}
        )),
        "num_failed_edges_source": int(sum(1 for x in scored if x["relation_family"] == "failed_diagnostic")),
        "num_failed_edges_selected": int(sum(1 for x in selected_scored if x["relation_family"] == "failed_diagnostic")),
    }

    return selected_subgraph, scored, summary


def audit_rows(rows: list[dict[str, Any]], split: str):
    bad_k = bad_q = bad_e = bad_sg = 0
    leaks = 0
    source_sizes = []
    selected_sizes = []
    coverage = []
    direct_source = []
    direct_selected = []
    failed_source = []
    failed_selected = []
    bad_display = 0

    for row in rows:
        if len(row.get("rank_entities_id", [])) != TOP_K:
            bad_k += 1

        if row.get("input", "").count("[QUERY]") != 1:
            bad_q += 1
        if row.get("input", "").count("[ENTITY]") != TOP_K:
            bad_e += 1

        sg = row.get("subgraph", [])
        if not isinstance(sg, list) or any((not isinstance(x, list) or len(x) != 3) for x in sg):
            bad_sg += 1

        if row.get("rank_entities") and str(row["rank_entities"][0]).startswith("Compound::"):
            bad_display += 1

        if split in {"valid", "test"}:
            gold = tuple(int(x) for x in row["triple_id"])
            if any(tuple(int(y) for y in e) == gold for e in sg):
                leaks += 1

        sm = row.get("subgraph_summary", {})
        source_sizes.append(sm.get("source_subgraph_size", 0))
        selected_sizes.append(len(sg))
        coverage.append(sm.get("candidate_coverage_preserved_rate", 1.0))
        direct_source.append(sm.get("num_direct_edges_source", 0))
        direct_selected.append(sm.get("num_direct_edges_selected", 0))
        failed_source.append(sm.get("num_failed_edges_source", 0))
        failed_selected.append(sm.get("num_failed_edges_selected", 0))

    return {
        "split": split,
        "num_rows": len(rows),
        "bad_candidate_len": int(bad_k),
        "bad_query_placeholder": int(bad_q),
        "bad_entity_placeholder": int(bad_e),
        "bad_subgraph": int(bad_sg),
        "bad_display_rank_entities": int(bad_display),
        "valid_test_exact_leak_count": int(leaks),
        "avg_source_subgraph_size": float(np.mean(source_sizes)) if source_sizes else 0.0,
        "avg_selected_subgraph_size": float(np.mean(selected_sizes)) if selected_sizes else 0.0,
        "min_selected_subgraph_size": int(np.min(selected_sizes)) if selected_sizes else 0,
        "max_selected_subgraph_size": int(np.max(selected_sizes)) if selected_sizes else 0,
        "avg_candidate_coverage_preserved_rate": float(np.mean(coverage)) if coverage else 1.0,
        "avg_direct_edges_source": float(np.mean(direct_source)) if direct_source else 0.0,
        "avg_direct_edges_selected": float(np.mean(direct_selected)) if direct_selected else 0.0,
        "avg_failed_edges_source": float(np.mean(failed_source)) if failed_source else 0.0,
        "avg_failed_edges_selected": float(np.mean(failed_selected)) if failed_selected else 0.0,
        "schema_pass": (
            bad_k == 0 and bad_q == 0 and bad_e == 0
            and bad_sg == 0 and bad_display == 0 and leaks == 0
        ),
    }


def compare_metrics(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = [
        "gold_present_at20",
        "mrr_at20",
        "hits1_at20",
        "hits3_at20",
        "hits10_at20",
        "hits20_at20",
        "rank21_count",
    ]
    for k in keys:
        if abs(float(a[k]) - float(b[k])) > 1e-12:
            return False
    return True


def copy_static_files():
    for name in [
        "entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl",
        "entity2id.json", "id2entity.json", "relation2id.json", "id2relation.json",
        "type_map.json", "entity_embeddings_rgcn.pt",
        "prompt_lexicon.json", "rules.json", "support_schema.json",
        "graph_summary.json", "leak_check.json",
    ]:
        src = INPUT_DIR / name
        if src.exists():
            shutil.copy2(src, E2E_DIR / name)


def write_report(summary: dict[str, Any]):
    path = REPORT_DIR / "day7_repodb_fuzzy_retrieval_rgcn.md"

    lines = []
    lines.append("# repoDB fuzzy retrieval on R-GCN raw-display-control")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append(f"- Input: `{summary['input_dir']}`")
    lines.append(f"- Output: `{summary['output_dir']}`")
    lines.append("")
    lines.append("## Candidate metrics and graph reduction")
    lines.append("")
    lines.append("| Split | Raw MRR | Fuzzy MRR | Preserved | Source graph | Fuzzy graph | Reduction | Coverage | Direct source | Direct selected | Failed source | Failed selected |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for split in ["train", "valid", "test"]:
        raw = summary["raw_metrics"][split]
        fuzzy = summary["fuzzy_metrics"][split]
        au = summary["audit"][split]
        reduction = 1.0 - au["avg_selected_subgraph_size"] / max(1.0, au["avg_source_subgraph_size"])

        lines.append(
            f"| {split} | {raw['mrr_at20']:.6f} | {fuzzy['mrr_at20']:.6f} | "
            f"{summary['metrics_preserved'][split]} | "
            f"{au['avg_source_subgraph_size']:.2f} | {au['avg_selected_subgraph_size']:.2f} | "
            f"{reduction:.3f} | {au['avg_candidate_coverage_preserved_rate']:.3f} | "
            f"{au['avg_direct_edges_source']:.2f} | {au['avg_direct_edges_selected']:.2f} | "
            f"{au['avg_failed_edges_source']:.2f} | {au['avg_failed_edges_selected']:.2f} |"
        )

    lines.append("")
    lines.append("## Schema audit")
    lines.append("")
    lines.append("| Split | Rows | Bad K | Bad prompt Q | Bad prompt E | Bad display | Leaks | Schema pass |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split, au in summary["audit"].items():
        lines.append(
            f"| {split} | {au['num_rows']} | {au['bad_candidate_len']} | "
            f"{au['bad_query_placeholder']} | {au['bad_entity_placeholder']} | "
            f"{au['bad_display_rank_entities']} | {au['valid_test_exact_leak_count']} | "
            f"{au['schema_pass']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Fuzzy retrieval preserves raw R-GCN candidate order.")
    lines.append("- This is the main repoDB graph-efficiency row after Day 6 soft support failed to improve ranking.")
    lines.append("- Day 8 should run E2E for raw-display-control and fuzzy_retrieval_main.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-name", default="fuzzy_retrieval_main_repodb_rgcn")
    parser.add_argument("--retain-ratio", type=float, default=0.55)
    parser.add_argument("--min-keep", type=int, default=45)
    parser.add_argument("--max-keep", type=int, default=60)

    parser.add_argument("--band-top", type=float, default=1.0)
    parser.add_argument("--band-mid", type=float, default=0.6)
    parser.add_argument("--band-tail", type=float, default=0.25)

    parser.add_argument("--touch-candidate", type=float, default=0.35)
    parser.add_argument("--touch-query", type=float, default=0.10)
    parser.add_argument("--top-candidate-bonus", type=float, default=0.05)

    parser.add_argument("--family-target-approved", type=float, default=0.10)
    parser.add_argument("--family-failed-diagnostic", type=float, default=0.02)
    parser.add_argument("--family-compound-gene", type=float, default=0.35)
    parser.add_argument("--family-gene-gene", type=float, default=0.25)
    parser.add_argument("--family-compound-disease-other", type=float, default=0.05)
    parser.add_argument("--family-other", type=float, default=0.05)

    parser.add_argument("--direct-shortcut-penalty", type=float, default=0.40)
    parser.add_argument("--failed-diagnostic-penalty", type=float, default=0.15)
    parser.add_argument("--density-weight", type=float, default=0.05)

    parser.add_argument("--family-min-compound-gene", type=int, default=8)
    parser.add_argument("--family-min-gene-gene", type=int, default=6)
    parser.add_argument("--family-min-target", type=int, default=1)
    parser.add_argument("--family-min-failed", type=int, default=0)

    args = parser.parse_args()

    for p in [FUZZY_DIR, E2E_DIR, RESULT_DIR, REPORT_DIR]:
        mkdir(p)

    if not INPUT_DIR.exists():
        raise FileNotFoundError(INPUT_DIR)

    id2relation = {int(k): v for k, v in read_json(INPUT_DIR / "id2relation.json").items()}

    raw_metrics = {}
    fuzzy_metrics = {}
    metrics_preserved = {}
    audit = {}

    for split in ["train", "valid", "test"]:
        print("=" * 100)
        print("[split]", split)

        rows = read_json(INPUT_DIR / f"{split}.json")
        fuzzy_rows = []

        for row in rows:
            out = dict(row)

            selected_subgraph, triple_score_rows, subgraph_summary = select_subgraph(
                row=out,
                id2relation=id2relation,
                args=args,
            )

            out["original_subgraph"] = out.get("subgraph", [])
            out["selected_subgraph"] = selected_subgraph
            out["subgraph"] = selected_subgraph
            out["triple_score_rows"] = triple_score_rows
            out["subgraph_summary"] = subgraph_summary
            out["selected_source_variant"] = args.variant_name
            out["variant_name"] = args.variant_name

            fuzzy_rows.append(out)

        raw_metrics[split] = compute_metrics(rows)
        fuzzy_metrics[split] = compute_metrics(fuzzy_rows)
        metrics_preserved[split] = compare_metrics(raw_metrics[split], fuzzy_metrics[split])
        audit[split] = audit_rows(fuzzy_rows, split)

        write_json(fuzzy_rows, FUZZY_DIR / f"{split}_fuzzy_retrieval_main.json")
        write_json(fuzzy_rows, E2E_DIR / f"{split}.json")

        print("raw   =", raw_metrics[split])
        print("fuzzy =", fuzzy_metrics[split])
        print("audit =", audit[split])

    copy_static_files()

    decision = "DAY7_REPODB_FUZZY_RETRIEVAL_READY"
    if not all(metrics_preserved.values()):
        decision = "DAY7_REPODB_FUZZY_RETRIEVAL_METRIC_MISMATCH"
    if not all(a["schema_pass"] for a in audit.values()):
        decision = "DAY7_REPODB_FUZZY_RETRIEVAL_SCHEMA_FIX_NEEDED"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_f_repodb",
        "dataset": "repoDB",
        "source": "rgcn_raw_display_control",
        "variant_name": args.variant_name,
        "params": vars(args),
        "input_dir": str(INPUT_DIR),
        "output_dir": str(E2E_DIR),
        "raw_metrics": raw_metrics,
        "fuzzy_metrics": fuzzy_metrics,
        "metrics_preserved": metrics_preserved,
        "audit": audit,
        "notes": [
            "Fuzzy retrieval preserves raw R-GCN candidate order.",
            "Day 6 soft support was diagnostic and not used as the main repoDB row.",
            "Display/canonical fields are preserved for E2E evaluation."
        ],
    }

    write_json(summary, RESULT_DIR / "day7_repodb_fuzzy_retrieval_rgcn_summary.json")
    write_json(summary, FUZZY_DIR / "fuzzy_retrieval_manifest.json")
    write_json(summary, E2E_DIR / "prep_manifest.json")
    write_report(summary)

    print("\n[DONE] Day 7 repoDB fuzzy retrieval")
    print(json.dumps({
        "decision": decision,
        "metrics_preserved": metrics_preserved,
        "audit": audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
