#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(".")
SETTING_DIR = ROOT / "dataset" / "setting_e_drkg"

SOURCE_DIR = SETTING_DIR / "backbone_raw_source" / "rgcn"
READY_DIR = SETTING_DIR / "softfuse_ready" / "rgcn"
SUPPORT_DIR = SETTING_DIR / "support_features" / "rgcn"

OUT_SOFT_ROOT = SETTING_DIR / "soft_support" / "rgcn_sweep_selected"
OUT_E2E_ROOT = SETTING_DIR / "e2e_soft_support_ready" / "rgcn_sweep_selected"

RESULT_DIR = ROOT / "outputs" / "drkg"
REPORT_DIR = ROOT / "outputs" / "drkg" / "reports"

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


def rank_gold(gold_id: int, candidate_ids: list[int]):
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


def minmax(xs: list[float]) -> list[float]:
    lo = min(xs)
    hi = max(xs)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def rerank_one(raw_row: dict[str, Any], support_row: dict[str, Any], beta: float, raw_weight: float, gamma: float):
    feats = support_row["candidate_feature_rows"]

    support_values = []
    raw_priors = []

    for f in feats:
        e = float(f.get("evidence_positive", 0.0))
        d = float(f.get("direct_shortcut_penalty", 0.0))
        x = float(f.get("contradiction_penalty", 0.0))
        support = e - beta * d - gamma * x
        support_values.append(support)

        old_rank = int(f["original_rank"])
        raw_prior = (TOP_K - old_rank + 1) / TOP_K
        raw_priors.append(raw_prior)

    support_norm = minmax(support_values)

    scored = []
    for f, support, sn, rp in zip(feats, support_values, support_norm, raw_priors):
        final = raw_weight * rp + (1.0 - raw_weight) * sn
        item = dict(f)
        item["sweep_support_score"] = float(support)
        item["sweep_support_norm"] = float(sn)
        item["sweep_raw_prior"] = float(rp)
        item["sweep_final_score"] = float(final)
        scored.append(item)

    scored = sorted(
        scored,
        key=lambda x: (
            -x["sweep_final_score"],
            -x["sweep_support_score"],
            int(x["original_rank"]),
        ),
    )

    new_ids = [int(x["candidate_id"]) for x in scored]
    new_names = [x["candidate_entity"] for x in scored]

    rank, present = rank_gold(int(raw_row["gold_entity_id"]), new_ids)

    out = dict(raw_row)
    out.update({
        "candidate_entities": new_names,
        "candidate_entity_ids": new_ids,
        "rank_entities": new_names,
        "rank_entities_id": new_ids,
        "rank": int(rank),
        "gold_rank_in_top20_or_21": int(rank),
        "gold_in_topk_raw": bool(present),
        "gold_present_top20": bool(present),
        "reviewer_safe_rr_item": float(1.0 / rank) if present and rank <= TOP_K else 0.0,
        "variant_name": f"sweep_beta{beta:.2f}_raw{raw_weight:.2f}",
        "support_scores": [float(x["sweep_final_score"]) for x in scored],
        "candidate_debug_rows": scored,
        "rank_delta_vs_backbone": int(int(raw_row["rank"]) - int(rank)),
        "change_label_vs_backbone": (
            "improved" if int(rank) < int(raw_row["rank"])
            else "worsened" if int(rank) > int(raw_row["rank"])
            else "unchanged"
        ),
    })

    return out


def run_variant(split: str, beta: float, raw_weight: float, gamma: float):
    raw_rows = read_json(SOURCE_DIR / f"{split}_top20_raw.json")
    support_rows = read_json(SUPPORT_DIR / f"{split}_support_features.json")

    if len(raw_rows) != len(support_rows):
        raise RuntimeError(f"row mismatch {split}: raw={len(raw_rows)} support={len(support_rows)}")

    out_rows = [
        rerank_one(raw, sup, beta=beta, raw_weight=raw_weight, gamma=gamma)
        for raw, sup in zip(raw_rows, support_rows)
    ]

    return out_rows, compute_metrics(out_rows)


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


def export_selected(selected: dict[str, Any], gamma: float):
    beta = float(selected["beta"])
    raw_weight = float(selected["raw_weight"])
    variant_name = selected["variant_name"]

    mkdir(OUT_SOFT_ROOT)
    mkdir(OUT_E2E_ROOT)

    for split in ["train", "valid", "test"]:
        rows, metrics = run_variant(split, beta=beta, raw_weight=raw_weight, gamma=gamma)
        write_json(rows, OUT_SOFT_ROOT / f"{split}_top20_soft_support_main.json")

        ready_rows = read_json(READY_DIR / f"{split}.json")
        ready_by_row_index = {r.get("row_index"): r for r in ready_rows}

        e2e_rows = []
        for sr in rows:
            base = ready_by_row_index.get(sr.get("row_index"))
            if base is None:
                raise KeyError(f"missing ready row: {split} row_index={sr.get('row_index')}")

            out = dict(base)
            for k, v in sr.items():
                out[k] = v

            out["subgraph"] = base["subgraph"]
            out["selected_source_variant"] = variant_name
            add_prompt(out)
            e2e_rows.append(out)

        write_json(e2e_rows, OUT_E2E_ROOT / f"{split}.json")

    for name in [
        "entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl",
        "entity2id.json", "id2entity.json", "relation2id.json", "id2relation.json",
        "type_map.json", "entity_embeddings_rgcn.pt",
        "prompt_lexicon.json", "rules.json", "support_schema.json",
        "graph_summary.json", "leak_check.json",
    ]:
        src = READY_DIR / name
        if src.exists():
            shutil.copy2(src, OUT_E2E_ROOT / name)


def audit_e2e():
    out = {}
    for split in ["train", "valid", "test"]:
        rows = read_json(OUT_E2E_ROOT / f"{split}.json")
        bad_k = sum(len(r["rank_entities_id"]) != TOP_K for r in rows)
        bad_q = sum(r["input"].count("[QUERY]") != 1 for r in rows)
        bad_e = sum(r["input"].count("[ENTITY]") != TOP_K for r in rows)
        avg_graph = sum(len(r["subgraph"]) for r in rows) / len(rows)
        out[split] = {
            "num_rows": len(rows),
            "bad_candidate_len": bad_k,
            "bad_query_placeholder": bad_q,
            "bad_entity_placeholder": bad_e,
            "avg_subgraph_size": avg_graph,
            "schema_pass": bad_k == 0 and bad_q == 0 and bad_e == 0,
        }
    return out


def write_report(summary: dict[str, Any]):
    path = REPORT_DIR / "day6b_drkg_soft_support_rgcn_sweep.md"
    lines = []
    lines.append("# DRKG R-GCN soft-support sweep")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append(f"- Selected variant: `{summary['selected_variant']['variant_name']}`")
    lines.append("")
    lines.append("## Raw baseline")
    lines.append("")
    for split, m in summary["raw_metrics"].items():
        lines.append(f"- {split}: MRR={m['mrr_at20']:.6f}, H@10={m['hits10_at20']:.3f}, Top1Dom={m['top1_dominance']:.3f}")
    lines.append("")
    lines.append("## Selected metrics")
    lines.append("")
    for split, m in summary["selected_metrics"].items():
        lines.append(f"- {split}: MRR={m['mrr_at20']:.6f}, H@10={m['hits10_at20']:.3f}, Top1Dom={m['top1_dominance']:.3f}, Rank21={m['rank21_count']}")
    lines.append("")
    lines.append("## Top valid variants")
    lines.append("")
    lines.append("| Variant | beta | raw_weight | Valid MRR | Valid H@10 | Valid Top1Dom | Test MRR | Test H@10 | Test Top1Dom |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in summary["top_valid_variants"][:20]:
        vm = v["metrics"]["valid"]
        tm = v["metrics"]["test"]
        lines.append(
            f"| {v['variant_name']} | {v['beta']:.2f} | {v['raw_weight']:.2f} | "
            f"{vm['mrr_at20']:.6f} | {vm['hits10_at20']:.3f} | {vm['top1_dominance']:.3f} | "
            f"{tm['mrr_at20']:.6f} | {tm['hits10_at20']:.3f} | {tm['top1_dominance']:.3f} |"
        )
    lines.append("")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.35, 0.50])
    parser.add_argument("--raw-weights", nargs="+", type=float, default=[0.35, 0.50, 0.65, 0.80, 0.90])
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--top1-threshold", type=float, default=0.70)
    parser.add_argument("--min-mrr-ratio", type=float, default=0.95)
    args = parser.parse_args()

    mkdir(RESULT_DIR)
    mkdir(REPORT_DIR)

    raw_metrics = {}
    for split in ["train", "valid", "test"]:
        raw_rows = read_json(SOURCE_DIR / f"{split}_top20_raw.json")
        raw_metrics[split] = compute_metrics(raw_rows)

    variants = []
    for beta in args.betas:
        for raw_weight in args.raw_weights:
            item = {
                "beta": beta,
                "raw_weight": raw_weight,
                "variant_name": f"sweep_beta{beta:.2f}_raw{raw_weight:.2f}",
                "metrics": {},
            }
            for split in ["train", "valid", "test"]:
                rows, metrics = run_variant(split, beta=beta, raw_weight=raw_weight, gamma=args.gamma)
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

    balanced_candidates = [
        v for v in variants
        if v["metrics"]["valid"]["top1_dominance"] <= args.top1_threshold
        and v["metrics"]["valid"]["mrr_at20"] >= raw_valid_mrr * args.min_mrr_ratio
    ]

    if balanced_candidates:
        selected = sorted(
            balanced_candidates,
            key=lambda x: (
                x["metrics"]["valid"]["mrr_at20"],
                x["metrics"]["valid"]["hits10_at20"],
                -x["metrics"]["valid"]["top1_dominance"],
            ),
            reverse=True,
        )[0]
        selection_reason = "best_valid_mrr_under_top1_constraint_and_min_mrr_ratio"
    else:
        selected = variants_by_valid[0]
        selection_reason = "best_valid_mrr_no_balanced_variant_met_constraints"

    export_selected(selected, gamma=args.gamma)
    e2e_audit = audit_e2e()

    decision = "DAY6B_DRKG_RGCN_SOFT_SWEEP_READY"
    if selected["metrics"]["valid"]["mrr_at20"] < raw_valid_mrr and selected["metrics"]["valid"]["top1_dominance"] > args.top1_threshold:
        decision = "DAY6B_DRKG_RGCN_SOFT_SWEEP_NO_ACCEPTABLE_VARIANT"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "source": "rgcn",
        "raw_metrics": raw_metrics,
        "sweep_params": {
            "betas": args.betas,
            "raw_weights": args.raw_weights,
            "gamma": args.gamma,
            "top1_threshold": args.top1_threshold,
            "min_mrr_ratio": args.min_mrr_ratio,
        },
        "selection_reason": selection_reason,
        "selected_variant": {
            "variant_name": selected["variant_name"],
            "beta": selected["beta"],
            "raw_weight": selected["raw_weight"],
        },
        "selected_metrics": selected["metrics"],
        "top_valid_variants": variants_by_valid[:30],
        "all_variants": variants,
        "selected_output": {
            "soft_dir": str(OUT_SOFT_ROOT),
            "e2e_dir": str(OUT_E2E_ROOT),
        },
        "e2e_audit": e2e_audit,
    }

    write_json(summary, RESULT_DIR / "day6b_drkg_soft_support_rgcn_sweep_summary.json")
    write_json(summary, OUT_E2E_ROOT / "prep_manifest.json")
    write_report(summary)

    print(json.dumps({
        "decision": decision,
        "selection_reason": selection_reason,
        "selected_variant": summary["selected_variant"],
        "raw_valid": raw_metrics["valid"],
        "raw_test": raw_metrics["test"],
        "selected_valid": selected["metrics"]["valid"],
        "selected_test": selected["metrics"]["test"],
        "e2e_audit": e2e_audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
