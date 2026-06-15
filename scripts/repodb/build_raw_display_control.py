#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(".")
SETTING_DIR = ROOT / "dataset" / "setting_f_repodb"

RAW_READY_DIR = SETTING_DIR / "softfuse_ready" / "rgcn"
OUT_DIR = SETTING_DIR / "e2e_soft_support_ready" / "rgcn_raw_display_control"

RESULT_DIR = ROOT / "outputs" / "repodb"
REPORT_DIR = ROOT / "outputs" / "repodb" / "reports"

TOP_K = 20
ABSENT_RANK = 21


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def display_name(entity, type_map):
    meta = type_map.get(entity, {})
    return str(meta.get("display_name") or meta.get("raw_name") or entity)


def rebuild_prompt(row, type_map):
    query_display = row.get("query_display") or row.get("query_name") or display_name(row["query_entity"], type_map)

    canonical = row.get("rank_entities_canonical")
    if canonical is None:
        old = list(row.get("rank_entities", []))
        if old and str(old[0]).startswith("Compound::"):
            canonical = old
        else:
            # fallback from canonical entity ids if needed
            id2entity = {int(k): v for k, v in read_json(RAW_READY_DIR / "id2entity.json").items()}
            canonical = [id2entity[int(x)] for x in row["rank_entities_id"]]

    displays = [display_name(x, type_map) for x in canonical]

    row["rank_entities_canonical"] = canonical
    row["candidate_entities_canonical"] = canonical

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

    row["selected_source_variant"] = "rgcn_raw_display_control"
    row["variant_name"] = "rgcn_raw_display_control"
    return row


def compute_metrics(rows):
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


def audit(rows, split):
    bad_k = sum(len(r["rank_entities_id"]) != TOP_K for r in rows)
    bad_q = sum(r["input"].count("[QUERY]") != 1 for r in rows)
    bad_e = sum(r["input"].count("[ENTITY]") != TOP_K for r in rows)
    bad_sg = sum(
        not isinstance(r.get("subgraph"), list)
        or any(not isinstance(x, list) or len(x) != 3 for x in r["subgraph"])
        for r in rows
    )
    leaks = 0
    if split in {"valid", "test"}:
        leaks = sum(any(tuple(e) == tuple(r["triple_id"]) for e in r["subgraph"]) for r in rows)

    return {
        "split": split,
        "num_rows": len(rows),
        "bad_candidate_len": bad_k,
        "bad_query_placeholder": bad_q,
        "bad_entity_placeholder": bad_e,
        "bad_subgraph": bad_sg,
        "valid_test_exact_leak_count": leaks,
        "avg_subgraph_size": float(np.mean([len(r["subgraph"]) for r in rows])),
        "schema_pass": bad_k == 0 and bad_q == 0 and bad_e == 0 and bad_sg == 0 and leaks == 0,
    }


def copy_static():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in [
        "entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl",
        "entity2id.json", "id2entity.json", "relation2id.json", "id2relation.json",
        "type_map.json", "entity_embeddings_rgcn.pt",
        "prompt_lexicon.json", "rules.json", "support_schema.json",
        "graph_summary.json", "leak_check.json",
    ]:
        src = RAW_READY_DIR / name
        if src.exists():
            shutil.copy2(src, OUT_DIR / name)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    type_map = read_json(RAW_READY_DIR / "type_map.json")

    metrics = {}
    audits = {}

    for split in ["train", "valid", "test"]:
        rows = read_json(RAW_READY_DIR / f"{split}.json")
        out_rows = [rebuild_prompt(dict(r), type_map) for r in rows]
        write_json(out_rows, OUT_DIR / f"{split}.json")
        metrics[split] = compute_metrics(out_rows)
        audits[split] = audit(out_rows, split)

    copy_static()

    decision = "DAY6C_REPODB_RGCN_RAW_DISPLAY_CONTROL_READY"
    if not all(a["schema_pass"] for a in audits.values()):
        decision = "DAY6C_REPODB_RGCN_RAW_DISPLAY_CONTROL_NEEDS_FIX"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_f_repodb",
        "dataset": "repoDB",
        "source": "rgcn",
        "variant": "rgcn_raw_display_control",
        "reason": "Day 6 soft support did not improve R-GCN. This package preserves raw R-GCN ranking while fixing display-name fields for E2E evaluation.",
        "metrics": metrics,
        "audit": audits,
        "output_dir": str(OUT_DIR),
        "display_patch": {
            "rank_entities": "display drug names for infer.py text matching",
            "rank_entities_id": "numeric entity IDs for graph embeddings",
            "rank_entities_canonical": "Compound::DBxxxxx IDs for audit",
        },
    }

    write_json(summary, RESULT_DIR / "day6c_repodb_rgcn_raw_display_control_summary.json")
    write_json(summary, OUT_DIR / "prep_manifest.json")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
