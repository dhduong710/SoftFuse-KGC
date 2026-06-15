#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build rule sensitivity variants.

Variants:
1. main_rules:
   Exact copy of retrieval_main.

2. no_rules:
   Soft-support candidate order with source/original subgraph.
   This disables the Week15/16 confidence-aware fuzzy rule selection layer.

3. random_rules:
   Retrieval_main candidate order, but subgraph is replaced by a
   coverage-preserving random sample from the source soft_support_raw subgraph
   with the same row-level budget as retrieval_main.

This script does NOT modify Week24 artifacts.
"""

from __future__ import annotations

import copy
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

PROTOCOL_PATH = ROOT / "outputs" / "sensitivity" / "protocol" / "sensitivity_manifest.json"

SRC_ROOT = ROOT / "dataset" / "setting_a" / "e2e_infer_ready"
SRC_MAIN = SRC_ROOT / "retrieval_main"
SRC_SOFT = SRC_ROOT / "soft_support_raw"

OUT_ROOT = ROOT / "dataset" / "setting_a" / "rule_sensitivity"
RESULTS_DIR = ROOT / "outputs" / "sensitivity" / "rule_sensitivity"
REPORTS_DIR = ROOT / "outputs" / "sensitivity" / "reports"

VARIANTS = ["main_rules", "no_rules", "random_rules"]
SPLITS = ["train", "valid", "test"]

RANDOM_SEED = 2502


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def edge_key(edge: Any) -> Tuple[int, int, int]:
    if isinstance(edge, dict):
        h = edge.get("h", edge.get("head", edge.get("src", edge.get("source"))))
        r = edge.get("r", edge.get("relation", edge.get("rel")))
        t = edge.get("t", edge.get("tail", edge.get("dst", edge.get("target"))))
        return (int(h), int(r), int(t))
    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
        return (int(edge[0]), int(edge[1]), int(edge[2]))
    raise ValueError(f"Cannot parse edge: {edge}")


def dedup_edges(edges: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for e in edges:
        try:
            k = edge_key(e)
        except Exception:
            continue
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def edge_touches(edge: Any, node_id: int) -> bool:
    try:
        h, _, t = edge_key(edge)
        return h == int(node_id) or t == int(node_id)
    except Exception:
        return False


def node_ids_in_edges(edges: List[Any]) -> set:
    nodes = set()
    for e in edges:
        try:
            h, _, t = edge_key(e)
            nodes.add(h)
            nodes.add(t)
        except Exception:
            continue
    return nodes


def coverage_preserving_random_sample(
    edges: List[Any],
    required_node_ids: List[int],
    budget: int,
    seed: int,
) -> List[Any]:
    """
    Random negative-control subgraph.
    It tries to keep query/candidate node coverage so GraphEnhancer will not fail,
    but it breaks confidence-aware rule/evidence selection.
    """
    edges = dedup_edges(edges)
    if not edges:
        return []

    budget = max(1, min(int(budget), len(edges)))
    rng = random.Random(seed)

    selected: List[Any] = []
    selected_keys = set()

    required = list(dict.fromkeys(int(x) for x in required_node_ids if x is not None))
    rng.shuffle(required)

    for node_id in required:
        if len(selected) >= budget:
            break
        touching = [e for e in edges if edge_touches(e, node_id) and edge_key(e) not in selected_keys]
        if not touching:
            continue
        e = rng.choice(touching)
        k = edge_key(e)
        selected.append(e)
        selected_keys.add(k)

    remaining = [e for e in edges if edge_key(e) not in selected_keys]
    rng.shuffle(remaining)

    for e in remaining:
        if len(selected) >= budget:
            break
        selected.append(e)
        selected_keys.add(edge_key(e))

    return selected


def safe_len(x: Any) -> int:
    return len(x) if isinstance(x, list) else 0


def extract_subgraph(row: Dict[str, Any]) -> List[Any]:
    if isinstance(row.get("subgraph"), list):
        return row["subgraph"]
    if isinstance(row.get("selected_subgraph"), list):
        return row["selected_subgraph"]
    return []


def validate_alignment(main_rows: List[Dict[str, Any]], soft_rows: List[Dict[str, Any]], split: str) -> None:
    if len(main_rows) != len(soft_rows):
        raise RuntimeError(f"{split}: row count mismatch: main={len(main_rows)} soft={len(soft_rows)}")

    for i, (m, s) in enumerate(zip(main_rows, soft_rows)):
        m_q = m.get("query_entity_id")
        s_q = s.get("query_entity_id")
        m_gold = m.get("gold_entity", m.get("output"))
        s_gold = s.get("gold_entity", s.get("output"))

        if m_q != s_q or m_gold != s_gold:
            raise RuntimeError(
                f"{split}: alignment mismatch at row {i}: "
                f"main_q={m_q}, soft_q={s_q}, main_gold={m_gold}, soft_gold={s_gold}"
            )


def annotate_row(row: Dict[str, Any], variant: str, split: str, source_desc: str) -> Dict[str, Any]:
    out = copy.deepcopy(row)
    out["rule_sensitivity_variant"] = variant
    out["rule_sensitivity_variant_source"] = source_desc
    out["sensitivity_split"] = split
    out["infer_row_name"] = f"rule_sensitivity_{variant}"
    return out


def build_main_rules(split: str, main_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in main_rows:
        r = annotate_row(
            row,
            variant="main_rules",
            split=split,
            source_desc="exact_copy_of_dataset/setting_a/e2e_infer_ready/retrieval_main",
        )
        if "selected_subgraph" in r and isinstance(r["selected_subgraph"], list):
            r["subgraph"] = r["selected_subgraph"]
        out.append(r)
    return out


def build_no_rules(split: str, soft_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in soft_rows:
        r = annotate_row(
            row,
            variant="no_rules",
            split=split,
            source_desc=(
                "soft_support_raw_package_without_confidence_aware_fuzzy_rule_selection"
            ),
        )
        source_subgraph = extract_subgraph(row)
        r["subgraph"] = source_subgraph
        r["selected_subgraph"] = source_subgraph
        r["selected_source_variant"] = "rule_sensitivity_no_rules_soft_support_raw_subgraph"
        r["subgraph_summary"] = {
            "variant": "no_rules",
            "subgraph_size": safe_len(source_subgraph),
            "interpretation": (
                "Candidate order is soft_support_raw; graph package is the pre-fuzzy source subgraph. "
                "This disables the Week15/16 confidence-aware fuzzy rule-selection layer."
            ),
        }
        out.append(r)
    return out


def build_random_rules(
    split: str,
    main_rows: List[Dict[str, Any]],
    soft_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out = []

    for i, (main_row, soft_row) in enumerate(zip(main_rows, soft_rows)):
        r = annotate_row(
            main_row,
            variant="random_rules",
            split=split,
            source_desc=(
                "retrieval_main_candidate_order_with_coverage_preserving_random_subgraph_from_soft_support_raw"
            ),
        )

        main_subgraph = extract_subgraph(main_row)
        source_subgraph = extract_subgraph(soft_row)

        budget = safe_len(main_subgraph)
        if budget <= 0:
            budget = max(1, min(32, safe_len(source_subgraph)))

        required_node_ids = []
        if main_row.get("query_entity_id") is not None:
            required_node_ids.append(int(main_row["query_entity_id"]))
        required_node_ids.extend([int(x) for x in main_row.get("rank_entities_id", [])])

        selected = coverage_preserving_random_sample(
            edges=source_subgraph,
            required_node_ids=required_node_ids,
            budget=budget,
            seed=RANDOM_SEED + i + (0 if split == "train" else 100000 if split == "valid" else 200000),
        )

        r["subgraph"] = selected
        r["selected_subgraph"] = selected
        r["selected_source_variant"] = "rule_sensitivity_random_rules_negative_control"
        r["triple_score_rows"] = []
        r["path_scores"] = []
        r["subgraph_summary"] = {
            "variant": "random_rules",
            "source_subgraph_size": safe_len(source_subgraph),
            "main_budget": budget,
            "selected_subgraph_size": safe_len(selected),
            "random_seed_base": RANDOM_SEED,
            "coverage_preserving": True,
            "interpretation": (
                "Negative control: candidate order is fixed, but confidence-aware rule/evidence "
                "selection is replaced by coverage-preserving random edge selection."
            ),
        }
        out.append(r)

    return out


def copy_sidecars(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    for p in src_dir.iterdir():
        if p.name in {"train.json", "valid.json", "test.json", "prep_manifest.json"}:
            continue
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)

    # Useful for infer pipeline / audit compatibility.
    for fname in ["entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl"]:
        src = src_dir / fname
        if src.exists() and src.is_file():
            shutil.copy2(src, dst_dir / fname)


def summarize_rows(rows: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
    sub_sizes = [safe_len(extract_subgraph(r)) for r in rows]
    cand_lens = [safe_len(r.get("rank_entities", [])) for r in rows]

    return {
        "split": split,
        "num_rows": len(rows),
        "avg_candidate_len": round(sum(cand_lens) / len(cand_lens), 6) if cand_lens else None,
        "min_candidate_len": min(cand_lens) if cand_lens else None,
        "max_candidate_len": max(cand_lens) if cand_lens else None,
        "avg_subgraph_size": round(sum(sub_sizes) / len(sub_sizes), 6) if sub_sizes else None,
        "min_subgraph_size": min(sub_sizes) if sub_sizes else None,
        "max_subgraph_size": max(sub_sizes) if sub_sizes else None,
        "empty_subgraph_rows": sum(1 for x in sub_sizes if x == 0),
    }


def main() -> None:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(
            f"Missing sensitivity manifest: {rel(PROTOCOL_PATH)}. Run scripts/sensitivity/build_sensitivity_manifest.py first."
        )

    protocol = load_json(PROTOCOL_PATH)
    if protocol.get("decision") != "SENSITIVITY_MANIFEST_READY":
        raise RuntimeError(
            f"sensitivity manifest is not READY: {protocol.get('decision')}"
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "week": 25,
        "day": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": None,
        "source_roots": {
            "retrieval_main": rel(SRC_MAIN),
            "soft_support_raw": rel(SRC_SOFT),
        },
        "output_root": rel(OUT_ROOT),
        "variants": {},
        "notes": [
            "main_rules is an exact copy of retrieval_main.",
            "no_rules uses soft_support_raw candidate order and source subgraph; it disables the Week15/16 fuzzy rule-selection layer.",
            "random_rules keeps retrieval_main candidate order but randomizes subgraph selection under the main row budget.",
            "Candidate metrics may be identical by construction; Day 3 E2E checks whether graph package changes affect LLM output.",
        ],
    }

    fatal_errors = []

    for variant in VARIANTS:
        variant_dir = OUT_ROOT / variant
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)

        copy_sidecars(SRC_MAIN, variant_dir)

    split_summaries: Dict[str, Dict[str, Any]] = {v: {} for v in VARIANTS}

    for split in SPLITS:
        main_rows = load_json(SRC_MAIN / f"{split}.json")
        soft_rows = load_json(SRC_SOFT / f"{split}.json")

        validate_alignment(main_rows, soft_rows, split)

        built = {
            "main_rules": build_main_rules(split, main_rows),
            "no_rules": build_no_rules(split, soft_rows),
            "random_rules": build_random_rules(split, main_rows, soft_rows),
        }

        for variant, rows in built.items():
            write_json(rows, OUT_ROOT / variant / f"{split}.json")
            split_summaries[variant][split] = summarize_rows(rows, split)

    for variant in VARIANTS:
        v_manifest = {
            "variant": variant,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_main": rel(SRC_MAIN),
            "source_soft": rel(SRC_SOFT),
            "splits": split_summaries[variant],
            "gold_injection": False,
            "reviewer_safe_policy": "RR=1/rank if rank<=20 else 0",
            "may_replace_main_result": False,
        }

        if variant == "main_rules":
            v_manifest["definition"] = "Exact copy of retrieval_main."
        elif variant == "no_rules":
            v_manifest["definition"] = (
                "Soft-support row without Week15/16 confidence-aware fuzzy rule-selection layer. "
                "Uses source soft_support_raw subgraph."
            )
        elif variant == "random_rules":
            v_manifest["definition"] = (
                "Negative-control graph package. Candidate order is retrieval_main; subgraph is "
                "coverage-preserving random sample from soft_support_raw source subgraph under main budget."
            )

        write_json(v_manifest, OUT_ROOT / variant / "manifest.json")
        manifest["variants"][variant] = {
            "dir": rel(OUT_ROOT / variant),
            "manifest": rel(OUT_ROOT / variant / "manifest.json"),
            "splits": split_summaries[variant],
        }

    # Final sanity checks.
    for variant in VARIANTS:
        for split in SPLITS:
            p = OUT_ROOT / variant / f"{split}.json"
            if not p.exists():
                fatal_errors.append(f"Missing output: {rel(p)}")
                continue
            rows = load_json(p)
            if not isinstance(rows, list) or not rows:
                fatal_errors.append(f"Bad output rows: {rel(p)}")
                continue
            if split in {"valid", "test"}:
                bad_cand = sum(1 for r in rows if safe_len(r.get("rank_entities", [])) != 20)
                if bad_cand:
                    fatal_errors.append(f"{rel(p)} has non-20 candidate rows: {bad_cand}")
            bad_sub = sum(1 for r in rows if safe_len(extract_subgraph(r)) == 0)
            if bad_sub:
                fatal_errors.append(f"{rel(p)} has empty subgraph rows: {bad_sub}")

    manifest["fatal_errors"] = fatal_errors
    manifest["decision"] = (
        "RULE_SENSITIVITY_VARIANTS_READY"
        if not fatal_errors
        else "RULE_SENSITIVITY_VARIANTS_BLOCKED"
    )

    write_json(manifest, RESULTS_DIR / "rule_sensitivity_variant_manifest.json")

    print("=" * 100)
    print("decision =", manifest["decision"])
    print("variant_manifest =", rel(RESULTS_DIR / "rule_sensitivity_variant_manifest.json"))
    print("output_root =", rel(OUT_ROOT))
    print("=" * 100)

    for variant in VARIANTS:
        print(f"[{variant}]")
        for split in SPLITS:
            s = split_summaries[variant][split]
            print(
                split,
                "rows =", s["num_rows"],
                "avg_subgraph =", s["avg_subgraph_size"],
                "empty_subgraph_rows =", s["empty_subgraph_rows"],
            )
        print("-" * 100)

    if fatal_errors:
        print("FATAL ERRORS:")
        for e in fatal_errors:
            print("-", e)


if __name__ == "__main__":
    main()