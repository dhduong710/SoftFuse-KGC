#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build small-noise robustness variants.

Variants:
- N0_no_noise: exact retrieval_main copy.
- N1/N2/N3 support_score_noise: add small deterministic noise to support_scores,
  re-rank candidates, and rebuild prompt.
- N4/N5/N6 subgraph_edge_dropout_5: drop 5% selected subgraph edges while
  preserving query/candidate coverage as much as possible.

This script does not modify Week24 artifacts.
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

PROTOCOL_PATH = ROOT / "outputs" / "sensitivity" / "protocol" / "sensitivity_manifest.json"

SRC_MAIN = ROOT / "dataset" / "setting_a" / "e2e_infer_ready" / "retrieval_main"
OUT_ROOT = ROOT / "dataset" / "setting_a" / "noise_robustness"
RESULTS_DIR = ROOT / "outputs" / "sensitivity" / "noise_robustness"
REPORTS_DIR = ROOT / "outputs" / "sensitivity" / "reports"

SPLITS = ["train", "valid", "test"]

NOISE_CONFIGS = {
    "N0_no_noise": {
        "kind": "none",
        "seed": None,
        "support_noise_std": 0.0,
        "edge_dropout_rate": 0.0,
    },
    "N1_support_score_noise_seed1": {
        "kind": "support_score_noise",
        "seed": 25051,
        "support_noise_std": 0.03,
        "edge_dropout_rate": 0.0,
    },
    "N2_support_score_noise_seed2": {
        "kind": "support_score_noise",
        "seed": 25052,
        "support_noise_std": 0.03,
        "edge_dropout_rate": 0.0,
    },
    "N3_support_score_noise_seed3": {
        "kind": "support_score_noise",
        "seed": 25053,
        "support_noise_std": 0.03,
        "edge_dropout_rate": 0.0,
    },
    "N4_subgraph_edge_dropout_5_seed1": {
        "kind": "subgraph_edge_dropout",
        "seed": 25054,
        "support_noise_std": 0.0,
        "edge_dropout_rate": 0.05,
    },
    "N5_subgraph_edge_dropout_5_seed2": {
        "kind": "subgraph_edge_dropout",
        "seed": 25055,
        "support_noise_std": 0.0,
        "edge_dropout_rate": 0.05,
    },
    "N6_subgraph_edge_dropout_5_seed3": {
        "kind": "subgraph_edge_dropout",
        "seed": 25056,
        "support_noise_std": 0.0,
        "edge_dropout_rate": 0.05,
    },
}


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


def safe_len(x: Any) -> int:
    return len(x) if isinstance(x, list) else 0


def edge_key(edge: Any) -> Tuple[int, int, int]:
    if isinstance(edge, dict):
        h = edge.get("h", edge.get("head", edge.get("src", edge.get("source"))))
        r = edge.get("r", edge.get("relation", edge.get("rel")))
        t = edge.get("t", edge.get("tail", edge.get("dst", edge.get("target"))))
        return (int(h), int(r), int(t))
    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
        return (int(edge[0]), int(edge[1]), int(edge[2]))
    raise ValueError(f"Cannot parse edge: {edge}")


def edge_touches(edge: Any, node_id: int) -> bool:
    try:
        h, _, t = edge_key(edge)
        return h == int(node_id) or t == int(node_id)
    except Exception:
        return False


def node_ids_in_edges(edges: List[Any]) -> set:
    nodes = set()
    for e in edges or []:
        try:
            h, _, t = edge_key(e)
            nodes.add(h)
            nodes.add(t)
        except Exception:
            continue
    return nodes


def dedup_edges(edges: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for e in edges or []:
        try:
            k = edge_key(e)
        except Exception:
            continue
        if k not in seen:
            seen.add(k)
            out.append(e)
    return out


def norm_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = re.sub(r"\s+", " ", s.strip())
    return s


def get_gold(row: Dict[str, Any]) -> str:
    return norm_text(row.get("gold_entity", row.get("output", "")))


def rank_of_gold(row: Dict[str, Any]) -> int:
    gold = get_gold(row)
    cands = [norm_text(x) for x in row.get("rank_entities", [])]
    for i, c in enumerate(cands):
        if c.lower() == gold.lower():
            return i + 1
    return 21


def rebuild_prompt(row: Dict[str, Any]) -> str:
    """
    Rebuild a DrKGC-style biomedical candidate-constrained prompt
    after candidate order changes.
    """
    query_entity = row["query_entity"]
    rank_entities = row["rank_entities"]
    question = row.get("question_text") or f"What drug is indicated for {query_entity}?"

    answer_options = "(" + ", ".join([f"'{name}'" for name in rank_entities]) + ")"

    refer_parts = [f"'{query_entity}': [QUERY]"]
    for name in rank_entities:
        refer_parts.append(f"'{name}': [ENTITY]")

    refer_str = ", ".join(refer_parts)

    prompt = (
        "You are a biomedical scientist. The task is to predict the answer based on the given question, "
        "and you only need to answer one entity. The answer must be in "
        + answer_options
        + ".\nYou can refer to the entity embeddings: "
        + refer_str
        + ".\n\nQuestion: "
        + question
        + "\nAnswer: "
    )
    return prompt


def reorder_list(values: Any, order: List[int]) -> Any:
    if isinstance(values, list) and len(values) == len(order):
        return [values[i] for i in order]
    return values


def reorder_candidate_aligned_fields(row: Dict[str, Any], order: List[int]) -> Dict[str, Any]:
    r = copy.deepcopy(row)

    aligned_fields = [
        "rank_entities",
        "rank_entities_id",
        "candidate_entities",
        "candidate_entity_ids",
        "support_scores",
        "candidate_support_bands",
        "contra_flags",
        "support_rank_order",
        "candidate_debug_rows",
    ]

    for f in aligned_fields:
        if f in r:
            r[f] = reorder_list(r[f], order)

    # After candidate reordering, recompute rank using rank_entities.
    r["rank"] = rank_of_gold(r)

    # Rebuild prompt so answer options and [ENTITY] order match rank_entities.
    r["input"] = rebuild_prompt(r)

    return r


def add_support_score_noise(row: Dict[str, Any], std: float, seed: int, row_idx: int, split: str) -> Dict[str, Any]:
    r = copy.deepcopy(row)

    scores = r.get("support_scores")
    if not isinstance(scores, list) or len(scores) != len(r.get("rank_entities", [])):
        r["noise_robustness_warning"] = "missing_or_misaligned_support_scores"
        return r

    split_offset = {"train": 0, "valid": 100000, "test": 200000}.get(split, 0)
    rng = random.Random(seed + split_offset + row_idx)

    noisy_scores = []
    noise_values = []
    for s in scores:
        try:
            base = float(s)
        except Exception:
            base = 0.0
        noise = rng.gauss(0.0, std)
        noisy_scores.append(base + noise)
        noise_values.append(noise)

    # Stable sort by noisy score desc, original order as tie-breaker.
    order = sorted(range(len(noisy_scores)), key=lambda i: (-noisy_scores[i], i))

    r = reorder_candidate_aligned_fields(r, order)

    r["support_scores_original"] = scores
    r["support_scores_noise"] = noise_values
    r["support_scores_noisy_before_reorder"] = noisy_scores
    r["support_noise_std"] = std
    r["support_noise_seed"] = seed
    r["noise_robustness_kind"] = "support_score_noise"

    # Keep a post-reorder noisy score aligned with new candidate order.
    r["support_scores_noisy"] = [noisy_scores[i] for i in order]

    return r


def dropout_edges_preserve_coverage(
    edges: List[Any],
    query_id: int,
    candidate_ids: List[int],
    dropout_rate: float,
    seed: int,
    row_idx: int,
    split: str,
) -> List[Any]:
    edges = dedup_edges(edges)
    if not edges:
        return []

    split_offset = {"train": 0, "valid": 100000, "test": 200000}.get(split, 0)
    rng = random.Random(seed + split_offset + row_idx)

    n = len(edges)
    target_keep = max(1, int(round(n * (1.0 - dropout_rate))))

    required_nodes = [int(query_id)] + [int(x) for x in candidate_ids]
    required_nodes = list(dict.fromkeys(required_nodes))

    selected = []
    selected_keys = set()

    # Coverage pass: keep at least one edge touching each required node when possible.
    shuffled_nodes = required_nodes[:]
    rng.shuffle(shuffled_nodes)

    for node_id in shuffled_nodes:
        touching = [e for e in edges if edge_touches(e, node_id) and edge_key(e) not in selected_keys]
        if not touching:
            continue
        e = rng.choice(touching)
        k = edge_key(e)
        selected.append(e)
        selected_keys.add(k)

    # If coverage pass already exceeds target, keep it; coverage is more important than exact 5%.
    remaining = [e for e in edges if edge_key(e) not in selected_keys]
    rng.shuffle(remaining)

    for e in remaining:
        if len(selected) >= target_keep:
            break
        k = edge_key(e)
        selected.append(e)
        selected_keys.add(k)

    return selected


def add_subgraph_dropout(row: Dict[str, Any], dropout_rate: float, seed: int, row_idx: int, split: str) -> Dict[str, Any]:
    r = copy.deepcopy(row)

    edges = r.get("subgraph", [])
    if not isinstance(edges, list):
        edges = r.get("selected_subgraph", [])
    if not isinstance(edges, list):
        edges = []

    selected = dropout_edges_preserve_coverage(
        edges=edges,
        query_id=int(r["query_entity_id"]),
        candidate_ids=[int(x) for x in r.get("rank_entities_id", [])],
        dropout_rate=dropout_rate,
        seed=seed,
        row_idx=row_idx,
        split=split,
    )

    r["subgraph_original_before_noise"] = edges
    r["subgraph"] = selected
    r["selected_subgraph"] = selected

    r["noise_robustness_kind"] = "subgraph_edge_dropout"
    r["edge_dropout_rate"] = dropout_rate
    r["edge_dropout_seed"] = seed
    r["subgraph_summary"] = {
        "variant": "subgraph_edge_dropout",
        "dropout_rate": dropout_rate,
        "source_subgraph_size": len(edges),
        "selected_subgraph_size": len(selected),
        "coverage_preserving": True,
    }

    return r


def annotate(row: Dict[str, Any], variant: str, split: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    r = copy.deepcopy(row)
    r["noise_robustness_variant"] = variant
    r["noise_robustness_config"] = cfg
    r["sensitivity_split"] = split
    r["infer_row_name"] = f"noise_robustness_{variant}"
    return r


def build_variant_rows(rows: List[Dict[str, Any]], split: str, variant: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []

    for idx, row in enumerate(rows):
        r = annotate(row, variant, split, cfg)

        if cfg["kind"] == "none":
            r["noise_robustness_kind"] = "none"

        elif cfg["kind"] == "support_score_noise":
            r = add_support_score_noise(
                r,
                std=float(cfg["support_noise_std"]),
                seed=int(cfg["seed"]),
                row_idx=idx,
                split=split,
            )

        elif cfg["kind"] == "subgraph_edge_dropout":
            r = add_subgraph_dropout(
                r,
                dropout_rate=float(cfg["edge_dropout_rate"]),
                seed=int(cfg["seed"]),
                row_idx=idx,
                split=split,
            )

        else:
            raise ValueError(f"Unknown noise kind: {cfg['kind']}")

        out.append(r)

    return out


def copy_sidecars(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    for p in src_dir.iterdir():
        if p.name in {"train.json", "valid.json", "test.json", "prep_manifest.json"}:
            continue
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)

    for fname in ["entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl"]:
        src = src_dir / fname
        if src.exists() and src.is_file():
            shutil.copy2(src, dst_dir / fname)


def summarize_rows(rows: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
    cand_lens = [safe_len(r.get("rank_entities", [])) for r in rows]
    sub_sizes = [safe_len(r.get("subgraph", [])) for r in rows]
    ranks = [rank_of_gold(r) for r in rows]

    return {
        "split": split,
        "num_rows": len(rows),
        "avg_candidate_len": round(sum(cand_lens) / len(cand_lens), 6) if cand_lens else None,
        "min_candidate_len": min(cand_lens) if cand_lens else None,
        "max_candidate_len": max(cand_lens) if cand_lens else None,
        "avg_subgraph_size": round(sum(sub_sizes) / len(sub_sizes), 6) if sub_sizes else None,
        "min_subgraph_size": min(sub_sizes) if sub_sizes else None,
        "max_subgraph_size": max(sub_sizes) if sub_sizes else None,
        "gold_at20": round(sum(1 for r in ranks if r <= 20) / len(ranks), 6) if ranks else None,
        "rank21_count": sum(1 for r in ranks if r == 21),
    }


def main() -> None:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"Missing sensitivity manifest: {rel(PROTOCOL_PATH)}")

    protocol = load_json(PROTOCOL_PATH)
    if protocol.get("decision") != "SENSITIVITY_MANIFEST_READY":
        raise RuntimeError(f"sensitivity manifest is not READY: {protocol.get('decision')}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    noise_config_manifest = {
        "week": 25,
        "day": 5,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_row": rel(SRC_MAIN),
        "output_root": rel(OUT_ROOT),
        "primary_role": "small-noise robustness, appendix only",
        "configs": NOISE_CONFIGS,
        "notes": [
            "Support-score noise uses Gaussian noise std=0.03 and stable re-ranking.",
            "Subgraph dropout removes approximately 5% of edges while preserving query/candidate coverage where possible.",
            "These variants do not replace Week24 main results.",
        ],
    }

    write_json(noise_config_manifest, RESULTS_DIR / "noise_configs.json")

    manifest = {
        "week": 25,
        "day": 5,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": None,
        "source_row": rel(SRC_MAIN),
        "output_root": rel(OUT_ROOT),
        "variants": {},
        "fatal_errors": [],
    }

    fatal_errors = []

    for variant, cfg in NOISE_CONFIGS.items():
        variant_dir = OUT_ROOT / variant
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)

        copy_sidecars(SRC_MAIN, variant_dir)

        split_summaries = {}

        for split in SPLITS:
            src_path = SRC_MAIN / f"{split}.json"
            rows = load_json(src_path)
            built = build_variant_rows(rows, split, variant, cfg)

            out_path = variant_dir / f"{split}.json"
            write_json(built, out_path)

            split_summaries[split] = summarize_rows(built, split)

        v_manifest = {
            "variant": variant,
            "config": cfg,
            "source_row": rel(SRC_MAIN),
            "splits": split_summaries,
            "may_replace_main_result": False,
        }

        write_json(v_manifest, variant_dir / "manifest.json")

        manifest["variants"][variant] = {
            "dir": rel(variant_dir),
            "manifest": rel(variant_dir / "manifest.json"),
            "config": cfg,
            "splits": split_summaries,
        }

    # Sanity checks.
    for variant in NOISE_CONFIGS:
        for split in SPLITS:
            p = OUT_ROOT / variant / f"{split}.json"
            if not p.exists():
                fatal_errors.append(f"Missing output: {rel(p)}")
                continue

            rows = load_json(p)
            if not isinstance(rows, list) or not rows:
                fatal_errors.append(f"Bad rows: {rel(p)}")
                continue

            if split in {"valid", "test"}:
                bad_cand = sum(1 for r in rows if safe_len(r.get("rank_entities", [])) != 20)
                if bad_cand:
                    fatal_errors.append(f"{rel(p)} has non-20 candidate rows: {bad_cand}")

            bad_sub = sum(1 for r in rows if safe_len(r.get("subgraph", [])) == 0)
            if bad_sub:
                fatal_errors.append(f"{rel(p)} has empty subgraph rows: {bad_sub}")

            bad_prompt = sum(
                1 for r in rows
                if "Question:" not in r.get("input", "") or "Answer:" not in r.get("input", "")
            )
            if bad_prompt:
                fatal_errors.append(f"{rel(p)} has bad prompt rows: {bad_prompt}")

    manifest["fatal_errors"] = fatal_errors
    manifest["decision"] = "NOISE_VARIANTS_READY" if not fatal_errors else "NOISE_VARIANTS_BLOCKED"

    write_json(manifest, RESULTS_DIR / "noise_variant_manifest.json")

    print("=" * 100)
    print("decision =", manifest["decision"])
    print("noise_configs =", rel(RESULTS_DIR / "noise_configs.json"))
    print("variant_manifest =", rel(RESULTS_DIR / "noise_variant_manifest.json"))
    print("output_root =", rel(OUT_ROOT))
    print("=" * 100)

    for variant, info in manifest["variants"].items():
        print(f"[{variant}] kind={info['config']['kind']}")
        for split, s in info["splits"].items():
            print(
                split,
                "rows =", s["num_rows"],
                "Gold@20 =", s["gold_at20"],
                "Rank21 =", s["rank21_count"],
                "avg_graph =", s["avg_subgraph_size"],
            )
        print("-" * 100)

    if fatal_errors:
        print("FATAL ERRORS:")
        for e in fatal_errors:
            print("-", e)


if __name__ == "__main__":
    main()