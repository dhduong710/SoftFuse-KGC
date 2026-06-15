#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(".")
SETTING_DIR = ROOT / "dataset" / "setting_e_drkg"
RAW_DIR = SETTING_DIR / "raw_inventory"
TASK_DIR = SETTING_DIR / "task_spec"
RESULT_DIR = ROOT / "outputs" / "drkg"
REPORT_DIR = ROOT / "outputs" / "drkg" / "reports"

DEFAULT_TARGET_RELATION = "DRUGBANK::treats::Compound:Disease"
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


def open_text_auto(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def parse_drkg_tsv(path: Path) -> Iterable[tuple[str, str, str]]:
    with open_text_auto(path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            h, r, t = row[0].strip(), row[1].strip(), row[2].strip()
            if not h or not r or not t:
                continue
            if h.lower() in {"head", "h"} and r.lower() in {"relation", "r"}:
                continue
            yield h, r, t


def infer_entity_type(entity: str) -> str:
    if "::" in entity:
        return entity.split("::", 1)[0]
    if ":" in entity:
        return entity.split(":", 1)[0]
    return "UNKNOWN"


def parse_relation_glossary(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}

    out = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row.get("Relation-name", "").strip()
            if not name:
                continue
            out[name] = dict(row)
    return out


def find_drkg_path_from_inventory() -> Path:
    inv_path = RAW_DIR / "raw_inventory.json"
    if not inv_path.exists():
        raise FileNotFoundError(f"Missing Day 1 inventory: {inv_path}")

    inv = read_json(inv_path)
    p = inv["inventory"]["detected_paths"]["drkg_tsv"]
    return Path(p)


def find_relation_glossary_from_inventory() -> Path | None:
    inv = read_json(RAW_DIR / "raw_inventory.json")
    p = inv["inventory"]["detected_paths"].get("relation_glossary_tsv")
    return Path(p) if p else None


def scan_target_relation(drkg_path: Path, target_relation: str, sample_limit: int = 20) -> dict[str, Any]:
    unique_edges = set()
    samples = []

    head_counter = Counter()
    tail_counter = Counter()

    bad_direction = 0
    duplicate_count = 0
    total_seen = 0

    for h, r, t in parse_drkg_tsv(drkg_path):
        if r != target_relation:
            continue

        total_seen += 1

        ht = infer_entity_type(h)
        tt = infer_entity_type(t)
        if ht != "Compound" or tt != "Disease":
            bad_direction += 1
            continue

        edge = (h, r, t)
        if edge in unique_edges:
            duplicate_count += 1
            continue

        unique_edges.add(edge)
        head_counter[h] += 1
        tail_counter[t] += 1

        if len(samples) < sample_limit:
            samples.append({
                "head": h,
                "relation": r,
                "tail": t,
                "head_type": ht,
                "tail_type": tt,
            })

    edges = sorted(unique_edges)

    head_degrees = list(head_counter.values())
    tail_degrees = list(tail_counter.values())

    def hist(vals):
        c = Counter(vals)
        return {str(k): int(v) for k, v in sorted(c.items())[:50]}

    return {
        "target_relation": target_relation,
        "num_rows_seen": total_seen,
        "num_unique_edges": len(edges),
        "num_duplicates_removed": duplicate_count,
        "bad_direction_count": bad_direction,
        "num_unique_compounds": len(head_counter),
        "num_unique_diseases": len(tail_counter),
        "compound_degree_min": min(head_degrees) if head_degrees else 0,
        "compound_degree_max": max(head_degrees) if head_degrees else 0,
        "disease_degree_min": min(tail_degrees) if tail_degrees else 0,
        "disease_degree_max": max(tail_degrees) if tail_degrees else 0,
        "compound_degree_hist_head": hist(head_degrees),
        "disease_degree_hist_head": hist(tail_degrees),
        "num_singleton_compounds": int(sum(1 for x in head_degrees if x == 1)),
        "num_singleton_diseases": int(sum(1 for x in tail_degrees if x == 1)),
        "sample_edges": samples,
        "edges": edges,
    }


def try_coverage_split(
    edges: list[tuple[str, str, str]],
    valid_size: int,
    test_size: int,
    seed: int,
    max_attempts: int = 500,
) -> dict[str, Any]:
    """
    Feasibility only. Day 3 will build the actual split.
    Held-out edge can be removed only if both compound and disease remain in train target edges.
    """
    target_holdout = valid_size + test_size
    unique_edges = sorted(set(edges))

    best_heldout = []
    best_attempt = None

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        shuffled = unique_edges[:]
        rng.shuffle(shuffled)

        train_set = set(unique_edges)
        h_count = Counter(h for h, _, _ in train_set)
        t_count = Counter(t for _, _, t in train_set)

        heldout = []

        for edge in shuffled:
            if len(heldout) >= target_holdout:
                break

            h, r, t = edge
            if h_count[h] <= 1:
                continue
            if t_count[t] <= 1:
                continue

            train_set.remove(edge)
            h_count[h] -= 1
            t_count[t] -= 1
            heldout.append(edge)

        if len(heldout) > len(best_heldout):
            best_heldout = heldout
            best_attempt = attempt

        if len(heldout) >= target_holdout:
            return {
                "requested_valid_size": valid_size,
                "requested_test_size": test_size,
                "requested_holdout": target_holdout,
                "feasible": True,
                "attempt_used": attempt,
                "actual_holdout_possible": len(heldout),
                "projected_train_size": len(unique_edges) - target_holdout,
                "projected_valid_size": valid_size,
                "projected_test_size": test_size,
                "coverage_policy": "valid/test compounds and diseases remain in train target relation",
            }

    return {
        "requested_valid_size": valid_size,
        "requested_test_size": test_size,
        "requested_holdout": target_holdout,
        "feasible": False,
        "best_attempt": best_attempt,
        "best_holdout_found": len(best_heldout),
        "projected_train_size_if_best": len(unique_edges) - len(best_heldout),
        "coverage_policy": "valid/test compounds and diseases remain in train target relation",
    }


def choose_recommended_split(edges: list[tuple[str, str, str]], seed: int) -> dict[str, Any]:
    candidates = [
        (500, 500),
        (400, 400),
        (300, 300),
        (250, 250),
        (200, 200),
        (100, 100),
    ]

    attempts = []
    for valid_size, test_size in candidates:
        res = try_coverage_split(edges, valid_size, test_size, seed=seed)
        attempts.append(res)
        if res["feasible"]:
            return {
                "recommended_valid_size": valid_size,
                "recommended_test_size": test_size,
                "recommended_train_size": len(edges) - valid_size - test_size,
                "attempts": attempts,
                "decision": "SPLIT_SIZE_FEASIBLE",
            }

    return {
        "recommended_valid_size": None,
        "recommended_test_size": None,
        "recommended_train_size": None,
        "attempts": attempts,
        "decision": "NO_CANDIDATE_SPLIT_SIZE_FEASIBLE",
    }


def write_target_edges(edges: list[tuple[str, str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for h, r, t in edges:
            writer.writerow([h, r, t])


def load_day1_cd_candidates() -> list[dict[str, Any]]:
    path = RAW_DIR / "compound_disease_relation_candidates.json"
    if not path.exists():
        return []
    return read_json(path)


def select_auxiliary_cd_relations(cd_candidates: list[dict[str, Any]], target_relation: str) -> list[dict[str, Any]]:
    """
    Keep candidates as protocol metadata. Day 3 can choose graph filtering.
    We do not use them as target.
    """
    aux = []
    for item in cd_candidates:
        rel = item["relation"]
        if rel == target_relation:
            continue

        source = item.get("source_hint", "")
        count = int(item.get("count", 0))

        role = "direct_cd_auxiliary"
        caution = []

        if source == "Hetionet":
            caution.append("overlaps_with_hetionet_source")
        if "contra" in rel.lower():
            role = "possible_contradiction_or_safety"
        if count > 50000:
            caution.append("very_large_relation_use_degree_cap_or_penalty")

        aux.append({
            "relation": rel,
            "direction": item.get("direction"),
            "count": count,
            "num_unique_compounds": item.get("num_unique_compounds"),
            "num_unique_diseases": item.get("num_unique_diseases"),
            "source_hint": source,
            "role": role,
            "caution": caution,
        })

    return aux


def write_report(path: Path, summary: dict[str, Any]) -> None:
    spec = summary["task_spec"]
    feas = summary["target_relation_feasibility"]
    split = summary["recommended_split"]
    protocol = summary["schema_manifest"]

    lines = []
    lines.append("# DRKG task schema selection")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append(f"- Target relation: `{spec['target_relation']}`")
    lines.append(f"- Task: `{spec['task_form']}`")
    lines.append(f"- Candidate universe policy: `{spec['candidate_universe_policy']}`")
    lines.append("")

    lines.append("## Target relation feasibility")
    lines.append("")
    lines.append(f"- Unique target edges: `{feas['num_unique_edges']}`")
    lines.append(f"- Unique compounds: `{feas['num_unique_compounds']}`")
    lines.append(f"- Unique diseases: `{feas['num_unique_diseases']}`")
    lines.append(f"- Singleton compounds: `{feas['num_singleton_compounds']}`")
    lines.append(f"- Singleton diseases: `{feas['num_singleton_diseases']}`")
    lines.append("")

    lines.append("## Recommended split")
    lines.append("")
    lines.append(f"- Decision: `{split['decision']}`")
    lines.append(f"- Train: `{split['recommended_train_size']}`")
    lines.append(f"- Valid: `{split['recommended_valid_size']}`")
    lines.append(f"- Test: `{split['recommended_test_size']}`")
    lines.append("")

    lines.append("## Relation semantics")
    lines.append("")
    lines.append(f"- Data source: `{spec['target_source']}`")
    lines.append(f"- Claim level: `{spec['claim_scope']}`")
    lines.append(f"- Wording: `{spec['recommended_wording']}`")
    lines.append("")

    lines.append("## DRKG large-graph policy")
    lines.append("")
    for k, v in protocol["graph_construction_policy"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    lines.append("## Auxiliary CompoundDisease relations")
    lines.append("")
    lines.append("| Relation | Count | Source | Role | Caution |")
    lines.append("|---|---:|---|---|---|")
    for item in protocol["auxiliary_compound_disease_relations"][:20]:
        caution = ", ".join(item.get("caution", []))
        lines.append(
            f"| `{item['relation']}` | {item['count']} | `{item['source_hint']}` | "
            f"`{item['role']}` | `{caution}` |"
        )
    lines.append("")

    lines.append("## Day 3 next step")
    lines.append("")
    lines.append("Build the actual coverage-safe split and train graph using this selected schema.")
    lines.append("The train graph should not blindly include all 5.87M DRKG triples; use relation filtering and degree caps.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--target-relation", default=DEFAULT_TARGET_RELATION)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    for p in [TASK_DIR, RESULT_DIR, REPORT_DIR]:
        mkdir(p)

    drkg_path = find_drkg_path_from_inventory()
    glossary_path = find_relation_glossary_from_inventory()
    glossary = parse_relation_glossary(glossary_path)

    cd_candidates = load_day1_cd_candidates()

    print("[target]", args.target_relation)
    print("[drkg]", drkg_path)

    target_scan = scan_target_relation(drkg_path, args.target_relation)
    edges = target_scan.pop("edges")

    if target_scan["num_unique_edges"] == 0:
        raise RuntimeError(f"Target relation has zero unique edges: {args.target_relation}")

    split_rec = choose_recommended_split(edges, seed=args.seed)

    target_glossary = glossary.get(args.target_relation, {})
    aux_cd = select_auxiliary_cd_relations(cd_candidates, args.target_relation)

    target_edges_path = TASK_DIR / "target_edges_unique.tsv"
    write_target_edges(edges, target_edges_path)

    target_samples_path = TASK_DIR / "target_relation_samples.json"
    write_json(target_scan["sample_edges"], target_samples_path)

    target_source = args.target_relation.split("::", 1)[0] if "::" in args.target_relation else "UNKNOWN"

    task_spec = {
        "created_at": now_iso(),
        "setting": "setting_e_drkg",
        "dataset": "DRKG",
        "status": "FROZEN_DAY2",
        "task_name": "drkg_drugbank_treats_disease_head_prediction",
        "task_form": "(?, DRUGBANK::treats, disease)",
        "prediction_type": "predicted_head",
        "target_relation": args.target_relation,
        "target_relation_normalized": "drugbank_treats",
        "target_relation_direction": "Compound->Disease",
        "target_source": target_source,
        "query_entity_type": "Disease",
        "missing_entity_type": "Compound",
        "candidate_universe_policy": "train_target_relation_compound_heads",
        "candidate_universe_description": (
            "Candidate compounds are compounds that appear as heads of the selected target relation in the training split. "
            "Coverage-safe split ensures valid/test gold compounds remain in this universe."
        ),
        "top_k": TOP_K,
        "gold_injection": False,
        "absent_gold_rank_sentinel": ABSENT_RANK,
        "claim_scope": "DrugBank treatment-relation prediction within DRKG; not clinical validation.",
        "recommended_wording": "treatment-like drugdisease prediction / DrugBank treats relation",
        "target_edges_path": str(target_edges_path),
        "target_relation_glossary": target_glossary,
    }

    graph_policy = {
        "do_not_use_full_graph_blindly": True,
        "reason": "DRKG has 5.87M triples; full graph is too large/noisy for fast baseline and subgraph retrieval.",
        "target_relation_handling": "Include only train target DRUGBANK::treats edges; remove valid/test target edges.",
        "candidate_universe": "Compounds from train target heads.",
        "initial_train_graph_relation_families": [
            "target train DRUGBANK::treats edges",
            "Compound-Gene relations",
            "Disease-Gene relations",
            "Gene-Gene relations",
            "selected auxiliary Compound-Disease evidence relations with shortcut penalties",
        ],
        "degree_cap_policy_day3": {
            "cap_high_degree_genes": True,
            "max_edges_target_for_graph": "Prefer <= 1.5M edges for initial DRKG Setting E graph.",
            "keep_all_target_train_edges": True,
            "keep_all_candidate_query_target_nodes": True,
            "apply_relation_and_node_degree_filtering": True,
        },
        "leakage_policy": "Exact valid/test target triples must be removed from train graph and subgraphs.",
    }

    schema_manifest = {
        "created_at": now_iso(),
        "setting": "setting_e_drkg",
        "dataset": "DRKG",
        "target_relation": args.target_relation,
        "reviewer_safe_policy": {
            "top_k": TOP_K,
            "gold_injection": False,
            "absent_gold_rank_sentinel": ABSENT_RANK,
            "rr_definition": "RR = 1/rank if rank <= 20 else 0",
        },
        "split_policy": {
            "coverage_safe": True,
            "valid_test_gold_compounds_in_train_candidate_universe": True,
            "valid_test_query_diseases_seen_in_train_target": True,
            "recommended_split": split_rec,
        },
        "graph_construction_policy": graph_policy,
        "auxiliary_compound_disease_relations": aux_cd,
        "baseline_policy": {
            "run_all_structure_baselines_if_compute_allows": [
                "TransE", "DistMult", "ComplEx", "RotatE", "R-GCN", "HRGAT"
            ],
            "source_selection_rule": (
                "Prefer a graph-compatible source such as R-GCN if it has non-trivial Gold@20; "
                "otherwise use best-valid non-degenerate structure model as a diagnostic source."
            ),
        },
    }

    feasibility = {
        "created_at": now_iso(),
        "target_relation": args.target_relation,
        **target_scan,
        "recommended_split": split_rec,
    }

    decision = "DAY2_DRKG_TASK_SCHEMA_READY"
    if split_rec["decision"] != "SPLIT_SIZE_FEASIBLE":
        decision = "DAY2_DRKG_TASK_SCHEMA_NEEDS_SMALLER_SPLIT_OR_NEW_RELATION"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "task_spec": task_spec,
        "target_relation_feasibility": feasibility,
        "recommended_split": split_rec,
        "schema_manifest": schema_manifest,
        "next_step": "Day 3 build actual split + filtered train graph + mappings.",
    }

    write_json(task_spec, TASK_DIR / "task_spec.json")
    write_json(schema_manifest, TASK_DIR / "schema_manifest.json")
    write_json(feasibility, TASK_DIR / "target_relation_feasibility.json")
    write_json(summary, RESULT_DIR / "day2_drkg_task_schema_summary.json")
    write_report(REPORT_DIR / "day2_drkg_task_schema.md", summary)

    print("\n[DONE] Day 2 DRKG task schema")
    print(json.dumps({
        "decision": decision,
        "target_relation": args.target_relation,
        "num_unique_edges": feasibility["num_unique_edges"],
        "num_unique_compounds": feasibility["num_unique_compounds"],
        "num_unique_diseases": feasibility["num_unique_diseases"],
        "recommended_split": split_rec,
        "target_edges_path": str(target_edges_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
