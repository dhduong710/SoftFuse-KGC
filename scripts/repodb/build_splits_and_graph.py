#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(".")
SETTING_DIR = ROOT / "dataset" / "setting_f_repodb"

TASK_DIR = SETTING_DIR / "task_spec"
SPLIT_DIR = SETTING_DIR / "splits"
GRAPH_DIR = SETTING_DIR / "graph"

RESULT_DIR = ROOT / "outputs" / "repodb"
REPORT_DIR = ROOT / "outputs" / "repodb" / "reports"

DRKG_GRAPH_DIR = ROOT / "dataset" / "setting_e_drkg" / "graph"
DRKG_TRAIN_GRAPH = DRKG_GRAPH_DIR / "train_enriched.tsv"

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


def write_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def read_edges_tsv(path: Path) -> list[tuple[str, str, str]]:
    edges = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            h, r, t = row[0].strip(), row[1].strip(), row[2].strip()
            if not h or not r or not t:
                continue
            if h.lower() in {"head", "h"} and r.lower() in {"relation", "r"}:
                continue
            edges.append((h, r, t))
    return edges


def write_edges_tsv(path: Path, edges: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for h, r, t in edges:
            writer.writerow([h, r, t])


def infer_entity_type(entity: str) -> str:
    if entity.startswith("Compound::"):
        return "Compound"
    if entity.startswith("Disease::"):
        return "Disease"
    if entity.startswith("Gene::"):
        return "Gene"
    if "::" in entity:
        return entity.split("::", 1)[0]
    return "UNKNOWN"


def relation_family(h: str, r: str, t: str) -> str:
    if r == TARGET_RELATION:
        return "target_approved"
    if r == FAILED_RELATION:
        return "failed_diagnostic"

    ht = infer_entity_type(h)
    tt = infer_entity_type(t)

    if {ht, tt} == {"Compound", "Gene"}:
        return "compound_gene"
    if ht == "Gene" and tt == "Gene":
        return "gene_gene"
    if {ht, tt} == {"Disease", "Gene"}:
        return "disease_gene"
    if {ht, tt} == {"Compound", "Disease"}:
        return "compound_disease_other"
    return "other"


def make_coverage_safe_split(
    edges: list[tuple[str, str, str]],
    valid_size: int,
    test_size: int,
    seed: int,
    max_attempts: int = 500,
):
    target_holdout = valid_size + test_size
    unique_edges = sorted(set(edges))

    best = []
    best_attempt = None

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        shuffled = unique_edges[:]
        rng.shuffle(shuffled)

        train_set = set(unique_edges)
        h_count = Counter(h for h, _, _ in train_set)
        t_count = Counter(t for _, _, t in train_set)

        heldout = []

        for e in shuffled:
            if len(heldout) >= target_holdout:
                break

            h, r, t = e
            if h_count[h] <= 1:
                continue
            if t_count[t] <= 1:
                continue

            train_set.remove(e)
            h_count[h] -= 1
            t_count[t] -= 1
            heldout.append(e)

        if len(heldout) > len(best):
            best = heldout[:]
            best_attempt = attempt

        if len(heldout) >= target_holdout:
            rng.shuffle(heldout)
            valid_edges = sorted(heldout[:valid_size])
            test_edges = sorted(heldout[valid_size:valid_size + test_size])
            train_edges = sorted(train_set)

            return train_edges, valid_edges, test_edges, {
                "requested_valid_size": valid_size,
                "requested_test_size": test_size,
                "actual_train_size": len(train_edges),
                "actual_valid_size": len(valid_edges),
                "actual_test_size": len(test_edges),
                "attempt_used": attempt,
                "coverage_policy": "valid/test drugs and diseases remain in train approved target relation",
            }

    raise RuntimeError(
        f"Could not build coverage-safe split. "
        f"best_holdout={len(best)}/{target_holdout}, best_attempt={best_attempt}"
    )


def check_coverage(train_edges, valid_edges, test_edges) -> dict[str, Any]:
    train_h = {h for h, _, _ in train_edges}
    train_t = {t for _, _, t in train_edges}

    valid_missing_h = sorted({h for h, _, _ in valid_edges if h not in train_h})
    valid_missing_t = sorted({t for _, _, t in valid_edges if t not in train_t})
    test_missing_h = sorted({h for h, _, _ in test_edges if h not in train_h})
    test_missing_t = sorted({t for _, _, t in test_edges if t not in train_t})

    return {
        "valid_missing_gold_compounds_in_train": valid_missing_h,
        "valid_missing_query_diseases_in_train": valid_missing_t,
        "test_missing_gold_compounds_in_train": test_missing_h,
        "test_missing_query_diseases_in_train": test_missing_t,
        "coverage_pass": (
            len(valid_missing_h) == 0
            and len(valid_missing_t) == 0
            and len(test_missing_h) == 0
            and len(test_missing_t) == 0
        ),
    }


def add_edge(edge_set: set[tuple[str, str, str]], edge_list: list[tuple[str, str, str]], edge: tuple[str, str, str]) -> bool:
    if edge in edge_set:
        return False
    edge_set.add(edge)
    edge_list.append(edge)
    return True


def load_pair_metadata() -> dict[tuple[str, str], dict[str, Any]]:
    meta = {}

    for name in ["approved_pairs_unique.json", "failed_pairs_unique.json"]:
        p = TASK_DIR / name
        if not p.exists():
            continue
        rows = read_json(p)
        for r in rows:
            key = (r["compound_entity"], r["disease_entity"])
            meta[key] = r

    return meta


def build_type_map(
    entities: set[str],
    pair_meta: dict[tuple[str, str], dict[str, Any]],
    train_edges: list[tuple[str, str, str]],
    valid_edges: list[tuple[str, str, str]],
    test_edges: list[tuple[str, str, str]],
    failed_edges: list[tuple[str, str, str]],
) -> dict[str, Any]:
    compound_meta = {}
    disease_meta = {}

    for r in pair_meta.values():
        compound_meta.setdefault(r["compound_entity"], {
            "kind": "Compound",
            "node_id": r["compound_entity"],
            "drugbank_id": r.get("drugbank_id", ""),
            "display_name": r.get("drug_name", r["compound_entity"]),
            "raw_name": r.get("drug_name", r["compound_entity"]),
            "source": "repoDB",
        })
        disease_meta.setdefault(r["disease_entity"], {
            "kind": "Disease",
            "node_id": r["disease_entity"],
            "umls_cui": r.get("umls_cui", ""),
            "display_name": r.get("disease_name", r["disease_entity"]),
            "raw_name": r.get("disease_name", r["disease_entity"]),
            "sem_type": r.get("sem_type", ""),
            "source": "repoDB",
        })

    type_map = {}

    for e in sorted(entities):
        kind = infer_entity_type(e)

        if e in compound_meta:
            type_map[e] = compound_meta[e]
        elif e in disease_meta:
            type_map[e] = disease_meta[e]
        else:
            type_map[e] = {
                "kind": kind,
                "node_id": e,
                "display_name": e,
                "raw_name": e,
                "source": "DRKG" if kind == "Gene" else "unknown",
            }

    return type_map


def build_graph(
    train_edges: list[tuple[str, str, str]],
    valid_edges: list[tuple[str, str, str]],
    test_edges: list[tuple[str, str, str]],
    failed_edges: list[tuple[str, str, str]],
    args,
):
    heldout_target = set(valid_edges) | set(test_edges)

    train_compounds = {h for h, _, _ in train_edges}
    all_task_compounds = {h for h, _, _ in train_edges + valid_edges + test_edges + failed_edges}
    all_task_diseases = {t for _, _, t in train_edges + valid_edges + test_edges + failed_edges}

    kept_edges = []
    kept_set = set()
    kept_by_family = Counter()

    # 1. Keep train approved target edges.
    for e in train_edges:
        if add_edge(kept_set, kept_edges, e):
            kept_by_family["target_approved"] += 1

    # 2. Keep failed-like diagnostic edges, after conflict removal.
    failed_cap_disease = Counter()
    failed_cap_compound = Counter()

    for h, r, t in failed_edges:
        if (h, TARGET_RELATION, t) in heldout_target:
            # Extra conservative: do not include diagnostic edge for a heldout positive pair.
            continue

        if failed_cap_disease[t] >= args.max_failed_per_disease:
            continue
        if failed_cap_compound[h] >= args.max_failed_per_compound:
            continue

        if add_edge(kept_set, kept_edges, (h, r, t)):
            kept_by_family["failed_diagnostic"] += 1
            failed_cap_disease[t] += 1
            failed_cap_compound[h] += 1

    # 3. Reuse DRKG compound-gene evidence for repoDB compounds.
    touched_genes = set()
    compound_gene_count = Counter()

    drkg_graph_found = DRKG_TRAIN_GRAPH.exists()

    if drkg_graph_found:
        print("[graph] scanning DRKG train_enriched for compound-gene evidence")
        with DRKG_TRAIN_GRAPH.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 3:
                    continue
                h, r, t = row[0].strip(), row[1].strip(), row[2].strip()

                fam = relation_family(h, r, t)
                if fam != "compound_gene":
                    continue

                ht = infer_entity_type(h)
                tt = infer_entity_type(t)

                if ht == "Compound" and tt == "Gene":
                    compound, gene = h, t
                elif ht == "Gene" and tt == "Compound":
                    compound, gene = t, h
                else:
                    continue

                if compound not in all_task_compounds:
                    continue
                if compound_gene_count[compound] >= args.max_compound_gene_per_compound:
                    continue

                if add_edge(kept_set, kept_edges, (h, r, t)):
                    kept_by_family["compound_gene"] += 1
                    compound_gene_count[compound] += 1
                    touched_genes.add(gene)

        print("[graph] touched genes =", len(touched_genes))

        # 4. Reuse DRKG gene-gene bridge evidence around touched genes.
        print("[graph] scanning DRKG train_enriched for gene-gene bridge evidence")
        gene_gene_count = Counter()
        gene_gene_total = 0

        with DRKG_TRAIN_GRAPH.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if gene_gene_total >= args.max_gene_gene_total:
                    break

                if len(row) < 3:
                    continue

                h, r, t = row[0].strip(), row[1].strip(), row[2].strip()

                if relation_family(h, r, t) != "gene_gene":
                    continue
                if h not in touched_genes or t not in touched_genes:
                    continue
                if gene_gene_count[h] >= args.max_gene_gene_per_gene:
                    continue
                if gene_gene_count[t] >= args.max_gene_gene_per_gene:
                    continue

                if add_edge(kept_set, kept_edges, (h, r, t)):
                    kept_by_family["gene_gene"] += 1
                    gene_gene_count[h] += 1
                    gene_gene_count[t] += 1
                    gene_gene_total += 1

    entities = set()
    relations = set()

    for h, r, t in kept_edges:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    # Ensure all task entities are in mappings even if no evidence edges.
    entities |= all_task_compounds
    entities |= all_task_diseases

    stats = {
        "drkg_graph_found": drkg_graph_found,
        "num_kept_edges": len(kept_edges),
        "kept_by_family": dict(kept_by_family),
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_train_compounds": len(train_compounds),
        "num_all_task_compounds": len(all_task_compounds),
        "num_all_task_diseases": len(all_task_diseases),
        "num_touched_genes": len(touched_genes),
        "caps": {
            "max_failed_per_disease": args.max_failed_per_disease,
            "max_failed_per_compound": args.max_failed_per_compound,
            "max_compound_gene_per_compound": args.max_compound_gene_per_compound,
            "max_gene_gene_per_gene": args.max_gene_gene_per_gene,
            "max_gene_gene_total": args.max_gene_gene_total,
        },
    }

    return kept_edges, kept_set, entities, relations, stats


def build_mappings(entities: set[str], relations: set[str]):
    entities_sorted = sorted(entities)
    relations_sorted = sorted(relations)

    entity2id = {e: i for i, e in enumerate(entities_sorted)}
    id2entity = {i: e for e, i in entity2id.items()}

    relation2id = {r: i for i, r in enumerate(relations_sorted)}
    id2relation = {i: r for r, i in relation2id.items()}

    return entity2id, id2entity, relation2id, id2relation


def write_graph_ids(
    kept_edges: list[tuple[str, str, str]],
    entity2id: dict[str, int],
    relation2id: dict[str, int],
):
    raw_path = GRAPH_DIR / "train_enriched.tsv"
    ids_path = GRAPH_DIR / "train_enriched_ids.tsv"

    write_edges_tsv(raw_path, kept_edges)

    with ids_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for h, r, t in kept_edges:
            writer.writerow([entity2id[h], relation2id[r], entity2id[t]])

    return {
        "train_enriched_path": str(raw_path),
        "train_enriched_ids_path": str(ids_path),
        "train_enriched_num_edges": len(kept_edges),
    }


def make_target_rows(
    split: str,
    edges: list[tuple[str, str, str]],
    entity2id: dict[str, int],
    relation2id: dict[str, int],
    type_map: dict[str, Any],
    candidate_universe_size: int,
    query_universe_size: int,
):
    rows = []
    rel_id = int(relation2id[TARGET_RELATION])

    for i, (h, r, t) in enumerate(edges):
        h_meta = type_map.get(h, {})
        t_meta = type_map.get(t, {})

        rows.append({
            "split": split,
            "row_index": i,
            "triple": [h, r, t],
            "triple_id": [int(entity2id[h]), rel_id, int(entity2id[t])],
            "type": "predicted_head",
            "relation": r,
            "relation_normalized": "repodb_approved_indication",
            "query_entity": t,
            "query_entity_id": int(entity2id[t]),
            "query_name": t_meta.get("display_name", t),
            "query_umls_cui": t_meta.get("umls_cui", ""),
            "gold_entity": h,
            "gold_entity_id": int(entity2id[h]),
            "gold_name": h_meta.get("display_name", h),
            "gold_drugbank_id": h_meta.get("drugbank_id", ""),
            "candidate_universe": "train_approved_relation_compound_heads",
            "candidate_universe_size": int(candidate_universe_size),
            "query_universe": "approved_relation_umls_diseases",
            "query_universe_size": int(query_universe_size),
            "gold_injection": False,
            "source_dataset": "repoDB",
            "setting": "setting_f_repodb",
        })

    return rows


def leak_check(kept_set, valid_edges, test_edges):
    valid_leaks = [e for e in valid_edges if e in kept_set]
    test_leaks = [e for e in test_edges if e in kept_set]

    return {
        "valid_heldout_size": len(valid_edges),
        "test_heldout_size": len(test_edges),
        "valid_exact_leak_count": len(valid_leaks),
        "test_exact_leak_count": len(test_leaks),
        "exact_leak_count": len(valid_leaks) + len(test_leaks),
        "exact_leak_pass": len(valid_leaks) == 0 and len(test_leaks) == 0,
        "valid_exact_leaks_sample": valid_leaks[:20],
        "test_exact_leaks_sample": test_leaks[:20],
    }


def write_report(summary: dict[str, Any]) -> None:
    path = REPORT_DIR / "day3_repodb_split_graph.md"
    s = summary["split_summary"]
    g = summary["graph_summary"]
    c = summary["checks"]

    lines = []
    lines.append("# repoDB split and evidence graph")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append("- Dataset: `repoDB`")
    lines.append("- Task: `(?, repoDB_approved_indication, disease)`")
    lines.append("")
    lines.append("## Split summary")
    lines.append("")
    lines.append(f"- Train: `{s['train_size']}`")
    lines.append(f"- Valid: `{s['valid_size']}`")
    lines.append(f"- Test: `{s['test_size']}`")
    lines.append(f"- Candidate compounds: `{s['candidate_universe_size']}`")
    lines.append(f"- Query diseases: `{s['query_universe_size']}`")
    lines.append(f"- Coverage pass: `{c['coverage']['coverage_pass']}`")
    lines.append("")
    lines.append("## Graph summary")
    lines.append("")
    lines.append(f"- Train enriched edges: `{g['train_enriched_num_edges']}`")
    lines.append(f"- Entities: `{g['num_entities']}`")
    lines.append(f"- Relations: `{g['num_relations']}`")
    lines.append(f"- graph_num_rels: `{g['graph_num_rels_for_future_training']}`")
    lines.append(f"- Exact leak count: `{c['leak_check']['exact_leak_count']}`")
    lines.append("")
    lines.append("## Kept edges by family")
    lines.append("")
    for k, v in g["kept_by_family"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- repoDB diseases are retained as local UMLS disease nodes.")
    lines.append("- DRKG evidence is reused mainly through mapped DrugBank compounds and compound-gene/gene-gene evidence.")
    lines.append("- Failed-like repoDB pairs are kept as diagnostic evidence after conflict removal.")
    lines.append("- Valid/test approved target triples are removed from the train graph.")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("Day 4 should run structure baselines on this graph and export fixed top-20 candidate rows.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--max-failed-per-disease", type=int, default=80)
    parser.add_argument("--max-failed-per-compound", type=int, default=80)
    parser.add_argument("--max-compound-gene-per-compound", type=int, default=80)
    parser.add_argument("--max-gene-gene-per-gene", type=int, default=40)
    parser.add_argument("--max-gene-gene-total", type=int, default=150000)
    args = parser.parse_args()

    for p in [SPLIT_DIR, GRAPH_DIR, RESULT_DIR, REPORT_DIR]:
        mkdir(p)

    target_edges = sorted(set(read_edges_tsv(TASK_DIR / "target_edges_unique.tsv")))
    failed_edges = sorted(set(read_edges_tsv(TASK_DIR / "failed_diagnostic_edges_unique.tsv")))

    print("[split] target edges =", len(target_edges))
    train_edges, valid_edges, test_edges, split_info = make_coverage_safe_split(
        target_edges,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    coverage = check_coverage(train_edges, valid_edges, test_edges)
    if not coverage["coverage_pass"]:
        raise RuntimeError(f"Coverage failed: {coverage}")

    write_edges_tsv(SPLIT_DIR / "train.tsv", train_edges)
    write_edges_tsv(SPLIT_DIR / "valid.tsv", valid_edges)
    write_edges_tsv(SPLIT_DIR / "test.tsv", test_edges)

    print("[graph] building graph")
    kept_edges, kept_set, entities, relations, graph_stats = build_graph(
        train_edges=train_edges,
        valid_edges=valid_edges,
        test_edges=test_edges,
        failed_edges=failed_edges,
        args=args,
    )

    entity2id, id2entity, relation2id, id2relation = build_mappings(entities, relations)

    pair_meta = load_pair_metadata()
    type_map = build_type_map(
        entities=entities,
        pair_meta=pair_meta,
        train_edges=train_edges,
        valid_edges=valid_edges,
        test_edges=test_edges,
        failed_edges=failed_edges,
    )

    graph_file_stats = write_graph_ids(kept_edges, entity2id, relation2id)

    # Save mappings.
    write_json(entity2id, GRAPH_DIR / "entity2id.json")
    write_json({str(k): v for k, v in id2entity.items()}, GRAPH_DIR / "id2entity.json")
    write_json(relation2id, GRAPH_DIR / "relation2id.json")
    write_json({str(k): v for k, v in id2relation.items()}, GRAPH_DIR / "id2relation.json")
    write_json(type_map, GRAPH_DIR / "type_map.json")

    write_pickle(entity2id, GRAPH_DIR / "entity2id.pkl")
    write_pickle(id2entity, GRAPH_DIR / "id2entity.pkl")
    write_pickle(relation2id, GRAPH_DIR / "relation2id.pkl")
    write_pickle(id2relation, GRAPH_DIR / "id2relation.pkl")

    candidate_compounds = sorted({h for h, _, _ in train_edges})
    query_diseases = sorted({t for _, _, t in train_edges + valid_edges + test_edges})

    candidate_obj = {
        "entity_type": "Compound",
        "candidate_universe_policy": "train_approved_relation_compound_heads",
        "num_candidates": len(candidate_compounds),
        "candidate_entities": candidate_compounds,
        "candidate_entity_ids": [int(entity2id[x]) for x in candidate_compounds],
    }

    query_obj = {
        "entity_type": "Disease",
        "query_universe_policy": "approved_relation_umls_diseases",
        "num_queries": len(query_diseases),
        "query_entities": query_diseases,
        "query_entity_ids": [int(entity2id[x]) for x in query_diseases],
    }

    write_json(candidate_obj, SPLIT_DIR / "candidate_universe_compound.json")
    write_json(query_obj, SPLIT_DIR / "query_universe_disease.json")

    train_rows = make_target_rows(
        "train", train_edges, entity2id, relation2id, type_map,
        len(candidate_compounds), len(query_diseases)
    )
    valid_rows = make_target_rows(
        "valid", valid_edges, entity2id, relation2id, type_map,
        len(candidate_compounds), len(query_diseases)
    )
    test_rows = make_target_rows(
        "test", test_edges, entity2id, relation2id, type_map,
        len(candidate_compounds), len(query_diseases)
    )

    write_json(train_rows, SPLIT_DIR / "train_target_rows.json")
    write_json(valid_rows, SPLIT_DIR / "valid_target_rows.json")
    write_json(test_rows, SPLIT_DIR / "test_target_rows.json")

    lc = leak_check(kept_set, valid_edges, test_edges)

    split_summary = {
        "created_at": now_iso(),
        "setting": "setting_f_repodb",
        "dataset": "repoDB",
        "target_relation": TARGET_RELATION,
        "target_relation_normalized": "repodb_approved_indication",
        "task_form": "(?, repoDB_approved_indication, disease)",
        "prediction_type": "predicted_head",
        "train_size": len(train_edges),
        "valid_size": len(valid_edges),
        "test_size": len(test_edges),
        "split_info": split_info,
        "candidate_universe": "train_approved_relation_compound_heads",
        "candidate_universe_size": len(candidate_compounds),
        "query_universe": "approved_relation_umls_diseases",
        "query_universe_size": len(query_diseases),
        "top_k": TOP_K,
        "gold_injection": False,
        "absent_gold_rank_sentinel": ABSENT_RANK,
    }

    graph_summary = {
        "created_at": now_iso(),
        "setting": "setting_f_repodb",
        "dataset": "repoDB",
        "target_relation": TARGET_RELATION,
        "failed_diagnostic_relation": FAILED_RELATION,
        "train_enriched_path": graph_file_stats["train_enriched_path"],
        "train_enriched_ids_path": graph_file_stats["train_enriched_ids_path"],
        "train_enriched_num_edges": graph_file_stats["train_enriched_num_edges"],
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
        "graph_num_rels_for_future_training": len(relation2id),
        "target_relation_id": int(relation2id[TARGET_RELATION]),
        "failed_relation_id": int(relation2id[FAILED_RELATION]),
        "kept_by_family": graph_stats["kept_by_family"],
        "num_candidate_compounds": len(candidate_compounds),
        "num_query_diseases": len(query_diseases),
        "graph_build_stats": graph_stats,
        "entity_id_range": [0, len(entity2id) - 1],
        "relation_id_range": [0, len(relation2id) - 1],
        "disease_mapping_policy": "repoDB local UMLS disease nodes",
        "drug_evidence_policy": "reuse DRKG compound-gene and gene-gene evidence for mapped Compound::DBxxxxx nodes",
    }

    checks = {
        "coverage": coverage,
        "leak_check": lc,
        "schema_check": {
            "target_relation_in_relation2id": TARGET_RELATION in relation2id,
            "failed_relation_in_relation2id": FAILED_RELATION in relation2id,
            "all_train_entities_mapped": all(h in entity2id and t in entity2id for h, _, t in train_edges),
            "all_valid_entities_mapped": all(h in entity2id and t in entity2id for h, _, t in valid_edges),
            "all_test_entities_mapped": all(h in entity2id and t in entity2id for h, _, t in test_edges),
        },
    }

    decision = "DAY3_REPODB_SPLIT_GRAPH_READY"
    if not coverage["coverage_pass"] or not lc["exact_leak_pass"]:
        decision = "DAY3_REPODB_SPLIT_GRAPH_NEEDS_FIX"
    if not all(checks["schema_check"].values()):
        decision = "DAY3_REPODB_SCHEMA_NEEDS_FIX"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "split_summary": split_summary,
        "graph_summary": graph_summary,
        "checks": checks,
        "important_next_arguments": {
            "graph_num_rels": len(relation2id),
            "target_relation_id": int(relation2id[TARGET_RELATION]),
            "failed_relation_id": int(relation2id[FAILED_RELATION]),
            "candidate_universe_size": len(candidate_compounds),
            "query_universe_size": len(query_diseases),
        },
    }

    write_json(split_summary, SPLIT_DIR / "split_summary.json")
    write_json(graph_summary, GRAPH_DIR / "graph_summary.json")
    write_json(lc, GRAPH_DIR / "leak_check.json")
    write_json(summary, RESULT_DIR / "day3_repodb_split_graph_summary.json")
    write_report(summary)

    print(json.dumps({
        "decision": decision,
        "split": {
            "train": len(train_edges),
            "valid": len(valid_edges),
            "test": len(test_edges),
        },
        "graph": {
            "edges": len(kept_edges),
            "entities": len(entity2id),
            "relations": len(relation2id),
            "graph_num_rels": len(relation2id),
            "target_relation_id": int(relation2id[TARGET_RELATION]),
            "failed_relation_id": int(relation2id[FAILED_RELATION]),
            "kept_by_family": graph_stats["kept_by_family"],
        },
        "checks": checks,
        "important_next_arguments": summary["important_next_arguments"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
