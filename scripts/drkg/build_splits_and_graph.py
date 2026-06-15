#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(".")
SETTING_DIR = ROOT / "dataset" / "setting_e_drkg"
RAW_DIR = SETTING_DIR / "raw_inventory"
TASK_DIR = SETTING_DIR / "task_spec"
SPLIT_DIR = SETTING_DIR / "splits"
GRAPH_DIR = SETTING_DIR / "graph"
RESULT_DIR = ROOT / "outputs" / "drkg"
REPORT_DIR = ROOT / "outputs" / "drkg" / "reports"

TARGET_RELATION = "DRUGBANK::treats::Compound:Disease"
TARGET_RELATION_NORMALIZED = "drugbank_treats"
TOP_K = 20
ABSENT_RANK = 21

DEFAULT_AUX_CD_RELATIONS = [
    "GNBR::T::Compound:Disease",
    "GNBR::Sa::Compound:Disease",
    "GNBR::Pa::Compound:Disease",
    "GNBR::C::Compound:Disease",
    "GNBR::J::Compound:Disease",
    "GNBR::Pr::Compound:Disease",
    "GNBR::Mp::Compound:Disease",
]


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


def open_text_auto(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def parse_tsv_edges(path: Path) -> Iterable[tuple[str, str, str]]:
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


def get_drkg_path() -> Path:
    inv = read_json(RAW_DIR / "raw_inventory.json")
    return Path(inv["inventory"]["detected_paths"]["drkg_tsv"])


def load_target_edges() -> list[tuple[str, str, str]]:
    path = TASK_DIR / "target_edges_unique.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    edges = sorted(set(parse_tsv_edges(path)))
    return edges


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
                "coverage_policy": "valid/test compounds and diseases remain in train target relation",
            }

    raise RuntimeError(
        f"Could not build coverage-safe split. "
        f"best_holdout={len(best)}/{target_holdout}, best_attempt={best_attempt}"
    )


def write_edges_tsv(path: Path, edges: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for h, r, t in edges:
            writer.writerow([h, r, t])


def check_coverage(train_edges, valid_edges, test_edges) -> dict[str, Any]:
    train_h = {h for h, _, _ in train_edges}
    train_t = {t for _, _, t in train_edges}

    def miss(edges):
        missing_h = sorted({h for h, _, _ in edges if h not in train_h})
        missing_t = sorted({t for _, _, t in edges if t not in train_t})
        return missing_h, missing_t

    vh, vt = miss(valid_edges)
    th, tt = miss(test_edges)

    return {
        "valid_missing_gold_compounds_in_train": vh,
        "valid_missing_query_diseases_in_train": vt,
        "test_missing_gold_compounds_in_train": th,
        "test_missing_query_diseases_in_train": tt,
        "coverage_pass": len(vh) == 0 and len(vt) == 0 and len(th) == 0 and len(tt) == 0,
    }


def relation_family(h: str, r: str, t: str, aux_cd_relations: set[str]) -> str | None:
    ht = infer_entity_type(h)
    tt = infer_entity_type(t)

    if r == TARGET_RELATION:
        return "target_treats"

    if r in aux_cd_relations and ht == "Compound" and tt == "Disease":
        return "aux_compound_disease"

    pair = {ht, tt}

    if pair == {"Compound", "Gene"}:
        return "compound_gene"

    if pair == {"Disease", "Gene"}:
        return "disease_gene"

    if ht == "Gene" and tt == "Gene":
        return "gene_gene"

    return None


def add_edge(edge_set: set[tuple[str, str, str]], edge_list: list[tuple[str, str, str]], edge: tuple[str, str, str]) -> bool:
    if edge in edge_set:
        return False
    edge_set.add(edge)
    edge_list.append(edge)
    return True


def build_filtered_graph(
    drkg_path: Path,
    train_edges: list[tuple[str, str, str]],
    valid_edges: list[tuple[str, str, str]],
    test_edges: list[tuple[str, str, str]],
    args,
) -> dict[str, Any]:
    train_target_set = set(train_edges)
    heldout_set = set(valid_edges) | set(test_edges)

    candidate_compounds = {h for h, _, _ in train_edges}
    target_diseases = {t for _, _, t in train_edges} | {t for _, _, t in valid_edges} | {t for _, _, t in test_edges}

    aux_cd_relations = set(args.aux_cd_relations)

    kept_edges: list[tuple[str, str, str]] = []
    kept_set: set[tuple[str, str, str]] = set()
    kept_by_family = Counter()
    skipped_heldout_target = 0
    exact_leak_seen = 0

    compound_gene_count = Counter()
    disease_gene_count = Counter()
    aux_cd_compound_count = Counter()
    aux_cd_disease_count = Counter()
    aux_cd_total = 0

    touched_genes: set[str] = set()

    print("[graph pass1] target + compound-gene + disease-gene + aux compound-disease")

    n_seen = 0
    for h, r, t in parse_tsv_edges(drkg_path):
        n_seen += 1
        edge = (h, r, t)

        if r == TARGET_RELATION and edge in heldout_set:
            skipped_heldout_target += 1
            continue

        if edge in heldout_set:
            exact_leak_seen += 1
            continue

        fam = relation_family(h, r, t, aux_cd_relations)
        if fam is None:
            continue

        if fam == "target_treats":
            if edge in train_target_set:
                if add_edge(kept_set, kept_edges, edge):
                    kept_by_family[fam] += 1
            continue

        ht = infer_entity_type(h)
        tt = infer_entity_type(t)

        if fam == "compound_gene":
            if ht == "Compound":
                compound, gene = h, t
            else:
                compound, gene = t, h

            if compound not in candidate_compounds:
                continue
            if compound_gene_count[compound] >= args.max_compound_gene_per_compound:
                continue

            if add_edge(kept_set, kept_edges, edge):
                kept_by_family[fam] += 1
                compound_gene_count[compound] += 1
                touched_genes.add(gene)
            continue

        if fam == "disease_gene":
            if ht == "Disease":
                disease, gene = h, t
            else:
                disease, gene = t, h

            if disease not in target_diseases:
                continue
            if disease_gene_count[disease] >= args.max_disease_gene_per_disease:
                continue

            if add_edge(kept_set, kept_edges, edge):
                kept_by_family[fam] += 1
                disease_gene_count[disease] += 1
                touched_genes.add(gene)
            continue

        if fam == "aux_compound_disease":
            compound, disease = h, t

            if compound not in candidate_compounds:
                continue
            if disease not in target_diseases:
                continue
            if aux_cd_total >= args.max_aux_cd_total:
                continue
            if aux_cd_compound_count[compound] >= args.max_aux_cd_per_compound:
                continue
            if aux_cd_disease_count[disease] >= args.max_aux_cd_per_disease:
                continue

            if add_edge(kept_set, kept_edges, edge):
                kept_by_family[fam] += 1
                aux_cd_total += 1
                aux_cd_compound_count[compound] += 1
                aux_cd_disease_count[disease] += 1
            continue

        if n_seen % 1_000_000 == 0:
            print(f"  pass1 scanned {n_seen:,}, kept={len(kept_edges):,}, touched_genes={len(touched_genes):,}")

    print(f"[pass1 done] scanned={n_seen:,}, kept={len(kept_edges):,}, touched_genes={len(touched_genes):,}")

    gene_gene_count = Counter()
    gene_gene_total = 0

    print("[graph pass2] gene-gene bridges over touched genes")

    n_seen2 = 0
    for h, r, t in parse_tsv_edges(drkg_path):
        n_seen2 += 1

        if gene_gene_total >= args.max_gene_gene_total:
            break

        ht = infer_entity_type(h)
        tt = infer_entity_type(t)
        if ht != "Gene" or tt != "Gene":
            continue

        if h not in touched_genes or t not in touched_genes:
            continue

        if gene_gene_count[h] >= args.max_gene_gene_per_gene:
            continue
        if gene_gene_count[t] >= args.max_gene_gene_per_gene:
            continue

        edge = (h, r, t)
        if add_edge(kept_set, kept_edges, edge):
            kept_by_family["gene_gene"] += 1
            gene_gene_total += 1
            gene_gene_count[h] += 1
            gene_gene_count[t] += 1

        if n_seen2 % 1_000_000 == 0:
            print(f"  pass2 scanned {n_seen2:,}, gene_gene={gene_gene_total:,}, kept={len(kept_edges):,}")

    print(f"[pass2 done] scanned={n_seen2:,}, gene_gene={gene_gene_total:,}, kept={len(kept_edges):,}")

    if len(kept_edges) > args.max_total_graph_edges:
        print(f"[trim] graph too large: {len(kept_edges):,} > {args.max_total_graph_edges:,}")
        # Keep all target first; trim non-target deterministically by family priority.
        priority = {
            "target_treats": 0,
            "compound_gene": 1,
            "disease_gene": 2,
            "aux_compound_disease": 3,
            "gene_gene": 4,
        }

        def edge_key(e):
            fam = relation_family(e[0], e[1], e[2], aux_cd_relations) or "other"
            return (priority.get(fam, 9), e[1], e[0], e[2])

        kept_edges = sorted(kept_edges, key=edge_key)[:args.max_total_graph_edges]
        kept_set = set(kept_edges)
        kept_by_family = Counter()
        for e in kept_edges:
            fam = relation_family(e[0], e[1], e[2], aux_cd_relations) or "other"
            kept_by_family[fam] += 1

    graph_entities = set()
    graph_relations = set()
    for h, r, t in kept_edges:
        graph_entities.add(h)
        graph_entities.add(t)
        graph_relations.add(r)

    # Ensure all task entities are mapped even if some have no non-target context.
    all_task_entities = set()
    for e in list(train_edges) + list(valid_edges) + list(test_edges):
        all_task_entities.add(e[0])
        all_task_entities.add(e[2])

    graph_entities |= all_task_entities

    return {
        "kept_edges": kept_edges,
        "kept_edge_set": kept_set,
        "graph_entities": graph_entities,
        "graph_relations": graph_relations,
        "candidate_compounds": sorted(candidate_compounds),
        "target_diseases": sorted(target_diseases),
        "touched_genes": sorted(touched_genes),
        "stats": {
            "num_kept_edges": len(kept_edges),
            "kept_by_family": dict(kept_by_family),
            "num_graph_entities": len(graph_entities),
            "num_graph_relations": len(graph_relations),
            "num_candidate_compounds": len(candidate_compounds),
            "num_target_diseases": len(target_diseases),
            "num_touched_genes": len(touched_genes),
            "skipped_heldout_target": skipped_heldout_target,
            "exact_leak_seen": exact_leak_seen,
            "caps": {
                "max_compound_gene_per_compound": args.max_compound_gene_per_compound,
                "max_disease_gene_per_disease": args.max_disease_gene_per_disease,
                "max_aux_cd_per_compound": args.max_aux_cd_per_compound,
                "max_aux_cd_per_disease": args.max_aux_cd_per_disease,
                "max_aux_cd_total": args.max_aux_cd_total,
                "max_gene_gene_per_gene": args.max_gene_gene_per_gene,
                "max_gene_gene_total": args.max_gene_gene_total,
                "max_total_graph_edges": args.max_total_graph_edges,
            },
        },
    }


def build_mappings(graph_entities: set[str], graph_relations: set[str]):
    entities = sorted(graph_entities)
    relations = sorted(graph_relations)

    entity2id = {e: i for i, e in enumerate(entities)}
    id2entity = {i: e for e, i in entity2id.items()}

    relation2id = {r: i for i, r in enumerate(relations)}
    id2relation = {i: r for r, i in relation2id.items()}

    type_map = {
        e: {
            "kind": infer_entity_type(e),
            "node_id": e,
            "raw_name": e,
            "display_name": e,
        }
        for e in entities
    }

    return entity2id, id2entity, relation2id, id2relation, type_map


def make_target_rows(
    split: str,
    edges: list[tuple[str, str, str]],
    entity2id: dict[str, int],
    relation2id: dict[str, int],
    candidate_universe_size: int,
    query_universe_size: int,
):
    rows = []
    for i, (h, r, t) in enumerate(edges):
        rows.append({
            "split": split,
            "row_index": i,
            "triple": [h, r, t],
            "triple_id": [int(entity2id[h]), int(relation2id[r]), int(entity2id[t])],
            "type": "predicted_head",
            "relation": r,
            "relation_normalized": TARGET_RELATION_NORMALIZED,
            "query_entity": t,
            "query_entity_id": int(entity2id[t]),
            "gold_entity": h,
            "gold_entity_id": int(entity2id[h]),
            "candidate_universe": "train_target_relation_compound_heads",
            "candidate_universe_size": int(candidate_universe_size),
            "query_universe": "target_relation_diseases",
            "query_universe_size": int(query_universe_size),
            "gold_injection": False,
            "source_dataset": "DRKG",
            "setting": "setting_e_drkg",
        })
    return rows


def write_graph_files(
    kept_edges: list[tuple[str, str, str]],
    entity2id: dict[str, int],
    relation2id: dict[str, int],
):
    raw_path = GRAPH_DIR / "train_enriched.tsv"
    ids_path = GRAPH_DIR / "train_enriched_ids.tsv"

    with raw_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for h, r, t in kept_edges:
            writer.writerow([h, r, t])

    min_ent, max_ent = 10**18, -1
    min_rel, max_rel = 10**18, -1
    n = 0

    with ids_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for h, r, t in kept_edges:
            hid = int(entity2id[h])
            rid = int(relation2id[r])
            tid = int(entity2id[t])
            writer.writerow([hid, rid, tid])
            min_ent = min(min_ent, hid, tid)
            max_ent = max(max_ent, hid, tid)
            min_rel = min(min_rel, rid)
            max_rel = max(max_rel, rid)
            n += 1

    return {
        "train_enriched_path": str(raw_path),
        "train_enriched_ids_path": str(ids_path),
        "train_enriched_num_edges": n,
        "entity_id_min": int(min_ent) if n else 0,
        "entity_id_max": int(max_ent) if n else 0,
        "relation_id_min": int(min_rel) if n else 0,
        "relation_id_max": int(max_rel) if n else 0,
    }


def leak_check(kept_set, valid_edges, test_edges):
    leaks_valid = sorted([e for e in valid_edges if e in kept_set])
    leaks_test = sorted([e for e in test_edges if e in kept_set])

    return {
        "valid_heldout_size": len(valid_edges),
        "test_heldout_size": len(test_edges),
        "valid_exact_leaks": leaks_valid[:20],
        "test_exact_leaks": leaks_test[:20],
        "valid_exact_leak_count": len(leaks_valid),
        "test_exact_leak_count": len(leaks_test),
        "exact_leak_count": len(leaks_valid) + len(leaks_test),
        "exact_leak_pass": len(leaks_valid) == 0 and len(leaks_test) == 0,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    s = summary["split_summary"]
    g = summary["graph_summary"]
    c = summary["checks"]

    lines = []
    lines.append("# DRKG split and filtered graph")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append(f"- Target relation: `{s['target_relation']}`")
    lines.append(f"- Task: `(? , DRUGBANK::treats, disease)`")
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
    lines.append("## Notes")
    lines.append("")
    lines.append("- Full DRKG graph is not used blindly.")
    lines.append("- Valid/test target DRUGBANK::treats triples are removed.")
    lines.append("- Hetionet CtD/CpD auxiliary relations are not included by default in this Day 3 graph.")
    lines.append("- Day 4 should run structure baselines using this filtered graph and fixed top-20 reviewer-safe protocol.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--valid-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-compound-gene-per-compound", type=int, default=80)
    parser.add_argument("--max-disease-gene-per-disease", type=int, default=120)
    parser.add_argument("--max-aux-cd-per-compound", type=int, default=80)
    parser.add_argument("--max-aux-cd-per-disease", type=int, default=80)
    parser.add_argument("--max-aux-cd-total", type=int, default=120000)
    parser.add_argument("--max-gene-gene-per-gene", type=int, default=40)
    parser.add_argument("--max-gene-gene-total", type=int, default=500000)
    parser.add_argument("--max-total-graph-edges", type=int, default=1500000)
    parser.add_argument("--aux-cd-relations", nargs="+", default=DEFAULT_AUX_CD_RELATIONS)
    args = parser.parse_args()

    for p in [SPLIT_DIR, GRAPH_DIR, RESULT_DIR, REPORT_DIR]:
        mkdir(p)

    drkg_path = get_drkg_path()
    target_edges = load_target_edges()

    print("[split] building coverage-safe split")
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

    print("[graph] building filtered DRKG graph")
    graph_obj = build_filtered_graph(
        drkg_path=drkg_path,
        train_edges=train_edges,
        valid_edges=valid_edges,
        test_edges=test_edges,
        args=args,
    )

    graph_entities = graph_obj["graph_entities"]
    graph_relations = graph_obj["graph_relations"]
    kept_edges = graph_obj["kept_edges"]
    kept_set = graph_obj["kept_edge_set"]

    entity2id, id2entity, relation2id, id2relation, type_map = build_mappings(graph_entities, graph_relations)

    graph_file_stats = write_graph_files(kept_edges, entity2id, relation2id)

    # Save mappings.
    write_pickle(entity2id, GRAPH_DIR / "entity2id.pkl")
    write_pickle(id2entity, GRAPH_DIR / "id2entity.pkl")
    write_pickle(relation2id, GRAPH_DIR / "relation2id.pkl")
    write_pickle(id2relation, GRAPH_DIR / "id2relation.pkl")

    write_json(entity2id, GRAPH_DIR / "entity2id.json")
    write_json({str(k): v for k, v in id2entity.items()}, GRAPH_DIR / "id2entity.json")
    write_json(relation2id, GRAPH_DIR / "relation2id.json")
    write_json({str(k): v for k, v in id2relation.items()}, GRAPH_DIR / "id2relation.json")
    write_json(type_map, GRAPH_DIR / "type_map.json")

    candidate_compounds = sorted({h for h, _, _ in train_edges})
    query_diseases = sorted({t for _, _, t in train_edges} | {t for _, _, t in valid_edges} | {t for _, _, t in test_edges})

    candidate_obj = {
        "entity_type": "Compound",
        "candidate_universe_policy": "train_target_relation_compound_heads",
        "num_candidates": len(candidate_compounds),
        "candidate_entities": candidate_compounds,
        "candidate_entity_ids": [int(entity2id[x]) for x in candidate_compounds],
    }
    query_obj = {
        "entity_type": "Disease",
        "query_universe_policy": "target_relation_diseases",
        "num_queries": len(query_diseases),
        "query_entities": query_diseases,
        "query_entity_ids": [int(entity2id[x]) for x in query_diseases],
    }

    write_json(candidate_obj, SPLIT_DIR / "candidate_universe_compound.json")
    write_json(query_obj, SPLIT_DIR / "query_universe_disease.json")

    train_rows = make_target_rows("train", train_edges, entity2id, relation2id, len(candidate_compounds), len(query_diseases))
    valid_rows = make_target_rows("valid", valid_edges, entity2id, relation2id, len(candidate_compounds), len(query_diseases))
    test_rows = make_target_rows("test", test_edges, entity2id, relation2id, len(candidate_compounds), len(query_diseases))

    write_json(train_rows, SPLIT_DIR / "train_target_rows.json")
    write_json(valid_rows, SPLIT_DIR / "valid_target_rows.json")
    write_json(test_rows, SPLIT_DIR / "test_target_rows.json")

    lc = leak_check(kept_set, valid_edges, test_edges)

    split_summary = {
        "created_at": now_iso(),
        "setting": "setting_e_drkg",
        "dataset": "DRKG",
        "target_relation": TARGET_RELATION,
        "target_relation_normalized": TARGET_RELATION_NORMALIZED,
        "task_form": "(?, DRUGBANK::treats, disease)",
        "prediction_type": "predicted_head",
        "train_size": len(train_edges),
        "valid_size": len(valid_edges),
        "test_size": len(test_edges),
        "split_info": split_info,
        "candidate_universe": "train_target_relation_compound_heads",
        "candidate_universe_size": len(candidate_compounds),
        "query_universe": "target_relation_diseases",
        "query_universe_size": len(query_diseases),
        "top_k": TOP_K,
        "gold_injection": False,
        "absent_gold_rank_sentinel": ABSENT_RANK,
    }

    graph_summary = {
        "created_at": now_iso(),
        "setting": "setting_e_drkg",
        "dataset": "DRKG",
        "target_relation": TARGET_RELATION,
        "train_enriched_path": graph_file_stats["train_enriched_path"],
        "train_enriched_ids_path": graph_file_stats["train_enriched_ids_path"],
        "train_enriched_num_edges": graph_file_stats["train_enriched_num_edges"],
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
        "graph_num_rels_for_future_training": len(relation2id),
        "target_relation_id": int(relation2id[TARGET_RELATION]),
        "kept_by_family": graph_obj["stats"]["kept_by_family"],
        "num_candidate_compounds": graph_obj["stats"]["num_candidate_compounds"],
        "num_target_diseases": graph_obj["stats"]["num_target_diseases"],
        "num_touched_genes": graph_obj["stats"]["num_touched_genes"],
        "graph_build_stats": graph_obj["stats"],
        "id_range_check": {
            "entity_id_min": graph_file_stats["entity_id_min"],
            "entity_id_max": graph_file_stats["entity_id_max"],
            "relation_id_min": graph_file_stats["relation_id_min"],
            "relation_id_max": graph_file_stats["relation_id_max"],
            "entity_id_range_pass": graph_file_stats["entity_id_min"] >= 0 and graph_file_stats["entity_id_max"] < len(entity2id),
            "relation_id_range_pass": graph_file_stats["relation_id_min"] >= 0 and graph_file_stats["relation_id_max"] < len(relation2id),
        },
        "aux_cd_relations_used": args.aux_cd_relations,
    }

    checks = {
        "coverage": coverage,
        "leak_check": lc,
        "schema_check": graph_summary["id_range_check"],
    }

    decision = "DAY3_DRKG_SPLIT_GRAPH_READY"
    if not coverage["coverage_pass"] or not lc["exact_leak_pass"] or not graph_summary["id_range_check"]["entity_id_range_pass"] or not graph_summary["id_range_check"]["relation_id_range_pass"]:
        decision = "DAY3_DRKG_SPLIT_GRAPH_NEEDS_FIX"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "split_summary": split_summary,
        "graph_summary": graph_summary,
        "checks": checks,
        "important_next_arguments": {
            "graph_num_rels": len(relation2id),
            "target_relation_id": int(relation2id[TARGET_RELATION]),
            "candidate_universe_size": len(candidate_compounds),
            "query_universe_size": len(query_diseases),
        },
    }

    write_json(split_summary, SPLIT_DIR / "split_summary.json")
    write_json(graph_summary, GRAPH_DIR / "graph_summary.json")
    write_json(lc, GRAPH_DIR / "leak_check.json")
    write_json(summary, RESULT_DIR / "day3_drkg_split_graph_summary.json")
    write_report(REPORT_DIR / "day3_drkg_split_graph.md", summary)

    print("\n[DONE] Day 3 DRKG split + graph")
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
            "kept_by_family": graph_obj["stats"]["kept_by_family"],
        },
        "checks": checks,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
