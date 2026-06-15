from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set


TARGET_REL = "CtD"
TARGET_REL_NAME = "Compound-treats-Disease"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj) -> None:
    mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_pickle(path: Path, obj) -> None:
    mkdir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def clean_text(x: str) -> str:
    if x is None:
        return ""
    return str(x).replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()


def open_text_auto(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def sniff_delimiter(path: Path) -> str:
    with open_text_auto(path) as f:
        sample = f.read(4096)
    return "\t" if "\t" in sample else ","


def parse_sif_edges(edges_path: Path) -> Iterable[Tuple[str, str, str]]:
    with open_text_auto(edges_path) as f:
        reader = csv.reader(f, delimiter="\t")
        first = next(reader, None)
        if first is None:
            return

        lower = [x.lower() for x in first]
        has_header = (
            len(lower) >= 3
            and ("source" in lower[0] or "subject" in lower[0] or "head" in lower[0])
            and ("target" in lower[2] or "object" in lower[2] or "tail" in lower[2])
        )

        if not has_header and len(first) >= 3:
            yield clean_text(first[0]), clean_text(first[1]), clean_text(first[2])

        for row in reader:
            if len(row) < 3:
                continue
            yield clean_text(row[0]), clean_text(row[1]), clean_text(row[2])


def load_nodes(nodes_path: Path) -> Tuple[Dict[str, Dict], Dict[str, str], Dict[str, List[str]]]:
    delim = sniff_delimiter(nodes_path)

    raw_rows = []
    with open_text_auto(nodes_path) as f:
        reader = csv.DictReader(f, delimiter=delim)
        fields = reader.fieldnames or []

        def pick(row, candidates):
            for c in candidates:
                if c in row:
                    return row[c]
            return ""

        for row in reader:
            node_id = clean_text(pick(row, ["id", "identifier", "node_id"]))
            name = clean_text(pick(row, ["name", "label"]))
            kind = clean_text(pick(row, ["kind", "metanode", "type", "category"]))

            if not node_id:
                raise ValueError(f"Node row missing id: {row}")
            if not name:
                name = node_id
            if not kind:
                kind = node_id.split("::", 1)[0] if "::" in node_id else "UNKNOWN"

            raw_rows.append(
                {
                    "node_id": node_id,
                    "name": name,
                    "kind": kind,
                    "raw": dict(row),
                }
            )

    name_counts = Counter(r["name"] for r in raw_rows)
    display_counts = Counter()
    node_meta = {}
    kind_to_node_ids = {}

    for r in raw_rows:
        node_id = r["node_id"]
        name = r["name"]
        kind = r["kind"]

        # Keep prompts readable, but avoid collisions.
        if name_counts[name] == 1:
            display = name
        else:
            display = f"{name} [{node_id}]"

        display = clean_text(display)
        display_counts[display] += 1
        if display_counts[display] > 1:
            display = f"{display} [{node_id}]"

        node_meta[node_id] = {
            "node_id": node_id,
            "name": name,
            "display_name": display,
            "kind": kind,
        }
        kind_to_node_ids.setdefault(kind, []).append(node_id)

    node_id_to_display = {nid: meta["display_name"] for nid, meta in node_meta.items()}
    return node_meta, node_id_to_display, kind_to_node_ids


def make_coverage_safe_split(
    target_edges: List[Tuple[str, str, str]],
    valid_size: int,
    test_size: int,
    seed: int,
    max_attempts: int = 500,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[str, str, str]], Dict]:
    """
    Remove heldout CtD edges only when both compound and disease still remain in train target edges.
    This preserves Setting-A-like target coverage for valid/test.
    """
    unique_edges = sorted(set(target_edges))
    target_holdout = valid_size + test_size

    best = None
    best_info = None

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        edges = unique_edges[:]
        rng.shuffle(edges)

        train_set = set(unique_edges)
        head_count = Counter(h for h, _, _ in train_set)
        tail_count = Counter(t for _, _, t in train_set)

        heldout = []

        for e in edges:
            if len(heldout) >= target_holdout:
                break
            h, r, t = e

            # Coverage condition after removing this edge.
            if head_count[h] <= 1:
                continue
            if tail_count[t] <= 1:
                continue

            train_set.remove(e)
            head_count[h] -= 1
            tail_count[t] -= 1
            heldout.append(e)

        info = {
            "attempt": attempt,
            "heldout_size": len(heldout),
            "train_size": len(train_set),
        }

        if best is None or len(heldout) > len(best):
            best = heldout[:]
            best_info = info

        if len(heldout) >= target_holdout:
            rng.shuffle(heldout)
            valid_edges = sorted(heldout[:valid_size])
            test_edges = sorted(heldout[valid_size:valid_size + test_size])
            train_edges = sorted(train_set)

            split_info = {
                "requested_valid_size": valid_size,
                "requested_test_size": test_size,
                "actual_valid_size": len(valid_edges),
                "actual_test_size": len(test_edges),
                "actual_train_size": len(train_edges),
                "attempt_used": attempt,
                "coverage_policy": "valid/test compounds and diseases remain in train CtD target edges",
            }
            return train_edges, valid_edges, test_edges, split_info

    raise RuntimeError(
        f"Could not build requested coverage-safe split. "
        f"Best heldout={len(best or [])}/{target_holdout}; best_info={best_info}"
    )


def write_triples_tsv(path: Path, edges: List[Tuple[str, str, str]], node_id_to_display: Dict[str, str]) -> None:
    mkdir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for h, r, t in edges:
            writer.writerow([node_id_to_display[h], r, node_id_to_display[t]])


def write_node_id_tsv(path: Path, edges: List[Tuple[str, str, str]]) -> None:
    mkdir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for h, r, t in edges:
            writer.writerow([h, r, t])


def build_entity_mappings(node_meta: Dict[str, Dict]) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, Dict]]:
    # Stable order: keep Hetionet node order as loaded by Python insertion order.
    entity2id = {}
    id2entity = {}
    type_map = {}

    for idx, (node_id, meta) in enumerate(node_meta.items()):
        display = meta["display_name"]
        entity2id[display] = idx
        id2entity[idx] = display
        type_map[display] = {
            "kind": meta["kind"],
            "node_id": node_id,
            "raw_name": meta["name"],
        }

    return entity2id, id2entity, type_map


def build_relation_mappings(relation_counts: Counter) -> Tuple[Dict[str, int], Dict[int, str]]:
    rels = sorted(relation_counts.keys())
    relation2id = {r: i for i, r in enumerate(rels)}
    id2relation = {i: r for r, i in relation2id.items()}
    return relation2id, id2relation


def make_target_rows(
    split_name: str,
    edges: List[Tuple[str, str, str]],
    node_id_to_display: Dict[str, str],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    candidate_universe_size: int,
    query_universe_size: int,
) -> List[Dict]:
    rows = []
    for i, (h, r, t) in enumerate(edges):
        h_name = node_id_to_display[h]
        t_name = node_id_to_display[t]

        row = {
            "split": split_name,
            "row_index": i,
            "triple": [h_name, r, t_name],
            "triple_id": [entity2id[h_name], relation2id[r], entity2id[t_name]],
            "type": "predicted_head",
            "relation": r,
            "relation_normalized": "compound_treats_disease",
            "query_entity": t_name,
            "query_entity_id": entity2id[t_name],
            "query_entity_node_id": t,
            "gold_entity": h_name,
            "gold_entity_id": entity2id[h_name],
            "gold_entity_node_id": h,
            "candidate_universe": "Compound",
            "candidate_universe_size": candidate_universe_size,
            "query_universe": "Disease",
            "query_universe_size": query_universe_size,
            "gold_injection": False,
            "source_dataset": "Hetionet v1.0",
            "setting": "setting_d_hetionet",
        }
        rows.append(row)
    return rows


def check_coverage(train_edges, valid_edges, test_edges) -> Dict:
    train_h = set(h for h, _, _ in train_edges)
    train_t = set(t for _, _, t in train_edges)

    def miss(edges):
        miss_h = sorted({h for h, _, _ in edges if h not in train_h})
        miss_t = sorted({t for _, _, t in edges if t not in train_t})
        return miss_h, miss_t

    valid_miss_h, valid_miss_t = miss(valid_edges)
    test_miss_h, test_miss_t = miss(test_edges)

    return {
        "valid_missing_gold_compounds_in_train_target": valid_miss_h,
        "valid_missing_query_diseases_in_train_target": valid_miss_t,
        "test_missing_gold_compounds_in_train_target": test_miss_h,
        "test_missing_query_diseases_in_train_target": test_miss_t,
        "coverage_pass": (
            len(valid_miss_h) == 0
            and len(valid_miss_t) == 0
            and len(test_miss_h) == 0
            and len(test_miss_t) == 0
        ),
    }


def write_markdown_report(path: Path, summary: Dict) -> None:
    lines = []
    lines.append("# Hetionet split, graph, and mappings\n")
    lines.append(f"- Created at: `{now_iso()}`")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append("")
    lines.append("## Split summary\n")
    s = summary["split_summary"]
    for k in [
        "target_relation",
        "num_target_edges_total",
        "train_size",
        "valid_size",
        "test_size",
        "num_train_compounds",
        "num_valid_compounds",
        "num_test_compounds",
        "num_train_diseases",
        "num_valid_diseases",
        "num_test_diseases",
    ]:
        lines.append(f"- {k}: `{s.get(k)}`")
    lines.append("")
    lines.append("## Graph summary\n")
    g = summary["graph_summary"]
    for k in [
        "train_enriched_num_edges",
        "num_entities",
        "num_relations",
        "target_train_edges_in_graph",
        "heldout_target_edges_removed",
    ]:
        lines.append(f"- {k}: `{g.get(k)}`")
    lines.append("")
    lines.append("## Checks\n")
    c = summary["checks"]
    lines.append(f"- coverage_pass: `{c['coverage']['coverage_pass']}`")
    lines.append(f"- exact_leak_count: `{c['leak_check']['exact_leak_count']}`")
    lines.append(f"- relation_id_range_pass: `{c['schema_check']['relation_id_range_pass']}`")
    lines.append(f"- entity_id_range_pass: `{c['schema_check']['entity_id_range_pass']}`")
    lines.append("")
    lines.append("## Notes\n")
    lines.append("- Target task is `(? , CtD, disease)`, predicted head.")
    lines.append("- Valid/test target CtD triples are removed from the train enriched graph.")
    lines.append("- CpD and other non-CtD edges remain as auxiliary graph evidence.")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--valid-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    repo = Path(args.root).resolve()
    setting_dir = repo / "dataset" / "setting_d_hetionet"
    raw_dir = setting_dir / "raw_inventory"
    split_dir = mkdir(setting_dir / "splits")
    graph_dir = mkdir(setting_dir / "graph")
    report_dir = mkdir(repo / "outputs" / "hetionet" / "reports")
    result_dir = mkdir(repo / "outputs" / "hetionet")

    inventory_path = raw_dir / "raw_inventory.json"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing Day 1 inventory: {inventory_path}")

    inventory = json.load(open(inventory_path, encoding="utf-8"))
    nodes_path = Path(inventory["detected_paths"]["nodes_path"])
    edges_path = Path(inventory["detected_paths"]["edges_path"])

    print(f"[load] nodes: {nodes_path}")
    node_meta, node_id_to_display, kind_to_node_ids = load_nodes(nodes_path)

    compound_node_ids = sorted(kind_to_node_ids.get("Compound", []))
    disease_node_ids = sorted(kind_to_node_ids.get("Disease", []))

    print(f"[nodes] total={len(node_meta)} compounds={len(compound_node_ids)} diseases={len(disease_node_ids)}")

    print(f"[scan] target relation {TARGET_REL} from edges")
    target_edges = []
    all_edge_count = 0
    raw_relation_counts = Counter()

    for h, r, t in parse_sif_edges(edges_path):
        all_edge_count += 1
        raw_relation_counts[r] += 1
        if r == TARGET_REL:
            target_edges.append((h, r, t))

    target_edges = sorted(set(target_edges))
    print(f"[target] unique {TARGET_REL} edges = {len(target_edges)}")

    train_edges, valid_edges, test_edges, split_info = make_coverage_safe_split(
        target_edges=target_edges,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    heldout_set = set(valid_edges) | set(test_edges)
    train_target_set = set(train_edges)

    coverage = check_coverage(train_edges, valid_edges, test_edges)
    if not coverage["coverage_pass"]:
        raise RuntimeError(f"Coverage check failed: {coverage}")

    print(f"[split] train={len(train_edges)} valid={len(valid_edges)} test={len(test_edges)}")
    print("[coverage] pass")

    write_triples_tsv(split_dir / "train.tsv", train_edges, node_id_to_display)
    write_triples_tsv(split_dir / "valid.tsv", valid_edges, node_id_to_display)
    write_triples_tsv(split_dir / "test.tsv", test_edges, node_id_to_display)

    write_node_id_tsv(split_dir / "train_node_ids.tsv", train_edges)
    write_node_id_tsv(split_dir / "valid_node_ids.tsv", valid_edges)
    write_node_id_tsv(split_dir / "test_node_ids.tsv", test_edges)

    # Entity mappings from all Hetionet nodes.
    entity2id, id2entity, type_map = build_entity_mappings(node_meta)

    unknown_node_ids = set()
    relation_counts_train_graph = Counter()
    train_enriched_edges = 0
    target_train_edges_in_graph = 0
    heldout_removed = 0
    exact_leak_count = 0

    train_enriched_path = graph_dir / "train_enriched.tsv"

    print(f"[write] train enriched graph: {train_enriched_path}")
    with train_enriched_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")

        for h, r, t in parse_sif_edges(edges_path):
            edge = (h, r, t)

            if h not in node_id_to_display:
                unknown_node_ids.add(h)
                continue
            if t not in node_id_to_display:
                unknown_node_ids.add(t)
                continue

            if r == TARGET_REL and edge in heldout_set:
                heldout_removed += 1
                continue

            if r == TARGET_REL and edge in train_target_set:
                target_train_edges_in_graph += 1

            if edge in heldout_set:
                exact_leak_count += 1

            h_name = node_id_to_display[h]
            t_name = node_id_to_display[t]

            writer.writerow([h_name, r, t_name])
            train_enriched_edges += 1
            relation_counts_train_graph[r] += 1

    if unknown_node_ids:
        raise RuntimeError(f"Unknown node IDs found in edges: {list(sorted(unknown_node_ids))[:20]}")

    relation2id, id2relation = build_relation_mappings(relation_counts_train_graph)

    # Now write ID graph after relation mappings exist.
    train_enriched_ids_path = graph_dir / "train_enriched_ids.tsv"
    print(f"[write] train enriched ID graph: {train_enriched_ids_path}")

    min_rel_id = 10**9
    max_rel_id = -1
    min_ent_id = 10**9
    max_ent_id = -1
    id_graph_edges = 0

    with train_enriched_ids_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")

        for h, r, t in parse_sif_edges(edges_path):
            edge = (h, r, t)
            if r == TARGET_REL and edge in heldout_set:
                continue

            h_name = node_id_to_display[h]
            t_name = node_id_to_display[t]

            hid = entity2id[h_name]
            rid = relation2id[r]
            tid = entity2id[t_name]

            writer.writerow([hid, rid, tid])
            id_graph_edges += 1

            min_rel_id = min(min_rel_id, rid)
            max_rel_id = max(max_rel_id, rid)
            min_ent_id = min(min_ent_id, hid, tid)
            max_ent_id = max(max_ent_id, hid, tid)

    # Save mappings.
    write_pickle(graph_dir / "entity2id.pkl", entity2id)
    write_pickle(graph_dir / "id2entity.pkl", id2entity)
    write_pickle(graph_dir / "relation2id.pkl", relation2id)
    write_pickle(graph_dir / "id2relation.pkl", id2relation)

    write_json(graph_dir / "entity2id.json", entity2id)
    write_json(graph_dir / "id2entity.json", {str(k): v for k, v in id2entity.items()})
    write_json(graph_dir / "relation2id.json", relation2id)
    write_json(graph_dir / "id2relation.json", {str(k): v for k, v in id2relation.items()})
    write_json(graph_dir / "type_map.json", type_map)
    write_json(graph_dir / "node_metadata.json", node_meta)

    compound_entities = [node_id_to_display[nid] for nid in compound_node_ids]
    disease_entities = [node_id_to_display[nid] for nid in disease_node_ids]

    write_json(
        split_dir / "candidate_universe_compound.json",
        {
            "entity_type": "Compound",
            "num_candidates": len(compound_entities),
            "candidate_entities": compound_entities,
            "candidate_entity_ids": [entity2id[x] for x in compound_entities],
            "candidate_node_ids": compound_node_ids,
        },
    )
    write_json(
        split_dir / "query_universe_disease.json",
        {
            "entity_type": "Disease",
            "num_queries": len(disease_entities),
            "query_entities": disease_entities,
            "query_entity_ids": [entity2id[x] for x in disease_entities],
            "query_node_ids": disease_node_ids,
        },
    )

    train_rows = make_target_rows(
        "train", train_edges, node_id_to_display, entity2id, relation2id,
        len(compound_entities), len(disease_entities)
    )
    valid_rows = make_target_rows(
        "valid", valid_edges, node_id_to_display, entity2id, relation2id,
        len(compound_entities), len(disease_entities)
    )
    test_rows = make_target_rows(
        "test", test_edges, node_id_to_display, entity2id, relation2id,
        len(compound_entities), len(disease_entities)
    )

    write_json(split_dir / "train_target_rows.json", train_rows)
    write_json(split_dir / "valid_target_rows.json", valid_rows)
    write_json(split_dir / "test_target_rows.json", test_rows)

    leak_check = {
        "heldout_valid_size": len(valid_edges),
        "heldout_test_size": len(test_edges),
        "heldout_total": len(heldout_set),
        "heldout_target_edges_removed": heldout_removed,
        "exact_leak_count": exact_leak_count,
        "exact_leak_pass": exact_leak_count == 0 and heldout_removed == len(heldout_set),
        "note": "Leak check means valid/test CtD target triples are absent from train_enriched.tsv.",
    }

    schema_check = {
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
        "entity_id_min": min_ent_id,
        "entity_id_max": max_ent_id,
        "relation_id_min": min_rel_id,
        "relation_id_max": max_rel_id,
        "entity_id_range_pass": min_ent_id >= 0 and max_ent_id < len(entity2id),
        "relation_id_range_pass": min_rel_id >= 0 and max_rel_id < len(relation2id),
        "target_relation_id": relation2id[TARGET_REL],
        "graph_num_rels_for_future_training": len(relation2id),
    }

    split_summary = {
        "created_at": now_iso(),
        "setting": "setting_d_hetionet",
        "dataset": "Hetionet v1.0",
        "target_relation": TARGET_REL,
        "target_relation_name": TARGET_REL_NAME,
        "task_form": "(?, CtD, disease)",
        "prediction_type": "predicted_head",
        "num_raw_edges_total": all_edge_count,
        "num_target_edges_total": len(target_edges),
        "train_size": len(train_edges),
        "valid_size": len(valid_edges),
        "test_size": len(test_edges),
        "split_info": split_info,
        "num_train_compounds": len(set(h for h, _, _ in train_edges)),
        "num_valid_compounds": len(set(h for h, _, _ in valid_edges)),
        "num_test_compounds": len(set(h for h, _, _ in test_edges)),
        "num_train_diseases": len(set(t for _, _, t in train_edges)),
        "num_valid_diseases": len(set(t for _, _, t in valid_edges)),
        "num_test_diseases": len(set(t for _, _, t in test_edges)),
        "candidate_universe": "Compound",
        "candidate_universe_size": len(compound_entities),
        "query_universe": "Disease",
        "query_universe_size": len(disease_entities),
        "gold_injection": False,
        "absent_gold_rank_sentinel": 21,
        "top_k_for_future_candidate_generation": 20,
    }

    graph_summary = {
        "created_at": now_iso(),
        "setting": "setting_d_hetionet",
        "train_enriched_path": str(train_enriched_path),
        "train_enriched_ids_path": str(train_enriched_ids_path),
        "train_enriched_num_edges": train_enriched_edges,
        "train_enriched_ids_num_edges": id_graph_edges,
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
        "relation_counts_train_graph": dict(relation_counts_train_graph.most_common()),
        "raw_relation_counts": dict(raw_relation_counts.most_common()),
        "target_train_edges_in_graph": target_train_edges_in_graph,
        "heldout_target_edges_removed": heldout_removed,
        "graph_num_rels_for_future_training": len(relation2id),
    }

    checks = {
        "coverage": coverage,
        "leak_check": leak_check,
        "schema_check": schema_check,
    }

    decision = "DAY2_HETIONET_SPLIT_GRAPH_READY"
    if not coverage["coverage_pass"] or not leak_check["exact_leak_pass"]:
        decision = "DAY2_NEEDS_FIX"

    summary = {
        "decision": decision,
        "split_summary": split_summary,
        "graph_summary": graph_summary,
        "checks": checks,
        "important_next_arguments": {
            "graph_num_rels": len(relation2id),
            "target_relation_id": relation2id[TARGET_REL],
            "candidate_universe_size": len(compound_entities),
            "query_universe_size": len(disease_entities),
        },
    }

    write_json(split_dir / "split_summary.json", split_summary)
    write_json(graph_dir / "graph_summary.json", graph_summary)
    write_json(graph_dir / "leak_check.json", leak_check)
    write_json(result_dir / "day2_hetionet_split_graph_summary.json", summary)
    write_markdown_report(report_dir / "day2_hetionet_split_graph.md", summary)

    print("\n[DONE] Day 2 artifacts:")
    print(f"  - {split_dir / 'split_summary.json'}")
    print(f"  - {split_dir / 'train_target_rows.json'}")
    print(f"  - {split_dir / 'valid_target_rows.json'}")
    print(f"  - {split_dir / 'test_target_rows.json'}")
    print(f"  - {graph_dir / 'train_enriched.tsv'}")
    print(f"  - {graph_dir / 'train_enriched_ids.tsv'}")
    print(f"  - {graph_dir / 'entity2id.pkl'}")
    print(f"  - {graph_dir / 'relation2id.pkl'}")
    print(f"  - {graph_dir / 'graph_summary.json'}")
    print(f"  - {graph_dir / 'leak_check.json'}")
    print(f"  - {result_dir / 'day2_hetionet_split_graph_summary.json'}")
    print(f"  - {report_dir / 'day2_hetionet_split_graph.md'}")

    print("\n[SUMMARY]")
    print(json.dumps({
        "decision": decision,
        "split": {
            "train": len(train_edges),
            "valid": len(valid_edges),
            "test": len(test_edges),
        },
        "graph": {
            "edges": train_enriched_edges,
            "entities": len(entity2id),
            "relations": len(relation2id),
            "graph_num_rels": len(relation2id),
        },
        "checks": {
            "coverage_pass": coverage["coverage_pass"],
            "exact_leak_pass": leak_check["exact_leak_pass"],
            "exact_leak_count": leak_check["exact_leak_count"],
            "heldout_removed": leak_check["heldout_target_edges_removed"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()