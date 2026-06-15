import json
import csv
from pathlib import Path


RAW_PATH = Path("dataset/setting_a/backbone_candidates/valid_top20_raw.json")
TYPE_FILTERED_PATH = Path("dataset/setting_a/ontology_control/valid_top20_type_filtered_raw.json")
ONTOLOGY_RAW_PATH = Path("dataset/setting_a/ontology_control/valid_top20_ontology_raw.json")
ALIGNED_EVIDENCE_PATH = Path("dataset/setting_a/aligned_evidence/valid_aligned_evidence.json")

TYPE_MAP_PATH = Path("dataset/setting_b/annotations/type_map.tsv")
SCHEMA_RULES_PATH = Path("dataset/setting_b/annotations/schema_rules.json")
VALID_B_PATH = Path("dataset/setting_b/contra_checked/valid_b_annotations_contra_checked.json")

OUT_DIR = Path("dataset/setting_a/support_features")
OUT_PATH = OUT_DIR / "valid_support_features.json"
MANIFEST_PATH = OUT_DIR / "support_feature_manifest.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_type_map(path: Path):
    """
    Supports either:
    - entity, raw_type, final_type, ...
    - entity, type, ...
    """
    result = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entity = row.get("entity")
            if not entity:
                continue
            final_type = row.get("final_type")
            coarse_type = row.get("type")
            raw_type = row.get("raw_type")
            val = final_type or coarse_type or raw_type or "Unknown"
            result[entity] = val
    return result


def make_query_key(query_entity, gold_entity, split):
    return f"{split}|||{query_entity}|||{gold_entity}"


def build_candidate_flag_map(candidates, values):
    out = {}
    if not isinstance(candidates, list) or not isinstance(values, list):
        return out
    for c, v in zip(candidates, values):
        out[c] = v
    return out


def count_touching_edges(subgraph, node_id):
    cnt = 0
    for triple in subgraph:
        if not isinstance(triple, list) or len(triple) != 3:
            continue
        h, _, t = triple
        if h == node_id or t == node_id:
            cnt += 1
    return cnt


def count_direct_edges_between(subgraph, a, b):
    cnt = 0
    for triple in subgraph:
        if not isinstance(triple, list) or len(triple) != 3:
            continue
        h, _, t = triple
        if (h == a and t == b) or (h == b and t == a):
            cnt += 1
    return cnt


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = load_json(RAW_PATH)
    type_filtered_rows = load_json(TYPE_FILTERED_PATH)
    ontology_rows = load_json(ONTOLOGY_RAW_PATH)
    aligned_rows = load_json(ALIGNED_EVIDENCE_PATH)
    valid_b_rows = load_json(VALID_B_PATH)

    type_map = load_type_map(TYPE_MAP_PATH)
    schema_rules = load_json(SCHEMA_RULES_PATH)

    # Build row maps by (split, query, gold)
    type_filtered_map = {}
    for row in type_filtered_rows:
        key = make_query_key(row["query_entity"], row["gold_entity"], row.get("split", "valid"))
        type_filtered_map[key] = row

    ontology_map = {}
    for row in ontology_rows:
        key = make_query_key(row["query_entity"], row["gold_entity"], row.get("split", "valid"))
        ontology_map[key] = row

    aligned_map = {}
    for row in aligned_rows:
        key = make_query_key(row["query_entity"], row["gold_entity"], row.get("split", "valid"))
        aligned_map[key] = row

    valid_b_map = {}
    for row in valid_b_rows:
        key = f"valid|||{row['query_disease']}|||{row['gold_drug']}"
        valid_b_map[key] = row

    out_rows = []
    total_queries = 0
    total_candidates = 0
    total_type_valid = 0
    total_schema_valid = 0
    total_type_filtered_keep = 0
    total_ontology_keep = 0
    total_contra = 0

    # coarse schema assumption for the main task
    # indication / treats: Drug -> Disease
    query_relation_schema = schema_rules.get("indication", ["Drug", "Disease"])
    if isinstance(query_relation_schema, list) and len(query_relation_schema) >= 2:
        expected_head_type = query_relation_schema[0]
        expected_tail_type = query_relation_schema[1]
    else:
        expected_head_type = "Drug"
        expected_tail_type = "Disease"

    for raw in raw_rows:
        split = raw.get("split", "valid")
        query_entity = raw["query_entity"]
        gold_entity = raw["gold_entity"]
        key = make_query_key(query_entity, gold_entity, split)

        tf_row = type_filtered_map.get(key, {})
        ont_row = ontology_map.get(key, {})
        ali_row = aligned_map.get(key, {})
        vb_row = valid_b_map.get(key, {})

        # maps from candidate -> metadata
        tf_keep_set = set(tf_row.get("candidate_entities", []))
        ont_keep_set = set(ont_row.get("candidate_entities", []))

        ont_support_map = {}
        if "candidate_support_types" in ont_row:
            ont_support_map = build_candidate_flag_map(
                ont_row.get("candidate_entities", []),
                ont_row.get("candidate_support_types", [])
            )

        b_type_map = build_candidate_flag_map(
            vb_row.get("candidate_drugs", []),
            vb_row.get("candidate_types", [])
        )
        contra_map = build_candidate_flag_map(
            vb_row.get("candidate_drugs", []),
            vb_row.get("contra_flags_lookup_checked", vb_row.get("contra_flags", []))
        )
        conflict_map = build_candidate_flag_map(
            vb_row.get("candidate_drugs", []),
            vb_row.get("conflict_flags", [])
        )

        aligned_candidate_names = ali_row.get("rank_entities", [])
        aligned_candidate_ids = ali_row.get("rank_entities_id", [])
        name_to_aligned_id = build_candidate_flag_map(aligned_candidate_names, aligned_candidate_ids)
        subgraph = ali_row.get("subgraph", [])
        query_entity_id = raw["query_entity_id"]

        query_type = type_map.get(query_entity, "Disease")

        candidate_feature_rows = []
        candidate_entities = raw["candidate_entities"]
        candidate_ids = raw["candidate_entity_ids"]

        for idx, (cand, cand_id) in enumerate(zip(candidate_entities, candidate_ids), start=1):
            candidate_type = b_type_map.get(cand, type_map.get(cand, "Unknown"))

            type_valid_flag = int(candidate_type == expected_head_type)
            schema_valid_flag = int(
                candidate_type == expected_head_type and query_type == expected_tail_type
            )
            type_filtered_keep_flag = int(cand in tf_keep_set)
            ontology_keep_flag = int(cand in ont_keep_set)

            ontology_support_type = ont_support_map.get(cand, "unknown")

            contra_flag = int(contra_map.get(cand, 0))
            conflict_flag = int(conflict_map.get(cand, 0))
            contra_penalty = float(contra_flag)

            aligned_cand_id = name_to_aligned_id.get(cand, cand_id)
            candidate_in_aligned_evidence = int(cand in aligned_candidate_names)
            evidence_edge_touch_count = count_touching_edges(subgraph, aligned_cand_id)
            query_edge_touch_count = count_touching_edges(subgraph, query_entity_id)
            candidate_query_edge_count = count_direct_edges_between(subgraph, aligned_cand_id, query_entity_id)

            feat = {
                "candidate_entity": cand,
                "candidate_entity_id": cand_id,
                "base_rank": idx,
                "type_valid_flag": type_valid_flag,
                "candidate_type": candidate_type,
                "query_type": query_type,
                "schema_valid_flag": schema_valid_flag,
                "type_filtered_keep_flag": type_filtered_keep_flag,
                "ontology_keep_flag": ontology_keep_flag,
                "ontology_support_type": ontology_support_type,
                "contra_flag": contra_flag,
                "conflict_flag": conflict_flag,
                "contra_penalty": contra_penalty,
                "candidate_in_aligned_evidence": candidate_in_aligned_evidence,
                "evidence_edge_touch_count": evidence_edge_touch_count,
                "query_edge_touch_count": query_edge_touch_count,
                "candidate_query_edge_count": candidate_query_edge_count,
                "support_feature_vector": {
                    "type": float(type_valid_flag),
                    "schema": float(schema_valid_flag),
                    "ontology": float(ontology_keep_flag),
                    "contra_penalty": float(contra_penalty),
                    "evidence_touch": float(evidence_edge_touch_count),
                    "candidate_query_link": float(candidate_query_edge_count)
                }
            }
            candidate_feature_rows.append(feat)

            total_candidates += 1
            total_type_valid += type_valid_flag
            total_schema_valid += schema_valid_flag
            total_type_filtered_keep += type_filtered_keep_flag
            total_ontology_keep += ontology_keep_flag
            total_contra += contra_flag

        out_rows.append({
            "split": split,
            "query_entity": query_entity,
            "query_entity_id": raw["query_entity_id"],
            "gold_entity": gold_entity,
            "gold_entity_id": raw["gold_entity_id"],
            "gold_rank_in_full_universe": raw.get("gold_rank_in_full_universe"),
            "gold_in_topk_raw": raw.get("gold_in_topk_raw"),
            "candidate_feature_rows": candidate_feature_rows
        })
        total_queries += 1

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    manifest = {
        "stage": "soft_support_feature_build",
        "decision_split": "valid",
        "main_truth": "raw_no_injection",
        "inputs": {
            "raw": str(RAW_PATH),
            "type_filtered_raw": str(TYPE_FILTERED_PATH),
            "ontology_raw": str(ONTOLOGY_RAW_PATH),
            "aligned_evidence": str(ALIGNED_EVIDENCE_PATH),
            "type_map": str(TYPE_MAP_PATH),
            "schema_rules": str(SCHEMA_RULES_PATH),
            "valid_b_annotations": str(VALID_B_PATH)
        },
        "outputs": {
            "valid_support_features": str(OUT_PATH)
        },
        "summary": {
            "num_queries": total_queries,
            "num_candidates": total_candidates,
            "avg_candidates_per_query": round(total_candidates / max(total_queries, 1), 4),
            "type_valid_rate": round(total_type_valid / max(total_candidates, 1), 6),
            "schema_valid_rate": round(total_schema_valid / max(total_candidates, 1), 6),
            "type_filtered_keep_rate": round(total_type_filtered_keep / max(total_candidates, 1), 6),
            "ontology_keep_rate": round(total_ontology_keep / max(total_candidates, 1), 6),
            "contra_rate": round(total_contra / max(total_candidates, 1), 6)
        }
    }

    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
