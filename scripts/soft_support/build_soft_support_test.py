import json
import csv
from pathlib import Path

ROOT = Path(".").resolve()

RAW_PATH = ROOT / "dataset/setting_a/backbone_candidates/test_top20_raw.json"
TYPE_FILTERED_PATH = ROOT / "dataset/setting_a/ontology_control/test_top20_type_filtered_raw.json"
ONTOLOGY_RAW_PATH = ROOT / "dataset/setting_a/ontology_control/test_top20_ontology_raw.json"
ALIGNED_EVIDENCE_PATH = ROOT / "dataset/setting_a/aligned_evidence/test_aligned_evidence.json"

TYPE_MAP_PATH = ROOT / "dataset/setting_b/annotations/type_map.tsv"
SCHEMA_RULES_PATH = ROOT / "dataset/setting_b/annotations/schema_rules.json"
TEST_B_PATH = ROOT / "dataset/setting_b/contra_checked/test_b_annotations_contra_checked.json"

VALID_MAIN_PATH = ROOT / "dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_main.json"
OUT_PATH = ROOT / "dataset/setting_a/soft_support_ranked_candidates/test_top20_soft_support_main.json"

SUMMARY_PATH = ROOT / "outputs/evaluation/soft_support_test_build_summary.json"
REPORT_PATH = ROOT / "outputs/evaluation/reports/build_soft_support_test.md"

# Main soft-support formula selected on the validation split.
MAIN_FORMULA = {
    "variant_name": "soft_support_raw_b050",
    "score_family": "B_evidence_minus_direct",
    "evidence_positive_weight": 1.0,
    "direct_link_penalty": 0.50,
    "contra_penalty_weight": 0.10,
    "use_capped_evidence": False,
    "evidence_cap": None,
}

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_type_map(path: Path):
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
            result[entity] = final_type or coarse_type or raw_type or "Unknown"
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

def compute_score(c):
    direct_penalty = 1.0 if c["candidate_query_edge_count"] > 0 else 0.0
    contra = float(c.get("contra_penalty", 0.0))

    if MAIN_FORMULA["use_capped_evidence"]:
        cap = float(MAIN_FORMULA["evidence_cap"])
        evidence_val = min(float(c["evidence_edge_touch_count"]), cap)
        evidence_term = evidence_val / max(cap, 1.0)
    else:
        evidence_term = 1.0 if c["evidence_edge_touch_count"] > 0 else 0.0

    score = (
        float(MAIN_FORMULA["evidence_positive_weight"]) * evidence_term
        - float(MAIN_FORMULA["direct_link_penalty"]) * direct_penalty
        - float(MAIN_FORMULA["contra_penalty_weight"]) * contra
    )
    return round(float(score), 6)

def main():
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = load_json(RAW_PATH)
    type_filtered_rows = load_json(TYPE_FILTERED_PATH)
    ontology_rows = load_json(ONTOLOGY_RAW_PATH)
    aligned_rows = load_json(ALIGNED_EVIDENCE_PATH)
    test_b_rows = load_json(TEST_B_PATH)
    valid_main_rows = load_json(VALID_MAIN_PATH)

    type_map = load_type_map(TYPE_MAP_PATH)
    schema_rules = load_json(SCHEMA_RULES_PATH)

    valid_top_keys = list(valid_main_rows[0].keys())
    valid_debug_keys = list(valid_main_rows[0]["candidate_debug_rows"][0].keys())
    valid_variant_name = valid_main_rows[0]["variant_name"]

    # If valid main row says another variant name, follow valid main exactly
    main_variant_name = valid_variant_name

    type_filtered_map = {}
    for row in type_filtered_rows:
        key = make_query_key(row["query_entity"], row["gold_entity"], row.get("split", "test"))
        type_filtered_map[key] = row

    ontology_map = {}
    for row in ontology_rows:
        key = make_query_key(row["query_entity"], row["gold_entity"], row.get("split", "test"))
        ontology_map[key] = row

    aligned_map = {}
    for row in aligned_rows:
        key = make_query_key(row["query_entity"], row["gold_entity"], row.get("split", "test"))
        aligned_map[key] = row

    test_b_map = {}
    for row in test_b_rows:
        key = f"test|||{row['query_disease']}|||{row['gold_drug']}"
        test_b_map[key] = row

    query_relation_schema = schema_rules.get("indication", ["Drug", "Disease"])
    if isinstance(query_relation_schema, list) and len(query_relation_schema) >= 2:
        expected_head_type = query_relation_schema[0]
        expected_tail_type = query_relation_schema[1]
    else:
        expected_head_type = "Drug"
        expected_tail_type = "Disease"

    out_rows = []
    total_queries = 0
    total_candidates = 0
    gold_present_count = 0
    exact20_count = 0
    avg_top5_direct = []
    avg_top5_evidence = []
    avg_top5_ontology = []
    avg_top5_contra = []

    for raw in raw_rows:
        split = raw.get("split", "test")
        query_entity = raw["query_entity"]
        gold_entity = raw["gold_entity"]
        key = make_query_key(query_entity, gold_entity, split)

        tf_row = type_filtered_map.get(key, {})
        ont_row = ontology_map.get(key, {})
        ali_row = aligned_map.get(key, {})
        tb_row = test_b_map.get(key, {})

        tf_keep_set = set(tf_row.get("candidate_entities", []))
        ont_keep_set = set(ont_row.get("candidate_entities", []))

        ont_support_map = {}
        if "candidate_support_types" in ont_row:
            ont_support_map = build_candidate_flag_map(
                ont_row.get("candidate_entities", []),
                ont_row.get("candidate_support_types", [])
            )

        b_type_map = build_candidate_flag_map(
            tb_row.get("candidate_drugs", []),
            tb_row.get("candidate_types", [])
        )
        contra_map = build_candidate_flag_map(
            tb_row.get("candidate_drugs", []),
            tb_row.get("contra_flags_lookup_checked", tb_row.get("contra_flags", []))
        )
        conflict_map = build_candidate_flag_map(
            tb_row.get("candidate_drugs", []),
            tb_row.get("conflict_flags", [])
        )

        aligned_candidate_names = ali_row.get("rank_entities", [])
        aligned_candidate_ids = ali_row.get("rank_entities_id", [])
        name_to_aligned_id = build_candidate_flag_map(aligned_candidate_names, aligned_candidate_ids)
        subgraph = ali_row.get("subgraph", [])
        query_entity_id = raw["query_entity_id"]

        query_type = type_map.get(query_entity, "Disease")

        cand_rows = []
        for idx, (cand, cand_id) in enumerate(zip(raw["candidate_entities"], raw["candidate_entity_ids"]), start=1):
            candidate_type = b_type_map.get(cand, type_map.get(cand, "Unknown"))

            type_valid_flag = int(candidate_type == expected_head_type)
            schema_valid_flag = int(candidate_type == expected_head_type and query_type == expected_tail_type)
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
            }
            feat["support_score"] = compute_score(feat)
            cand_rows.append(feat)

        # Stable reorder only
        cand_rows.sort(key=lambda x: (-x["support_score"], x["base_rank"]))

        candidate_entities = []
        candidate_entity_ids = []
        support_scores = []
        support_rank_order = []
        candidate_debug_rows = []

        for i, c in enumerate(cand_rows, start=1):
            c["support_rank"] = i
            candidate_entities.append(c["candidate_entity"])
            candidate_entity_ids.append(c["candidate_entity_id"])
            support_scores.append(c["support_score"])
            support_rank_order.append(i)
            candidate_debug_rows.append({
                "candidate_entity": c["candidate_entity"],
                "candidate_entity_id": c["candidate_entity_id"],
                "base_rank": c["base_rank"],
                "support_rank": c["support_rank"],
                "support_score": c["support_score"],
                "evidence_edge_touch_count": c["evidence_edge_touch_count"],
                "candidate_query_edge_count": c["candidate_query_edge_count"],
                "ontology_keep_flag": c["ontology_keep_flag"],
                "contra_flag": c["contra_flag"],
            })

        if len(candidate_entities) == 20:
            exact20_count += 1
        if gold_entity in candidate_entities:
            gold_present_count += 1

        top5 = cand_rows[:5]
        avg_top5_direct.append(sum(1 for c in top5 if c["candidate_query_edge_count"] > 0) / max(len(top5), 1))
        avg_top5_evidence.append(sum(1 for c in top5 if c["evidence_edge_touch_count"] > 0) / max(len(top5), 1))
        avg_top5_ontology.append(sum(c["ontology_keep_flag"] for c in top5) / max(len(top5), 1))
        avg_top5_contra.append(sum(c["contra_flag"] for c in top5) / max(len(top5), 1))

        out_row = {
            "split": split,
            "query_entity": query_entity,
            "query_entity_id": raw["query_entity_id"],
            "gold_entity": gold_entity,
            "gold_entity_id": raw["gold_entity_id"],
            "gold_rank_in_full_universe": raw.get("gold_rank_in_full_universe"),
            "gold_in_topk_raw": raw.get("gold_in_topk_raw"),
            "variant_name": main_variant_name,
            "candidate_entities": candidate_entities,
            "candidate_entity_ids": candidate_entity_ids,
            "support_scores": support_scores,
            "support_rank_order": support_rank_order,
            "candidate_debug_rows": candidate_debug_rows,
        }
        out_rows.append(out_row)

        total_queries += 1
        total_candidates += len(candidate_entities)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    out_top_keys = list(out_rows[0].keys())
    out_debug_keys = list(out_rows[0]["candidate_debug_rows"][0].keys())

    summary = {
        "stage": "build_soft_support_test",
        "status": "BUILT",
        "input_paths": {
            "raw_test": str(RAW_PATH),
            "type_filtered_test": str(TYPE_FILTERED_PATH),
            "ontology_raw_test": str(ONTOLOGY_RAW_PATH),
            "aligned_evidence_test": str(ALIGNED_EVIDENCE_PATH),
            "test_b_annotations": str(TEST_B_PATH),
            "valid_main_reference": str(VALID_MAIN_PATH),
        },
        "main_formula": MAIN_FORMULA,
        "variant_name_from_valid_main": valid_variant_name,
        "variant_name_written_to_test": main_variant_name,
        "output_path": str(OUT_PATH),
        "summary": {
            "num_rows": total_queries,
            "num_candidates_total": total_candidates,
            "avg_candidates_per_query": round(total_candidates / max(total_queries, 1), 6),
            "rows_with_exactly_20_candidates": exact20_count,
            "gold_present_rate_after_reorder": round(gold_present_count / max(total_queries, 1), 6),
            "avg_top5_direct_link_rate": round(sum(avg_top5_direct) / max(total_queries, 1), 6),
            "avg_top5_evidence_positive_rate": round(sum(avg_top5_evidence) / max(total_queries, 1), 6),
            "avg_top5_ontology_keep_rate": round(sum(avg_top5_ontology) / max(total_queries, 1), 6),
            "avg_top5_contra_rate": round(sum(avg_top5_contra) / max(total_queries, 1), 6),
        },
        "schema_check": {
            "valid_main_top_keys": valid_top_keys,
            "test_main_top_keys": out_top_keys,
            "top_keys_match": valid_top_keys == out_top_keys,
            "valid_main_debug_keys": valid_debug_keys,
            "test_main_debug_keys": out_debug_keys,
            "debug_keys_match": valid_debug_keys == out_debug_keys,
        },
        "policy_checks": {
            "reorder_only": True,
            "no_pruning": exact20_count == total_queries,
            "formula_changed": False,
            "row_naming_changed": valid_variant_name != main_variant_name,
        },
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    md = []
    md.append("# Build soft_support_raw_test")
    md.append("")
    md.append(f"- status: **{summary['status']}**")
    md.append(f"- output: `{OUT_PATH}`")
    md.append(f"- variant_name: `{main_variant_name}`")
    md.append("")
    md.append("## 1. Main formula")
    md.append(f"- score_family: `{MAIN_FORMULA['score_family']}`")
    md.append(f"- evidence_positive_weight: `{MAIN_FORMULA['evidence_positive_weight']}`")
    md.append(f"- direct_link_penalty: `{MAIN_FORMULA['direct_link_penalty']}`")
    md.append(f"- contra_penalty_weight: `{MAIN_FORMULA['contra_penalty_weight']}`")
    md.append(f"- use_capped_evidence: `{MAIN_FORMULA['use_capped_evidence']}`")
    md.append("")
    md.append("## 2. Build summary")
    for k, v in summary["summary"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 3. Schema check vs valid main")
    for k, v in summary["schema_check"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 4. Policy checks")
    for k, v in summary["policy_checks"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 5. Conclusion")
    md.append(
        "Built `test_top20_soft_support_main.json` by applying the selected soft-support "
        "reorder logic to the test raw/no-injection source without pruning, formula change, "
        "or row-role change."
    )

    REPORT_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
