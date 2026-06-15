from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "dataset/setting_a/backbone_candidates/valid_top20_raw.json"
EVID_PATH = ROOT / "dataset/setting_a/aligned_evidence/valid_aligned_evidence.json"
SUPPORT_PATH = ROOT / "dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_main.json"

TYPE_MAP_PATH = ROOT / "dataset/setting_b/annotations/type_map.tsv"
SCHEMA_RULES_PATH = ROOT / "dataset/setting_b/annotations/schema_rules.json"
VALID_B_PATH = ROOT / "dataset/setting_b/contra_checked/valid_b_annotations_contra_checked.json"

# Optional: nếu có thì dùng để map relation_id -> relation_name
ID2REL_CANDIDATES = [
    ROOT / "dataset/setting_a/drkgc_json/id2relation.pkl",
    ROOT / "dataset/setting_a/backbone_ready/id2relation.pkl",
]

OUT_FEATURES = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_path_features.json"
OUT_SUMMARY = ROOT / "outputs/fuzzy_retrieval/path_feature_summary.json"
OUT_REPORT = ROOT / "outputs/fuzzy_retrieval/reports/path_feature_build.md"

TOP_BAND_K = 5


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_type_map(path: Path) -> Dict[str, str]:
    type_map: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            entity, ent_type = parts[0], parts[1]
            if line_idx == 0 and entity.lower() == "entity" and ent_type.lower() == "type":
                continue
            type_map[entity] = ent_type
    return type_map


def maybe_load_id2relation(paths: List[Path]) -> Optional[Dict[Any, Any]]:
    for p in paths:
        if p.exists():
            with p.open("rb") as f:
                return pickle.load(f)
    return None


def get_row_uid(row_index: int, query_id: Any, gold_id: Any) -> str:
    return f"valid::{row_index}::{query_id}::{gold_id}"


def band_from_rank(rank_1based: int) -> str:
    if rank_1based <= TOP_BAND_K:
        return "top_band"
    if rank_1based <= 10:
        return "mid_band"
    return "tail_band"


def build_candidate_bands(candidate_ids: List[Any], support_scores: List[float]) -> Tuple[List[str], Dict[Any, int]]:
    """
    Stable sort theo score giảm dần, tie giữ thứ tự gốc.
    Trả:
      - bands theo thứ tự candidate hiện tại
      - rank map (candidate_id -> support_rank_1based) nếu candidate_id unique
    """
    indexed = list(enumerate(support_scores))
    ranked = sorted(indexed, key=lambda x: (-float(x[1]), x[0]))

    rank_by_pos: Dict[int, int] = {}
    for rank_1based, (pos, _) in enumerate(ranked, start=1):
        rank_by_pos[pos] = rank_1based

    bands = [band_from_rank(rank_by_pos[i]) for i in range(len(candidate_ids))]

    # Nếu candidate_id trùng nhau (không kỳ vọng), map cuối cùng sẽ overwrite.
    rank_by_candidate_id = {candidate_ids[i]: rank_by_pos[i] for i in range(len(candidate_ids))}
    return bands, rank_by_candidate_id


def get_valid_b_row_info(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Support a few annotation naming styles used by earlier data-preparation steps.
    """
    return {
        "query_name": row.get("query_disease") or row.get("query_entity"),
        "gold_name": row.get("gold_drug") or row.get("gold_entity"),
        "candidate_names": row.get("candidate_drugs") or row.get("candidate_entities") or [],
        "candidate_types": row.get("candidate_types") or row.get("candidate_support_types") or [],
        "contra_flags": row.get("contra_flags") or row.get("contra_flags_final") or [],
    }


def safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y"}
    return False


def parse_schema_rules(schema_rules: Any) -> Dict[str, Dict[str, str]]:
    """
    Chuẩn hóa schema rules về dạng:
      relation_name -> {"domain": "...", "range": "..."}
    Hỗ trợ một vài shape phổ biến.
    """
    parsed: Dict[str, Dict[str, str]] = {}

    if isinstance(schema_rules, dict):
        for rel, value in schema_rules.items():
            if isinstance(value, dict):
                domain = (
                    value.get("domain")
                    or value.get("head")
                    or value.get("source_type")
                    or value.get("subject_type")
                )
                rng = (
                    value.get("range")
                    or value.get("tail")
                    or value.get("target_type")
                    or value.get("object_type")
                )
                if domain or rng:
                    parsed[str(rel)] = {"domain": domain, "range": rng}
    elif isinstance(schema_rules, list):
        for item in schema_rules:
            if not isinstance(item, dict):
                continue
            rel = item.get("relation") or item.get("rel") or item.get("name")
            if rel is None:
                continue
            domain = (
                item.get("domain")
                or item.get("head")
                or item.get("source_type")
                or item.get("subject_type")
            )
            rng = (
                item.get("range")
                or item.get("tail")
                or item.get("target_type")
                or item.get("object_type")
            )
            parsed[str(rel)] = {"domain": domain, "range": rng}

    return parsed


def relation_name_from_id(rel_id: Any, id2relation: Optional[Dict[Any, Any]]) -> Optional[str]:
    if id2relation is None:
        return None
    if rel_id in id2relation:
        return str(id2relation[rel_id])
    try:
        rel_id_int = int(rel_id)
        if rel_id_int in id2relation:
            return str(id2relation[rel_id_int])
    except Exception:
        pass
    return None


def schema_consistency_flag(
    head_name: Optional[str],
    rel_name: Optional[str],
    tail_name: Optional[str],
    type_map: Dict[str, str],
    parsed_rules: Dict[str, Dict[str, str]],
) -> Optional[bool]:
    """
    Trả None nếu không đủ thông tin.
    """
    if rel_name is None:
        return None
    rule = parsed_rules.get(rel_name)
    if rule is None:
        return None

    head_type = type_map.get(head_name) if head_name is not None else None
    tail_type = type_map.get(tail_name) if tail_name is not None else None
    if head_type is None or tail_type is None:
        return None

    expect_domain = rule.get("domain")
    expect_range = rule.get("range")
    if expect_domain is None or expect_range is None:
        return None

    return (head_type == expect_domain) and (tail_type == expect_range)


def main() -> None:
    raw_rows = load_json(RAW_PATH)
    evid_rows = load_json(EVID_PATH)
    support_rows = load_json(SUPPORT_PATH)
    valid_b_rows = load_json(VALID_B_PATH)

    assert isinstance(raw_rows, list) and len(raw_rows) == 500, "raw_rows must be a 500-row list"
    assert isinstance(evid_rows, list) and len(evid_rows) == 500, "evid_rows must be a 500-row list"
    assert isinstance(support_rows, list) and len(support_rows) == 500, "support_rows must be a 500-row list"
    assert isinstance(valid_b_rows, list) and len(valid_b_rows) == 500, "valid_b_rows must be a 500-row list"

    type_map = load_type_map(TYPE_MAP_PATH)
    schema_rules_raw = load_json(SCHEMA_RULES_PATH)
    parsed_rules = parse_schema_rules(schema_rules_raw)
    id2relation = maybe_load_id2relation(ID2REL_CANDIDATES)

    out_rows: List[Dict[str, Any]] = []

    total_triples = 0
    total_direct_shortcut = 0
    total_touch_candidate = 0
    total_touch_query = 0
    total_touch_top_band = 0
    total_contra = 0
    total_schema_known = 0
    total_schema_consistent = 0
    total_subgraph_size = 0

    band_counter = Counter()
    relation_counter = Counter()

    for i, (raw_row, evid_row, supp_row, vb_row_raw) in enumerate(zip(raw_rows, evid_rows, support_rows, valid_b_rows)):
        # Row-level alignment check: tuyệt đối không join theo query_id một mình
        assert raw_row["query_entity_id"] == evid_row["query_entity_id"] == supp_row["query_entity_id"], \
            f"query_entity_id mismatch at row {i}"
        assert raw_row["gold_entity_id"] == evid_row["gold_entity_id"] == supp_row["gold_entity_id"], \
            f"gold_entity_id mismatch at row {i}"

        query_name = supp_row["query_entity"]
        query_id = supp_row["query_entity_id"]
        gold_name = supp_row["gold_entity"]
        gold_id = supp_row["gold_entity_id"]

        row_uid = get_row_uid(i, query_id, gold_id)

        candidate_names = supp_row["candidate_entities"]
        candidate_ids = supp_row["candidate_entity_ids"]
        support_scores = supp_row["support_scores"]
        candidate_debug_rows = supp_row.get("candidate_debug_rows", [])

        assert len(candidate_names) == len(candidate_ids) == len(support_scores), \
            f"candidate/support length mismatch at row {i}"

        candidate_bands, rank_by_candidate_id = build_candidate_bands(candidate_ids, support_scores)

        vb = get_valid_b_row_info(vb_row_raw)
        contra_flags = vb["contra_flags"]
        if not isinstance(contra_flags, list) or len(contra_flags) != len(candidate_ids):
            # Soft fallback: recover contra flags from candidate_debug_rows if annotations do not align.
            fallback = []
            for dbg in candidate_debug_rows:
                fallback.append(safe_bool(dbg.get("contra_flag", False)) if isinstance(dbg, dict) else False)
            contra_flags = fallback if len(fallback) == len(candidate_ids) else [False] * len(candidate_ids)

        # Map candidate id/name -> info
        candidate_info_by_id: Dict[Any, Dict[str, Any]] = {}
        for pos, (cname, cid, cscore, cband, ccontra) in enumerate(
            zip(candidate_names, candidate_ids, support_scores, candidate_bands, contra_flags)
        ):
            band_counter[cband] += 1
            candidate_info_by_id[cid] = {
                "position": pos,
                "name": cname,
                "score": cscore,
                "band": cband,
                "contra": safe_bool(ccontra),
            }

        subgraph = evid_row.get("subgraph", [])
        if not isinstance(subgraph, list):
            subgraph = []

        total_subgraph_size += len(subgraph)

        triple_counts = Counter(tuple(tr) for tr in subgraph if isinstance(tr, list) and len(tr) == 3)

        # local density = node appearance count in row subgraph
        node_counts = Counter()
        for triple in subgraph:
            if isinstance(triple, list) and len(triple) == 3:
                h, _, t = triple
                node_counts[h] += 1
                node_counts[t] += 1

        triple_feature_rows = []
        for triple_index, triple in enumerate(subgraph):
            if not (isinstance(triple, list) and len(triple) == 3):
                continue

            head_id, rel_id, tail_id = triple
            rel_name = relation_name_from_id(rel_id, id2relation)
            if rel_name is not None:
                relation_counter[rel_name] += 1
            else:
                relation_counter[f"rel_id::{rel_id}"] += 1

            touches_query = (head_id == query_id) or (tail_id == query_id)

            touched_candidate_positions = []
            touched_candidate_ids = []
            touched_candidate_names = []
            touched_top_band_candidate = False
            direct_candidate_query_flag = False
            triple_contra_flag = False

            for node_id in [head_id, tail_id]:
                if node_id in candidate_info_by_id:
                    info = candidate_info_by_id[node_id]
                    touched_candidate_positions.append(info["position"])
                    touched_candidate_ids.append(node_id)
                    touched_candidate_names.append(info["name"])
                    touched_top_band_candidate = touched_top_band_candidate or (info["band"] == "top_band")
                    triple_contra_flag = triple_contra_flag or info["contra"]

                    # Direct shortcut: the triple links a candidate directly to the query.
                    if touches_query:
                        direct_candidate_query_flag = True

            touches_candidate = len(touched_candidate_ids) > 0

            # Only check schema consistency when node names are available.
            head_name = query_name if head_id == query_id else candidate_info_by_id.get(head_id, {}).get("name")
            tail_name = query_name if tail_id == query_id else candidate_info_by_id.get(tail_id, {}).get("name")
            ont_flag = schema_consistency_flag(
                head_name=head_name,
                rel_name=rel_name,
                tail_name=tail_name,
                type_map=type_map,
                parsed_rules=parsed_rules,
            )

            local_density_hint = int(node_counts[head_id] + node_counts[tail_id])
            triple_frequency = int(triple_counts[tuple(triple)])

            triple_feature_rows.append({
                "triple_index": triple_index,
                "head_id": head_id,
                "relation_id": rel_id,
                "relation_name": rel_name,
                "tail_id": tail_id,
                "touches_query": touches_query,
                "touches_candidate": touches_candidate,
                "touched_candidate_positions": touched_candidate_positions,
                "touched_candidate_ids": touched_candidate_ids,
                "touched_candidate_names": touched_candidate_names,
                "touches_top_band_candidate": touched_top_band_candidate,
                "direct_candidate_query_flag": direct_candidate_query_flag,
                "contra_flag": triple_contra_flag,
                "ontology_consistency_flag": ont_flag,
                "triple_frequency_in_subgraph": triple_frequency,
                "local_density_hint": local_density_hint,
            })

            total_triples += 1
            total_touch_query += int(touches_query)
            total_touch_candidate += int(touches_candidate)
            total_touch_top_band += int(touched_top_band_candidate)
            total_direct_shortcut += int(direct_candidate_query_flag)
            total_contra += int(triple_contra_flag)
            if ont_flag is not None:
                total_schema_known += 1
                total_schema_consistent += int(bool(ont_flag))

        out_rows.append({
            "row_index": i,
            "row_uid": row_uid,
            "split": supp_row.get("split", "valid"),
            "query_entity": query_name,
            "query_entity_id": query_id,
            "gold_entity": gold_name,
            "gold_entity_id": gold_id,
            "variant_name": supp_row.get("variant_name", "soft_support_raw"),
            "candidate_entities": candidate_names,
            "candidate_entity_ids": candidate_ids,
            "support_scores": support_scores,
            "candidate_support_bands": candidate_bands,
            "contra_flags": [safe_bool(x) for x in contra_flags],
            "support_rank_order": supp_row.get("support_rank_order"),
            "subgraph_num_triples": len(subgraph),
            "triple_feature_rows": triple_feature_rows,
        })

    avg_subgraph_size = total_subgraph_size / len(out_rows) if out_rows else 0.0
    avg_triples_per_row = total_triples / len(out_rows) if out_rows else 0.0

    summary = {
        "stage": "path_feature_build",
        "decision_split": "valid",
        "num_rows": len(out_rows),
        "num_unique_query_ids": len({row["query_entity_id"] for row in out_rows}),
        "num_total_triples": total_triples,
        "avg_subgraph_size": round(avg_subgraph_size, 6),
        "avg_triple_feature_rows_per_row": round(avg_triples_per_row, 6),
        "touch_query_rate": round(total_touch_query / total_triples, 6) if total_triples else 0.0,
        "touch_candidate_rate": round(total_touch_candidate / total_triples, 6) if total_triples else 0.0,
        "touch_top_band_candidate_rate": round(total_touch_top_band / total_triples, 6) if total_triples else 0.0,
        "direct_candidate_query_rate": round(total_direct_shortcut / total_triples, 6) if total_triples else 0.0,
        "contra_flag_rate": round(total_contra / total_triples, 6) if total_triples else 0.0,
        "schema_known_count": total_schema_known,
        "schema_consistent_rate_among_known": round(total_schema_consistent / total_schema_known, 6) if total_schema_known else None,
        "candidate_band_counts": dict(band_counter),
        "top_relations": relation_counter.most_common(20),
        "top_band_k": TOP_BAND_K,
        "notes": [
            "Rows are aligned by row index, not by query_entity_id only.",
            "query_entity_id is duplicated across rows on valid.",
            "ontology_consistency_flag is None when relation-name or node-type resolution is unavailable."
        ],
        "sample_row_uid": out_rows[0]["row_uid"] if out_rows else None,
    }

    save_json(OUT_FEATURES, out_rows)
    save_json(OUT_SUMMARY, summary)

    report_lines = [
        "# Path/Triple Feature Build",
        "",
        "## Main decision",
        "- Feature table was built on valid only.",
        "- Row alignment used row index, not query_entity_id only.",
        f"- Total rows: {summary['num_rows']}",
        f"- Unique query ids: {summary['num_unique_query_ids']}",
        "",
        "## Summary",
        f"- avg_subgraph_size: {summary['avg_subgraph_size']}",
        f"- avg_triple_feature_rows_per_row: {summary['avg_triple_feature_rows_per_row']}",
        f"- touch_query_rate: {summary['touch_query_rate']}",
        f"- touch_candidate_rate: {summary['touch_candidate_rate']}",
        f"- touch_top_band_candidate_rate: {summary['touch_top_band_candidate_rate']}",
        f"- direct_candidate_query_rate: {summary['direct_candidate_query_rate']}",
        f"- contra_flag_rate: {summary['contra_flag_rate']}",
        f"- schema_known_count: {summary['schema_known_count']}",
        f"- schema_consistent_rate_among_known: {summary['schema_consistent_rate_among_known']}",
        "",
        "## Candidate band counts",
        json.dumps(summary["candidate_band_counts"], ensure_ascii=False, indent=2),
        "",
        "## Notes",
        "- This step only builds retrieval-ready features.",
        "- No path/triple score has been finalized yet.",
        "- No subgraph was reselected yet.",
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 80)
    print("PATH FEATURE BUILD DONE")
    print("features:", OUT_FEATURES)
    print("summary :", OUT_SUMMARY)
    print("report  :", OUT_REPORT)
    print("=" * 80)
    print("num_rows                =", summary["num_rows"])
    print("num_unique_query_ids    =", summary["num_unique_query_ids"])
    print("num_total_triples       =", summary["num_total_triples"])
    print("avg_subgraph_size       =", summary["avg_subgraph_size"])
    print("touch_candidate_rate    =", summary["touch_candidate_rate"])
    print("direct_candidate_query_rate =", summary["direct_candidate_query_rate"])
    print("contra_flag_rate        =", summary["contra_flag_rate"])
    print("schema_known_count      =", summary["schema_known_count"])
    print("schema_consistent_rate  =", summary["schema_consistent_rate_among_known"])
    print("=" * 80)


if __name__ == "__main__":
    main()
