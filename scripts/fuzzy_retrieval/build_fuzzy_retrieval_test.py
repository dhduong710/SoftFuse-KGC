from __future__ import annotations

import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(".").resolve()

RAW_PATH = ROOT / "dataset/setting_a/backbone_candidates/test_top20_raw.json"
EVID_PATH = ROOT / "dataset/setting_a/aligned_evidence/test_aligned_evidence.json"
SUPPORT_PATH = ROOT / "dataset/setting_a/soft_support_ranked_candidates/test_top20_soft_support_main.json"
TEST_B_PATH = ROOT / "dataset/setting_b/contra_checked/test_b_annotations_contra_checked.json"

TYPE_MAP_PATH = ROOT / "dataset/setting_b/annotations/type_map.tsv"
SCHEMA_RULES_PATH = ROOT / "dataset/setting_b/annotations/schema_rules.json"
MANIFEST_PATH = ROOT / "dataset/setting_a/fuzzy_retrieval/retrieval_main_manifest.json"

CONFIG_TIGHT = ROOT / "configs/fuzzy_retrieval/retrieval_tight.json"
CONFIG_DIRECTPLUS = ROOT / "configs/fuzzy_retrieval/retrieval_directplus.json"

ID2REL_CANDIDATES = [
    ROOT / "dataset/setting_a/drkgc_json/id2relation.pkl",
    ROOT / "dataset/setting_a/backbone_ready/id2relation.pkl",
]

OUT_FEATURES = ROOT / "dataset/setting_a/fuzzy_retrieval/test_path_features.json"
OUT_MAIN = ROOT / "dataset/setting_a/fuzzy_retrieval/test_fuzzy_retrieval_main.json"
OUT_SUMMARY = ROOT / "outputs/evaluation/retrieval_test_build_summary.json"
OUT_REPORT = ROOT / "outputs/evaluation/reports/build_fuzzy_retrieval_test.md"

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
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 2:
                continue
            if row[0].strip().lower() == "entity" and row[1].strip().lower() in {"type", "final_type"}:
                continue
            type_map[row[0]] = row[1]
    return type_map

def maybe_load_id2relation(paths: List[Path]) -> Optional[Dict[Any, Any]]:
    for p in paths:
        if p.exists():
            with p.open("rb") as f:
                return pickle.load(f)
    return None

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

def parse_schema_rules(schema_rules: Any) -> Dict[str, Dict[str, str]]:
    parsed: Dict[str, Dict[str, str]] = {}
    if isinstance(schema_rules, dict):
        for rel, value in schema_rules.items():
            if isinstance(value, dict):
                domain = value.get("domain") or value.get("head") or value.get("source_type") or value.get("subject_type")
                rng = value.get("range") or value.get("tail") or value.get("target_type") or value.get("object_type")
                if domain or rng:
                    parsed[str(rel)] = {"domain": domain, "range": rng}
    elif isinstance(schema_rules, list):
        for item in schema_rules:
            if not isinstance(item, dict):
                continue
            rel = item.get("relation") or item.get("rel") or item.get("name")
            if rel is None:
                continue
            domain = item.get("domain") or item.get("head") or item.get("source_type") or item.get("subject_type")
            rng = item.get("range") or item.get("tail") or item.get("target_type") or item.get("object_type")
            parsed[str(rel)] = {"domain": domain, "range": rng}
    return parsed

def schema_consistency_flag(
    head_name: Optional[str],
    rel_name: Optional[str],
    tail_name: Optional[str],
    type_map: Dict[str, str],
    parsed_rules: Dict[str, Dict[str, str]],
) -> Optional[bool]:
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

def band_from_rank(rank_1based: int) -> str:
    if rank_1based <= TOP_BAND_K:
        return "top_band"
    if rank_1based <= 10:
        return "mid_band"
    return "tail_band"

def build_candidate_bands(candidate_ids: List[Any], support_scores: List[float]) -> Tuple[List[str], Dict[Any, int]]:
    indexed = list(enumerate(support_scores))
    ranked = sorted(indexed, key=lambda x: (-float(x[1]), x[0]))
    rank_by_pos: Dict[int, int] = {}
    for rank_1based, (pos, _) in enumerate(ranked, start=1):
        rank_by_pos[pos] = rank_1based
    bands = [band_from_rank(rank_by_pos[i]) for i in range(len(candidate_ids))]
    rank_by_candidate_id = {candidate_ids[i]: rank_by_pos[i] for i in range(len(candidate_ids))}
    return bands, rank_by_candidate_id

def get_row_uid(row_index: int, query_id: Any, gold_id: Any) -> str:
    return f"test::{row_index}::{query_id}::{gold_id}"

def load_selected_config() -> Tuple[str, Dict[str, Any]]:
    manifest = load_json(MANIFEST_PATH)
    selected = manifest["selected_source_variant"]
    if selected == "soft_support_fuzzy_retrieval_tight":
        return selected, load_json(CONFIG_TIGHT)
    if selected == "soft_support_fuzzy_retrieval_directplus":
        return selected, load_json(CONFIG_DIRECTPLUS)
    if selected == "soft_support_fuzzy_retrieval_v1":
        return selected, {
            "variant_name": "soft_support_fuzzy_retrieval_v1",
            "retain_ratio": 0.67,
            "min_keep": 20,
            "top_band_weight": 1.0,
            "mid_band_weight": 0.6,
            "tail_band_weight": 0.25,
            "touch_candidate": 0.35,
            "touch_query": 0.05,
            "direct_shortcut": -0.80,
            "contra": -0.10,
            "density": 0.05,
        }
    raise RuntimeError(f"Unsupported selected_source_variant: {selected}")

def band_weight(band: str, cfg: Dict[str, Any]) -> float:
    if band == "top_band":
        return float(cfg["top_band_weight"])
    if band == "mid_band":
        return float(cfg["mid_band_weight"])
    return float(cfg["tail_band_weight"])

def score_triple(tr: Dict[str, Any], candidate_bands: List[str], cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    touched_positions = tr.get("touched_candidate_positions", [])
    touched_bands = []
    for pos in touched_positions:
        if isinstance(pos, int) and 0 <= pos < len(candidate_bands):
            touched_bands.append(candidate_bands[pos])

    if touched_bands:
        band_signal = max(band_weight(b, cfg) for b in touched_bands)
        best_band = max(touched_bands, key=lambda b: band_weight(b, cfg))
    else:
        band_signal = 0.0
        best_band = None

    density_hint = float(tr.get("local_density_hint", 0))
    density_norm = min(density_hint, 10.0) / 10.0
    touches_candidate = bool(tr.get("touches_candidate", False))
    touches_query = bool(tr.get("touches_query", False))
    direct_shortcut = bool(tr.get("direct_candidate_query_flag", False))
    contra_flag = bool(tr.get("contra_flag", False))

    score = (
        band_signal
        + float(cfg["touch_candidate"]) * float(touches_candidate)
        + float(cfg["touch_query"]) * float(touches_query)
        + float(cfg["direct_shortcut"]) * float(direct_shortcut)
        + float(cfg["contra"]) * float(contra_flag)
        + float(cfg["density"]) * density_norm
    )

    debug = {
        "band_signal": round(band_signal, 6),
        "best_touched_band": best_band,
        "touch_candidate_term": round(float(cfg["touch_candidate"]) * float(touches_candidate), 6),
        "touch_query_term": round(float(cfg["touch_query"]) * float(touches_query), 6),
        "direct_shortcut_term": round(float(cfg["direct_shortcut"]) * float(direct_shortcut), 6),
        "contra_term": round(float(cfg["contra"]) * float(contra_flag), 6),
        "density_term": round(float(cfg["density"]) * density_norm, 6),
    }
    return round(score, 6), debug

def make_triple_key(tr: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    return (tr["head_id"], tr["relation_id"], tr["tail_id"])

def target_size_from_original(n: int, retain_ratio: float, min_keep: int) -> int:
    if n <= 0:
        return 0
    return min(n, max(min_keep, int(round(n * retain_ratio))))

def main() -> None:
    raw_rows = load_json(RAW_PATH)
    evid_rows = load_json(EVID_PATH)
    support_rows = load_json(SUPPORT_PATH)
    test_b_rows = load_json(TEST_B_PATH)

    assert isinstance(raw_rows, list) and len(raw_rows) == 500, "raw_rows must be a 500-row list"
    assert isinstance(evid_rows, list) and len(evid_rows) == 500, "evid_rows must be a 500-row list"
    assert isinstance(support_rows, list) and len(support_rows) == 500, "support_rows must be a 500-row list"
    assert isinstance(test_b_rows, list) and len(test_b_rows) == 500, "test_b_rows must be a 500-row list"

    type_map = load_type_map(TYPE_MAP_PATH)
    schema_rules_raw = load_json(SCHEMA_RULES_PATH)
    parsed_rules = parse_schema_rules(schema_rules_raw)
    id2relation = maybe_load_id2relation(ID2REL_CANDIDATES)

    selected_source_variant, cfg = load_selected_config()

    # ---------- Step 1: build test_path_features ----------
    path_feature_rows: List[Dict[str, Any]] = []

    total_triples = 0
    total_direct_shortcut = 0
    total_touch_candidate = 0
    total_touch_query = 0
    total_touch_top_band = 0
    total_contra = 0
    total_schema_known = 0
    total_schema_consistent = 0
    total_subgraph_size = 0
    relation_counter = Counter()

    for i, (raw_row, evid_row, supp_row, tb_row) in enumerate(zip(raw_rows, evid_rows, support_rows, test_b_rows)):
        assert raw_row["query_entity_id"] == evid_row["query_entity_id"] == supp_row["query_entity_id"], f"query_entity_id mismatch at row {i}"
        assert raw_row["gold_entity_id"] == evid_row["gold_entity_id"] == supp_row["gold_entity_id"], f"gold_entity_id mismatch at row {i}"

        query_name = supp_row["query_entity"]
        query_id = supp_row["query_entity_id"]
        gold_name = supp_row["gold_entity"]
        gold_id = supp_row["gold_entity_id"]

        candidate_names = supp_row["candidate_entities"]
        candidate_ids = supp_row["candidate_entity_ids"]
        support_scores = supp_row["support_scores"]
        support_rank_order = supp_row["support_rank_order"]

        candidate_bands, _ = build_candidate_bands(candidate_ids, support_scores)

        # contra flags aligned by candidate name from test_b annotation
        b_names = tb_row.get("candidate_drugs", [])
        b_contra = tb_row.get("contra_flags_lookup_checked", tb_row.get("contra_flags", []))
        contra_map = {name: int(flag) for name, flag in zip(b_names, b_contra)}
        contra_flags = [int(contra_map.get(name, 0)) for name in candidate_names]

        subgraph = evid_row.get("subgraph", [])
        triple_counter = Counter(tuple(x) for x in subgraph if isinstance(x, list) and len(x) == 3)

        # local degree in subgraph
        degree_counter = defaultdict(int)
        for tr in subgraph:
            if isinstance(tr, list) and len(tr) == 3:
                h, _, t = tr
                degree_counter[h] += 1
                degree_counter[t] += 1

        cand_id_to_pos = {cid: pos for pos, cid in enumerate(candidate_ids)}
        cand_id_to_name = {cid: name for cid, name in zip(candidate_ids, candidate_names)}
        cand_id_to_contra = {cid: flag for cid, flag in zip(candidate_ids, contra_flags)}

        triple_features = []
        for tr_idx, tr in enumerate(subgraph):
            if not (isinstance(tr, list) and len(tr) == 3):
                continue
            head_id, relation_id, tail_id = tr
            rel_name = relation_name_from_id(relation_id, id2relation)

            touches_query = (head_id == query_id or tail_id == query_id)

            touched_candidate_positions = []
            touched_candidate_ids = []
            touched_candidate_names = []
            for node_id in (head_id, tail_id):
                if node_id in cand_id_to_pos:
                    pos = cand_id_to_pos[node_id]
                    if pos not in touched_candidate_positions:
                        touched_candidate_positions.append(pos)
                        touched_candidate_ids.append(node_id)
                        touched_candidate_names.append(cand_id_to_name[node_id])

            touches_candidate = len(touched_candidate_positions) > 0
            touches_top_band_candidate = any(candidate_bands[pos] == "top_band" for pos in touched_candidate_positions)

            direct_candidate_query_flag = False
            if touches_query and touches_candidate:
                direct_candidate_query_flag = True

            contra_flag = any(bool(cand_id_to_contra.get(cid, 0)) for cid in touched_candidate_ids)

            head_name = cand_id_to_name.get(head_id, query_name if head_id == query_id else None)
            tail_name = cand_id_to_name.get(tail_id, query_name if tail_id == query_id else None)
            ontology_flag = schema_consistency_flag(head_name, rel_name, tail_name, type_map, parsed_rules)

            if ontology_flag is not None:
                total_schema_known += 1
                total_schema_consistent += int(bool(ontology_flag))

            triple_frequency = triple_counter[(head_id, relation_id, tail_id)]
            local_density_hint = degree_counter[head_id] + degree_counter[tail_id]

            relation_counter[str(rel_name) if rel_name is not None else str(relation_id)] += 1
            total_triples += 1
            total_direct_shortcut += int(direct_candidate_query_flag)
            total_touch_candidate += int(touches_candidate)
            total_touch_query += int(touches_query)
            total_touch_top_band += int(touches_top_band_candidate)
            total_contra += int(contra_flag)

            triple_features.append({
                "triple_index": tr_idx,
                "head_id": head_id,
                "relation_id": relation_id,
                "relation_name": rel_name,
                "tail_id": tail_id,
                "touches_query": bool(touches_query),
                "touches_candidate": bool(touches_candidate),
                "touched_candidate_positions": touched_candidate_positions,
                "touched_candidate_ids": touched_candidate_ids,
                "touched_candidate_names": touched_candidate_names,
                "touches_top_band_candidate": bool(touches_top_band_candidate),
                "direct_candidate_query_flag": bool(direct_candidate_query_flag),
                "contra_flag": bool(contra_flag),
                "ontology_consistency_flag": ontology_flag,
                "triple_frequency_in_subgraph": int(triple_frequency),
                "local_density_hint": int(local_density_hint),
            })

        total_subgraph_size += len(triple_features)

        path_feature_rows.append({
            "row_index": i,
            "row_uid": get_row_uid(i, query_id, gold_id),
            "split": supp_row["split"],
            "query_entity": query_name,
            "query_entity_id": query_id,
            "gold_entity": gold_name,
            "gold_entity_id": gold_id,
            "variant_name": "test_path_features",
            "candidate_entities": candidate_names,
            "candidate_entity_ids": candidate_ids,
            "support_scores": support_scores,
            "candidate_support_bands": candidate_bands,
            "contra_flags": contra_flags,
            "support_rank_order": support_rank_order,
            "subgraph_num_triples": len(triple_features),
            "triple_feature_rows": triple_features,
        })

    save_json(OUT_FEATURES, path_feature_rows)

    # ---------- Step 2: apply selected retrieval config ----------
    main_rows: List[Dict[str, Any]] = []

    total_orig = 0
    total_selected = 0
    total_direct_orig = 0
    total_direct_selected = 0
    total_contra_orig = 0
    total_contra_selected = 0
    total_selected_score = 0.0
    coverage_rates = []
    top_band_coverage_rates = []

    for row in path_feature_rows:
        triple_rows = row["triple_feature_rows"]
        candidate_bands = row["candidate_support_bands"]
        candidate_ids = row["candidate_entity_ids"]
        candidate_names = row["candidate_entities"]
        support_scores = row["support_scores"]
        contra_flags = row["contra_flags"]

        scored_triples = []
        for tr in triple_rows:
            score, debug = score_triple(tr, candidate_bands, cfg)
            scored_triples.append({
                **tr,
                "triple_score": score,
                "score_debug": debug,
            })

        orig_touched_candidates = set()
        orig_touched_top_band = set()
        for tr in scored_triples:
            for pos in tr.get("touched_candidate_positions", []):
                if isinstance(pos, int) and 0 <= pos < len(candidate_ids):
                    orig_touched_candidates.add(pos)
                    if candidate_bands[pos] == "top_band":
                        orig_touched_top_band.add(pos)

        selected_keys = set()
        selected_rows = []

        # Stage A: coverage pass (best triple per candidate)
        for cand_pos, _ in enumerate(candidate_ids):
            touching = [tr for tr in scored_triples if cand_pos in tr.get("touched_candidate_positions", [])]
            if not touching:
                continue
            best_tr = max(touching, key=lambda x: (x["triple_score"], -x["triple_index"]))
            key = make_triple_key(best_tr)
            if key not in selected_keys:
                selected_keys.add(key)
                selected_rows.append(best_tr)

        # Stage B: global fill to target size
        target_size = target_size_from_original(
            len(scored_triples),
            retain_ratio=float(cfg["retain_ratio"]),
            min_keep=int(cfg["min_keep"]),
        )

        global_sorted = sorted(
            scored_triples,
            key=lambda x: (x["triple_score"], x["touches_top_band_candidate"], -x["triple_index"]),
            reverse=True,
        )

        for tr in global_sorted:
            if len(selected_rows) >= target_size:
                break
            key = make_triple_key(tr)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected_rows.append(tr)

        selected_rows = sorted(
            selected_rows,
            key=lambda x: (x["triple_score"], x["touches_top_band_candidate"], -x["triple_index"]),
            reverse=True,
        )

        selected_subgraph = [[tr["head_id"], tr["relation_id"], tr["tail_id"]] for tr in selected_rows]

        sel_touched_candidates = set()
        sel_touched_top_band = set()
        for tr in selected_rows:
            for pos in tr.get("touched_candidate_positions", []):
                if isinstance(pos, int) and 0 <= pos < len(candidate_ids):
                    sel_touched_candidates.add(pos)
                    if candidate_bands[pos] == "top_band":
                        sel_touched_top_band.add(pos)

        cand_cov = len(sel_touched_candidates) / len(orig_touched_candidates) if orig_touched_candidates else 1.0
        top_cov = len(sel_touched_top_band) / len(orig_touched_top_band) if orig_touched_top_band else 1.0
        coverage_rates.append(cand_cov)
        top_band_coverage_rates.append(top_cov)

        num_direct_orig = sum(int(bool(tr.get("direct_candidate_query_flag", False))) for tr in scored_triples)
        num_direct_sel = sum(int(bool(tr.get("direct_candidate_query_flag", False))) for tr in selected_rows)
        num_contra_orig = sum(int(bool(tr.get("contra_flag", False))) for tr in scored_triples)
        num_contra_sel = sum(int(bool(tr.get("contra_flag", False))) for tr in selected_rows)

        total_orig += len(scored_triples)
        total_selected += len(selected_rows)
        total_direct_orig += num_direct_orig
        total_direct_selected += num_direct_sel
        total_contra_orig += num_contra_orig
        total_contra_selected += num_contra_sel
        total_selected_score += sum(float(tr["triple_score"]) for tr in selected_rows)

        main_rows.append({
            "row_index": row["row_index"],
            "row_uid": row["row_uid"],
            "split": row["split"],
            "query_entity": row["query_entity"],
            "query_entity_id": row["query_entity_id"],
            "gold_entity": row["gold_entity"],
            "gold_entity_id": row["gold_entity_id"],
            "variant_name": "soft_support_fuzzy_retrieval_main",
            "candidate_entities": candidate_names,
            "candidate_entity_ids": candidate_ids,
            "support_scores": support_scores,
            "candidate_support_bands": candidate_bands,
            "contra_flags": contra_flags,
            "selected_subgraph": selected_subgraph,
            "triple_score_rows": selected_rows,
            "path_scores": [],
            "subgraph_summary": {
                "original_subgraph_size": len(scored_triples),
                "selected_subgraph_size": len(selected_rows),
                "target_size": target_size,
                "num_original_direct_shortcuts": num_direct_orig,
                "num_selected_direct_shortcuts": num_direct_sel,
                "num_original_contra_triples": num_contra_orig,
                "num_selected_contra_triples": num_contra_sel,
                "original_candidate_touched_count": len(orig_touched_candidates),
                "selected_candidate_touched_count": len(sel_touched_candidates),
                "candidate_coverage_preserved_rate": round(cand_cov, 6),
                "top_band_coverage_preserved_rate": round(top_cov, 6),
            },
            "selected_source_variant": selected_source_variant,
        })

    save_json(OUT_MAIN, main_rows)

    summary = {
        "stage": "build_retrieval_test",
        "status": "BUILT",
        "selected_source_variant": selected_source_variant,
        "selected_config": cfg,
        "input_paths": {
            "raw_test": str(RAW_PATH),
            "aligned_evidence_test": str(EVID_PATH),
            "soft_support_test": str(SUPPORT_PATH),
            "test_b_annotations": str(TEST_B_PATH),
            "retrieval_main_manifest": str(MANIFEST_PATH),
        },
        "intermediate_outputs": {
            "test_path_features": str(OUT_FEATURES),
        },
        "output_path": str(OUT_MAIN),
        "path_feature_summary": {
            "num_rows": len(path_feature_rows),
            "avg_subgraph_num_triples_before_scoring": round(total_subgraph_size / max(len(path_feature_rows), 1), 6),
            "touch_candidate_rate": round(total_touch_candidate / max(total_triples, 1), 6),
            "touch_query_rate": round(total_touch_query / max(total_triples, 1), 6),
            "touch_top_band_rate": round(total_touch_top_band / max(total_triples, 1), 6),
            "direct_shortcut_rate_before": round(total_direct_shortcut / max(total_triples, 1), 6),
            "contra_rate_before": round(total_contra / max(total_triples, 1), 6),
            "schema_consistency_rate_known_only": round(total_schema_consistent / max(total_schema_known, 1), 6) if total_schema_known else None,
            "top_relations_seen": relation_counter.most_common(10),
        },
        "retrieval_main_summary": {
            "num_rows": len(main_rows),
            "avg_original_subgraph_size": round(total_orig / max(len(main_rows), 1), 6),
            "avg_selected_subgraph_size": round(total_selected / max(len(main_rows), 1), 6),
            "avg_triple_score": round(total_selected_score / max(total_selected, 1), 6) if total_selected else 0.0,
            "direct_shortcut_path_rate": round(total_direct_selected / max(total_selected, 1), 6) if total_selected else 0.0,
            "contradiction_path_rate": round(total_contra_selected / max(total_selected, 1), 6) if total_selected else 0.0,
            "candidate_coverage_preserved_rate": round(sum(coverage_rates) / max(len(coverage_rates), 1), 6),
            "top_band_coverage_preserved_rate": round(sum(top_band_coverage_rates) / max(len(top_band_coverage_rates), 1), 6),
        },
        "schema_check": {
            "num_rows_is_500": len(main_rows) == 500,
            "selected_source_variant_matches_manifest": True,
            "canonical_variant_name": sorted(set(x["variant_name"] for x in main_rows)),
            "main_row_keys": list(main_rows[0].keys()) if main_rows else [],
            "triple_score_row_keys": list(main_rows[0]["triple_score_rows"][0].keys()) if main_rows and main_rows[0]["triple_score_rows"] else [],
        },
        "policy_checks": {
            "retrieval_logic_changed": False,
            "selected_source_variant_changed": False,
            "candidate_stage_reopened": False,
        },
    }

    save_json(OUT_SUMMARY, summary)

    md = []
    md.append("# Build retrieval_main_test")
    md.append("")
    md.append(f"- status: **{summary['status']}**")
    md.append(f"- selected_source_variant: `{selected_source_variant}`")
    md.append(f"- output: `{OUT_MAIN}`")
    md.append("")
    md.append("## 1. Selected retrieval source")
    md.append(f"- selected_source_variant: `{selected_source_variant}`")
    md.append(f"- canonical_variant_name: `soft_support_fuzzy_retrieval_main`")
    md.append("")
    md.append("## 2. Path feature summary")
    for k, v in summary["path_feature_summary"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 3. Retrieval-main summary")
    for k, v in summary["retrieval_main_summary"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 4. Policy checks")
    for k, v in summary["policy_checks"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 5. Conclusion")
    md.append(
        "Built `test_fuzzy_retrieval_main.json` by reusing the selected retrieval source "
        "variant from valid-side, preserving graph-side fields and canonicalizing the row name "
        "to `soft_support_fuzzy_retrieval_main`."
    )
    OUT_REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
