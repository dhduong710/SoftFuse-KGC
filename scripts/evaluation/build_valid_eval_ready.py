from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(".").resolve()

INPUTS = {
    "backbone_raw": ROOT / "dataset/setting_a/backbone_candidates/valid_top20_raw.json",
    "ontology_raw": ROOT / "dataset/setting_a/ontology_control/valid_top20_ontology_raw.json",
    "soft_support_raw": ROOT / "dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_main.json",
    "retrieval_main": ROOT / "dataset/setting_a/fuzzy_retrieval/valid_fuzzy_retrieval_main.json",
    "valid_b": ROOT / "dataset/setting_b/contra_checked/valid_b_annotations_contra_checked.json",
}

OUT_DIR = ROOT / "dataset/setting_b/eval_valid"
OUT_REPORT = ROOT / "outputs/evaluation/reports/valid_eval_ready.md"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_row_uid(row: dict, idx: int) -> str:
    if "row_uid" in row and row["row_uid"] not in (None, ""):
        return str(row["row_uid"])
    if "row_index" in row and row["row_index"] is not None:
        return f"row_index::{row['row_index']}"
    return f"fallback::{idx}::{row.get('query_entity', 'UNK')}::{row.get('gold_entity', 'UNK')}"


def build_valid_b_index(valid_b_rows: list[dict]) -> dict:
    idx = {}
    for row in valid_b_rows:
        key = (row["query_disease"], row["gold_drug"])
        idx[key] = row
    return idx


def reorder_setting_b_fields(valid_b_row: dict | None, candidate_entities: list[str]) -> dict:
    if valid_b_row is None:
        n = len(candidate_entities)
        return {
            "candidate_types": [None] * n,
            "contra_flags": [None] * n,
            "conflict_flags": [None] * n,
            "has_any_contra_candidate": None,
            "gold_is_contraindicated": None,
            "gold_conflict_flag": None,
        }

    cand_names = valid_b_row.get("candidate_drugs", [])
    cand_types = valid_b_row.get("candidate_types_lookup_checked", valid_b_row.get("candidate_types", []))
    contra_flags = valid_b_row.get("contra_flags_lookup_checked", valid_b_row.get("contra_flags", []))
    conflict_flags = valid_b_row.get("conflict_flags_lookup_checked", valid_b_row.get("conflict_flags", []))

    type_map = {k: v for k, v in zip(cand_names, cand_types)}
    contra_map = {k: v for k, v in zip(cand_names, contra_flags)}
    conflict_map = {k: v for k, v in zip(cand_names, conflict_flags)}

    return {
        "candidate_types": [type_map.get(c) for c in candidate_entities],
        "contra_flags": [contra_map.get(c) for c in candidate_entities],
        "conflict_flags": [conflict_map.get(c) for c in candidate_entities],
        "has_any_contra_candidate": valid_b_row.get("has_any_contra_candidate"),
        "gold_is_contraindicated": valid_b_row.get("gold_is_contraindicated"),
        "gold_conflict_flag": valid_b_row.get("gold_conflict_flag"),
    }


def extract_candidates(row: dict) -> tuple[list[str], list[int]]:
    if "candidate_entities" in row and "candidate_entity_ids" in row:
        return row["candidate_entities"], row["candidate_entity_ids"]
    if "rank_entities" in row and "rank_entities_id" in row:
        return row["rank_entities"], row["rank_entities_id"]
    raise KeyError("Cannot find candidate fields in row.")

def get_gold_present_and_rank(row_name: str, row: dict, candidate_ids: list[int], gold_id: int) -> tuple[bool, int, str]:
    """
    Return:
      gold_present, gold_rank, gold_rank_source

    Rule:
    - ontology_raw must trust ontology-specific metadata first.
    - if ontology says gold is absent, rank must be 21 (not candidate_list fallback).
    """

    if row_name == "ontology_raw":
        meta_present = row.get("gold_in_topk_ontology")
        meta_rank = row.get("gold_rank_in_ontology_candidates")

        # Case 1: ontology explicitly says gold is absent
        if meta_present is False:
            return False, 21, "ontology_metadata_absent"

        # Case 2: ontology explicitly gives a rank
        if meta_rank is not None:
            rank = int(meta_rank)
            present = True if meta_present is None else bool(meta_present)
            return present, rank, "ontology_metadata"

        # Case 3: ontology says present but rank missing -> defensive fallback
        if meta_present is True:
            if gold_id in candidate_ids:
                return True, candidate_ids.index(gold_id) + 1, "ontology_present_candidate_fallback"
            return True, 21, "ontology_present_missing_rank_fallback"

    # Default path for backbone / soft support / retrieval
    if gold_id in candidate_ids:
        return True, candidate_ids.index(gold_id) + 1, "candidate_list_position"

    return False, len(candidate_ids) + 1, "candidate_list_position"

def stage_specific_payload(row_name: str, row: dict) -> dict:
    if row_name == "backbone_raw":
        return {
            "source_stage": "raw_candidate_source",
            "gold_rank_in_full_universe": row.get("gold_rank_in_full_universe"),
            "gold_in_topk_raw": row.get("gold_in_topk_raw"),
        }

    if row_name == "ontology_raw":
        return {
            "source_stage": "ontology_negative_control",
            "type_filter_target_type": row.get("type_filter_target_type"),
            "num_candidates_before_type_filter": row.get("num_candidates_before_type_filter"),
            "num_candidates_after_type_filter": row.get("num_candidates_after_type_filter"),
            "num_candidates_removed_by_type_filter": row.get("num_candidates_removed_by_type_filter"),
            "empty_after_type_filter": row.get("empty_after_type_filter"),
            "ontology_filter_applied": row.get("ontology_filter_applied"),
            "ontology_filter_mode": row.get("ontology_filter_mode"),
            "ontology_fallback_used": row.get("ontology_fallback_used"),
            "num_candidates_before_ontology_filter": row.get("num_candidates_before_ontology_filter"),
            "num_candidates_after_ontology_filter": row.get("num_candidates_after_ontology_filter"),
            "num_candidates_removed_by_ontology_filter": row.get("num_candidates_removed_by_ontology_filter"),
            "gold_in_topk_ontology": row.get("gold_in_topk_ontology"),
            "gold_rank_in_ontology_candidates": row.get("gold_rank_in_ontology_candidates"),
        }

    if row_name == "soft_support_raw":
        return {
            "source_stage": "candidate_stage_main_intermediate",
            "variant_name": row.get("variant_name"),
            "support_scores": row.get("support_scores"),
            "support_rank_order": row.get("support_rank_order"),
            "candidate_debug_rows": row.get("candidate_debug_rows"),
        }

    if row_name == "retrieval_main":
        sg_summary = row.get("subgraph_summary", {})
        ts_rows = row.get("triple_score_rows", [])
        path_scores = row.get("path_scores", [])

        return {
            "source_stage": "retrieval_stage_main_row",
            "variant_name": row.get("variant_name"),
            "selected_source_variant": row.get("selected_source_variant"),
            "subgraph_summary": sg_summary,
            "num_selected_triples": sg_summary.get("num_selected_triples", len(row.get("selected_subgraph", []))),
            "num_triple_score_rows": len(ts_rows),
            "num_path_scores": len(path_scores),
            "avg_triple_score_row_value": (
                round(mean([float(x.get("triple_score", 0.0)) for x in ts_rows]), 6) if ts_rows else 0.0
            ),
                        "direct_shortcut_path_rate": (
                round(
                    float(sg_summary.get("num_selected_direct_shortcuts", 0)) /
                    max(int(sg_summary.get("selected_subgraph_size", 0)), 1),
                    6
                )
                if sg_summary.get("selected_subgraph_size") is not None
                else None
            ),
            "contradiction_path_rate": (
                round(
                    float(sg_summary.get("num_selected_contra_triples", 0)) /
                    max(int(sg_summary.get("selected_subgraph_size", 0)), 1),
                    6
                )
                if sg_summary.get("selected_subgraph_size") is not None
                else None
            ),
            "candidate_coverage_preserved_rate": sg_summary.get("candidate_coverage_preserved_rate"),
        }

    return {"source_stage": "unknown"}


def build_eval_rows(row_name: str, rows: list[dict], valid_b_index: dict) -> tuple[list[dict], dict]:
    out_rows = []

    ranks = []
    missing_valid_b = 0

    for i, row in enumerate(rows):
        query = row["query_entity"]
        gold = row["gold_entity"]
        gold_id = row["gold_entity_id"]
        candidates, candidate_ids = extract_candidates(row)

        gold_present, rank, gold_rank_source = get_gold_present_and_rank(
            row_name=row_name,
            row=row,
            candidate_ids=candidate_ids,
            gold_id=gold_id,
        )

        valid_b_row = valid_b_index.get((query, gold))
        if valid_b_row is None:
            missing_valid_b += 1

        setting_b_payload = reorder_setting_b_fields(valid_b_row, candidates)

        eval_row = {
            "eval_row_name": row_name,
            "row_uid": get_row_uid(row, i),
            "row_index": row.get("row_index", i),
            "split": row.get("split", "valid"),
            "query_entity": query,
            "query_entity_id": row["query_entity_id"],
            "gold_entity": gold,
            "gold_entity_id": gold_id,
            "candidate_entities": candidates,
            "candidate_entity_ids": candidate_ids,
            "num_candidates": len(candidate_ids),
            "gold_present": gold_present,
            "gold_rank": rank,
            "gold_rank_source": gold_rank_source,
            "top1_candidate": candidates[0],
            "top5_candidates": candidates[:5],
            "setting_b_annotations": setting_b_payload,
            "row_metrics_ready": {
                "gold_present_rate_item": 1 if gold_present else 0,
                "reciprocal_rank_item": round(1.0 / rank, 8),
                "hits1_item": 1 if rank <= 1 else 0,
                "hits3_item": 1 if rank <= 3 else 0,
                "hits10_item": 1 if rank <= 10 else 0,
            },
            "stage_specific": stage_specific_payload(row_name, row),
        }

        out_rows.append(eval_row)
        ranks.append(rank)

    summary = {
        "eval_row_name": row_name,
        "num_rows": len(out_rows),
        "missing_valid_b_rows": missing_valid_b,
        "gold_present_rate": round(sum(1 for r in ranks if r <= 20) / len(ranks), 6) if ranks else 0.0,
        "mrr_like": round(sum(1.0 / r for r in ranks) / len(ranks), 6) if ranks else 0.0,
        "hits1_like": round(sum(1 for r in ranks if r <= 1) / len(ranks), 6) if ranks else 0.0,
        "hits3_like": round(sum(1 for r in ranks if r <= 3) / len(ranks), 6) if ranks else 0.0,
        "hits10_like": round(sum(1 for r in ranks if r <= 10) / len(ranks), 6) if ranks else 0.0,
        "avg_gold_rank": round(sum(ranks) / len(ranks), 6) if ranks else 0.0,
    }

    return out_rows, summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)

    valid_b_rows = load_json(INPUTS["valid_b"])
    valid_b_index = build_valid_b_index(valid_b_rows)

    all_summaries = {}
    out_file_map = {
        "backbone_raw": OUT_DIR / "valid_backbone_raw_eval.json",
        "ontology_raw": OUT_DIR / "valid_ontology_raw_eval.json",
        "soft_support_raw": OUT_DIR / "valid_soft_support_raw_eval.json",
        "retrieval_main": OUT_DIR / "valid_retrieval_main_eval.json",
    }

    for row_name in ["backbone_raw", "ontology_raw", "soft_support_raw", "retrieval_main"]:
        rows = load_json(INPUTS[row_name])
        eval_rows, summary = build_eval_rows(row_name, rows, valid_b_index)
        write_json(out_file_map[row_name], eval_rows)
        all_summaries[row_name] = summary
        print(f"[OK] wrote {out_file_map[row_name]}")

    lines = []
    lines.append("# Valid Eval-Ready Package\n")
    lines.append("## Goal")
    lines.append("- Standardize 4 valid rows into one eval-ready schema.")
    lines.append("- Prepare clean inputs for valid main table building.")
    lines.append("")
    lines.append("## Output files")
    for name, path in out_file_map.items():
        lines.append(f"- `{name}` -> `{path.relative_to(ROOT)}`")
    lines.append("")
    lines.append("## Summaries")
    for name, summary in all_summaries.items():
        lines.append(f"### {name}")
        for k, v in summary.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")
    lines.append("## Notes")
    lines.append("- No table or ablation decision is made in this step.")
    lines.append("- This script only standardizes row schemas and joins Setting-B annotations.")
    lines.append("")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {OUT_REPORT}")
    print(json.dumps(all_summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
