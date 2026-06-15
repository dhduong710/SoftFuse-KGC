from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(".").resolve()

# ---------- input test sources ----------
BACKBONE_TEST = ROOT / "dataset/setting_a/backbone_candidates/test_top20_raw.json"
ONTOLOGY_TEST = ROOT / "dataset/setting_a/ontology_control/test_top20_ontology_raw.json"
SOFT_TEST = ROOT / "dataset/setting_a/soft_support_ranked_candidates/test_top20_soft_support_main.json"
RETR_TEST = ROOT / "dataset/setting_a/fuzzy_retrieval/test_fuzzy_retrieval_main.json"
TEST_B = ROOT / "dataset/setting_b/contra_checked/test_b_annotations_contra_checked.json"

# ---------- valid eval-ready references ----------
VALID_REF = {
    "backbone_raw": ROOT / "dataset/setting_b/eval_valid/valid_backbone_raw_eval.json",
    "ontology_raw": ROOT / "dataset/setting_b/eval_valid/valid_ontology_raw_eval.json",
    "soft_support_raw": ROOT / "dataset/setting_b/eval_valid/valid_soft_support_raw_eval.json",
    "retrieval_main": ROOT / "dataset/setting_b/eval_valid/valid_retrieval_main_eval.json",
}

# ---------- outputs ----------
OUT_DIR = ROOT / "dataset/setting_b/eval_test"
OUT_PATHS = {
    "backbone_raw": OUT_DIR / "test_backbone_raw_eval.json",
    "ontology_raw": OUT_DIR / "test_ontology_raw_eval.json",
    "soft_support_raw": OUT_DIR / "test_soft_support_raw_eval.json",
    "retrieval_main": OUT_DIR / "test_retrieval_main_eval.json",
}
SUMMARY_PATH = ROOT / "outputs/evaluation/test_eval_ready_summary.json"
REPORT_PATH = ROOT / "outputs/evaluation/reports/test_eval_ready.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def key_for(query_entity: str, gold_entity: str, split: str = "test") -> str:
    return f"{split}|||{query_entity}|||{gold_entity}"


def align_candidate_values(
    source_names: List[str],
    source_values: List[Any],
    target_names: List[str],
    default=None,
) -> List[Any]:
    mp = {n: v for n, v in zip(source_names, source_values)}
    return [mp.get(n, default) for n in target_names]


def build_setting_b_annotations(tb_row: Dict[str, Any], target_candidates: List[str]) -> Dict[str, Any]:
    candidate_drugs = tb_row.get("candidate_drugs", [])
    candidate_types = tb_row.get("candidate_types", [])
    contra_flags = tb_row.get("contra_flags_lookup_checked", tb_row.get("contra_flags", []))
    conflict_flags = tb_row.get("conflict_flags", [])

    aligned_types = align_candidate_values(candidate_drugs, candidate_types, target_candidates, default=None)
    aligned_contra = align_candidate_values(candidate_drugs, contra_flags, target_candidates, default=0)
    aligned_conflict = align_candidate_values(candidate_drugs, conflict_flags, target_candidates, default=0)

    return {
        "query_disease": tb_row.get("query_disease"),
        "gold_drug": tb_row.get("gold_drug"),
        "setting_a_relation": tb_row.get("setting_a_relation"),
        "candidate_types": aligned_types,
        "contra_flags": aligned_contra,
        "conflict_flags": aligned_conflict,
        "gold_type": tb_row.get("gold_type"),
        "gold_is_contraindicated": tb_row.get("gold_is_contraindicated"),
        "gold_conflict_flag": tb_row.get("gold_conflict_flag"),
        "has_any_contra_candidate": any(bool(x) for x in aligned_contra),
    }


def compute_gold_metrics(candidate_entities: List[str], gold_entity: str) -> Tuple[bool, int, float, int, int, int]:
    if gold_entity in candidate_entities:
        rank = candidate_entities.index(gold_entity) + 1
        present = True
        rr = 1.0 / rank
    else:
        rank = 21
        present = False
        rr = 0.0

    hits1 = int(rank <= 1)
    hits3 = int(rank <= 3)
    hits10 = int(rank <= 10)
    return present, rank, rr, hits1, hits3, hits10


def build_top5_diag_from_debug(candidate_debug_rows: List[Dict[str, Any]]) -> Dict[str, float]:
    top5 = candidate_debug_rows[:5]
    denom = max(len(top5), 1)
    return {
        "avg_top5_direct_link_rate": round(
            sum(1 for x in top5 if x.get("candidate_query_edge_count", 0) > 0) / denom, 6
        ),
        "avg_top5_evidence_positive_rate": round(
            sum(1 for x in top5 if x.get("evidence_edge_touch_count", 0) > 0) / denom, 6
        ),
        "avg_top5_ontology_keep_rate": round(
            sum(int(x.get("ontology_keep_flag", 0)) for x in top5) / denom, 6
        ),
        "avg_top5_contra_rate": round(
            sum(int(x.get("contra_flag", 0)) for x in top5) / denom, 6
        ),
    }


def build_retrieval_diag(row: Dict[str, Any]) -> Dict[str, Any]:
    triple_rows = row.get("triple_score_rows", [])
    denom = max(len(triple_rows), 1)

    avg_triple_score = (
        round(sum(float(x.get("triple_score", 0.0)) for x in triple_rows) / denom, 6)
        if triple_rows else 0.0
    )
    direct_shortcut_path_rate = (
        round(sum(1 for x in triple_rows if x.get("direct_candidate_query_flag", False)) / denom, 6)
        if triple_rows else 0.0
    )
    contradiction_path_rate = (
        round(sum(1 for x in triple_rows if x.get("contra_flag", False)) / denom, 6)
        if triple_rows else 0.0
    )

    return {
        "avg_triple_score": avg_triple_score,
        "direct_shortcut_path_rate": direct_shortcut_path_rate,
        "contradiction_path_rate": contradiction_path_rate,
        "subgraph_summary": row.get("subgraph_summary", {}),
    }


def build_stage_specific(row_name: str, row: Dict[str, Any]) -> Dict[str, Any]:
    if row_name == "backbone_raw":
        return {
            "source_variant": "backbone_raw",
            "gold_in_topk_raw": row.get("gold_in_topk_raw"),
            "gold_rank_in_full_universe": row.get("gold_rank_in_full_universe"),
        }

    if row_name == "ontology_raw":
        extras = {}
        for k in [
            "ontology_filter_applied",
            "ontology_filter_mode",
            "ontology_fallback_used",
            "gold_in_topk_ontology",
            "gold_rank_in_ontology_candidates",
            "num_candidates_before_ontology_filter",
            "num_candidates_after_ontology_filter",
            "num_candidates_removed_by_ontology_filter",
            "candidate_support_types",
        ]:
            if k in row:
                extras[k] = row[k]
        extras["source_variant"] = "ontology_raw"
        return extras

    if row_name == "soft_support_raw":
        return {
            "variant_name": row.get("variant_name"),
            "support_scores": row.get("support_scores", []),
            "support_rank_order": row.get("support_rank_order", []),
            **build_top5_diag_from_debug(row.get("candidate_debug_rows", [])),
        }

    if row_name == "retrieval_main":
        return {
            "variant_name": row.get("variant_name"),
            "selected_source_variant": row.get("selected_source_variant"),
            "support_scores": row.get("support_scores", []),
            "candidate_support_bands": row.get("candidate_support_bands", []),
            "contra_flags": row.get("contra_flags", []),
            **build_retrieval_diag(row),
        }

    return {}


def metric_get(metrics: Dict[str, Any], *names: str, default=None):
    for name in names:
        if name in metrics and metrics[name] is not None:
            return metrics[name]
    return default


def avg(vals: List[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def mrr_present_only_from_row(row: Dict[str, Any]) -> float | None:
    if row.get("gold_present"):
        rank = int(row.get("gold_rank", 21))
        if rank <= 20:
            return 1.0 / rank
    return None


def make_metric_payload_by_valid_ref(
    row_name: str,
    present: bool,
    gold_rank: int,
    rr: float,
    hits1: int,
    hits3: int,
    hits10: int,
    ref_metrics_keys: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Build row_metrics_ready to match exactly the valid-side reference key set/order.
    Supports both old valid schema:
      - gold_present_rate_item, reciprocal_rank_item, hits1_item, ...
    and newer schema:
      - mrr_at20, hits1_at20, ...
    """
    canonical = {
        "gold_present_rate_item": int(bool(present)),
        "reciprocal_rank_item": round(rr, 8),
        "hits1_item": int(hits1),
        "hits3_item": int(hits3),
        "hits10_item": int(hits10),

        "gold_present_rate": int(bool(present)),
        "mrr_at20": round(rr, 8),
        "rr_at20": round(rr, 8),
        "mrr_present_only": round(rr, 8) if present else None,
        "hits1_at20": int(hits1),
        "hits3_at20": int(hits3),
        "hits10_at20": int(hits10),
        "avg_gold_rank": int(gold_rank),

        "mrr": round(rr, 8),
        "hits1": int(hits1),
        "hits3": int(hits3),
        "hits10": int(hits10),
    }

    keys = ref_metrics_keys[row_name]
    return {k: canonical.get(k) for k in keys}


def build_eval_row(
    row_name: str,
    row_index: int,
    row: Dict[str, Any],
    tb_row: Dict[str, Any],
    ref_metrics_keys: Dict[str, List[str]],
) -> Dict[str, Any]:
    split = row.get("split", "test")
    query_entity = row["query_entity"]
    query_entity_id = row["query_entity_id"]
    gold_entity = row["gold_entity"]
    gold_entity_id = row["gold_entity_id"]
    candidate_entities = row["candidate_entities"]
    candidate_entity_ids = row["candidate_entity_ids"]

    present, gold_rank, rr, hits1, hits3, hits10 = compute_gold_metrics(candidate_entities, gold_entity)
    setting_b_annotations = build_setting_b_annotations(tb_row, candidate_entities)

    eval_row = {
        "eval_row_name": row_name,
        "row_uid": f"{row_name}::{row_index}::{query_entity_id}::{gold_entity_id}",
        "row_index": row_index,
        "split": split,
        "query_entity": query_entity,
        "query_entity_id": query_entity_id,
        "gold_entity": gold_entity,
        "gold_entity_id": gold_entity_id,
        "candidate_entities": candidate_entities,
        "candidate_entity_ids": candidate_entity_ids,
        "num_candidates": len(candidate_entities),
        "gold_present": bool(present),
        "gold_rank": int(gold_rank),
        "gold_rank_source": "candidate_entities_order_top20",
        "top1_candidate": candidate_entities[0] if candidate_entities else None,
        "top5_candidates": candidate_entities[:5],
        "setting_b_annotations": setting_b_annotations,
        "row_metrics_ready": make_metric_payload_by_valid_ref(
            row_name=row_name,
            present=present,
            gold_rank=gold_rank,
            rr=rr,
            hits1=hits1,
            hits3=hits3,
            hits10=hits10,
            ref_metrics_keys=ref_metrics_keys,
        ),
        "stage_specific": build_stage_specific(row_name, row),
    }
    return eval_row


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    backbone = load_json(BACKBONE_TEST)
    ontology = load_json(ONTOLOGY_TEST)
    soft = load_json(SOFT_TEST)
    retr = load_json(RETR_TEST)
    tb = load_json(TEST_B)

    tb_map = {}
    for row in tb:
        tb_map[key_for(row["query_disease"], row["gold_drug"], "test")] = row

    source_rows = {
        "backbone_raw": backbone,
        "ontology_raw": ontology,
        "soft_support_raw": soft,
        "retrieval_main": retr,
    }

    outputs = {}
    summaries = {}

    ref_top_keys = {}
    ref_metrics_keys = {}
    for row_name, ref_path in VALID_REF.items():
        ref_data = load_json(ref_path)
        ref_top_keys[row_name] = list(ref_data[0].keys())
        ref_metrics_keys[row_name] = list(ref_data[0]["row_metrics_ready"].keys())

    common_query_sets = {}

    for row_name, rows in source_rows.items():
        assert isinstance(rows, list) and len(rows) == 500, f"{row_name} must have 500 rows"

        eval_rows = []
        for i, row in enumerate(rows):
            key = key_for(row["query_entity"], row["gold_entity"], row.get("split", "test"))
            tb_row = tb_map.get(key)
            if tb_row is None:
                raise KeyError(f"Missing test_b annotation for key={key}")

            eval_row = build_eval_row(row_name, i, row, tb_row, ref_metrics_keys)
            eval_rows.append(eval_row)

        outputs[row_name] = eval_rows
        save_json(OUT_PATHS[row_name], eval_rows)

        present_count = sum(1 for x in eval_rows if x["gold_present"])

        mrr_at20_vals = [
            float(metric_get(
                x["row_metrics_ready"],
                "mrr_at20", "rr_at20", "reciprocal_rank_item", "mrr_like", "mrr",
                default=0.0
            ))
            for x in eval_rows
        ]
        hits1_vals = [
            int(metric_get(
                x["row_metrics_ready"],
                "hits1_at20", "hits1_item", "hits1_like", "hits1",
                default=0
            ))
            for x in eval_rows
        ]
        hits3_vals = [
            int(metric_get(
                x["row_metrics_ready"],
                "hits3_at20", "hits3_item", "hits3_like", "hits3",
                default=0
            ))
            for x in eval_rows
        ]
        hits10_vals = [
            int(metric_get(
                x["row_metrics_ready"],
                "hits10_at20", "hits10_item", "hits10_like", "hits10",
                default=0
            ))
            for x in eval_rows
        ]
        avg_gold_rank_vals = [float(x["gold_rank"]) for x in eval_rows]

        mrr_present_only_vals = []
        for x in eval_rows:
            val = metric_get(x["row_metrics_ready"], "mrr_present_only", default=None)
            if val is None:
                val = mrr_present_only_from_row(x)
            if val is not None:
                mrr_present_only_vals.append(float(val))

        mrr_at20 = avg(mrr_at20_vals)
        hits1 = avg(hits1_vals)
        hits3 = avg(hits3_vals)
        hits10 = avg(hits10_vals)
        avg_gold_rank = avg(avg_gold_rank_vals)

        query_set = [(x["query_entity_id"], x["gold_entity_id"]) for x in eval_rows]
        common_query_sets[row_name] = query_set

        summaries[row_name] = {
            "num_rows": len(eval_rows),
            "gold_present_rate": round(present_count / len(eval_rows), 6),
            "mrr_at20": round(mrr_at20, 8),
            "mrr_present_only": round(avg(mrr_present_only_vals), 8) if mrr_present_only_vals else None,
            "hits1_at20": round(hits1, 6),
            "hits3_at20": round(hits3, 6),
            "hits10_at20": round(hits10, 6),
            "avg_gold_rank": round(avg_gold_rank, 6),
            "top_keys_match_valid": list(eval_rows[0].keys()) == ref_top_keys[row_name],
            "metrics_keys_match_valid": list(eval_rows[0]["row_metrics_ready"].keys()) == ref_metrics_keys[row_name],
            "gold_rank_21_count": sum(1 for x in eval_rows if x["gold_rank"] == 21),
        }

    row_names = list(outputs.keys())
    query_set_equal = all(common_query_sets[row_names[0]] == common_query_sets[r] for r in row_names[1:])
    top_keys_equal = all(list(outputs[row_names[0]][0].keys()) == list(outputs[r][0].keys()) for r in row_names[1:])

    summary = {
        "stage": "build_test_eval_ready",
        "status": "BUILT",
        "output_paths": {k: str(v) for k, v in OUT_PATHS.items()},
        "row_summaries": summaries,
        "global_checks": {
            "all_rows_have_500_examples": all(summaries[r]["num_rows"] == 500 for r in summaries),
            "all_query_sets_match": query_set_equal,
            "all_top_level_keys_match_across_test_rows": top_keys_equal,
            "all_gold_rank_use_21_sentinel_when_missing": all(
                x["gold_rank"] == 21
                for r in outputs.values()
                for x in r
                if not x["gold_present"]
            ),
            "all_row_metrics_keys_match_valid_refs": all(summaries[r]["metrics_keys_match_valid"] for r in summaries),
            "all_top_keys_match_valid_refs": all(summaries[r]["top_keys_match_valid"] for r in summaries),
        },
        "metric_policy": {
            "main_metric": "mrr_at20",
            "rr_rule": "1/rank if rank <= 20 else 0",
            "gold_rank_out_of_top20": 21,
            "secondary_metrics": [
                "gold_present_rate",
                "mrr_present_only",
                "hits1_at20",
                "hits3_at20",
                "hits10_at20",
                "avg_gold_rank",
            ],
        },
    }

    save_json(SUMMARY_PATH, summary)

    md = []
    md.append("# Test Eval-Ready Package")
    md.append("")
    md.append(f"- status: **{summary['status']}**")
    md.append("")
    md.append("## 1. Output files")
    for k, v in summary["output_paths"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 2. Row summaries")
    for row_name, row_sum in summary["row_summaries"].items():
        md.append(f"### {row_name}")
        for k, v in row_sum.items():
            md.append(f"- {k}: `{v}`")
        md.append("")
    md.append("## 3. Global checks")
    for k, v in summary["global_checks"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 4. Metric policy")
    md.append(f"- main_metric: `{summary['metric_policy']['main_metric']}`")
    md.append(f"- rr_rule: `{summary['metric_policy']['rr_rule']}`")
    md.append(f"- gold_rank_out_of_top20: `{summary['metric_policy']['gold_rank_out_of_top20']}`")
    md.append("")
    md.append("## 5. Conclusion")
    md.append(
        "Built test eval-ready package for all four scientific rows under the reviewer-safe "
        "policy, with unified schema, gold_rank sentinel=21 for out-of-top20, and row-level metrics "
        "ready for locked-test aggregation."
    )
    REPORT_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
