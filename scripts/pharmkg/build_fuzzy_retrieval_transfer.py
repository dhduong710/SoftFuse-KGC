#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week 23 Day 4: Build PharmKG fuzzy retrieval / confidence-aware subgraph selection.

Input:
- dataset/setting_c_pharmkg/softfuse_ready/{valid,test}.json
- dataset/setting_c_pharmkg/soft_support/{valid,test}_top20_soft_support_main.json
- dataset/setting_c_pharmkg/support_features/{valid,test}_support_features.json
- outputs/pharmkg/soft_support_raw_eval_{valid,test}.json

Output:
- dataset/setting_c_pharmkg/fuzzy_retrieval/{valid,test}_path_features.json
- dataset/setting_c_pharmkg/fuzzy_retrieval/{valid,test}_fuzzy_retrieval_main.json
- dataset/setting_c_pharmkg/fuzzy_retrieval/retrieval_config.json
- dataset/setting_c_pharmkg/fuzzy_retrieval/retrieval_summary.json
- outputs/pharmkg/fuzzy_retrieval_eval_{valid,test}.json
- outputs/pharmkg/reports/day4_fuzzy_retrieval_transfer.md
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(".")

SOFTFUSE_READY_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "softfuse_ready"
SOFT_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "soft_support"
SUPPORT_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "support_features"
RETRIEVAL_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "fuzzy_retrieval"
GRAPH_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "graph"

RESULT_DIR = ROOT / "outputs" / "pharmkg"
REPORT_DIR = ROOT / "outputs" / "pharmkg" / "reports"

VALID_FUZZY_EVAL_PATH = RESULT_DIR / "fuzzy_retrieval_eval_valid.json"
TEST_FUZZY_EVAL_PATH = RESULT_DIR / "fuzzy_retrieval_eval_test.json"
REPORT_PATH = REPORT_DIR / "day4_fuzzy_retrieval_transfer.md"

TOP_K = 20
ABSENT_RANK = 21

TARGET_RELATION = "T"
TARGET_RELATION_NORMALIZED = "therapeutic_association_proxy"

VARIANT_NAME = "fuzzy_retrieval_main"
SOURCE_VARIANT_NAME = "soft_support_pharmkg_b050"

CONFIG = {
    "variant_name": VARIANT_NAME,
    "source_variant_name": SOURCE_VARIANT_NAME,
    "retain_ratio": 0.55,
    "min_keep": 18,
    "top_band_weight": 1.0,
    "mid_band_weight": 0.6,
    "tail_band_weight": 0.25,
    "touch_candidate_weight": 0.35,
    "touch_query_weight": 0.05,
    "direct_shortcut_penalty": -0.80,
    "contra_penalty": 0.0,
    "density_weight": 0.05,
    "selection_policy": "coverage_pass_best_triple_per_candidate_then_global_fill",
    "candidate_bands": {
        "top_band": [1, 5],
        "mid_band": [6, 10],
        "tail_band": [11, 20],
    },
    "ranking_policy": "do_not_reorder_candidates; preserve soft_support ranking",
    "gold_injection": False,
    "target_relation": TARGET_RELATION,
    "target_relation_normalized": TARGET_RELATION_NORMALIZED,
}


def ensure_dirs() -> None:
    for path in [RETRIEVAL_DIR, RESULT_DIR, REPORT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_id2relation() -> dict[int, str]:
    raw = read_json(GRAPH_DIR / "id2relation.json")
    return {int(k): str(v) for k, v in raw.items()}


def band_for_position(pos: int) -> str:
    if 1 <= pos <= 5:
        return "top_band"
    if 6 <= pos <= 10:
        return "mid_band"
    return "tail_band"


def band_weight(band: str) -> float:
    if band == "top_band":
        return float(CONFIG["top_band_weight"])
    if band == "mid_band":
        return float(CONFIG["mid_band_weight"])
    return float(CONFIG["tail_band_weight"])


def compute_rank_and_rr(gold_entity: str, candidates: list[str]) -> tuple[int, bool, float]:
    if gold_entity in candidates:
        rank = candidates.index(gold_entity) + 1
        if rank <= TOP_K:
            return rank, True, 1.0 / rank
    return ABSENT_RANK, False, 0.0


def build_ready_by_index(split: str) -> dict[int, dict[str, Any]]:
    rows = read_json(SOFTFUSE_READY_DIR / f"{split}.json")
    return {int(r["row_index"]): r for r in rows}


def build_support_by_index(split: str) -> dict[int, dict[str, Any]]:
    rows = read_json(SUPPORT_DIR / f"{split}_support_features.json")
    return {int(r["row_index"]): r for r in rows}


def build_candidate_info(soft_row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    candidate_entities = list(soft_row["candidate_entities"])
    candidate_ids = [int(x) for x in soft_row["candidate_entity_ids"]]
    support_scores = [float(x) for x in soft_row["support_scores"]]

    debug_by_id = {
        int(d["candidate_entity_id"]): d
        for d in soft_row.get("candidate_debug_rows", [])
    }

    out: dict[int, dict[str, Any]] = {}

    for pos, (name, cid, score) in enumerate(
        zip(candidate_entities, candidate_ids, support_scores),
        start=1,
    ):
        band = band_for_position(pos)
        debug = debug_by_id.get(cid, {})

        out[cid] = {
            "candidate_entity": name,
            "candidate_entity_id": int(cid),
            "soft_rank": int(pos),
            "support_score": float(score),
            "support_band": band,
            "support_band_weight": float(band_weight(band)),
            "base_rank": int(debug.get("base_rank", pos)),
            "direct_T_candidate_query_flag": int(debug.get("direct_T_candidate_query_flag", 0)),
            "evidence_positive": int(debug.get("evidence_positive", 1)),
            "contra_flag": int(debug.get("contra_flag", 0)),
        }

    return out


def make_triple_feature_rows(
    ready_row: dict[str, Any],
    soft_row: dict[str, Any],
    id2relation: dict[int, str],
) -> list[dict[str, Any]]:
    query_id = int(ready_row["query_entity_id"])
    candidate_info = build_candidate_info(soft_row)
    candidate_ids = set(candidate_info.keys())

    original_subgraph = [
        (int(h), int(r), int(t))
        for h, r, t in ready_row["subgraph"]
    ]

    triple_counter = Counter(original_subgraph)

    rows: list[dict[str, Any]] = []

    for idx, (h, r, t) in enumerate(original_subgraph):
        touched_candidate_ids = []
        for node in [h, t]:
            if node in candidate_ids and node not in touched_candidate_ids:
                touched_candidate_ids.append(node)

        touched_candidate_positions = [
            int(candidate_info[cid]["soft_rank"])
            for cid in touched_candidate_ids
        ]

        touched_candidate_names = [
            candidate_info[cid]["candidate_entity"]
            for cid in touched_candidate_ids
        ]

        touched_bands = [
            candidate_info[cid]["support_band"]
            for cid in touched_candidate_ids
        ]

        touches_candidate = int(len(touched_candidate_ids) > 0)
        touches_query = int(h == query_id or t == query_id)
        touches_top_band_candidate = int(
            any(candidate_info[cid]["soft_rank"] <= 5 for cid in touched_candidate_ids)
        )

        direct_candidate_query_flag = int(touches_candidate and touches_query)
        relation_name = id2relation.get(int(r), str(r))
        direct_T_candidate_query_flag = int(
            direct_candidate_query_flag and relation_name == TARGET_RELATION
        )

        best_band_score = 0.0
        if touched_bands:
            best_band_score = max(band_weight(b) for b in touched_bands)

        local_density_hint = int(len(touched_candidate_ids) + touches_query)

        triple_score = (
            best_band_score
            + CONFIG["touch_candidate_weight"] * touches_candidate
            + CONFIG["touch_query_weight"] * touches_query
            + CONFIG["direct_shortcut_penalty"] * direct_candidate_query_flag
            + CONFIG["contra_penalty"] * 0
            + CONFIG["density_weight"] * local_density_hint
        )

        rows.append(
            {
                "triple_index": int(idx),
                "head_id": int(h),
                "relation_id": int(r),
                "relation_name": relation_name,
                "tail_id": int(t),
                "triple": [int(h), int(r), int(t)],
                "touches_query": int(touches_query),
                "touches_candidate": int(touches_candidate),
                "touched_candidate_positions": touched_candidate_positions,
                "touched_candidate_ids": [int(x) for x in touched_candidate_ids],
                "touched_candidate_names": touched_candidate_names,
                "touched_candidate_bands": touched_bands,
                "touches_top_band_candidate": int(touches_top_band_candidate),
                "direct_candidate_query_flag": int(direct_candidate_query_flag),
                "direct_T_candidate_query_flag": int(direct_T_candidate_query_flag),
                "contra_flag": 0,
                "triple_frequency_in_subgraph": int(triple_counter[(h, r, t)]),
                "local_density_hint": int(local_density_hint),
                "band_score": float(best_band_score),
                "triple_score": float(triple_score),
            }
        )

    return rows


def select_triples(
    triple_rows: list[dict[str, Any]],
    candidate_ids_in_order: list[int],
    original_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_size = max(
        int(CONFIG["min_keep"]),
        int(round(original_size * float(CONFIG["retain_ratio"]))),
    )
    target_size = min(target_size, original_size)

    selected_by_index: dict[int, dict[str, Any]] = {}

    by_candidate: dict[int, list[dict[str, Any]]] = {int(cid): [] for cid in candidate_ids_in_order}

    for tr in triple_rows:
        for cid in tr["touched_candidate_ids"]:
            if cid in by_candidate:
                by_candidate[cid].append(tr)

    for cid in candidate_ids_in_order:
        cand_triples = by_candidate.get(cid, [])
        if not cand_triples:
            continue

        best = sorted(
            cand_triples,
            key=lambda x: (-float(x["triple_score"]), int(x["triple_index"])),
        )[0]

        selected_by_index[int(best["triple_index"])] = best

        if len(selected_by_index) >= target_size:
            break

    global_sorted = sorted(
        triple_rows,
        key=lambda x: (-float(x["triple_score"]), int(x["triple_index"])),
    )

    for tr in global_sorted:
        if len(selected_by_index) >= target_size:
            break
        selected_by_index[int(tr["triple_index"])] = tr

    selected_rows = sorted(
        selected_by_index.values(),
        key=lambda x: (-float(x["triple_score"]), int(x["triple_index"])),
    )

    summary = summarize_selection(
        triple_rows=triple_rows,
        selected_rows=selected_rows,
        candidate_ids_in_order=candidate_ids_in_order,
        target_size=target_size,
    )

    return selected_rows, summary


def touched_candidate_set(rows: list[dict[str, Any]]) -> set[int]:
    out = set()
    for r in rows:
        for cid in r["touched_candidate_ids"]:
            out.add(int(cid))
    return out


def summarize_selection(
    triple_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    candidate_ids_in_order: list[int],
    target_size: int,
) -> dict[str, Any]:
    original_touched = touched_candidate_set(triple_rows)
    selected_touched = touched_candidate_set(selected_rows)

    top_band_ids = set(candidate_ids_in_order[:5])
    mid_band_ids = set(candidate_ids_in_order[5:10])
    tail_band_ids = set(candidate_ids_in_order[10:20])

    original_top_touched = original_touched & top_band_ids
    selected_top_touched = selected_touched & top_band_ids

    original_mid_touched = original_touched & mid_band_ids
    selected_mid_touched = selected_touched & mid_band_ids

    original_tail_touched = original_touched & tail_band_ids
    selected_tail_touched = selected_touched & tail_band_ids

    def rate(num: int, den: int) -> float:
        return float(num / den) if den else 1.0

    original_direct = sum(int(r["direct_candidate_query_flag"]) for r in triple_rows)
    selected_direct = sum(int(r["direct_candidate_query_flag"]) for r in selected_rows)

    original_direct_t = sum(int(r["direct_T_candidate_query_flag"]) for r in triple_rows)
    selected_direct_t = sum(int(r["direct_T_candidate_query_flag"]) for r in selected_rows)

    selected_scores = [float(r["triple_score"]) for r in selected_rows]

    return {
        "original_subgraph_size": int(len(triple_rows)),
        "selected_subgraph_size": int(len(selected_rows)),
        "target_size": int(target_size),
        "retain_ratio_actual": float(len(selected_rows) / max(1, len(triple_rows))),
        "num_original_direct_shortcuts": int(original_direct),
        "num_selected_direct_shortcuts": int(selected_direct),
        "num_original_direct_T_shortcuts": int(original_direct_t),
        "num_selected_direct_T_shortcuts": int(selected_direct_t),
        "direct_shortcut_path_rate_original": float(original_direct / max(1, len(triple_rows))),
        "direct_shortcut_path_rate_selected": float(selected_direct / max(1, len(selected_rows))),
        "direct_T_shortcut_rate_original": float(original_direct_t / max(1, len(triple_rows))),
        "direct_T_shortcut_rate_selected": float(selected_direct_t / max(1, len(selected_rows))),
        "original_candidate_touched_count": int(len(original_touched)),
        "selected_candidate_touched_count": int(len(selected_touched)),
        "candidate_coverage_preserved_rate": rate(
            len(selected_touched & original_touched),
            len(original_touched),
        ),
        "original_top_band_touched_count": int(len(original_top_touched)),
        "selected_top_band_touched_count": int(len(selected_top_touched)),
        "top_band_coverage_preserved_rate": rate(
            len(selected_top_touched & original_top_touched),
            len(original_top_touched),
        ),
        "mid_band_coverage_preserved_rate": rate(
            len(selected_mid_touched & original_mid_touched),
            len(original_mid_touched),
        ),
        "tail_band_coverage_preserved_rate": rate(
            len(selected_tail_touched & original_tail_touched),
            len(original_tail_touched),
        ),
        "avg_selected_triple_score": (
            float(sum(selected_scores) / len(selected_scores))
            if selected_scores
            else 0.0
        ),
        "min_selected_triple_score": float(min(selected_scores)) if selected_scores else 0.0,
        "max_selected_triple_score": float(max(selected_scores)) if selected_scores else 0.0,
    }


def build_split(
    split: str,
    id2relation: dict[int, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    ready_by_idx = build_ready_by_index(split)
    support_by_idx = build_support_by_index(split)

    soft_rows = read_json(SOFT_DIR / f"{split}_top20_soft_support_main.json")

    path_feature_rows: list[dict[str, Any]] = []
    fuzzy_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for soft in soft_rows:
        row_index = int(soft["row_index"])
        ready = ready_by_idx[row_index]
        support = support_by_idx[row_index]

        if int(ready["query_entity_id"]) != int(soft["query_entity_id"]):
            raise ValueError(f"{split} row {row_index}: query mismatch.")
        if int(ready["gold_entity_id"]) != int(soft["gold_entity_id"]):
            raise ValueError(f"{split} row {row_index}: gold mismatch.")

        triple_rows = make_triple_feature_rows(
            ready_row=ready,
            soft_row=soft,
            id2relation=id2relation,
        )

        candidate_ids = [int(x) for x in soft["candidate_entity_ids"]]

        selected_rows, subgraph_summary = select_triples(
            triple_rows=triple_rows,
            candidate_ids_in_order=candidate_ids,
            original_size=len(triple_rows),
        )

        selected_subgraph = [
            list(map(int, tr["triple"]))
            for tr in selected_rows
        ]

        candidate_entities = list(soft["candidate_entities"])
        candidate_entity_ids = [int(x) for x in soft["candidate_entity_ids"]]
        support_scores = [float(x) for x in soft["support_scores"]]

        rank, present, rr = compute_rank_and_rr(
            gold_entity=soft["gold_entity"],
            candidates=candidate_entities,
        )

        path_feature_rows.append(
            {
                "split": split,
                "row_index": row_index,
                "row_uid": f"{split}_{row_index}",
                "query_entity": soft["query_entity"],
                "query_entity_id": int(soft["query_entity_id"]),
                "gold_entity": soft["gold_entity"],
                "gold_entity_id": int(soft["gold_entity_id"]),
                "candidate_entities": candidate_entities,
                "candidate_entity_ids": candidate_entity_ids,
                "support_scores": support_scores,
                "candidate_support_bands": {
                    str(cid): band_for_position(pos)
                    for pos, cid in enumerate(candidate_entity_ids, start=1)
                },
                "source_support_row_summary": support.get("row_summary", {}),
                "triple_feature_rows": triple_rows,
            }
        )

        fuzzy_row = {
            "split": split,
            "row_index": row_index,
            "query_entity": soft["query_entity"],
            "query_entity_id": int(soft["query_entity_id"]),
            "gold_entity": soft["gold_entity"],
            "gold_entity_id": int(soft["gold_entity_id"]),
            "candidate_entities": candidate_entities,
            "candidate_entity_ids": candidate_entity_ids,
            "rank_entities": candidate_entities,
            "rank_entities_id": candidate_entity_ids,
            "support_scores": support_scores,
            "gold_rank_in_top20_or_21": int(rank),
            "gold_in_topk_fuzzy_retrieval": bool(present),
            "candidate_universe": "drug_only_from_train_T_heads",
            "gold_injection": False,
            "variant_name": VARIANT_NAME,
            "source_variant_name": SOURCE_VARIANT_NAME,
            "target_relation": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "original_subgraph": ready["subgraph"],
            "selected_subgraph": selected_subgraph,
            "triple_score_rows": selected_rows,
            "subgraph_summary": subgraph_summary,
        }

        eval_row = {
            "eval_row_name": "fuzzy_retrieval_main",
            "row_index": row_index,
            "split": split,
            "query_entity": soft["query_entity"],
            "query_entity_id": int(soft["query_entity_id"]),
            "gold_entity": soft["gold_entity"],
            "gold_entity_id": int(soft["gold_entity_id"]),
            "candidate_entities": candidate_entities,
            "candidate_entity_ids": candidate_entity_ids,
            "num_candidates": int(len(candidate_entities)),
            "gold_present": bool(present),
            "gold_rank": int(rank),
            "gold_rank_source": VARIANT_NAME,
            "reciprocal_rank_item": float(rr),
            "hits1_item": int(rank <= 1),
            "hits3_item": int(rank <= 3),
            "hits10_item": int(rank <= 10),
            "hits20_item": int(rank <= 20),
            "candidate_universe": "drug_only_from_train_T_heads",
            "gold_injection": False,
            "target_relation": TARGET_RELATION,
            "target_relation_normalized": TARGET_RELATION_NORMALIZED,
            "stage_specific": {
                "stage": "fuzzy_retrieval_main",
                "variant_name": VARIANT_NAME,
                "source_variant_name": SOURCE_VARIANT_NAME,
                "ranking_policy": "candidate ranking copied from soft_support_raw",
                "subgraph_summary": subgraph_summary,
            },
        }

        fuzzy_rows.append(fuzzy_row)
        eval_rows.append(eval_row)

    metrics = summarize_eval_rows(eval_rows, "fuzzy_retrieval_main")
    retrieval_summary = summarize_retrieval_rows(fuzzy_rows, split)

    return path_feature_rows, fuzzy_rows, metrics, retrieval_summary


def summarize_eval_rows(rows: list[dict[str, Any]], row_name: str) -> dict[str, Any]:
    n = len(rows)
    ranks = [int(r["gold_rank"]) for r in rows]
    rr = [float(r["reciprocal_rank_item"]) for r in rows]
    present = [bool(r["gold_present"]) for r in rows]
    present_ranks = [rank for rank, is_present in zip(ranks, present) if is_present]
    candidate_sizes = [int(r["num_candidates"]) for r in rows]

    return {
        "eval_row_name": row_name,
        "num_rows": int(n),
        "gold_present_at20": float(sum(present) / max(1, n)),
        "mrr_at20": float(sum(rr) / max(1, n)),
        "mrr_present_only": (
            float(sum(1.0 / r for r in present_ranks) / len(present_ranks))
            if present_ranks
            else 0.0
        ),
        "hits1_at20": float(sum(r <= 1 for r in ranks) / max(1, n)),
        "hits3_at20": float(sum(r <= 3 for r in ranks) / max(1, n)),
        "hits10_at20": float(sum(r <= 10 for r in ranks) / max(1, n)),
        "hits20_at20": float(sum(r <= 20 for r in ranks) / max(1, n)),
        "avg_gold_rank_absent_as_21": float(sum(ranks) / max(1, n)),
        "gold_rank_21_count": int(sum(r == ABSENT_RANK for r in ranks)),
        "avg_candidate_size": float(sum(candidate_sizes) / max(1, n)),
        "min_candidate_size": int(min(candidate_sizes)) if candidate_sizes else 0,
        "max_candidate_size": int(max(candidate_sizes)) if candidate_sizes else 0,
        "top_k": TOP_K,
        "gold_injection": False,
        "rr_policy": "RR = 1/rank if gold is present in top-20 else 0",
        "rank_absent_sentinel": ABSENT_RANK,
    }


def summarize_retrieval_rows(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    summaries = [r["subgraph_summary"] for r in rows]

    def avg(key: str) -> float:
        if not summaries:
            return 0.0
        return float(sum(float(s[key]) for s in summaries) / len(summaries))

    def minv(key: str) -> float:
        if not summaries:
            return 0.0
        return float(min(float(s[key]) for s in summaries))

    def maxv(key: str) -> float:
        if not summaries:
            return 0.0
        return float(max(float(s[key]) for s in summaries))

    return {
        "split": split,
        "num_rows": int(len(rows)),
        "avg_original_subgraph_size": avg("original_subgraph_size"),
        "avg_selected_subgraph_size": avg("selected_subgraph_size"),
        "min_selected_subgraph_size": minv("selected_subgraph_size"),
        "max_selected_subgraph_size": maxv("selected_subgraph_size"),
        "avg_retain_ratio_actual": avg("retain_ratio_actual"),
        "avg_candidate_coverage_preserved_rate": avg("candidate_coverage_preserved_rate"),
        "avg_top_band_coverage_preserved_rate": avg("top_band_coverage_preserved_rate"),
        "avg_mid_band_coverage_preserved_rate": avg("mid_band_coverage_preserved_rate"),
        "avg_tail_band_coverage_preserved_rate": avg("tail_band_coverage_preserved_rate"),
        "avg_direct_shortcut_path_rate_original": avg("direct_shortcut_path_rate_original"),
        "avg_direct_shortcut_path_rate_selected": avg("direct_shortcut_path_rate_selected"),
        "avg_direct_T_shortcut_rate_original": avg("direct_T_shortcut_rate_original"),
        "avg_direct_T_shortcut_rate_selected": avg("direct_T_shortcut_rate_selected"),
        "avg_selected_triple_score": avg("avg_selected_triple_score"),
    }


def compare_with_soft(fuzzy_metrics: dict[str, Any], split: str) -> dict[str, Any]:
    soft = read_json(RESULT_DIR / f"soft_support_raw_eval_{split}.json")

    return {
        "split": split,
        "soft_gold_present_at20": soft["gold_present_at20"],
        "fuzzy_gold_present_at20": fuzzy_metrics["gold_present_at20"],
        "delta_gold_present_at20": fuzzy_metrics["gold_present_at20"] - soft["gold_present_at20"],
        "soft_mrr_at20": soft["mrr_at20"],
        "fuzzy_mrr_at20": fuzzy_metrics["mrr_at20"],
        "delta_mrr_at20": fuzzy_metrics["mrr_at20"] - soft["mrr_at20"],
        "soft_hits1": soft["hits1_at20"],
        "fuzzy_hits1": fuzzy_metrics["hits1_at20"],
        "delta_hits1": fuzzy_metrics["hits1_at20"] - soft["hits1_at20"],
        "soft_hits3": soft["hits3_at20"],
        "fuzzy_hits3": fuzzy_metrics["hits3_at20"],
        "delta_hits3": fuzzy_metrics["hits3_at20"] - soft["hits3_at20"],
        "soft_hits10": soft["hits10_at20"],
        "fuzzy_hits10": fuzzy_metrics["hits10_at20"],
        "delta_hits10": fuzzy_metrics["hits10_at20"] - soft["hits10_at20"],
    }


def decide(
    valid_compare: dict[str, Any],
    test_compare: dict[str, Any],
    valid_retrieval: dict[str, Any],
    test_retrieval: dict[str, Any],
) -> str:
    ranking_preserved = (
        abs(valid_compare["delta_mrr_at20"]) < 1e-12
        and abs(test_compare["delta_mrr_at20"]) < 1e-12
    )

    compressed = (
        valid_retrieval["avg_selected_subgraph_size"] < valid_retrieval["avg_original_subgraph_size"]
        and test_retrieval["avg_selected_subgraph_size"] < test_retrieval["avg_original_subgraph_size"]
    )

    coverage_ok = (
        valid_retrieval["avg_top_band_coverage_preserved_rate"] >= 0.99
        and test_retrieval["avg_top_band_coverage_preserved_rate"] >= 0.99
    )

    if ranking_preserved and compressed and coverage_ok:
        return "FUZZY_RETRIEVAL_TRANSFER_READY"

    if compressed and coverage_ok:
        return "FUZZY_RETRIEVAL_TRANSFER_READY_WITH_RANKING_DELTA"

    return "FUZZY_RETRIEVAL_TRANSFER_DIAGNOSTIC_ONLY"


def write_report(
    valid_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    valid_compare: dict[str, Any],
    test_compare: dict[str, Any],
    valid_retrieval: dict[str, Any],
    test_retrieval: dict[str, Any],
    decision: str,
) -> None:
    md = f"""# Week 23 Day 4 — Fuzzy Retrieval Transfer on PharmKG

## Decision

`{decision}`

## Variant

- Variant name: `{VARIANT_NAME}`
- Source variant: `{SOURCE_VARIANT_NAME}`
- Ranking policy: preserve soft-support candidate order
- Gold injection: false
- Target relation: `T`
- Relation label: `therapeutic_association_proxy`

## Retrieval config

- retain_ratio: `{CONFIG["retain_ratio"]}`
- min_keep: `{CONFIG["min_keep"]}`
- top/mid/tail band weights: `{CONFIG["top_band_weight"]}`, `{CONFIG["mid_band_weight"]}`, `{CONFIG["tail_band_weight"]}`
- touch_candidate_weight: `{CONFIG["touch_candidate_weight"]}`
- touch_query_weight: `{CONFIG["touch_query_weight"]}`
- direct_shortcut_penalty: `{CONFIG["direct_shortcut_penalty"]}`
- contra_penalty: `{CONFIG["contra_penalty"]}`
- density_weight: `{CONFIG["density_weight"]}`

## Ranking metrics

| Split | Gold@20 | MRR@20 | MRR present-only | H@1 | H@3 | H@10 | H@20 | Rank21 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valid | {valid_metrics["gold_present_at20"]:.3f} | {valid_metrics["mrr_at20"]:.12f} | {valid_metrics["mrr_present_only"]:.12f} | {valid_metrics["hits1_at20"]:.3f} | {valid_metrics["hits3_at20"]:.3f} | {valid_metrics["hits10_at20"]:.3f} | {valid_metrics["hits20_at20"]:.3f} | {valid_metrics["gold_rank_21_count"]} |
| test | {test_metrics["gold_present_at20"]:.3f} | {test_metrics["mrr_at20"]:.12f} | {test_metrics["mrr_present_only"]:.12f} | {test_metrics["hits1_at20"]:.3f} | {test_metrics["hits3_at20"]:.3f} | {test_metrics["hits10_at20"]:.3f} | {test_metrics["hits20_at20"]:.3f} | {test_metrics["gold_rank_21_count"]} |

## Deltas vs soft_support_raw

| Split | Delta Gold@20 | Delta MRR@20 | Delta H@1 | Delta H@3 | Delta H@10 |
|---|---:|---:|---:|---:|---:|
| valid | {valid_compare["delta_gold_present_at20"]:.6f} | {valid_compare["delta_mrr_at20"]:.12f} | {valid_compare["delta_hits1"]:.6f} | {valid_compare["delta_hits3"]:.6f} | {valid_compare["delta_hits10"]:.6f} |
| test | {test_compare["delta_gold_present_at20"]:.6f} | {test_compare["delta_mrr_at20"]:.12f} | {test_compare["delta_hits1"]:.6f} | {test_compare["delta_hits3"]:.6f} | {test_compare["delta_hits10"]:.6f} |

## Retrieval compression and coverage

| Split | Original size | Selected size | Retain ratio | Candidate coverage | Top-band coverage | Direct shortcut original | Direct shortcut selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| valid | {valid_retrieval["avg_original_subgraph_size"]:.2f} | {valid_retrieval["avg_selected_subgraph_size"]:.2f} | {valid_retrieval["avg_retain_ratio_actual"]:.3f} | {valid_retrieval["avg_candidate_coverage_preserved_rate"]:.3f} | {valid_retrieval["avg_top_band_coverage_preserved_rate"]:.3f} | {valid_retrieval["avg_direct_shortcut_path_rate_original"]:.3f} | {valid_retrieval["avg_direct_shortcut_path_rate_selected"]:.3f} |
| test | {test_retrieval["avg_original_subgraph_size"]:.2f} | {test_retrieval["avg_selected_subgraph_size"]:.2f} | {test_retrieval["avg_retain_ratio_actual"]:.3f} | {test_retrieval["avg_candidate_coverage_preserved_rate"]:.3f} | {test_retrieval["avg_top_band_coverage_preserved_rate"]:.3f} | {test_retrieval["avg_direct_shortcut_path_rate_original"]:.3f} | {test_retrieval["avg_direct_shortcut_path_rate_selected"]:.3f} |

## Interpretation

Fuzzy retrieval is not intended to improve ranking directly on Day 4. Its goal is to preserve the Day 3 soft-support ranking while reducing evidence budget and prioritizing high-confidence candidate-support triples.

If MRR is unchanged and selected subgraph size is substantially smaller, this is a positive SoftFuse result: cleaner evidence without ranking collapse.

## Files written

- `dataset/setting_c_pharmkg/fuzzy_retrieval/valid_path_features.json`
- `dataset/setting_c_pharmkg/fuzzy_retrieval/test_path_features.json`
- `dataset/setting_c_pharmkg/fuzzy_retrieval/valid_fuzzy_retrieval_main.json`
- `dataset/setting_c_pharmkg/fuzzy_retrieval/test_fuzzy_retrieval_main.json`
- `dataset/setting_c_pharmkg/fuzzy_retrieval/retrieval_config.json`
- `dataset/setting_c_pharmkg/fuzzy_retrieval/retrieval_summary.json`
- `outputs/pharmkg/fuzzy_retrieval_eval_valid.json`
- `outputs/pharmkg/fuzzy_retrieval_eval_test.json`

## Next step

Day 5 will build the final reviewer-safe PharmKG table comparing backbone_raw, hard_support_raw, soft_support_raw, fuzzy_retrieval_main, and six Week 22 structure baselines.
"""
    REPORT_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    ensure_dirs()

    write_json(CONFIG, RETRIEVAL_DIR / "retrieval_config.json")

    id2relation = load_id2relation()

    valid_path_features, valid_fuzzy, valid_metrics, valid_retrieval = build_split(
        split="valid",
        id2relation=id2relation,
    )

    test_path_features, test_fuzzy, test_metrics, test_retrieval = build_split(
        split="test",
        id2relation=id2relation,
    )

    valid_compare = compare_with_soft(valid_metrics, "valid")
    test_compare = compare_with_soft(test_metrics, "test")

    decision = decide(
        valid_compare=valid_compare,
        test_compare=test_compare,
        valid_retrieval=valid_retrieval,
        test_retrieval=test_retrieval,
    )

    retrieval_summary = {
        "week": 23,
        "day": 4,
        "decision": decision,
        "variant_name": VARIANT_NAME,
        "source_variant_name": SOURCE_VARIANT_NAME,
        "config": CONFIG,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "valid_compare_to_soft": valid_compare,
        "test_compare_to_soft": test_compare,
        "valid_retrieval_summary": valid_retrieval,
        "test_retrieval_summary": test_retrieval,
    }

    write_json(valid_path_features, RETRIEVAL_DIR / "valid_path_features.json")
    write_json(test_path_features, RETRIEVAL_DIR / "test_path_features.json")

    write_json(valid_fuzzy, RETRIEVAL_DIR / "valid_fuzzy_retrieval_main.json")
    write_json(test_fuzzy, RETRIEVAL_DIR / "test_fuzzy_retrieval_main.json")

    write_json(retrieval_summary, RETRIEVAL_DIR / "retrieval_summary.json")

    write_json(valid_metrics, VALID_FUZZY_EVAL_PATH)
    write_json(test_metrics, TEST_FUZZY_EVAL_PATH)

    write_report(
        valid_metrics=valid_metrics,
        test_metrics=test_metrics,
        valid_compare=valid_compare,
        test_compare=test_compare,
        valid_retrieval=valid_retrieval,
        test_retrieval=test_retrieval,
        decision=decision,
    )

    if len(valid_fuzzy) != 500 or len(test_fuzzy) != 500:
        raise RuntimeError("Expected 500 valid/test fuzzy rows.")

    if abs(valid_compare["delta_mrr_at20"]) > 1e-12:
        raise RuntimeError("Valid ranking changed unexpectedly.")
    if abs(test_compare["delta_mrr_at20"]) > 1e-12:
        raise RuntimeError("Test ranking changed unexpectedly.")

    print("Saved:")
    print(f"  {RETRIEVAL_DIR / 'valid_path_features.json'}")
    print(f"  {RETRIEVAL_DIR / 'test_path_features.json'}")
    print(f"  {RETRIEVAL_DIR / 'valid_fuzzy_retrieval_main.json'}")
    print(f"  {RETRIEVAL_DIR / 'test_fuzzy_retrieval_main.json'}")
    print(f"  {RETRIEVAL_DIR / 'retrieval_config.json'}")
    print(f"  {RETRIEVAL_DIR / 'retrieval_summary.json'}")
    print(f"  {VALID_FUZZY_EVAL_PATH}")
    print(f"  {TEST_FUZZY_EVAL_PATH}")
    print(f"  {REPORT_PATH}")

    print("\nDecision:", decision)

    print("\nVALID fuzzy metrics:")
    print(json.dumps(valid_metrics, ensure_ascii=False, indent=2))

    print("\nTEST fuzzy metrics:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))

    print("\nRetrieval summary:")
    print(json.dumps(
        {
            "valid": valid_retrieval,
            "test": test_retrieval,
        },
        ensure_ascii=False,
        indent=2,
    ))

    print("\nCompare to soft:")
    print(json.dumps(
        {
            "valid": valid_compare,
            "test": test_compare,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()