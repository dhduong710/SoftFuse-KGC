from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "dataset/setting_a/backbone_candidates/valid_top20_raw.json"
SUPPORT_PATH = ROOT / "dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_main.json"
FEATURE_PATH = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_path_features.json"
FUZZY_PATH = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_fuzzy_retrieval_v1.json"

OUT_JSON = ROOT / "outputs/fuzzy_retrieval/retrieval_valid_compare_v1.json"
OUT_REPORT = ROOT / "outputs/fuzzy_retrieval/reports/fuzzy_retrieval_v1_compare.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def rank_metrics_from_ranks(ranks: List[int]) -> Dict[str, float]:
    n = len(ranks)
    return {
        "mrr_like": round(sum((1.0 / r if r <= 20 else 0.0) for r in ranks) / n, 6),
        "hits1_like": round(sum(r <= 1 for r in ranks) / n, 6),
        "hits3_like": round(sum(r <= 3 for r in ranks) / n, 6),
        "hits10_like": round(sum(r <= 10 for r in ranks) / n, 6),
        "avg_gold_rank": round(sum(ranks) / n, 6),
    }


def gold_rank(candidate_ids: List[Any], gold_id: Any) -> int:
    try:
        return candidate_ids.index(gold_id) + 1
    except ValueError:
        return 21


def build_original_candidate_maps(feature_row: Dict[str, Any]) -> Tuple[Dict[Any, bool], Dict[Any, bool], Dict[Any, bool]]:
    """
    Return:
      direct_flag_by_candidate
      evidence_positive_by_candidate
      contra_flag_by_candidate
    """
    candidate_ids = feature_row["candidate_entity_ids"]
    contra_flags = feature_row["contra_flags"]

    direct_flag_by_id = {cid: False for cid in candidate_ids}
    evidence_flag_by_id = {cid: False for cid in candidate_ids}
    contra_flag_by_id = {cid: bool(cflag) for cid, cflag in zip(candidate_ids, contra_flags)}

    for tr in feature_row["triple_feature_rows"]:
        touched_ids = tr.get("touched_candidate_ids", [])
        for cid in touched_ids:
            if cid in evidence_flag_by_id:
                evidence_flag_by_id[cid] = True
            if cid in direct_flag_by_id and bool(tr.get("direct_candidate_query_flag", False)):
                direct_flag_by_id[cid] = True

    return direct_flag_by_id, evidence_flag_by_id, contra_flag_by_id


def build_selected_candidate_maps(fuzzy_row: Dict[str, Any]) -> Tuple[Dict[Any, bool], Dict[Any, float], Dict[Any, bool]]:
    """
    Return:
      direct_flag_by_candidate
      evidence_bonus_by_candidate
      contra_flag_by_candidate
    """
    candidate_ids = fuzzy_row["candidate_entity_ids"]
    contra_flags = fuzzy_row["contra_flags"]
    triple_rows = fuzzy_row["triple_score_rows"]

    direct_flag_by_id = {cid: False for cid in candidate_ids}
    evidence_bonus_by_id = {cid: 0.0 for cid in candidate_ids}
    contra_flag_by_id = {cid: bool(cflag) for cid, cflag in zip(candidate_ids, contra_flags)}

    max_score = 0.0
    for tr in triple_rows:
        max_score = max(max_score, float(tr.get("triple_score", 0.0)))
    if max_score <= 0:
        max_score = 1.0

    for tr in triple_rows:
        score_norm = float(tr.get("triple_score", 0.0)) / max_score
        touched_ids = tr.get("touched_candidate_ids", [])
        direct_flag = bool(tr.get("direct_candidate_query_flag", False))
        for cid in touched_ids:
            if cid in evidence_bonus_by_id:
                evidence_bonus_by_id[cid] = max(evidence_bonus_by_id[cid], score_norm)
            if cid in direct_flag_by_id and direct_flag:
                direct_flag_by_id[cid] = True

    return direct_flag_by_id, evidence_bonus_by_id, contra_flag_by_id


def rerank_fuzzy_candidates(fuzzy_row: Dict[str, Any]) -> List[Any]:
    candidate_ids = fuzzy_row["candidate_entity_ids"]
    direct_flag_by_id, evidence_bonus_by_id, contra_flag_by_id = build_selected_candidate_maps(fuzzy_row)

    n = len(candidate_ids)
    scored = []
    for pos, cid in enumerate(candidate_ids):
        support_prior = 1.0 - (pos / (n - 1)) if n > 1 else 1.0
        evidence_bonus = evidence_bonus_by_id[cid]
        shortcut_penalty = 1.0 if direct_flag_by_id[cid] else 0.0
        contra_penalty = 1.0 if contra_flag_by_id[cid] else 0.0

        score = (
            0.70 * support_prior
            + 0.30 * evidence_bonus
            - 0.15 * shortcut_penalty
            - 0.05 * contra_penalty
        )
        scored.append((cid, score, pos))

    scored.sort(key=lambda x: (-x[1], x[2]))
    return [cid for cid, _, _ in scored]


def top5_rate(candidate_ids: List[Any], flag_by_id: Dict[Any, bool]) -> float:
    top5 = candidate_ids[:5]
    if not top5:
        return 0.0
    return sum(bool(flag_by_id.get(cid, False)) for cid in top5) / len(top5)


def summarize_rowwise(
    raw_rows: List[Dict[str, Any]],
    support_rows: List[Dict[str, Any]],
    feature_rows: List[Dict[str, Any]],
    fuzzy_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    assert len(raw_rows) == len(support_rows) == len(feature_rows) == len(fuzzy_rows) == 500

    ranks_backbone = []
    ranks_support = []
    ranks_fuzzy = []

    top5_direct_backbone = []
    top5_direct_support = []
    top5_direct_fuzzy = []

    top5_evidence_backbone = []
    top5_evidence_support = []
    top5_evidence_fuzzy = []

    top5_contra_backbone = []
    top5_contra_support = []
    top5_contra_fuzzy = []

    fuzzy_selected_sizes = []
    fuzzy_avg_triple_scores = []
    fuzzy_direct_rates = []
    fuzzy_contra_rates = []
    fuzzy_coverage_rates = []

    original_subgraph_sizes = []
    original_direct_rates = []
    original_contra_rates = []

    improved_vs_backbone = 0
    worsened_vs_backbone = 0
    improved_vs_support = 0
    worsened_vs_support = 0

    sample_cases = []

    for i, (raw_row, supp_row, feat_row, fuzzy_row) in enumerate(zip(raw_rows, support_rows, feature_rows, fuzzy_rows)):
        gold_id = raw_row["gold_entity_id"]

        raw_ids = raw_row["candidate_entity_ids"]
        support_ids = supp_row["candidate_entity_ids"]
        fuzzy_ids = rerank_fuzzy_candidates(fuzzy_row)

        rank_b = gold_rank(raw_ids, gold_id)
        rank_s = gold_rank(support_ids, gold_id)
        rank_f = gold_rank(fuzzy_ids, gold_id)

        ranks_backbone.append(rank_b)
        ranks_support.append(rank_s)
        ranks_fuzzy.append(rank_f)

        orig_direct_by_id, orig_evidence_by_id, orig_contra_by_id = build_original_candidate_maps(feat_row)
        sel_direct_by_id, sel_evidence_bonus_by_id, sel_contra_by_id = build_selected_candidate_maps(fuzzy_row)
        sel_evidence_by_id = {cid: (score > 0.0) for cid, score in sel_evidence_bonus_by_id.items()}

        top5_direct_backbone.append(top5_rate(raw_ids, orig_direct_by_id))
        top5_direct_support.append(top5_rate(support_ids, orig_direct_by_id))
        top5_direct_fuzzy.append(top5_rate(fuzzy_ids, sel_direct_by_id))

        top5_evidence_backbone.append(top5_rate(raw_ids, orig_evidence_by_id))
        top5_evidence_support.append(top5_rate(support_ids, orig_evidence_by_id))
        top5_evidence_fuzzy.append(top5_rate(fuzzy_ids, sel_evidence_by_id))

        top5_contra_backbone.append(top5_rate(raw_ids, orig_contra_by_id))
        top5_contra_support.append(top5_rate(support_ids, orig_contra_by_id))
        top5_contra_fuzzy.append(top5_rate(fuzzy_ids, sel_contra_by_id))

        orig_triples = feat_row["triple_feature_rows"]
        sel_triples = fuzzy_row["triple_score_rows"]

        original_subgraph_sizes.append(len(orig_triples))
        fuzzy_selected_sizes.append(len(sel_triples))

        orig_direct = sum(bool(tr.get("direct_candidate_query_flag", False)) for tr in orig_triples)
        sel_direct = sum(bool(tr.get("direct_candidate_query_flag", False)) for tr in sel_triples)
        orig_contra = sum(bool(tr.get("contra_flag", False)) for tr in orig_triples)
        sel_contra = sum(bool(tr.get("contra_flag", False)) for tr in sel_triples)

        original_direct_rates.append(orig_direct / len(orig_triples) if orig_triples else 0.0)
        original_contra_rates.append(orig_contra / len(orig_triples) if orig_triples else 0.0)
        fuzzy_direct_rates.append(sel_direct / len(sel_triples) if sel_triples else 0.0)
        fuzzy_contra_rates.append(sel_contra / len(sel_triples) if sel_triples else 0.0)

        fuzzy_avg_triple_scores.append(
            sum(float(tr.get("triple_score", 0.0)) for tr in sel_triples) / len(sel_triples)
            if sel_triples else 0.0
        )
        fuzzy_coverage_rates.append(float(fuzzy_row["subgraph_summary"]["candidate_coverage_preserved_rate"]))

        if rank_f < rank_b:
            improved_vs_backbone += 1
        elif rank_f > rank_b:
            worsened_vs_backbone += 1

        if rank_f < rank_s:
            improved_vs_support += 1
        elif rank_f > rank_s:
            worsened_vs_support += 1

        if len(sample_cases) < 20 and (rank_f != rank_s or rank_f != rank_b):
            sample_cases.append({
                "row_index": i,
                "row_uid": fuzzy_row["row_uid"],
                "query_entity": fuzzy_row["query_entity"],
                "gold_entity": fuzzy_row["gold_entity"],
                "rank_backbone": rank_b,
                "rank_soft_support": rank_s,
                "rank_fuzzy_v1": rank_f,
            })

    backbone_metrics = {
        **rank_metrics_from_ranks(ranks_backbone),
        "avg_subgraph_size": round(sum(original_subgraph_sizes) / len(original_subgraph_sizes), 6),
        "avg_triple_score": None,
        "direct_shortcut_path_rate": round(sum(original_direct_rates) / len(original_direct_rates), 6),
        "contradiction_path_rate": round(sum(original_contra_rates) / len(original_contra_rates), 6),
        "candidate_coverage_preserved_rate": 1.0,
        "avg_top5_direct_link_rate": round(sum(top5_direct_backbone) / len(top5_direct_backbone), 6),
        "avg_top5_evidence_positive_rate": round(sum(top5_evidence_backbone) / len(top5_evidence_backbone), 6),
        "avg_top5_contra_rate": round(sum(top5_contra_backbone) / len(top5_contra_backbone), 6),
    }

    support_metrics = {
        **rank_metrics_from_ranks(ranks_support),
        "avg_subgraph_size": round(sum(original_subgraph_sizes) / len(original_subgraph_sizes), 6),
        "avg_triple_score": None,
        "direct_shortcut_path_rate": round(sum(original_direct_rates) / len(original_direct_rates), 6),
        "contradiction_path_rate": round(sum(original_contra_rates) / len(original_contra_rates), 6),
        "candidate_coverage_preserved_rate": 1.0,
        "avg_top5_direct_link_rate": round(sum(top5_direct_support) / len(top5_direct_support), 6),
        "avg_top5_evidence_positive_rate": round(sum(top5_evidence_support) / len(top5_evidence_support), 6),
        "avg_top5_contra_rate": round(sum(top5_contra_support) / len(top5_contra_support), 6),
    }

    fuzzy_metrics = {
        **rank_metrics_from_ranks(ranks_fuzzy),
        "avg_subgraph_size": round(sum(fuzzy_selected_sizes) / len(fuzzy_selected_sizes), 6),
        "avg_triple_score": round(sum(fuzzy_avg_triple_scores) / len(fuzzy_avg_triple_scores), 6),
        "direct_shortcut_path_rate": round(sum(fuzzy_direct_rates) / len(fuzzy_direct_rates), 6),
        "contradiction_path_rate": round(sum(fuzzy_contra_rates) / len(fuzzy_contra_rates), 6),
        "candidate_coverage_preserved_rate": round(sum(fuzzy_coverage_rates) / len(fuzzy_coverage_rates), 6),
        "avg_top5_direct_link_rate": round(sum(top5_direct_fuzzy) / len(top5_direct_fuzzy), 6),
        "avg_top5_evidence_positive_rate": round(sum(top5_evidence_fuzzy) / len(top5_evidence_fuzzy), 6),
        "avg_top5_contra_rate": round(sum(top5_contra_fuzzy) / len(top5_contra_fuzzy), 6),
    }

    compare = {
        "stage": "fuzzy_retrieval_v1_compare",
        "rows": {
            "backbone_raw": backbone_metrics,
            "soft_support_raw": support_metrics,
            "soft_support_fuzzy_retrieval_v1": fuzzy_metrics,
        },
        "delta": {
            "fuzzy_vs_backbone": {
                "mrr_like": round(fuzzy_metrics["mrr_like"] - backbone_metrics["mrr_like"], 6),
                "hits1_like": round(fuzzy_metrics["hits1_like"] - backbone_metrics["hits1_like"], 6),
                "hits3_like": round(fuzzy_metrics["hits3_like"] - backbone_metrics["hits3_like"], 6),
                "hits10_like": round(fuzzy_metrics["hits10_like"] - backbone_metrics["hits10_like"], 6),
                "avg_gold_rank": round(fuzzy_metrics["avg_gold_rank"] - backbone_metrics["avg_gold_rank"], 6),
                "direct_shortcut_path_rate": round(fuzzy_metrics["direct_shortcut_path_rate"] - backbone_metrics["direct_shortcut_path_rate"], 6),
                "avg_top5_direct_link_rate": round(fuzzy_metrics["avg_top5_direct_link_rate"] - backbone_metrics["avg_top5_direct_link_rate"], 6),
            },
            "fuzzy_vs_soft_support": {
                "mrr_like": round(fuzzy_metrics["mrr_like"] - support_metrics["mrr_like"], 6),
                "hits1_like": round(fuzzy_metrics["hits1_like"] - support_metrics["hits1_like"], 6),
                "hits3_like": round(fuzzy_metrics["hits3_like"] - support_metrics["hits3_like"], 6),
                "hits10_like": round(fuzzy_metrics["hits10_like"] - support_metrics["hits10_like"], 6),
                "avg_gold_rank": round(fuzzy_metrics["avg_gold_rank"] - support_metrics["avg_gold_rank"], 6),
                "direct_shortcut_path_rate": round(fuzzy_metrics["direct_shortcut_path_rate"] - support_metrics["direct_shortcut_path_rate"], 6),
                "avg_top5_direct_link_rate": round(fuzzy_metrics["avg_top5_direct_link_rate"] - support_metrics["avg_top5_direct_link_rate"], 6),
            },
        },
        "case_level": {
            "improved_vs_backbone": improved_vs_backbone,
            "worsened_vs_backbone": worsened_vs_backbone,
            "improved_vs_soft_support": improved_vs_support,
            "worsened_vs_soft_support": worsened_vs_support,
            "sample_cases": sample_cases,
        },
        "notes": [
            "backbone_raw uses raw candidate order + original evidence package",
            "soft_support_raw uses support-stage candidate order + original evidence package",
            "soft_support_fuzzy_retrieval_v1 uses analysis-only retrieval-aware candidate rerank on top of selected subgraph",
        ],
    }

    return compare


def main() -> None:
    raw_rows = load_json(RAW_PATH)
    support_rows = load_json(SUPPORT_PATH)
    feature_rows = load_json(FEATURE_PATH)
    fuzzy_rows = load_json(FUZZY_PATH)

    compare = summarize_rowwise(raw_rows, support_rows, feature_rows, fuzzy_rows)
    save_json(OUT_JSON, compare)

    rows = compare["rows"]
    delta = compare["delta"]
    case_level = compare["case_level"]

    report_lines = [
        "# Fuzzy Retrieval v1 Valid Compare",
        "",
        "## Main rows",
        "- backbone_raw",
        "- soft_support_raw",
        "- soft_support_fuzzy_retrieval_v1",
        "",
        "## backbone_raw",
        json.dumps(rows["backbone_raw"], ensure_ascii=False, indent=2),
        "",
        "## soft_support_raw",
        json.dumps(rows["soft_support_raw"], ensure_ascii=False, indent=2),
        "",
        "## soft_support_fuzzy_retrieval_v1",
        json.dumps(rows["soft_support_fuzzy_retrieval_v1"], ensure_ascii=False, indent=2),
        "",
        "## Delta: fuzzy vs backbone",
        json.dumps(delta["fuzzy_vs_backbone"], ensure_ascii=False, indent=2),
        "",
        "## Delta: fuzzy vs soft_support",
        json.dumps(delta["fuzzy_vs_soft_support"], ensure_ascii=False, indent=2),
        "",
        "## Case-level",
        json.dumps({
            "improved_vs_backbone": case_level["improved_vs_backbone"],
            "worsened_vs_backbone": case_level["worsened_vs_backbone"],
            "improved_vs_soft_support": case_level["improved_vs_soft_support"],
            "worsened_vs_soft_support": case_level["worsened_vs_soft_support"],
        }, ensure_ascii=False, indent=2),
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 80)
    print("FUZZY RETRIEVAL V1 VALID COMPARE DONE")
    print("json  :", OUT_JSON)
    print("report:", OUT_REPORT)
    print("=" * 80)
    print(json.dumps(compare["rows"], ensure_ascii=False, indent=2))
    print("=" * 80)
    print(json.dumps(compare["delta"], ensure_ascii=False, indent=2))
    print("=" * 80)
    print(json.dumps(compare["case_level"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
