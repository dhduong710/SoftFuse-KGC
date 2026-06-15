from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "dataset/setting_a/backbone_candidates/valid_top20_raw.json"
SUPPORT_PATH = ROOT / "dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_main.json"
FEATURE_PATH = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_path_features.json"

FUZZY_VARIANT_PATHS = {
    "soft_support_fuzzy_retrieval_v1": ROOT / "dataset/setting_a/fuzzy_retrieval/valid_fuzzy_retrieval_v1.json",
    "soft_support_fuzzy_retrieval_tight": ROOT / "dataset/setting_a/fuzzy_retrieval/valid_soft_support_fuzzy_retrieval_tight.json",
    "soft_support_fuzzy_retrieval_directplus": ROOT / "dataset/setting_a/fuzzy_retrieval/valid_soft_support_fuzzy_retrieval_directplus.json",
}

OUT_COMPARE = ROOT / "outputs/fuzzy_retrieval/retrieval_valid_compare.json"
OUT_SUMMARY = ROOT / "outputs/fuzzy_retrieval/retrieval_sweep_summary.json"
OUT_REPORT = ROOT / "outputs/fuzzy_retrieval/reports/fuzzy_retrieval_variant_compare.md"


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


def selected_subgraph_signature(fuzzy_row: Dict[str, Any]) -> Tuple[Tuple[Any, Any, Any], ...]:
    return tuple(tuple(x) for x in fuzzy_row["selected_subgraph"])


def summarize_baseline_rows(
    raw_rows: List[Dict[str, Any]],
    support_rows: List[Dict[str, Any]],
    feature_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    ranks_backbone = []
    ranks_support = []

    top5_direct_backbone = []
    top5_direct_support = []
    top5_evidence_backbone = []
    top5_evidence_support = []
    top5_contra_backbone = []
    top5_contra_support = []

    original_subgraph_sizes = []
    original_direct_rates = []
    original_contra_rates = []
    original_query_touch_counts = []
    original_non_direct_query_touch_counts = []

    for raw_row, supp_row, feat_row in zip(raw_rows, support_rows, feature_rows):
        gold_id = raw_row["gold_entity_id"]
        raw_ids = raw_row["candidate_entity_ids"]
        support_ids = supp_row["candidate_entity_ids"]

        ranks_backbone.append(gold_rank(raw_ids, gold_id))
        ranks_support.append(gold_rank(support_ids, gold_id))

        orig_direct_by_id, orig_evidence_by_id, orig_contra_by_id = build_original_candidate_maps(feat_row)

        top5_direct_backbone.append(top5_rate(raw_ids, orig_direct_by_id))
        top5_direct_support.append(top5_rate(support_ids, orig_direct_by_id))
        top5_evidence_backbone.append(top5_rate(raw_ids, orig_evidence_by_id))
        top5_evidence_support.append(top5_rate(support_ids, orig_evidence_by_id))
        top5_contra_backbone.append(top5_rate(raw_ids, orig_contra_by_id))
        top5_contra_support.append(top5_rate(support_ids, orig_contra_by_id))

        orig_triples = feat_row["triple_feature_rows"]
        original_subgraph_sizes.append(len(orig_triples))
        original_direct_rates.append(
            sum(bool(tr.get("direct_candidate_query_flag", False)) for tr in orig_triples) / len(orig_triples)
            if orig_triples else 0.0
        )
        original_contra_rates.append(
            sum(bool(tr.get("contra_flag", False)) for tr in orig_triples) / len(orig_triples)
            if orig_triples else 0.0
        )
        original_query_touch_counts.append(
            sum(bool(tr.get("touches_query", False)) for tr in orig_triples)
        )
        original_non_direct_query_touch_counts.append(
            sum(bool(tr.get("touches_query", False)) and not bool(tr.get("direct_candidate_query_flag", False)) for tr in orig_triples)
        )

    backbone = {
        **rank_metrics_from_ranks(ranks_backbone),
        "avg_subgraph_size": round(sum(original_subgraph_sizes) / len(original_subgraph_sizes), 6),
        "avg_triple_score": None,
        "direct_shortcut_path_rate": round(sum(original_direct_rates) / len(original_direct_rates), 6),
        "contradiction_path_rate": round(sum(original_contra_rates) / len(original_contra_rates), 6),
        "candidate_coverage_preserved_rate": 1.0,
        "avg_top5_direct_link_rate": round(sum(top5_direct_backbone) / len(top5_direct_backbone), 6),
        "avg_top5_evidence_positive_rate": round(sum(top5_evidence_backbone) / len(top5_evidence_backbone), 6),
        "avg_top5_contra_rate": round(sum(top5_contra_backbone) / len(top5_contra_backbone), 6),
        "avg_query_touch_count": round(sum(original_query_touch_counts) / len(original_query_touch_counts), 6),
        "avg_non_direct_query_touch_count": round(sum(original_non_direct_query_touch_counts) / len(original_non_direct_query_touch_counts), 6),
    }

    support = {
        **rank_metrics_from_ranks(ranks_support),
        "avg_subgraph_size": round(sum(original_subgraph_sizes) / len(original_subgraph_sizes), 6),
        "avg_triple_score": None,
        "direct_shortcut_path_rate": round(sum(original_direct_rates) / len(original_direct_rates), 6),
        "contradiction_path_rate": round(sum(original_contra_rates) / len(original_contra_rates), 6),
        "candidate_coverage_preserved_rate": 1.0,
        "avg_top5_direct_link_rate": round(sum(top5_direct_support) / len(top5_direct_support), 6),
        "avg_top5_evidence_positive_rate": round(sum(top5_evidence_support) / len(top5_evidence_support), 6),
        "avg_top5_contra_rate": round(sum(top5_contra_support) / len(top5_contra_support), 6),
        "avg_query_touch_count": round(sum(original_query_touch_counts) / len(original_query_touch_counts), 6),
        "avg_non_direct_query_touch_count": round(sum(original_non_direct_query_touch_counts) / len(original_non_direct_query_touch_counts), 6),
    }

    return {
        "backbone_raw": backbone,
        "soft_support_raw": support,
    }


def summarize_fuzzy_variant(
    variant_name: str,
    raw_rows: List[Dict[str, Any]],
    support_rows: List[Dict[str, Any]],
    feature_rows: List[Dict[str, Any]],
    fuzzy_rows: List[Dict[str, Any]],
    v1_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ranks_backbone = []
    ranks_support = []
    ranks_fuzzy = []

    top5_direct = []
    top5_evidence = []
    top5_contra = []

    fuzzy_selected_sizes = []
    fuzzy_avg_triple_scores = []
    fuzzy_direct_rates = []
    fuzzy_contra_rates = []
    fuzzy_coverage_rates = []
    fuzzy_query_touch_counts = []
    fuzzy_non_direct_query_touch_counts = []

    improved_vs_backbone = 0
    worsened_vs_backbone = 0
    improved_vs_soft_support = 0
    worsened_vs_soft_support = 0
    identical_to_v1 = 0

    for raw_row, supp_row, feat_row, fuzzy_row, v1_row in zip(raw_rows, support_rows, feature_rows, fuzzy_rows, v1_rows):
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

        sel_direct_by_id, sel_evidence_bonus_by_id, sel_contra_by_id = build_selected_candidate_maps(fuzzy_row)
        sel_evidence_by_id = {cid: (score > 0.0) for cid, score in sel_evidence_bonus_by_id.items()}

        top5_direct.append(top5_rate(fuzzy_ids, sel_direct_by_id))
        top5_evidence.append(top5_rate(fuzzy_ids, sel_evidence_by_id))
        top5_contra.append(top5_rate(fuzzy_ids, sel_contra_by_id))

        sel_triples = fuzzy_row["triple_score_rows"]
        fuzzy_selected_sizes.append(len(sel_triples))
        fuzzy_direct_rates.append(
            sum(bool(tr.get("direct_candidate_query_flag", False)) for tr in sel_triples) / len(sel_triples)
            if sel_triples else 0.0
        )
        fuzzy_contra_rates.append(
            sum(bool(tr.get("contra_flag", False)) for tr in sel_triples) / len(sel_triples)
            if sel_triples else 0.0
        )
        fuzzy_avg_triple_scores.append(
            sum(float(tr.get("triple_score", 0.0)) for tr in sel_triples) / len(sel_triples)
            if sel_triples else 0.0
        )
        fuzzy_coverage_rates.append(float(fuzzy_row["subgraph_summary"]["candidate_coverage_preserved_rate"]))
        fuzzy_query_touch_counts.append(
            sum(bool(tr.get("touches_query", False)) for tr in sel_triples)
        )
        fuzzy_non_direct_query_touch_counts.append(
            sum(bool(tr.get("touches_query", False)) and not bool(tr.get("direct_candidate_query_flag", False)) for tr in sel_triples)
        )

        if rank_f < rank_b:
            improved_vs_backbone += 1
        elif rank_f > rank_b:
            worsened_vs_backbone += 1

        if rank_f < rank_s:
            improved_vs_soft_support += 1
        elif rank_f > rank_s:
            worsened_vs_soft_support += 1

        if selected_subgraph_signature(fuzzy_row) == selected_subgraph_signature(v1_row):
            identical_to_v1 += 1

    metrics = {
        **rank_metrics_from_ranks(ranks_fuzzy),
        "avg_subgraph_size": round(sum(fuzzy_selected_sizes) / len(fuzzy_selected_sizes), 6),
        "avg_triple_score": round(sum(fuzzy_avg_triple_scores) / len(fuzzy_avg_triple_scores), 6),
        "direct_shortcut_path_rate": round(sum(fuzzy_direct_rates) / len(fuzzy_direct_rates), 6),
        "contradiction_path_rate": round(sum(fuzzy_contra_rates) / len(fuzzy_contra_rates), 6),
        "candidate_coverage_preserved_rate": round(sum(fuzzy_coverage_rates) / len(fuzzy_coverage_rates), 6),
        "avg_top5_direct_link_rate": round(sum(top5_direct) / len(top5_direct), 6),
        "avg_top5_evidence_positive_rate": round(sum(top5_evidence) / len(top5_evidence), 6),
        "avg_top5_contra_rate": round(sum(top5_contra) / len(top5_contra), 6),
        "avg_query_touch_count": round(sum(fuzzy_query_touch_counts) / len(fuzzy_query_touch_counts), 6),
        "avg_non_direct_query_touch_count": round(sum(fuzzy_non_direct_query_touch_counts) / len(fuzzy_non_direct_query_touch_counts), 6),
        "improved_vs_backbone": improved_vs_backbone,
        "worsened_vs_backbone": worsened_vs_backbone,
        "improved_vs_soft_support": improved_vs_soft_support,
        "worsened_vs_soft_support": worsened_vs_soft_support,
        "selected_subgraph_identical_to_v1_rate": round(identical_to_v1 / len(fuzzy_rows), 6),
    }
    return metrics


def delta(a: Dict[str, Any], b: Dict[str, Any], keys: List[str]) -> Dict[str, float]:
    out = {}
    for k in keys:
        va = a[k]
        vb = b[k]
        out[k] = round(va - vb, 6)
    return out


def main() -> None:
    raw_rows = load_json(RAW_PATH)
    support_rows = load_json(SUPPORT_PATH)
    feature_rows = load_json(FEATURE_PATH)
    fuzzy_variants = {name: load_json(path) for name, path in FUZZY_VARIANT_PATHS.items()}

    assert len(raw_rows) == len(support_rows) == len(feature_rows) == 500
    for name, rows in fuzzy_variants.items():
        assert len(rows) == 500, f"{name} must have 500 rows"

    baseline_rows = summarize_baseline_rows(raw_rows, support_rows, feature_rows)

    variant_metrics = {}
    v1_rows = fuzzy_variants["soft_support_fuzzy_retrieval_v1"]
    for name, rows in fuzzy_variants.items():
        variant_metrics[name] = summarize_fuzzy_variant(
            name, raw_rows, support_rows, feature_rows, rows, v1_rows
        )

    compare = {
        "stage": "fuzzy_retrieval_variant_compare",
        "rows": {
            **baseline_rows,
            **variant_metrics,
        },
        "delta_vs_soft_support": {
            name: delta(
                variant_metrics[name],
                baseline_rows["soft_support_raw"],
                [
                    "mrr_like",
                    "hits1_like",
                    "hits3_like",
                    "hits10_like",
                    "avg_gold_rank",
                    "avg_subgraph_size",
                    "direct_shortcut_path_rate",
                    "avg_top5_direct_link_rate",
                    "avg_query_touch_count",
                    "avg_non_direct_query_touch_count",
                ],
            )
            for name in variant_metrics
        },
        "delta_vs_v1": {
            name: delta(
                variant_metrics[name],
                variant_metrics["soft_support_fuzzy_retrieval_v1"],
                [
                    "mrr_like",
                    "hits1_like",
                    "hits3_like",
                    "hits10_like",
                    "avg_gold_rank",
                    "avg_subgraph_size",
                    "direct_shortcut_path_rate",
                    "avg_top5_direct_link_rate",
                    "avg_query_touch_count",
                    "avg_non_direct_query_touch_count",
                ],
            )
            for name in variant_metrics
            if name != "soft_support_fuzzy_retrieval_v1"
        },
    }

    # Sweep summary for easier reading
    evidence_order = sorted(
        variant_metrics.items(),
        key=lambda kv: (
            kv[1]["direct_shortcut_path_rate"],
            kv[1]["avg_subgraph_size"],
            -kv[1]["candidate_coverage_preserved_rate"],
        ),
    )
    proxy_order = sorted(
        variant_metrics.items(),
        key=lambda kv: (
            -kv[1]["mrr_like"],
            kv[1]["avg_gold_rank"],
            kv[1]["worsened_vs_soft_support"],
        ),
    )

    sweep_summary = {
        "stage": "fuzzy_retrieval_variant_compare",
        "retrieval_variants": list(variant_metrics.keys()),
        "evidence_cleanliness_order": [
            {
                "variant_name": name,
                "direct_shortcut_path_rate": metrics["direct_shortcut_path_rate"],
                "avg_subgraph_size": metrics["avg_subgraph_size"],
                "candidate_coverage_preserved_rate": metrics["candidate_coverage_preserved_rate"],
            }
            for name, metrics in evidence_order
        ],
        "ranking_proxy_order": [
            {
                "variant_name": name,
                "mrr_like": metrics["mrr_like"],
                "avg_gold_rank": metrics["avg_gold_rank"],
                "worsened_vs_soft_support": metrics["worsened_vs_soft_support"],
            }
            for name, metrics in proxy_order
        ],
        "redundancy_check": {
            name: metrics["selected_subgraph_identical_to_v1_rate"]
            for name, metrics in variant_metrics.items()
        },
        "provisional_read": [
            "A strong main-row candidate should reduce shortcut-heavy evidence more than v1 without introducing worsened_vs_soft_support.",
            "If a variant is nearly identical to v1, it should not be kept as a distinct candidate.",
            "If tight wins on evidence cleanliness but clearly hurts ranking proxy, it may remain a supporting row rather than the main row."
        ],
    }

    save_json(OUT_COMPARE, compare)
    save_json(OUT_SUMMARY, sweep_summary)

    lines = [
        "# Fuzzy Retrieval Variant Compare",
        "",
        "## Baselines",
        json.dumps(baseline_rows, ensure_ascii=False, indent=2),
        "",
        "## Retrieval variants",
        json.dumps(variant_metrics, ensure_ascii=False, indent=2),
        "",
        "## Delta vs soft_support_raw",
        json.dumps(compare["delta_vs_soft_support"], ensure_ascii=False, indent=2),
        "",
        "## Delta vs v1",
        json.dumps(compare["delta_vs_v1"], ensure_ascii=False, indent=2),
        "",
        "## Sweep summary",
        json.dumps(sweep_summary, ensure_ascii=False, indent=2),
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 80)
    print("FUZZY RETRIEVAL VARIANT COMPARE DONE")
    print("compare:", OUT_COMPARE)
    print("summary:", OUT_SUMMARY)
    print("report :", OUT_REPORT)
    print("=" * 80)
    print(json.dumps(sweep_summary, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
