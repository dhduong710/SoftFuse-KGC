from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "dataset/setting_a/backbone_candidates/valid_top20_raw.json"
SUPPORT_PATH = ROOT / "dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_main.json"
FEATURE_PATH = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_path_features.json"
FUZZY_PATH = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_fuzzy_retrieval_v1.json"
COMPARE_PATH = ROOT / "outputs/fuzzy_retrieval/retrieval_valid_compare_v1.json"

OUT_JSON = ROOT / "outputs/fuzzy_retrieval/retrieval_case_samples_v1.json"
OUT_REPORT = ROOT / "outputs/fuzzy_retrieval/reports/fuzzy_retrieval_v1_case_review.md"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def gold_rank(candidate_ids: List[Any], gold_id: Any) -> int:
    try:
        return candidate_ids.index(gold_id) + 1
    except ValueError:
        return 21


def build_selected_candidate_maps(fuzzy_row: Dict[str, Any]):
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


def case_payload(
    i: int,
    raw_row: Dict[str, Any],
    support_row: Dict[str, Any],
    feature_row: Dict[str, Any],
    fuzzy_row: Dict[str, Any],
    rank_b: int,
    rank_s: int,
    rank_f: int,
    bucket_name: str,
    rationale: str,
) -> Dict[str, Any]:
    orig_direct = sum(bool(tr.get("direct_candidate_query_flag", False)) for tr in feature_row["triple_feature_rows"])
    sel_direct = sum(bool(tr.get("direct_candidate_query_flag", False)) for tr in fuzzy_row["triple_score_rows"])
    orig_contra = sum(bool(tr.get("contra_flag", False)) for tr in feature_row["triple_feature_rows"])
    sel_contra = sum(bool(tr.get("contra_flag", False)) for tr in fuzzy_row["triple_score_rows"])

    return {
        "bucket": bucket_name,
        "row_index": i,
        "row_uid": fuzzy_row["row_uid"],
        "query_entity": fuzzy_row["query_entity"],
        "query_entity_id": fuzzy_row["query_entity_id"],
        "gold_entity": fuzzy_row["gold_entity"],
        "gold_entity_id": fuzzy_row["gold_entity_id"],
        "rank_backbone": rank_b,
        "rank_soft_support": rank_s,
        "rank_fuzzy_v1": rank_f,
        "original_subgraph_size": len(feature_row["triple_feature_rows"]),
        "selected_subgraph_size": len(fuzzy_row["triple_score_rows"]),
        "original_direct_shortcuts": orig_direct,
        "selected_direct_shortcuts": sel_direct,
        "original_contra_triples": orig_contra,
        "selected_contra_triples": sel_contra,
        "candidate_coverage_preserved_rate": fuzzy_row["subgraph_summary"]["candidate_coverage_preserved_rate"],
        "top_band_coverage_preserved_rate": fuzzy_row["subgraph_summary"]["top_band_coverage_preserved_rate"],
        "rationale": rationale,
        "backbone_top10": raw_row["candidate_entities"][:10],
        "soft_support_top10": support_row["candidate_entities"][:10],
        "fuzzy_top10_candidate_ids": rerank_fuzzy_candidates(fuzzy_row)[:10],
        "selected_top_scored_triples": [
            {
                "triple": [tr["head_id"], tr["relation_id"], tr["tail_id"]],
                "relation_name": tr["relation_name"],
                "triple_score": tr["triple_score"],
                "touches_query": tr["touches_query"],
                "touches_candidate": tr["touches_candidate"],
                "touches_top_band_candidate": tr["touches_top_band_candidate"],
                "direct_candidate_query_flag": tr["direct_candidate_query_flag"],
                "contra_flag": tr["contra_flag"],
            }
            for tr in fuzzy_row["triple_score_rows"][:5]
        ],
    }


def main() -> None:
    raw_rows = load_json(RAW_PATH)
    support_rows = load_json(SUPPORT_PATH)
    feature_rows = load_json(FEATURE_PATH)
    fuzzy_rows = load_json(FUZZY_PATH)
    compare = load_json(COMPARE_PATH)

    assert len(raw_rows) == len(support_rows) == len(feature_rows) == len(fuzzy_rows) == 500

    preserved_improvement_vs_backbone = []
    same_rank_cleaner_subgraph = []
    unchanged_bad = []
    raw_bottleneck_failure = []

    for i, (raw_row, support_row, feature_row, fuzzy_row) in enumerate(zip(raw_rows, support_rows, feature_rows, fuzzy_rows)):
        gold_id = raw_row["gold_entity_id"]

        raw_ids = raw_row["candidate_entity_ids"]
        support_ids = support_row["candidate_entity_ids"]
        fuzzy_ids = rerank_fuzzy_candidates(fuzzy_row)

        rank_b = gold_rank(raw_ids, gold_id)
        rank_s = gold_rank(support_ids, gold_id)
        rank_f = gold_rank(fuzzy_ids, gold_id)

        orig_size = len(feature_row["triple_feature_rows"])
        sel_size = len(fuzzy_row["triple_score_rows"])

        orig_direct = sum(bool(tr.get("direct_candidate_query_flag", False)) for tr in feature_row["triple_feature_rows"])
        sel_direct = sum(bool(tr.get("direct_candidate_query_flag", False)) for tr in fuzzy_row["triple_score_rows"])

        # A. backbone improvement preserved
        if rank_s < rank_b and rank_f == rank_s:
            preserved_improvement_vs_backbone.append(
                case_payload(
                    i, raw_row, support_row, feature_row, fuzzy_row,
                    rank_b, rank_s, rank_f,
                    "preserved_improvement_vs_backbone",
                    "soft_support improved over backbone and fuzzy retrieval preserved that gain while using a smaller / cleaner selected subgraph."
                )
            )

        # B. same rank but cleaner subgraph
        if rank_f == rank_s and sel_size < orig_size and sel_direct < orig_direct:
            same_rank_cleaner_subgraph.append(
                case_payload(
                    i, raw_row, support_row, feature_row, fuzzy_row,
                    rank_b, rank_s, rank_f,
                    "same_rank_cleaner_subgraph",
                    "ranking stayed unchanged, but retrieval reduced shortcut-heavy evidence and compressed the subgraph."
                )
            )

        # C. unchanged bad
        if rank_f == rank_s and rank_f >= 10:
            unchanged_bad.append(
                case_payload(
                    i, raw_row, support_row, feature_row, fuzzy_row,
                    rank_b, rank_s, rank_f,
                    "unchanged_bad",
                    "retrieval stayed stable but did not rescue the candidate ranking; this is a useful failure bucket for manual inspection."
                )
            )

        # D. raw bottleneck failure
        if rank_b == 21 and rank_s == 21 and rank_f == 21:
            raw_bottleneck_failure.append(
                case_payload(
                    i, raw_row, support_row, feature_row, fuzzy_row,
                    rank_b, rank_s, rank_f,
                    "raw_bottleneck_failure",
                    "gold is absent from raw candidate ranking, so retrieval cannot recover it at this stage."
                )
            )
        elif rank_s >= 15 and rank_f == rank_s and sel_size < orig_size:
            raw_bottleneck_failure.append(
                case_payload(
                    i, raw_row, support_row, feature_row, fuzzy_row,
                    rank_b, rank_s, rank_f,
                    "raw_bottleneck_failure",
                    "retrieval cleaned evidence but the remaining error still appears dominated by candidate-stage weakness or weak evidence."
                )
            )

    # Keep sample buckets small enough for manual review.
    out = {
        "stage": "fuzzy_retrieval_v1_case_review",
        "summary_from_compare": {
            "improved_vs_backbone": compare["case_level"]["improved_vs_backbone"],
            "worsened_vs_backbone": compare["case_level"]["worsened_vs_backbone"],
            "improved_vs_soft_support": compare["case_level"]["improved_vs_soft_support"],
            "worsened_vs_soft_support": compare["case_level"]["worsened_vs_soft_support"],
        },
        "buckets": {
            "preserved_improvement_vs_backbone": preserved_improvement_vs_backbone[:12],
            "same_rank_cleaner_subgraph": same_rank_cleaner_subgraph[:12],
            "unchanged_bad": unchanged_bad[:12],
            "raw_bottleneck_failure": raw_bottleneck_failure[:12],
        },
        "bucket_counts": {
            "preserved_improvement_vs_backbone": len(preserved_improvement_vs_backbone),
            "same_rank_cleaner_subgraph": len(same_rank_cleaner_subgraph),
            "unchanged_bad": len(unchanged_bad),
            "raw_bottleneck_failure": len(raw_bottleneck_failure),
        },
        "case_review_takeaway": [
            "There were no ranking improvements over soft_support_raw in the v1 comparison.",
            "Therefore, this review focuses on evidence cleanliness, preservation of prior gains, unchanged-bad cases, and raw bottleneck failures.",
        ],
    }

    save_json(OUT_JSON, out)

    report_lines = [
        "# Fuzzy Retrieval v1 Case Review",
        "",
        "## Main framing",
        "- fuzzy retrieval v1 did not change ranking-like proxy vs soft_support_raw",
        "- therefore, manual review focuses on subgraph quality and failure interpretation",
        "",
        "## Bucket counts",
        json.dumps(out["bucket_counts"], ensure_ascii=False, indent=2),
        "",
        "## Case Review Takeaway",
        "- preserved_improvement_vs_backbone: check whether fuzzy retrieval preserves soft-support gains",
        "- same_rank_cleaner_subgraph: check whether evidence becomes cleaner without rank movement",
        "- unchanged_bad: inspect stable failures",
        "- raw_bottleneck_failure: separate candidate-source bottlenecks from retrieval-stage issues",
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 80)
    print("FUZZY RETRIEVAL V1 CASE REVIEW BUILT")
    print("json  :", OUT_JSON)
    print("report:", OUT_REPORT)
    print("=" * 80)
    print(json.dumps(out["bucket_counts"], ensure_ascii=False, indent=2))
    print("=" * 80)
    print(json.dumps(out["summary_from_compare"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
