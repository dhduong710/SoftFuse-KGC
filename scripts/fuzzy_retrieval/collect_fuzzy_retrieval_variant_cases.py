from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

RAW_PATH = ROOT / "dataset/setting_a/backbone_candidates/valid_top20_raw.json"
SUPPORT_PATH = ROOT / "dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_main.json"
V1_PATH = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_fuzzy_retrieval_v1.json"
TIGHT_PATH = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_soft_support_fuzzy_retrieval_tight.json"
DIRECTPLUS_PATH = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_soft_support_fuzzy_retrieval_directplus.json"
COMPARE_PATH = ROOT / "outputs/fuzzy_retrieval/retrieval_valid_compare.json"

OUT_JSON = ROOT / "outputs/fuzzy_retrieval/retrieval_case_samples_main.json"
OUT_MD = ROOT / "outputs/fuzzy_retrieval/reports/fuzzy_retrieval_variant_case_review.md"


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


def subgraph_signature(row: Dict[str, Any]) -> Tuple[Tuple[Any, Any, Any], ...]:
    return tuple(tuple(x) for x in row["selected_subgraph"])


def count_direct_shortcuts(row: Dict[str, Any]) -> int:
    return sum(bool(tr.get("direct_candidate_query_flag", False)) for tr in row["triple_score_rows"])


def count_non_direct_query_touch(row: Dict[str, Any]) -> int:
    return sum(
        bool(tr.get("touches_query", False)) and not bool(tr.get("direct_candidate_query_flag", False))
        for tr in row["triple_score_rows"]
    )


def top_scored_triples(row: Dict[str, Any], n: int = 5) -> List[Dict[str, Any]]:
    return [
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
        for tr in row["triple_score_rows"][:n]
    ]


def make_case(
    bucket: str,
    rationale: str,
    i: int,
    raw_row: Dict[str, Any],
    support_row: Dict[str, Any],
    v1_row: Dict[str, Any],
    tight_row: Dict[str, Any],
    directplus_row: Dict[str, Any],
) -> Dict[str, Any]:
    gold_id = raw_row["gold_entity_id"]

    rank_backbone = gold_rank(raw_row["candidate_entity_ids"], gold_id)
    rank_soft = gold_rank(support_row["candidate_entity_ids"], gold_id)
    rank_v1 = gold_rank(rerank_fuzzy_candidates(v1_row), gold_id)
    rank_tight = gold_rank(rerank_fuzzy_candidates(tight_row), gold_id)
    rank_directplus = gold_rank(rerank_fuzzy_candidates(directplus_row), gold_id)

    return {
        "bucket": bucket,
        "row_index": i,
        "row_uid": tight_row["row_uid"],
        "query_entity": tight_row["query_entity"],
        "query_entity_id": tight_row["query_entity_id"],
        "gold_entity": tight_row["gold_entity"],
        "gold_entity_id": tight_row["gold_entity_id"],
        "rank_backbone": rank_backbone,
        "rank_soft_support": rank_soft,
        "rank_v1": rank_v1,
        "rank_tight": rank_tight,
        "rank_directplus": rank_directplus,
        "v1_selected_subgraph_size": len(v1_row["triple_score_rows"]),
        "tight_selected_subgraph_size": len(tight_row["triple_score_rows"]),
        "directplus_selected_subgraph_size": len(directplus_row["triple_score_rows"]),
        "v1_direct_shortcuts": count_direct_shortcuts(v1_row),
        "tight_direct_shortcuts": count_direct_shortcuts(tight_row),
        "directplus_direct_shortcuts": count_direct_shortcuts(directplus_row),
        "v1_non_direct_query_touch_count": count_non_direct_query_touch(v1_row),
        "tight_non_direct_query_touch_count": count_non_direct_query_touch(tight_row),
        "directplus_non_direct_query_touch_count": count_non_direct_query_touch(directplus_row),
        "candidate_coverage_preserved_rate_tight": tight_row["subgraph_summary"]["candidate_coverage_preserved_rate"],
        "top_band_coverage_preserved_rate_tight": tight_row["subgraph_summary"]["top_band_coverage_preserved_rate"],
        "rationale": rationale,
        "soft_support_top10": support_row["candidate_entities"][:10],
        "v1_top_scored_triples": top_scored_triples(v1_row, 5),
        "tight_top_scored_triples": top_scored_triples(tight_row, 5),
    }


def main() -> None:
    raw_rows = load_json(RAW_PATH)
    support_rows = load_json(SUPPORT_PATH)
    v1_rows = load_json(V1_PATH)
    tight_rows = load_json(TIGHT_PATH)
    directplus_rows = load_json(DIRECTPLUS_PATH)
    compare = load_json(COMPARE_PATH)

    assert len(raw_rows) == len(support_rows) == len(v1_rows) == len(tight_rows) == len(directplus_rows) == 500

    buckets = {
        "tight_same_rank_cleaner_than_v1": [],
        "tight_preserved_backbone_gain": [],
        "tight_anchor_caution_same_rank": [],
        "directplus_redundant_vs_v1": [],
        "raw_bottleneck_failure": [],
    }

    for i, (raw_row, support_row, v1_row, tight_row, directplus_row) in enumerate(
        zip(raw_rows, support_rows, v1_rows, tight_rows, directplus_rows)
    ):
        gold_id = raw_row["gold_entity_id"]

        rank_backbone = gold_rank(raw_row["candidate_entity_ids"], gold_id)
        rank_soft = gold_rank(support_row["candidate_entity_ids"], gold_id)
        rank_v1 = gold_rank(rerank_fuzzy_candidates(v1_row), gold_id)
        rank_tight = gold_rank(rerank_fuzzy_candidates(tight_row), gold_id)
        rank_directplus = gold_rank(rerank_fuzzy_candidates(directplus_row), gold_id)

        v1_size = len(v1_row["triple_score_rows"])
        tight_size = len(tight_row["triple_score_rows"])

        v1_direct = count_direct_shortcuts(v1_row)
        tight_direct = count_direct_shortcuts(tight_row)

        v1_ndqt = count_non_direct_query_touch(v1_row)
        tight_ndqt = count_non_direct_query_touch(tight_row)

        # A. tight same-rank but cleaner than v1
        if rank_tight == rank_v1 == rank_soft and tight_size < v1_size and tight_direct < v1_direct:
            buckets["tight_same_rank_cleaner_than_v1"].append(
                make_case(
                    "tight_same_rank_cleaner_than_v1",
                    "tight preserves the same ranking as v1 / soft_support while using a smaller and less shortcut-heavy evidence bundle.",
                    i, raw_row, support_row, v1_row, tight_row, directplus_row
                )
            )

        # B. tight preserved backbone gain
        if rank_soft < rank_backbone and rank_tight == rank_soft:
            buckets["tight_preserved_backbone_gain"].append(
                make_case(
                    "tight_preserved_backbone_gain",
                    "soft_support already improved over backbone, and tight preserves that gain while remaining retrieval-clean.",
                    i, raw_row, support_row, v1_row, tight_row, directplus_row
                )
            )

        # C. anchor caution, same rank
        if rank_tight == rank_v1 and tight_size < v1_size and tight_ndqt < v1_ndqt:
            buckets["tight_anchor_caution_same_rank"].append(
                make_case(
                    "tight_anchor_caution_same_rank",
                    "tight stays ranking-stable but compresses non-direct query-touch evidence more than v1, so this row should be reviewed for possible over-compression.",
                    i, raw_row, support_row, v1_row, tight_row, directplus_row
                )
            )

        # D. directplus redundancy
        if subgraph_signature(directplus_row) == subgraph_signature(v1_row):
            buckets["directplus_redundant_vs_v1"].append(
                make_case(
                    "directplus_redundant_vs_v1",
                    "directplus yields the same selected evidence bundle as v1 for this row, so it adds little distinct value.",
                    i, raw_row, support_row, v1_row, tight_row, directplus_row
                )
            )

        # E. raw bottleneck
        if rank_backbone == 21 and rank_soft == 21 and rank_v1 == 21 and rank_tight == 21:
            buckets["raw_bottleneck_failure"].append(
                make_case(
                    "raw_bottleneck_failure",
                    "gold is absent from the usable candidate ranking path, so retrieval-stage refinement cannot recover it here.",
                    i, raw_row, support_row, v1_row, tight_row, directplus_row
                )
            )

    out = {
        "stage": "fuzzy_retrieval_variant_case_review",
        "summary_from_variant_compare": {
            "v1": compare["rows"]["soft_support_fuzzy_retrieval_v1"],
            "tight": compare["rows"]["soft_support_fuzzy_retrieval_tight"],
            "directplus": compare["rows"]["soft_support_fuzzy_retrieval_directplus"],
            "delta_vs_v1": compare["delta_vs_v1"],
        },
        "bucket_counts": {k: len(v) for k, v in buckets.items()},
        "buckets": {k: v[:12] for k, v in buckets.items()},
        "case_review_takeaway": [
            "tight is the main candidate under review against v1.",
            "directplus is treated mainly as a redundancy check rather than a serious main-row competitor.",
            "anchor caution should be carried to main-row selection but should not be confused with a ranking collapse."
        ]
    }

    save_json(OUT_JSON, out)

    md_lines = [
        "# Fuzzy Retrieval Variant Case Review",
        "",
        "## Main framing",
        "- review focus: v1 vs tight",
        "- directplus is reviewed mainly for redundancy",
        "",
        "## Bucket counts",
        json.dumps(out["bucket_counts"], ensure_ascii=False, indent=2),
        "",
        "## Case Review Takeaway",
        "\n".join([f"- {x}" for x in out["case_review_takeaway"]]),
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print("=" * 80)
    print("FUZZY RETRIEVAL VARIANT CASE REVIEW BUILT")
    print("json  :", OUT_JSON)
    print("md    :", OUT_MD)
    print("=" * 80)
    print(json.dumps(out["bucket_counts"], ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
