import json
import csv
from pathlib import Path
from statistics import mean

IN_PATH = Path("dataset/setting_a/support_features/valid_support_features.json")
OUT_JSON = Path("outputs/soft_support/support_feature_summary.json")
OUT_TSV = Path("outputs/soft_support/support_feature_distribution.tsv")
OUT_QUERY_TSV = Path("outputs/soft_support/support_feature_query_summary.tsv")


BINARY_FEATURES = [
    "type_valid_flag",
    "schema_valid_flag",
    "type_filtered_keep_flag",
    "ontology_keep_flag",
    "contra_flag",
    "conflict_flag",
    "candidate_in_aligned_evidence",
]

NUMERIC_FEATURES = [
    "evidence_edge_touch_count",
    "query_edge_touch_count",
    "candidate_query_edge_count",
    "contra_penalty",
]

ALL_FEATURES = BINARY_FEATURES + NUMERIC_FEATURES


def safe_mean(xs):
    return mean(xs) if xs else 0.0


def safe_rate(xs):
    if not xs:
        return 0.0
    return sum(1 for x in xs if x) / len(xs)


def main():
    with IN_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    feature_all = {k: [] for k in ALL_FEATURES}
    feature_gold = {k: [] for k in ALL_FEATURES}
    feature_non_gold = {k: [] for k in ALL_FEATURES}

    ontology_support_type_all = {}
    ontology_support_type_gold = {}

    num_queries = len(data)
    num_queries_gold_in_topk = 0
    num_queries_gold_ontology_keep = 0
    num_queries_any_candidate_query_link = 0
    num_queries_any_evidence_touch = 0

    query_rows = []

    for row in data:
        gold_entity = row["gold_entity"]
        cand_rows = row["candidate_feature_rows"]

        gold_rows = [c for c in cand_rows if c["candidate_entity"] == gold_entity]
        gold_in_topk = len(gold_rows) > 0

        if gold_in_topk:
            num_queries_gold_in_topk += 1
            gold_row = gold_rows[0]
            if gold_row["ontology_keep_flag"] == 1:
                num_queries_gold_ontology_keep += 1
        else:
            gold_row = None

        any_candidate_query_link = any(c["candidate_query_edge_count"] > 0 for c in cand_rows)
        any_evidence_touch = any(c["evidence_edge_touch_count"] > 0 for c in cand_rows)

        if any_candidate_query_link:
            num_queries_any_candidate_query_link += 1
        if any_evidence_touch:
            num_queries_any_evidence_touch += 1

        # query-level summary row
        query_rows.append({
            "query_entity": row["query_entity"],
            "gold_entity": gold_entity,
            "gold_in_topk_raw": row["gold_in_topk_raw"],
            "gold_present_in_feature_table": int(gold_in_topk),
            "gold_ontology_keep_flag": gold_row["ontology_keep_flag"] if gold_row else None,
            "gold_candidate_in_aligned_evidence": gold_row["candidate_in_aligned_evidence"] if gold_row else None,
            "gold_evidence_edge_touch_count": gold_row["evidence_edge_touch_count"] if gold_row else None,
            "gold_candidate_query_edge_count": gold_row["candidate_query_edge_count"] if gold_row else None,
            "num_candidates": len(cand_rows),
            "num_ontology_keep": sum(c["ontology_keep_flag"] for c in cand_rows),
            "num_contra": sum(c["contra_flag"] for c in cand_rows),
            "num_evidence_touch_positive": sum(1 for c in cand_rows if c["evidence_edge_touch_count"] > 0),
            "num_candidate_query_link_positive": sum(1 for c in cand_rows if c["candidate_query_edge_count"] > 0),
        })

        for c in cand_rows:
            is_gold = c["candidate_entity"] == gold_entity

            for feat in ALL_FEATURES:
                val = c[feat]
                feature_all[feat].append(val)
                if is_gold:
                    feature_gold[feat].append(val)
                else:
                    feature_non_gold[feat].append(val)

            ost = c.get("ontology_support_type", "unknown")
            ontology_support_type_all[ost] = ontology_support_type_all.get(ost, 0) + 1
            if is_gold:
                ontology_support_type_gold[ost] = ontology_support_type_gold.get(ost, 0) + 1

    summary_rows = []
    for feat in ALL_FEATURES:
        all_vals = feature_all[feat]
        gold_vals = feature_gold[feat]
        non_gold_vals = feature_non_gold[feat]

        unique_vals = sorted(set(all_vals))
        summary_rows.append({
            "feature": feat,
            "count_all": len(all_vals),
            "count_gold": len(gold_vals),
            "count_non_gold": len(non_gold_vals),
            "mean_all": round(safe_mean(all_vals), 6),
            "mean_gold": round(safe_mean(gold_vals), 6),
            "mean_non_gold": round(safe_mean(non_gold_vals), 6),
            "nonzero_rate_all": round(safe_rate([x != 0 for x in all_vals]), 6),
            "nonzero_rate_gold": round(safe_rate([x != 0 for x in gold_vals]), 6),
            "nonzero_rate_non_gold": round(safe_rate([x != 0 for x in non_gold_vals]), 6),
            "unique_count": len(unique_vals),
            "unique_values_preview": unique_vals[:10],
        })

    # high-level conclusions heuristic
    dropped_features = []
    keep_primary = []
    keep_supporting = []

    for r in summary_rows:
        feat = r["feature"]
        if r["unique_count"] <= 1:
            dropped_features.append(feat)
            continue

        gold_lift = r["mean_gold"] - r["mean_non_gold"]
        nonzero_lift = r["nonzero_rate_gold"] - r["nonzero_rate_non_gold"]

        if feat in {"contra_flag", "conflict_flag", "contra_penalty"}:
            keep_supporting.append(feat)
        elif gold_lift > 0 or nonzero_lift > 0:
            keep_primary.append(feat)
        else:
            keep_supporting.append(feat)

    summary = {
        "num_queries": num_queries,
        "num_queries_gold_in_topk_raw": num_queries_gold_in_topk,
        "gold_in_topk_raw_rate": round(num_queries_gold_in_topk / max(num_queries, 1), 6),
        "gold_ontology_keep_rate_given_gold_in_topk": round(
            num_queries_gold_ontology_keep / max(num_queries_gold_in_topk, 1), 6
        ),
        "query_any_candidate_query_link_rate": round(
            num_queries_any_candidate_query_link / max(num_queries, 1), 6
        ),
        "query_any_evidence_touch_rate": round(
            num_queries_any_evidence_touch / max(num_queries, 1), 6
        ),
        "feature_summary_rows": summary_rows,
        "ontology_support_type_all": ontology_support_type_all,
        "ontology_support_type_gold": ontology_support_type_gold,
        "recommended_drop_from_main_score": dropped_features,
        "recommended_primary_features": keep_primary,
        "recommended_supporting_only_features": keep_supporting,
        "notes": [
            "Dropped features are constant or near-constant on valid raw candidates.",
            "Primary features are those with at least some gold-vs-non-gold separation.",
            "Contra-related features are kept as supporting/tiny-penalty signals unless later evidence suggests otherwise."
        ]
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "feature",
                "count_all",
                "count_gold",
                "count_non_gold",
                "mean_all",
                "mean_gold",
                "mean_non_gold",
                "nonzero_rate_all",
                "nonzero_rate_gold",
                "nonzero_rate_non_gold",
                "unique_count",
                "unique_values_preview",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for r in summary_rows:
            r = dict(r)
            r["unique_values_preview"] = json.dumps(r["unique_values_preview"], ensure_ascii=False)
            writer.writerow(r)

    with OUT_QUERY_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_entity",
                "gold_entity",
                "gold_in_topk_raw",
                "gold_present_in_feature_table",
                "gold_ontology_keep_flag",
                "gold_candidate_in_aligned_evidence",
                "gold_evidence_edge_touch_count",
                "gold_candidate_query_edge_count",
                "num_candidates",
                "num_ontology_keep",
                "num_contra",
                "num_evidence_touch_positive",
                "num_candidate_query_link_positive",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for r in query_rows:
            writer.writerow(r)

    print(json.dumps({
        "num_queries": summary["num_queries"],
        "gold_in_topk_raw_rate": summary["gold_in_topk_raw_rate"],
        "gold_ontology_keep_rate_given_gold_in_topk": summary["gold_ontology_keep_rate_given_gold_in_topk"],
        "query_any_candidate_query_link_rate": summary["query_any_candidate_query_link_rate"],
        "query_any_evidence_touch_rate": summary["query_any_evidence_touch_rate"],
        "recommended_drop_from_main_score": summary["recommended_drop_from_main_score"],
        "recommended_primary_features": summary["recommended_primary_features"],
        "recommended_supporting_only_features": summary["recommended_supporting_only_features"]
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()