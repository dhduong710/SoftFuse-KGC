import json
from pathlib import Path


FEATURE_PATH = Path("dataset/setting_a/support_features/valid_support_features.json")
CONFIG_DIR = Path("configs/soft_support")
OUT_DIR = Path("dataset/setting_a/soft_support_ranked_candidates")
SUMMARY_PATH = Path("outputs/soft_support/soft_support_build_summary.json")


CONFIG_PATHS = [
    CONFIG_DIR / "soft_support_b025.json",
    CONFIG_DIR / "soft_support_b050.json",
    CONFIG_DIR / "soft_support_bcap.json",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_score(c, formula_cfg):
    direct_penalty = 1.0 if c["candidate_query_edge_count"] > 0 else 0.0
    contra = float(c.get("contra_penalty", 0.0))

    if formula_cfg["use_capped_evidence"]:
        evidence_val = min(float(c["evidence_edge_touch_count"]), float(formula_cfg["evidence_cap"]))
        # normalize capped evidence to [0,1]
        evidence_term = evidence_val / max(float(formula_cfg["evidence_cap"]), 1.0)
    else:
        evidence_term = 1.0 if c["evidence_edge_touch_count"] > 0 else 0.0

    score = (
        float(formula_cfg["evidence_positive_weight"]) * evidence_term
        - float(formula_cfg["direct_link_penalty"]) * direct_penalty
        - float(formula_cfg["contra_penalty_weight"]) * contra
    )
    return round(float(score), 6)


def build_variant_rows(data, cfg):
    variant_name = cfg["variant_name"]
    formula_cfg = cfg["formula"]

    out_rows = []
    total_queries = 0
    total_candidates = 0
    gold_present = 0
    avg_top5_direct = []
    avg_top5_evidence_pos = []
    avg_top5_ontology_keep = []
    avg_top5_contra = []

    for row in data:
        cand_rows = []
        for c in row["candidate_feature_rows"]:
            x = dict(c)
            x["support_score"] = compute_score(c, formula_cfg)
            cand_rows.append(x)

        # stable sort: score desc, then base_rank asc
        cand_rows.sort(key=lambda x: (-x["support_score"], x["base_rank"]))

        candidate_entities = []
        candidate_entity_ids = []
        support_scores = []
        support_rank_order = []

        for i, c in enumerate(cand_rows, start=1):
            c["support_rank"] = i
            candidate_entities.append(c["candidate_entity"])
            candidate_entity_ids.append(c["candidate_entity_id"])
            support_scores.append(c["support_score"])
            support_rank_order.append(i)

        gold = row["gold_entity"]
        if gold in candidate_entities:
            gold_present += 1

        top5 = cand_rows[:5]
        avg_top5_direct.append(
            sum(1 for c in top5 if c["candidate_query_edge_count"] > 0) / max(len(top5), 1)
        )
        avg_top5_evidence_pos.append(
            sum(1 for c in top5 if c["evidence_edge_touch_count"] > 0) / max(len(top5), 1)
        )
        avg_top5_ontology_keep.append(
            sum(c["ontology_keep_flag"] for c in top5) / max(len(top5), 1)
        )
        avg_top5_contra.append(
            sum(c["contra_flag"] for c in top5) / max(len(top5), 1)
        )

        out_rows.append({
            "split": row["split"],
            "query_entity": row["query_entity"],
            "query_entity_id": row["query_entity_id"],
            "gold_entity": row["gold_entity"],
            "gold_entity_id": row["gold_entity_id"],
            "gold_rank_in_full_universe": row["gold_rank_in_full_universe"],
            "gold_in_topk_raw": row["gold_in_topk_raw"],
            "variant_name": variant_name,
            "candidate_entities": candidate_entities,
            "candidate_entity_ids": candidate_entity_ids,
            "support_scores": support_scores,
            "support_rank_order": support_rank_order,
            "candidate_debug_rows": [
                {
                    "candidate_entity": c["candidate_entity"],
                    "candidate_entity_id": c["candidate_entity_id"],
                    "base_rank": c["base_rank"],
                    "support_rank": c["support_rank"],
                    "support_score": c["support_score"],
                    "evidence_edge_touch_count": c["evidence_edge_touch_count"],
                    "candidate_query_edge_count": c["candidate_query_edge_count"],
                    "ontology_keep_flag": c["ontology_keep_flag"],
                    "contra_flag": c["contra_flag"],
                }
                for c in cand_rows
            ]
        })

        total_queries += 1
        total_candidates += len(cand_rows)

    summary = {
        "variant_name": variant_name,
        "num_queries": total_queries,
        "num_candidates": total_candidates,
        "avg_candidates_per_query": round(total_candidates / max(total_queries, 1), 6),
        "gold_present_rate": round(gold_present / max(total_queries, 1), 6),
        "avg_top5_direct_link_rate": round(sum(avg_top5_direct) / max(total_queries, 1), 6),
        "avg_top5_evidence_positive_rate": round(sum(avg_top5_evidence_pos) / max(total_queries, 1), 6),
        "avg_top5_ontology_keep_rate": round(sum(avg_top5_ontology_keep) / max(total_queries, 1), 6),
        "avg_top5_contra_rate": round(sum(avg_top5_contra) / max(total_queries, 1), 6),
    }

    return out_rows, summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(FEATURE_PATH)

    all_summary = {
        "stage": "soft_support_variant_build",
        "input_feature_path": str(FEATURE_PATH),
        "variant_summaries": []
    }

    for cfg_path in CONFIG_PATHS:
        cfg = load_json(cfg_path)
        rows, summary = build_variant_rows(data, cfg)

        out_path = Path(cfg["output_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

        all_summary["variant_summaries"].append(summary)

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(all_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(all_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
