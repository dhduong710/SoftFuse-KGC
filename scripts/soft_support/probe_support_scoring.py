import json
from pathlib import Path

IN_PATH = Path("dataset/setting_a/support_features/valid_support_features.json")
OUT_JSON = Path("dataset/setting_a/support_features/valid_support_probe.json")
OUT_SUMMARY = Path("outputs/soft_support/support_formula_probe.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def score_formula_a(c):
    evidence_pos = 1.0 if c["evidence_edge_touch_count"] > 0 else 0.0
    return evidence_pos - 0.1 * float(c["contra_penalty"])


def score_formula_b(c):
    evidence_pos = 1.0 if c["evidence_edge_touch_count"] > 0 else 0.0
    direct_penalty = 1.0 if c["candidate_query_edge_count"] > 0 else 0.0
    return evidence_pos - 0.5 * direct_penalty - 0.1 * float(c["contra_penalty"])


def score_formula_c(c):
    evidence_cap = float(min(c["evidence_edge_touch_count"], 2))
    direct_penalty = 1.0 if c["candidate_query_edge_count"] > 0 else 0.0
    return evidence_cap - 0.5 * direct_penalty - 0.1 * float(c["contra_penalty"])


FORMULAS = {
    "A_evidence_positive": score_formula_a,
    "B_evidence_minus_direct": score_formula_b,
    "C_capped_evidence_minus_direct": score_formula_c,
}


def rank_candidates(cand_rows, score_fn):
    rescored = []
    for c in cand_rows:
        x = dict(c)
        x["support_score"] = round(float(score_fn(c)), 6)
        rescored.append(x)

    # stable sort: score desc, then base_rank asc
    rescored.sort(key=lambda x: (-x["support_score"], x["base_rank"]))

    for i, c in enumerate(rescored, start=1):
        c["support_rank"] = i

    return rescored


def summarize_formula(rows, formula_name):
    gold_rank_list = []
    top1_hit = 0
    top3_hit = 0
    top10_hit = 0
    avg_direct_penalty_in_top5 = []
    avg_evidence_positive_in_top5 = []
    avg_ontology_keep_in_top5 = []
    avg_contra_in_top5 = []

    for row in rows:
        gold = row["gold_entity"]
        cand_rows = row["formula_outputs"][formula_name]

        gold_rank = None
        for c in cand_rows:
            if c["candidate_entity"] == gold:
                gold_rank = c["support_rank"]
                break

        if gold_rank is None:
            gold_rank = len(cand_rows) + 1

        gold_rank_list.append(gold_rank)
        top1_hit += int(gold_rank <= 1)
        top3_hit += int(gold_rank <= 3)
        top10_hit += int(gold_rank <= 10)

        top5 = cand_rows[:5]
        avg_direct_penalty_in_top5.append(
            sum(1 for c in top5 if c["candidate_query_edge_count"] > 0) / max(len(top5), 1)
        )
        avg_evidence_positive_in_top5.append(
            sum(1 for c in top5 if c["evidence_edge_touch_count"] > 0) / max(len(top5), 1)
        )
        avg_ontology_keep_in_top5.append(
            sum(c["ontology_keep_flag"] for c in top5) / max(len(top5), 1)
        )
        avg_contra_in_top5.append(
            sum(c["contra_flag"] for c in top5) / max(len(top5), 1)
        )

    n = len(rows)
    mrr = sum(1.0 / r for r in gold_rank_list) / max(n, 1)

    return {
        "num_queries": n,
        "mrr_like": round(mrr, 6),
        "hits1_like": round(top1_hit / max(n, 1), 6),
        "hits3_like": round(top3_hit / max(n, 1), 6),
        "hits10_like": round(top10_hit / max(n, 1), 6),
        "avg_gold_rank": round(sum(gold_rank_list) / max(n, 1), 6),
        "avg_top5_direct_link_rate": round(sum(avg_direct_penalty_in_top5) / max(n, 1), 6),
        "avg_top5_evidence_positive_rate": round(sum(avg_evidence_positive_in_top5) / max(n, 1), 6),
        "avg_top5_ontology_keep_rate": round(sum(avg_ontology_keep_in_top5) / max(n, 1), 6),
        "avg_top5_contra_rate": round(sum(avg_contra_in_top5) / max(n, 1), 6),
    }


def main():
    data = load_json(IN_PATH)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    out_rows = []
    summary = {}

    for row in data:
        out_row = {
            "split": row["split"],
            "query_entity": row["query_entity"],
            "query_entity_id": row["query_entity_id"],
            "gold_entity": row["gold_entity"],
            "gold_entity_id": row["gold_entity_id"],
            "gold_rank_in_full_universe": row["gold_rank_in_full_universe"],
            "gold_in_topk_raw": row["gold_in_topk_raw"],
            "formula_outputs": {}
        }

        for name, fn in FORMULAS.items():
            out_row["formula_outputs"][name] = rank_candidates(row["candidate_feature_rows"], fn)

        out_rows.append(out_row)

    for name in FORMULAS:
        summary[name] = summarize_formula(out_rows, name)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)

    best_formula = max(summary.items(), key=lambda kv: kv[1]["mrr_like"])[0]

    final_summary = {
        "stage": "soft_support_score_probe",
        "input": str(IN_PATH),
        "output": str(OUT_JSON),
        "formula_summaries": summary,
        "best_formula_by_mrr_like": best_formula,
        "notes": [
            "This is a validation-only soft-support scoring probe.",
            "The outputs are not the final soft_support_raw row.",
            "No test decision is allowed at this stage."
        ]
    }

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
