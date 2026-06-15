import json
from pathlib import Path

IN_PATH = Path("dataset/setting_a/support_features/valid_support_probe.json")
OUT_DIR = Path("outputs/soft_support")
OUT_CASES = OUT_DIR / "support_case_samples.json"
OUT_COMPARE = OUT_DIR / "support_case_compare_summary.json"


FORMULA_BASE = "A_evidence_positive"
FORMULA_BEST = "B_evidence_minus_direct"


def get_gold_rank(cand_rows, gold_entity):
    for c in cand_rows:
        if c["candidate_entity"] == gold_entity:
            return c["support_rank"], c
    return len(cand_rows) + 1, None


def topk_names(cand_rows, k=5):
    return [c["candidate_entity"] for c in cand_rows[:k]]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with IN_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    improved = []
    worsened = []
    unchanged_good = []
    unchanged_bad = []

    for row in data:
        gold = row["gold_entity"]
        out_a = row["formula_outputs"][FORMULA_BASE]
        out_b = row["formula_outputs"][FORMULA_BEST]

        rank_a, gold_a = get_gold_rank(out_a, gold)
        rank_b, gold_b = get_gold_rank(out_b, gold)

        item = {
            "query_entity": row["query_entity"],
            "gold_entity": gold,
            "gold_rank_formula_a": rank_a,
            "gold_rank_formula_b": rank_b,
            "rank_delta_b_minus_a": rank_b - rank_a,
            "top5_formula_a": topk_names(out_a, 5),
            "top5_formula_b": topk_names(out_b, 5),
            "gold_features_formula_a": {
                "support_score": gold_a["support_score"] if gold_a else None,
                "ontology_keep_flag": gold_a["ontology_keep_flag"] if gold_a else None,
                "evidence_edge_touch_count": gold_a["evidence_edge_touch_count"] if gold_a else None,
                "candidate_query_edge_count": gold_a["candidate_query_edge_count"] if gold_a else None,
                "contra_flag": gold_a["contra_flag"] if gold_a else None,
            },
            "gold_features_formula_b": {
                "support_score": gold_b["support_score"] if gold_b else None,
                "ontology_keep_flag": gold_b["ontology_keep_flag"] if gold_b else None,
                "evidence_edge_touch_count": gold_b["evidence_edge_touch_count"] if gold_b else None,
                "candidate_query_edge_count": gold_b["candidate_query_edge_count"] if gold_b else None,
                "contra_flag": gold_b["contra_flag"] if gold_b else None,
            },
        }

        if rank_b < rank_a:
            improved.append(item)
        elif rank_b > rank_a:
            worsened.append(item)
        else:
            if rank_b <= 10:
                unchanged_good.append(item)
            else:
                unchanged_bad.append(item)

    # sort
    improved.sort(key=lambda x: (x["gold_rank_formula_b"] - x["gold_rank_formula_a"], x["gold_rank_formula_b"]))
    worsened.sort(key=lambda x: (x["gold_rank_formula_b"] - x["gold_rank_formula_a"]), reverse=True)
    unchanged_bad.sort(key=lambda x: x["gold_rank_formula_b"], reverse=True)

    out = {
        "formula_base": FORMULA_BASE,
        "formula_best": FORMULA_BEST,
        "summary": {
            "num_queries": len(data),
            "num_improved": len(improved),
            "num_worsened": len(worsened),
            "num_unchanged_good": len(unchanged_good),
            "num_unchanged_bad": len(unchanged_bad),
        },
        "samples": {
            "top_improved": improved[:10],
            "top_worsened": worsened[:10],
            "top_unchanged_bad": unchanged_bad[:10],
            "top_unchanged_good": unchanged_good[:10],
        }
    }

    with OUT_CASES.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    compare = {
        "num_queries": len(data),
        "num_improved": len(improved),
        "num_worsened": len(worsened),
        "num_unchanged_good": len(unchanged_good),
        "num_unchanged_bad": len(unchanged_bad),
        "improved_rate": round(len(improved) / max(len(data), 1), 6),
        "worsened_rate": round(len(worsened) / max(len(data), 1), 6),
    }

    with OUT_COMPARE.open("w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)

    print(json.dumps(compare, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
