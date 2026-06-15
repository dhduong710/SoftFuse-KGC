import json
from pathlib import Path

BACKBONE_PATH = Path("dataset/setting_a/backbone_candidates/valid_top20_raw.json")
B050_PATH = Path("dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_b050.json")
BCAP_PATH = Path("dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_bcap.json")

OUT_CASES = Path("outputs/soft_support/soft_support_case_samples.json")
OUT_SUMMARY = Path("outputs/soft_support/soft_support_case_summary.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def qkey(row):
    return f"{row.get('split','valid')}|||{row['query_entity']}|||{row['gold_entity']}"


def get_candidate_order(row):
    return row["candidate_entities"]


def get_gold_rank(order, gold, missing_rank=21):
    if gold in order:
        return order.index(gold) + 1
    return missing_rank


def topk(order, k=5):
    return order[:k]


def debug_map(row):
    return {x["candidate_entity"]: x for x in row.get("candidate_debug_rows", [])}


def main():
    backbone_rows = load_json(BACKBONE_PATH)
    b050_rows = load_json(B050_PATH)
    bcap_rows = load_json(BCAP_PATH)

    backbone_map = {qkey(r): r for r in backbone_rows}
    b050_map = {qkey(r): r for r in b050_rows}
    bcap_map = {qkey(r): r for r in bcap_rows}

    improved_vs_backbone = []
    worsened_vs_backbone = []
    unchanged_bad_vs_backbone = []
    improved_vs_bcap = []
    worsened_vs_bcap = []

    for key in b050_map:
        bb = backbone_map[key]
        s5 = b050_map[key]
        sc = bcap_map[key]

        gold = s5["gold_entity"]

        bb_order = get_candidate_order(bb)
        s5_order = get_candidate_order(s5)
        sc_order = get_candidate_order(sc)

        bb_rank = get_gold_rank(bb_order, gold, missing_rank=21)
        s5_rank = get_gold_rank(s5_order, gold, missing_rank=21)
        sc_rank = get_gold_rank(sc_order, gold, missing_rank=21)

        s5_dbg = debug_map(s5).get(gold, {})
        sc_dbg = debug_map(sc).get(gold, {})

        item = {
            "query_entity": s5["query_entity"],
            "gold_entity": gold,
            "backbone_rank": bb_rank,
            "b050_rank": s5_rank,
            "bcap_rank": sc_rank,
            "b050_minus_backbone": s5_rank - bb_rank,
            "b050_minus_bcap": s5_rank - sc_rank,
            "top5_backbone": topk(bb_order, 5),
            "top5_b050": topk(s5_order, 5),
            "top5_bcap": topk(sc_order, 5),
            "gold_debug_b050": {
                "support_score": s5_dbg.get("support_score"),
                "base_rank": s5_dbg.get("base_rank"),
                "support_rank": s5_dbg.get("support_rank"),
                "evidence_edge_touch_count": s5_dbg.get("evidence_edge_touch_count"),
                "candidate_query_edge_count": s5_dbg.get("candidate_query_edge_count"),
                "ontology_keep_flag": s5_dbg.get("ontology_keep_flag"),
                "contra_flag": s5_dbg.get("contra_flag"),
            },
            "gold_debug_bcap": {
                "support_score": sc_dbg.get("support_score"),
                "base_rank": sc_dbg.get("base_rank"),
                "support_rank": sc_dbg.get("support_rank"),
                "evidence_edge_touch_count": sc_dbg.get("evidence_edge_touch_count"),
                "candidate_query_edge_count": sc_dbg.get("candidate_query_edge_count"),
                "ontology_keep_flag": sc_dbg.get("ontology_keep_flag"),
                "contra_flag": sc_dbg.get("contra_flag"),
            }
        }

        if s5_rank < bb_rank:
            improved_vs_backbone.append(item)
        elif s5_rank > bb_rank:
            worsened_vs_backbone.append(item)
        elif s5_rank > 10:
            unchanged_bad_vs_backbone.append(item)

        if s5_rank < sc_rank:
            improved_vs_bcap.append(item)
        elif s5_rank > sc_rank:
            worsened_vs_bcap.append(item)

    improved_vs_backbone.sort(key=lambda x: (x["b050_minus_backbone"], x["b050_rank"]))
    worsened_vs_backbone.sort(key=lambda x: (x["b050_minus_backbone"]), reverse=True)
    unchanged_bad_vs_backbone.sort(key=lambda x: x["b050_rank"], reverse=True)
    improved_vs_bcap.sort(key=lambda x: (x["b050_minus_bcap"], x["b050_rank"]))
    worsened_vs_bcap.sort(key=lambda x: (x["b050_minus_bcap"]), reverse=True)

    summary = {
        "num_queries": len(b050_map),
        "num_improved_vs_backbone": len(improved_vs_backbone),
        "num_worsened_vs_backbone": len(worsened_vs_backbone),
        "num_unchanged_bad_vs_backbone": len(unchanged_bad_vs_backbone),
        "num_improved_vs_bcap": len(improved_vs_bcap),
        "num_worsened_vs_bcap": len(worsened_vs_bcap),
        "improved_rate_vs_backbone": round(len(improved_vs_backbone) / max(len(b050_map), 1), 6),
        "worsened_rate_vs_backbone": round(len(worsened_vs_backbone) / max(len(b050_map), 1), 6),
        "improved_rate_vs_bcap": round(len(improved_vs_bcap) / max(len(b050_map), 1), 6),
        "worsened_rate_vs_bcap": round(len(worsened_vs_bcap) / max(len(b050_map), 1), 6),
    }

    out = {
        "summary": summary,
        "samples": {
            "top_improved_vs_backbone": improved_vs_backbone[:12],
            "top_worsened_vs_backbone": worsened_vs_backbone[:12],
            "top_unchanged_bad_vs_backbone": unchanged_bad_vs_backbone[:12],
            "top_improved_vs_bcap": improved_vs_bcap[:12],
            "top_worsened_vs_bcap": worsened_vs_bcap[:12],
        }
    }

    with OUT_CASES.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()