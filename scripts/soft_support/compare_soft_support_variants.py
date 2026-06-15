import json
from pathlib import Path


FEATURE_PATH = Path("dataset/setting_a/support_features/valid_support_features.json")

ROW_PATHS = {
    "backbone_raw": Path("dataset/setting_a/backbone_candidates/valid_top20_raw.json"),
    "ontology_raw": Path("dataset/setting_a/ontology_control/valid_top20_ontology_raw.json"),
    "soft_support_raw_b025": Path("dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_b025.json"),
    "soft_support_raw_b050": Path("dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_b050.json"),
    "soft_support_raw_bcap": Path("dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_bcap.json"),
}

OUT_COMPARE = Path("outputs/soft_support/soft_support_valid_compare.json")
OUT_PAIRWISE = Path("outputs/soft_support/soft_support_pairwise_order_match.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def qkey(split, query_entity, gold_entity):
    return f"{split}|||{query_entity}|||{gold_entity}"


def get_candidate_order(row):
    # raw / ontology rows
    if "candidate_entities" in row:
        return row["candidate_entities"]
    raise KeyError("candidate_entities not found")


def get_gold_rank(candidate_order, gold_entity, missing_rank=21):
    if gold_entity in candidate_order:
        return candidate_order.index(gold_entity) + 1
    return missing_rank


def build_feature_maps(feature_rows):
    fmap = {}
    for row in feature_rows:
        key = qkey(row["split"], row["query_entity"], row["gold_entity"])
        cand_map = {}
        for c in row["candidate_feature_rows"]:
            cand_map[c["candidate_entity"]] = c
        fmap[key] = cand_map
    return fmap


def summarize_row(name, rows, feature_map, missing_rank=21):
    gold_ranks = []
    avg_candidate_size = []
    top5_direct = []
    top5_evidence = []
    top5_ontology = []
    top5_contra = []
    gold_present = 0

    row_orders = {}

    for row in rows:
        split = row.get("split", "valid")
        query_entity = row["query_entity"]
        gold_entity = row["gold_entity"]
        key = qkey(split, query_entity, gold_entity)

        candidate_order = get_candidate_order(row)
        row_orders[key] = candidate_order

        cand_size = len(candidate_order)
        avg_candidate_size.append(cand_size)

        if gold_entity in candidate_order:
            gold_present += 1

        gold_rank = get_gold_rank(candidate_order, gold_entity, missing_rank=missing_rank)
        gold_ranks.append(gold_rank)

        feat_map = feature_map[key]
        top5 = candidate_order[:5]

        top5_direct.append(
            sum(1 for c in top5 if feat_map[c]["candidate_query_edge_count"] > 0) / max(len(top5), 1)
        )
        top5_evidence.append(
            sum(1 for c in top5 if feat_map[c]["evidence_edge_touch_count"] > 0) / max(len(top5), 1)
        )
        top5_ontology.append(
            sum(feat_map[c]["ontology_keep_flag"] for c in top5) / max(len(top5), 1)
        )
        top5_contra.append(
            sum(feat_map[c]["contra_flag"] for c in top5) / max(len(top5), 1)
        )

    n = len(rows)
    mrr_like = sum(1.0 / r for r in gold_ranks) / max(n, 1)

    summary = {
        "row_name": name,
        "num_queries": n,
        "avg_candidate_size": round(sum(avg_candidate_size) / max(n, 1), 6),
        "gold_present_rate": round(gold_present / max(n, 1), 6),
        "mrr_like": round(mrr_like, 6),
        "hits1_like": round(sum(1 for r in gold_ranks if r <= 1) / max(n, 1), 6),
        "hits3_like": round(sum(1 for r in gold_ranks if r <= 3) / max(n, 1), 6),
        "hits10_like": round(sum(1 for r in gold_ranks if r <= 10) / max(n, 1), 6),
        "avg_gold_rank": round(sum(gold_ranks) / max(n, 1), 6),
        "avg_top5_direct_link_rate": round(sum(top5_direct) / max(n, 1), 6),
        "avg_top5_evidence_positive_rate": round(sum(top5_evidence) / max(n, 1), 6),
        "avg_top5_ontology_keep_rate": round(sum(top5_ontology) / max(n, 1), 6),
        "avg_top5_contra_rate": round(sum(top5_contra) / max(n, 1), 6),
    }
    return summary, row_orders, gold_ranks

def compare_vs_backbone(target_orders, target_gold_ranks, backbone_orders, backbone_gold_ranks):
    keys = list(target_orders.keys())
    top1_changed = 0
    improved = 0
    worsened = 0

    for i, key in enumerate(keys):
        if target_orders[key][0] != backbone_orders[key][0]:
            top1_changed += 1
        if target_gold_ranks[i] < backbone_gold_ranks[i]:
            improved += 1
        elif target_gold_ranks[i] > backbone_gold_ranks[i]:
            worsened += 1

    n = len(keys)
    return {
        "num_queries_top1_changed_vs_backbone_raw": top1_changed,
        "top1_changed_rate_vs_backbone_raw": round(top1_changed / max(n, 1), 6),
        "improved_rate_vs_backbone_raw": round(improved / max(n, 1), 6),
        "worsened_rate_vs_backbone_raw": round(worsened / max(n, 1), 6),
    }


def pairwise_order_match(orders_a, orders_b):
    keys = list(orders_a.keys())
    exact_match = 0
    top5_match = 0
    top1_match = 0

    for key in keys:
        a = orders_a[key]
        b = orders_b[key]
        if a == b:
            exact_match += 1
        if a[:5] == b[:5]:
            top5_match += 1
        if a[0] == b[0]:
            top1_match += 1

    n = len(keys)
    return {
        "exact_order_match_rate": round(exact_match / max(n, 1), 6),
        "top5_order_match_rate": round(top5_match / max(n, 1), 6),
        "top1_match_rate": round(top1_match / max(n, 1), 6),
    }


def main():
    feature_rows = load_json(FEATURE_PATH)
    feature_map = build_feature_maps(feature_rows)

    row_summaries = {}
    row_orders = {}
    row_gold_ranks = {}

    for name, path in ROW_PATHS.items():
        rows = load_json(path)
        summary, orders, gold_ranks = summarize_row(name, rows, feature_map, missing_rank=21)
        row_summaries[name] = summary
        row_orders[name] = orders
        row_gold_ranks[name] = gold_ranks

    backbone_orders = row_orders["backbone_raw"]
    backbone_gold_ranks = row_gold_ranks["backbone_raw"]

    for name in ["ontology_raw", "soft_support_raw_b025", "soft_support_raw_b050", "soft_support_raw_bcap"]:
        delta = compare_vs_backbone(
            row_orders[name],
            row_gold_ranks[name],
            backbone_orders,
            backbone_gold_ranks
        )
        row_summaries[name].update(delta)

    row_summaries["backbone_raw"].update({
        "num_queries_top1_changed_vs_backbone_raw": 0,
        "top1_changed_rate_vs_backbone_raw": 0.0,
        "improved_rate_vs_backbone_raw": 0.0,
        "worsened_rate_vs_backbone_raw": 0.0,
    })

    pairwise = {
        "b025_vs_b050": pairwise_order_match(
            row_orders["soft_support_raw_b025"],
            row_orders["soft_support_raw_b050"]
        ),
        "b025_vs_bcap": pairwise_order_match(
            row_orders["soft_support_raw_b025"],
            row_orders["soft_support_raw_bcap"]
        ),
        "b050_vs_bcap": pairwise_order_match(
            row_orders["soft_support_raw_b050"],
            row_orders["soft_support_raw_bcap"]
        ),
    }

    out = {
        "stage": "soft_support_valid_compare",
        "row_summaries": row_summaries,
        "pairwise_order_match": pairwise,
        "notes": [
            "This comparison is candidate-stage only.",
            "No test decision is allowed here.",
            "Pairwise order match is used to see whether b025 and b050 are effectively the same ordering family."
        ]
    }

    with OUT_COMPARE.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    with OUT_PAIRWISE.open("w", encoding="utf-8") as f:
        json.dump(pairwise, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
