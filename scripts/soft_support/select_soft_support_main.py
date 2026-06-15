import json
import shutil
from pathlib import Path

SRC_MAIN = Path("dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_b050.json")
SRC_CONTRAST = Path("dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_bcap.json")
SRC_COMPARE = Path("outputs/soft_support/soft_support_valid_compare.json")
SRC_CASES = Path("outputs/soft_support/soft_support_case_summary.json")

OUT_MAIN = Path("dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_main.json")
OUT_MANIFEST = Path("dataset/setting_a/soft_support_ranked_candidates/support_main_manifest.json")
OUT_DECISION = Path("outputs/soft_support/soft_support_main_decision.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    compare = load_json(SRC_COMPARE)
    case_summary = load_json(SRC_CASES)

    # Export chosen main row
    shutil.copyfile(SRC_MAIN, OUT_MAIN)

    chosen_row = compare["row_summaries"]["soft_support_raw_b050"]
    contrast_row = compare["row_summaries"]["soft_support_raw_bcap"]
    backbone_row = compare["row_summaries"]["backbone_raw"]
    ontology_row = compare["row_summaries"]["ontology_raw"]

    manifest = {
        "stage": "soft_support_main_selection",
        "selected_variant": "soft_support_raw_b050",
        "selected_output": str(OUT_MAIN),
        "selected_from": str(SRC_MAIN),
        "contrast_variant": "soft_support_raw_bcap",
        "contrast_from": str(SRC_CONTRAST),
        "why_selected": [
            "b050 is the surviving representative of the Formula-B family.",
            "b025 is exactly identical in ordering to b050, so b050 is used as the canonical row.",
            "b050 improves over backbone_raw with zero worsened cases.",
            "b050 clearly outperforms bcap and avoids the evidence-overreward pattern."
        ],
        "selected_metrics": chosen_row,
        "backbone_reference_metrics": backbone_row,
        "ontology_negative_control_metrics": ontology_row,
        "contrast_metrics": contrast_row,
        "case_summary": case_summary,
        "scientific_role": [
            "backbone_raw = reference row",
            "ontology_raw = negative control",
            "soft_support_raw = main intermediate row"
        ]
    }

    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    decision = {
        "stage": "soft_support_main_selection",
        "decision": "SELECT_SOFT_SUPPORT_MAIN",
        "selected_variant": "soft_support_raw_b050",
        "selected_output": str(OUT_MAIN),
        "reason": {
            "exact_match_with_b025": True,
            "better_than_backbone_raw": {
                "mrr_like_delta": round(chosen_row["mrr_like"] - backbone_row["mrr_like"], 6),
                "hits1_like_delta": round(chosen_row["hits1_like"] - backbone_row["hits1_like"], 6),
                "avg_gold_rank_delta": round(chosen_row["avg_gold_rank"] - backbone_row["avg_gold_rank"], 6),
                "worsened_rate_vs_backbone_raw": chosen_row["worsened_rate_vs_backbone_raw"]
            },
            "better_than_bcap": {
                "mrr_like_delta": round(chosen_row["mrr_like"] - contrast_row["mrr_like"], 6),
                "avg_gold_rank_delta": round(chosen_row["avg_gold_rank"] - contrast_row["avg_gold_rank"], 6),
                "top5_direct_link_rate_delta": round(
                    chosen_row["avg_top5_direct_link_rate"] - contrast_row["avg_top5_direct_link_rate"], 6
                )
            },
            "ontology_raw_is_negative_control": {
                "ontology_mrr_like": ontology_row["mrr_like"],
                "ontology_gold_present_rate": ontology_row["gold_present_rate"],
                "ontology_worsened_rate_vs_backbone_raw": ontology_row["worsened_rate_vs_backbone_raw"]
            }
        },
        "next_stage_policy": [
            "use soft_support_raw as the main intermediate row",
            "do not revisit b025",
            "keep bcap only as a contrast row",
            "prepare transition toward retrieval-stage work without changing the fixed raw source"
        ]
    }

    with OUT_DECISION.open("w", encoding="utf-8") as f:
        json.dump(decision, f, ensure_ascii=False, indent=2)

    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
