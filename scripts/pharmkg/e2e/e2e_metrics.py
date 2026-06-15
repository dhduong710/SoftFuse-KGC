import json
import re
from pathlib import Path
from statistics import mean


ROOT = Path("dataset/setting_c_pharmkg/e2e_infer_ready")
PRED_ROOT = Path("outputs/pharmkg/e2e/main/main_checkpoint")
OUT_ROOT = Path("outputs/pharmkg/e2e/main/reviewer_safe")
REPORT_DIR = Path("outputs/pharmkg/reports")

ROWS = ["backbone_raw", "soft_support_raw", "fuzzy_retrieval_main"]
SPLITS = ["valid", "test"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_text(x):
    if x is None:
        return ""
    x = str(x)
    if "Answer:" in x:
        x = x.split("Answer:")[-1]
    x = x.strip()
    x = x.splitlines()[0].strip() if x.splitlines() else x
    x = x.strip().strip("'").strip('"').strip("`")
    x = re.sub(r"\s+", " ", x)
    x = x.strip(" .,:;，。")
    return x.lower()


def get_prediction_payload(row_name, split):
    path = PRED_ROOT / f"prediction_{split}_{row_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    obj = load_json(path)
    preds = obj.get("prediction", None)
    if preds is None:
        raise KeyError(f"Missing key 'prediction' in {path}")
    return path, preds


def compute_candidate_ceiling(ready_rows):
    rrs = []
    hits1 = []
    hits3 = []
    hits10 = []
    rank21 = 0

    for r in ready_rows:
        rank = int(r["rank"])
        if rank <= 20:
            rr = 1.0 / rank
        else:
            rr = 0.0
            rank21 += 1

        rrs.append(rr)
        hits1.append(1.0 if rank <= 1 else 0.0)
        hits3.append(1.0 if rank <= 3 else 0.0)
        hits10.append(1.0 if rank <= 10 else 0.0)

    return {
        "gold_at20": round(mean([1.0 if int(r["rank"]) <= 20 else 0.0 for r in ready_rows]), 12),
        "candidate_mrr_at20": round(mean(rrs), 12),
        "candidate_hits1_at20": round(mean(hits1), 12),
        "candidate_hits3_at20": round(mean(hits3), 12),
        "candidate_hits10_at20": round(mean(hits10), 12),
        "rank21_count": int(rank21),
    }


def compute_e2e_metrics(pred_rows):
    out_rows = []

    rr_items = []
    h1_items = []
    h3_items = []
    h10_items = []

    pred_in_candidate = []
    exact_target = []
    invalid_pred = []
    gold_present_items = []

    for i, r in enumerate(pred_rows):
        target = r.get("target", r.get("output", ""))
        pred_raw = r.get("pred", "")
        topk = r["rank_entities"]

        target_norm = norm_text(target)
        pred_norm = norm_text(pred_raw)

        norm_to_first_idx = {}
        for idx, name in enumerate(topk):
            n = norm_text(name)
            if n not in norm_to_first_idx:
                norm_to_first_idx[n] = idx

        pred_candidate_idx0 = norm_to_first_idx.get(pred_norm, None)
        pred_candidate_name = None
        if pred_candidate_idx0 is not None:
            pred_candidate_name = topk[pred_candidate_idx0]

        gold_rank = int(r.get("rank", r.get("gold_rank_in_top20_or_21", 21)))
        gold_present = gold_rank <= 20

        is_exact_target = pred_norm == target_norm
        is_pred_in_candidate = pred_candidate_idx0 is not None

        if not gold_present:
            reviewer_safe_pred_rank = 21
            rr = 0.0
        else:
            if is_exact_target:
                reviewer_safe_pred_rank = 1
            else:
                reviewer_safe_pred_rank = gold_rank

                # Mimic DrKGC-style generated-answer reranking logic,
                # but force RR=0 if the resulting rank goes beyond top-20.
                if (not is_pred_in_candidate) or (pred_candidate_idx0 >= gold_rank):
                    reviewer_safe_pred_rank += 1

            rr = 1.0 / reviewer_safe_pred_rank if reviewer_safe_pred_rank <= 20 else 0.0

        item = dict(r)
        item["pred_raw"] = pred_raw
        item["pred_normalized"] = pred_norm
        item["target_normalized"] = target_norm
        item["pred_candidate_name"] = pred_candidate_name
        item["pred_in_candidate"] = bool(is_pred_in_candidate)
        item["exact_target_match"] = bool(is_exact_target)
        item["gold_present_reviewer_safe"] = bool(gold_present)
        item["reviewer_safe_pred_rank"] = int(reviewer_safe_pred_rank)
        item["reviewer_safe_rr"] = float(rr)
        item["reviewer_safe_hits1"] = bool(reviewer_safe_pred_rank <= 1)
        item["reviewer_safe_hits3"] = bool(reviewer_safe_pred_rank <= 3)
        item["reviewer_safe_hits10"] = bool(reviewer_safe_pred_rank <= 10)

        out_rows.append(item)

        rr_items.append(rr)
        h1_items.append(1.0 if reviewer_safe_pred_rank <= 1 else 0.0)
        h3_items.append(1.0 if reviewer_safe_pred_rank <= 3 else 0.0)
        h10_items.append(1.0 if reviewer_safe_pred_rank <= 10 else 0.0)

        pred_in_candidate.append(1.0 if is_pred_in_candidate else 0.0)
        exact_target.append(1.0 if is_exact_target else 0.0)
        invalid_pred.append(0.0 if is_pred_in_candidate else 1.0)
        gold_present_items.append(1.0 if gold_present else 0.0)

    metrics = {
        "num_examples": len(pred_rows),
        "reviewer_safe_mrr_at20": round(mean(rr_items), 12),
        "reviewer_safe_hits1_at20": round(mean(h1_items), 12),
        "reviewer_safe_hits3_at20": round(mean(h3_items), 12),
        "reviewer_safe_hits10_at20": round(mean(h10_items), 12),
        "gold_present_rate": round(mean(gold_present_items), 12),
        "pred_in_candidate_rate": round(mean(pred_in_candidate), 12),
        "exact_target_match_rate": round(mean(exact_target), 12),
        "invalid_prediction_rate": round(mean(invalid_pred), 12),
        "rank21_count": int(sum(1 for r in pred_rows if int(r.get("rank", r.get("gold_rank_in_top20_or_21", 21))) > 20)),
    }

    return out_rows, metrics


def markdown_table(rows):
    headers = [
        "Split",
        "Row",
        "Gold@20",
        "Candidate MRR@20",
        "E2E MRR@20",
        "E2E H@1",
        "E2E H@3",
        "E2E H@10",
        "Pred-in-cand",
        "Invalid pred",
        "Rank21",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["split"],
                    r["row_name"],
                    f"{r['gold_at20']:.3f}",
                    f"{r['candidate_mrr_at20']:.6f}",
                    f"{r['reviewer_safe_mrr_at20']:.6f}",
                    f"{r['reviewer_safe_hits1_at20']:.3f}",
                    f"{r['reviewer_safe_hits3_at20']:.3f}",
                    f"{r['reviewer_safe_hits10_at20']:.3f}",
                    f"{r['pred_in_candidate_rate']:.3f}",
                    f"{r['invalid_prediction_rate']:.3f}",
                    str(r["rank21_count"]),
                ]
            )
            + " |"
        )

    return "\n".join(lines)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for split in SPLITS:
        for row_name in ROWS:
            ready_path = ROOT / row_name / f"{split}.json"
            ready_rows = load_json(ready_path)
            ceiling = compute_candidate_ceiling(ready_rows)

            pred_path, pred_rows = get_prediction_payload(row_name, split)
            safe_rows, e2e = compute_e2e_metrics(pred_rows)

            row_summary = {
                "split": split,
                "row_name": row_name,
                "ready_path": str(ready_path),
                "prediction_path": str(pred_path),
                **ceiling,
                **e2e,
            }
            summary_rows.append(row_summary)

            save_json(
                {
                    "row_summary": row_summary,
                    "predictions": safe_rows,
                },
                OUT_ROOT / f"reviewer_safe_predictions_{split}_{row_name}.json",
            )
            save_json(
                row_summary,
                OUT_ROOT / f"reviewer_safe_metrics_{split}_{row_name}.json",
            )

    save_json(summary_rows, OUT_ROOT / "e2e_pharmkg_reviewer_safe_summary.json")

    report_lines = []
    report_lines.append("# Week 23 Day 2 — PharmKG E2E Reviewer-Safe Metrics")
    report_lines.append("")
    report_lines.append("## Protocol")
    report_lines.append("")
    report_lines.append("- Dataset: PharmKG therapeutic-association proxy task.")
    report_lines.append("- Missing entity: drug/chemical head.")
    report_lines.append("- Relation: `T`, normalized as `therapeutic_association_proxy`.")
    report_lines.append("- Candidate size: top-20.")
    report_lines.append("- No valid/test gold injection.")
    report_lines.append("- Reviewer-safe RR: `1/rank` if rank <= 20, otherwise `0`.")
    report_lines.append("")
    report_lines.append("## Summary table")
    report_lines.append("")
    report_lines.append(markdown_table(summary_rows))
    report_lines.append("")
    report_lines.append("## Notes")
    report_lines.append("")
    report_lines.append("- Candidate MRR@20 is computed from the fixed candidate order before generation.")
    report_lines.append("- E2E MRR@20 is computed from generated predictions using reviewer-safe rank handling.")
    report_lines.append("- Rows with rank 21 receive RR = 0.")
    report_lines.append("- `fuzzy_retrieval_main` should preserve soft-support candidate order but use compressed subgraphs.")

    report_path = REPORT_DIR / "day2_e2e_pharmkg_reviewer_safe_metrics.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("[DONE] Reviewer-safe E2E metrics complete.")
    print(f"Summary JSON: {OUT_ROOT / 'e2e_pharmkg_reviewer_safe_summary.json'}")
    print(f"Report MD: {report_path}")
    print("")
    print(markdown_table(summary_rows))


if __name__ == "__main__":
    main()