import json
import re
from pathlib import Path
from statistics import mean


READY_ROOT = Path("dataset/setting_c_pharmkg/e2e_infer_ready")
RESULT_ROOT = Path("outputs/pharmkg/e2e/model_compare")
OUT_DIR = RESULT_ROOT / "reviewer_safe"
REPORT_DIR = Path("outputs/pharmkg/reports/model_compare")

ROWS = ["backbone_raw", "soft_support_raw", "fuzzy_retrieval_main"]
SPLITS = ["valid", "test"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_text(x):
    x = "" if x is None else str(x)
    if "Answer:" in x:
        x = x.split("Answer:")[-1]
    x = x.strip()
    x = x.splitlines()[0].strip() if x.splitlines() else x
    x = x.strip().strip("'").strip('"').strip("`")
    x = re.sub(r"\s+", " ", x)
    x = x.strip(" .,:;，。")
    return x.lower()


def candidate_ceiling(rows):
    rr = []
    h1 = []
    h3 = []
    h10 = []
    for r in rows:
        rank = int(r["rank"])
        rr.append(1.0 / rank if rank <= 20 else 0.0)
        h1.append(1.0 if rank <= 1 else 0.0)
        h3.append(1.0 if rank <= 3 else 0.0)
        h10.append(1.0 if rank <= 10 else 0.0)
    return {
        "gold_at20": mean([1.0 if int(r["rank"]) <= 20 else 0.0 for r in rows]),
        "candidate_mrr_at20": mean(rr),
        "candidate_hits1_at20": mean(h1),
        "candidate_hits3_at20": mean(h3),
        "candidate_hits10_at20": mean(h10),
        "rank21_count": sum(1 for r in rows if int(r["rank"]) > 20),
    }


def e2e_metrics(pred_rows):
    rr = []
    h1 = []
    h3 = []
    h10 = []
    pred_in_cand = []
    invalid = []
    exact = []

    for r in pred_rows:
        target = norm_text(r.get("target", r.get("output", "")))
        pred = norm_text(r.get("pred", ""))

        topk = r["rank_entities"]
        norm_to_idx = {}
        for i, name in enumerate(topk):
            n = norm_text(name)
            if n not in norm_to_idx:
                norm_to_idx[n] = i

        pred_idx = norm_to_idx.get(pred)
        gold_rank = int(r.get("rank", r.get("gold_rank_in_top20_or_21", 21)))

        is_pred_in = pred_idx is not None
        is_exact = pred == target

        if gold_rank > 20:
            safe_rank = 21
        else:
            if is_exact:
                safe_rank = 1
            else:
                safe_rank = gold_rank
                if (not is_pred_in) or (pred_idx >= gold_rank):
                    safe_rank += 1

        this_rr = 1.0 / safe_rank if safe_rank <= 20 else 0.0

        rr.append(this_rr)
        h1.append(1.0 if safe_rank <= 1 else 0.0)
        h3.append(1.0 if safe_rank <= 3 else 0.0)
        h10.append(1.0 if safe_rank <= 10 else 0.0)
        pred_in_cand.append(1.0 if is_pred_in else 0.0)
        invalid.append(0.0 if is_pred_in else 1.0)
        exact.append(1.0 if is_exact else 0.0)

    return {
        "reviewer_safe_mrr_at20": mean(rr),
        "reviewer_safe_hits1_at20": mean(h1),
        "reviewer_safe_hits3_at20": mean(h3),
        "reviewer_safe_hits10_at20": mean(h10),
        "pred_in_candidate_rate": mean(pred_in_cand),
        "invalid_prediction_rate": mean(invalid),
        "exact_target_match_rate": mean(exact),
    }


def md_table(rows):
    headers = [
        "Model", "Split", "Row", "Gold@20", "Cand MRR",
        "E2E MRR", "H@1", "H@3", "H@10",
        "Pred-in-cand", "Invalid", "Rank21"
    ]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append(
            "| "
            + " | ".join([
                r["model_tag"],
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
            ])
            + " |"
        )
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    model_dirs = []
    for p in RESULT_ROOT.iterdir():
        if p.is_dir() and (p / "main_checkpoint" / "checkpoint-final").exists():
            model_dirs.append(p)

    if not model_dirs:
        raise RuntimeError(f"No model dirs found under {RESULT_ROOT}")

    for model_dir in sorted(model_dirs):
        model_tag = model_dir.name
        pred_root = model_dir / "main_checkpoint"

        for split in SPLITS:
            for row_name in ROWS:
                ready_rows = load_json(READY_ROOT / row_name / f"{split}.json")
                pred_path = pred_root / f"prediction_{split}_{row_name}.json"

                if not pred_path.exists():
                    print(f"[SKIP] missing {pred_path}")
                    continue

                pred_obj = load_json(pred_path)
                pred_rows = pred_obj["prediction"]

                row = {
                    "model_tag": model_tag,
                    "split": split,
                    "row_name": row_name,
                    **candidate_ceiling(ready_rows),
                    **e2e_metrics(pred_rows),
                    "prediction_path": str(pred_path),
                }
                summary.append(row)

    save_json(summary, OUT_DIR / "model_compare_reviewer_safe_summary.json")

    report = []
    report.append("# Week 23 Day 3 — PharmKG E2E Model Comparison")
    report.append("")
    report.append("## Summary")
    report.append("")
    report.append(md_table(summary))
    report.append("")
    report.append("## Paper-safe interpretation")
    report.append("")
    report.append("- Compare models mainly by test split and fuzzy_retrieval_main row.")
    report.append("- Candidate MRR is fixed per row and does not depend on LLM.")
    report.append("- E2E MRR reflects whether the LLM follows candidate constraints and graph embeddings.")
    report.append("- High invalid prediction rate should be discussed as a generation-side limitation.")

    report_path = REPORT_DIR / "day3_model_compare_reviewer_safe.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print(md_table(summary))
    print(f"\nSaved: {OUT_DIR / 'model_compare_reviewer_safe_summary.json'}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()