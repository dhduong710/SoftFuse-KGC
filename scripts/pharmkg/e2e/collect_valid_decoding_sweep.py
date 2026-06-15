from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


ROOT = Path(".").resolve()

CONFIG_JSON = ROOT / "outputs/pharmkg/e2e/decoding_sweep_valid/configs.json"
PRED_ROOT = ROOT / "outputs/pharmkg/e2e/decoding_sweep_valid/decoding_sweep_predictions"

OUT_SUMMARY = ROOT / "outputs/pharmkg/e2e/decoding_sweep_valid/pharmkg_decoding_sweep_summary.json"
OUT_BEST = ROOT / "outputs/pharmkg/e2e/decoding_sweep_valid/pharmkg_decoding_sweep_best_config.json"
OUT_MD = ROOT / "outputs/pharmkg/e2e/reports/day6b_pharmkg_valid_decoding_sweep.md"

ROWS = ["backbone_raw", "soft_support_raw", "fuzzy_retrieval_main"]
K = 20
TOP1_COLLAPSE_THRESHOLD = 0.90


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def norm_strict(x: Any) -> str:
    return str(x).strip()


def norm_loose(x: Any) -> str:
    x = "" if x is None else str(x)
    if "Answer:" in x:
        x = x.split("Answer:")[-1]
    x = x.strip()
    x = x.splitlines()[0].strip() if x.splitlines() else x
    x = x.strip().strip("'").strip('"').strip("`")
    x = re.sub(r"\s+", " ", x)
    return x.strip(" .,:;，。").lower()


def get_target(row: Dict[str, Any]) -> str:
    if "target" in row:
        return norm_strict(row["target"])
    if "output" in row:
        return norm_strict(row["output"])
    raise KeyError("Prediction row has neither target nor output")


def get_candidates(row: Dict[str, Any]) -> List[str]:
    return [norm_strict(x) for x in row["rank_entities"][:K]]


def compute_gold_rank(row: Dict[str, Any]) -> Tuple[int, bool]:
    target = get_target(row)
    cands = get_candidates(row)

    if target in cands:
        return cands.index(target) + 1, True

    return K + 1, False


def compute_adjusted_rank(row: Dict[str, Any]) -> Tuple[int, bool]:
    target = get_target(row)
    pred = norm_strict(row.get("pred", ""))
    cands = get_candidates(row)

    gold_rank, gold_present = compute_gold_rank(row)

    if not gold_present:
        return K + 1, False

    if pred == target:
        return 1, True

    adjusted = gold_rank

    if pred not in set(cands):
        adjusted += 1
    else:
        pred_pos = cands.index(pred) + 1
        if pred_pos >= gold_rank:
            adjusted += 1

    return min(adjusted, K + 1), True


def is_candidate_list_fragment(pred: str, candidates: List[str]) -> bool:
    p = str(pred)

    if len(p) > 120:
        return True
    if p.count(",") >= 2:
        return True
    if p.count("'") >= 4:
        return True
    if "####" in p:
        return True

    loose = norm_loose(p)
    hits = 0
    for c in candidates:
        cn = norm_loose(c)
        if cn and cn in loose:
            hits += 1

    return hits >= 3


def summarize_prediction_file(path: Path) -> Dict[str, Any]:
    obj = load_json(path)
    rows = obj["prediction"]
    n = len(rows)

    if n != 500:
        raise RuntimeError(f"{path} has {n} rows, expected 500")

    gold_ranks = []
    adjusted_ranks = []
    gold_present_flags = []

    exact = 0
    pred_in_candidate = 0
    invalid = 0
    top1_copy = 0
    list_frag = 0
    empty_pred = 0

    for row in rows:
        pred = norm_strict(row.get("pred", ""))
        target = get_target(row)
        cands = get_candidates(row)

        gr, present = compute_gold_rank(row)
        ar, _ = compute_adjusted_rank(row)

        gold_ranks.append(gr)
        adjusted_ranks.append(ar)
        gold_present_flags.append(present)

        if pred == target:
            exact += 1
        if pred in set(cands):
            pred_in_candidate += 1
        else:
            invalid += 1
        if cands and pred == cands[0]:
            top1_copy += 1
        if is_candidate_list_fragment(pred, cands):
            list_frag += 1
        if pred == "":
            empty_pred += 1

    cand_rr = [(1.0 / r) if r <= K else 0.0 for r in gold_ranks]
    e2e_rr = [(1.0 / r) if r <= K else 0.0 for r in adjusted_ranks]

    return {
        "num_examples": n,

        "gold_at20": round(sum(gold_present_flags) / n, 8),
        "candidate_mrr_at20": round(sum(cand_rr) / n, 8),
        "candidate_hits1_at20": round(sum(1 for r in gold_ranks if r <= 1) / n, 8),
        "candidate_hits3_at20": round(sum(1 for r in gold_ranks if r <= 3) / n, 8),
        "candidate_hits10_at20": round(sum(1 for r in gold_ranks if r <= 10) / n, 8),
        "candidate_rank21_count": int(sum(1 for r in gold_ranks if r == K + 1)),

        "reviewer_safe_mrr_at20": round(sum(e2e_rr) / n, 8),
        "reviewer_safe_hits1_at20": round(sum(1 for r in adjusted_ranks if r <= 1) / n, 8),
        "reviewer_safe_hits3_at20": round(sum(1 for r in adjusted_ranks if r <= 3) / n, 8),
        "reviewer_safe_hits10_at20": round(sum(1 for r in adjusted_ranks if r <= 10) / n, 8),
        "e2e_rank21_count": int(sum(1 for r in adjusted_ranks if r == K + 1)),

        "exact_target_match_rate": round(exact / n, 8),
        "pred_in_candidate_rate": round(pred_in_candidate / n, 8),
        "invalid_prediction_rate": round(invalid / n, 8),
        "top1_copy_rate": round(top1_copy / n, 8),
        "candidate_list_fragment_rate": round(list_frag / n, 8),
        "empty_prediction_rate": round(empty_pred / n, 8),

        "rr_rule": "1/adjusted_rank if adjusted_rank <= 20 else 0",
        "absent_gold_policy": "RR@20 = 0",
    }


def md_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Config", "Row", "Gold@20", "Cand MRR", "E2E MRR",
        "H@1", "H@3", "H@10", "Pred-in-cand", "Invalid",
        "Top1-copy", "List-frag"
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        lines.append(
            "| "
            + " | ".join([
                r["config_id"],
                r["row_name"],
                f"{r['gold_at20']:.3f}",
                f"{r['candidate_mrr_at20']:.6f}",
                f"{r['reviewer_safe_mrr_at20']:.6f}",
                f"{r['reviewer_safe_hits1_at20']:.3f}",
                f"{r['reviewer_safe_hits3_at20']:.3f}",
                f"{r['reviewer_safe_hits10_at20']:.3f}",
                f"{r['pred_in_candidate_rate']:.3f}",
                f"{r['invalid_prediction_rate']:.3f}",
                f"{r['top1_copy_rate']:.3f}",
                f"{r['candidate_list_fragment_rate']:.3f}",
            ])
            + " |"
        )

    return "\n".join(lines)


def main() -> None:
    config_obj = load_json(CONFIG_JSON)
    configs = config_obj["configs"]

    summary_rows = []

    for cfg in configs:
        cfg_id = cfg["config_id"]

        for row_name in ROWS:
            pred_path = PRED_ROOT / cfg_id / f"prediction_valid_{row_name}.json"
            metrics = summarize_prediction_file(pred_path)

            summary_rows.append({
                "config_id": cfg_id,
                "row_name": row_name,
                "split": "valid",
                **cfg,
                **metrics,
                "prediction_path": str(pred_path),
            })

    config_rollup = []

    for cfg in configs:
        cfg_id = cfg["config_id"]
        cfg_rows = [r for r in summary_rows if r["config_id"] == cfg_id]
        main = [r for r in cfg_rows if r["row_name"] == "fuzzy_retrieval_main"][0]

        avg_mrr = mean([r["reviewer_safe_mrr_at20"] for r in cfg_rows])
        avg_invalid = mean([r["invalid_prediction_rate"] for r in cfg_rows])
        avg_pred_in = mean([r["pred_in_candidate_rate"] for r in cfg_rows])
        avg_top1 = mean([r["top1_copy_rate"] for r in cfg_rows])
        avg_list_frag = mean([r["candidate_list_fragment_rate"] for r in cfg_rows])

        top1_collapsed = main["top1_copy_rate"] >= TOP1_COLLAPSE_THRESHOLD

        config_rollup.append({
            "config_id": cfg_id,
            **cfg,
            "fuzzy_valid_mrr": main["reviewer_safe_mrr_at20"],
            "fuzzy_invalid": main["invalid_prediction_rate"],
            "fuzzy_pred_in_candidate": main["pred_in_candidate_rate"],
            "fuzzy_top1_copy": main["top1_copy_rate"],
            "fuzzy_list_fragment": main["candidate_list_fragment_rate"],
            "avg_valid_mrr_all_rows": round(avg_mrr, 8),
            "avg_invalid_all_rows": round(avg_invalid, 8),
            "avg_pred_in_candidate_all_rows": round(avg_pred_in, 8),
            "avg_top1_copy_all_rows": round(avg_top1, 8),
            "avg_list_fragment_all_rows": round(avg_list_frag, 8),
            "top1_collapsed": top1_collapsed,
        })

    eligible = [r for r in config_rollup if not r["top1_collapsed"]]
    if not eligible:
        eligible = config_rollup

    ranked = sorted(
        eligible,
        key=lambda x: (
            x["fuzzy_valid_mrr"],
            -x["fuzzy_invalid"],
            x["fuzzy_pred_in_candidate"],
            -x["fuzzy_list_fragment"],
            -x["fuzzy_top1_copy"],
            x["avg_valid_mrr_all_rows"],
        ),
        reverse=True,
    )

    best = ranked[0]

    all_ranked = sorted(
        config_rollup,
        key=lambda x: (
            x["fuzzy_valid_mrr"],
            -x["fuzzy_invalid"],
            x["fuzzy_pred_in_candidate"],
            -x["fuzzy_list_fragment"],
            -x["fuzzy_top1_copy"],
            x["avg_valid_mrr_all_rows"],
        ),
        reverse=True,
    )

    decision = "PHARMKG_DECODING_CONFIG_FROZEN_FROM_VALID"

    payload = {
        "decision": "PHARMKG_VALID_DECODING_SWEEP_COMPLETE",
        "dataset": "PharmKG therapeutic_association_proxy task",
        "selection_split": "valid",
        "selection_row": "fuzzy_retrieval_main",
        "graph_num_rels": 28,
        "top1_copy_collapse_threshold": TOP1_COLLAPSE_THRESHOLD,
        "summary_rows": summary_rows,
        "config_rollup": config_rollup,
        "eligible_config_rollup": eligible,
        "ranked_eligible_configs": ranked,
        "ranked_all_configs": all_ranked,
        "best_config": best,
        "selection_rule": config_obj["selection_rule"],
    }

    save_json(OUT_SUMMARY, payload)

    save_json(OUT_BEST, {
        "decision": decision,
        "best_config": best,
        "use_for_day6c": {
            "max_new_tokens": best["max_new_tokens"],
            "repetition_penalty": best["repetition_penalty"],
            "no_repeat_ngram_size": best["no_repeat_ngram_size"],
            "do_sample": False,
            "num_beams": 1,
            "temperature": 1.0,
        },
        "important_note": "Selected on PharmKG valid only. Do not change after looking at test.",
    })

    lines = []
    lines.append("# PharmKG Valid Decoding Sweep")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"**{decision}**")
    lines.append("")
    lines.append("## Best config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(best, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Ranked configs")
    lines.append("")
    lines.append("| Rank | Config | max_new | rep penalty | ngram | fuzzy MRR | invalid | pred-in-cand | top1-copy | list-frag | avg MRR | top1 collapsed |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for i, r in enumerate(all_ranked, start=1):
        lines.append(
            f"| {i} | {r['config_id']} | {r['max_new_tokens']} | {r['repetition_penalty']} | "
            f"{r['no_repeat_ngram_size']} | {r['fuzzy_valid_mrr']:.6f} | "
            f"{r['fuzzy_invalid']:.3f} | {r['fuzzy_pred_in_candidate']:.3f} | "
            f"{r['fuzzy_top1_copy']:.3f} | {r['fuzzy_list_fragment']:.3f} | "
            f"{r['avg_valid_mrr_all_rows']:.6f} | {r['top1_collapsed']} |"
        )

    lines.append("")
    lines.append("## Full row-level table")
    lines.append("")
    lines.append(md_table(summary_rows))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This sweep uses PharmKG valid only.")
    lines.append("- Test split is not used for selection.")
    lines.append("- Raw `infer.py` metrics are ignored for reviewer-safe selection.")
    lines.append("- `top1_copy_rate >= 0.90` is treated as top-1-copy collapse diagnostic.")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("decision =", decision)
    print(f"summary = {OUT_SUMMARY}")
    print(f"best_config = {OUT_BEST}")
    print(f"report = {OUT_MD}")
    print("")
    print("Best config:")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("")
    print("Top configs:")
    for i, r in enumerate(all_ranked[:8], start=1):
        print(
            i,
            r["config_id"],
            "fuzzy_mrr=", r["fuzzy_valid_mrr"],
            "invalid=", r["fuzzy_invalid"],
            "pred_in=", r["fuzzy_pred_in_candidate"],
            "top1=", r["fuzzy_top1_copy"],
            "list_frag=", r["fuzzy_list_fragment"],
            "avg_mrr=", r["avg_valid_mrr_all_rows"],
            "collapsed=", r["top1_collapsed"],
        )


if __name__ == "__main__":
    main()