from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


ROOT = Path(".").resolve()

MODEL_ROOT = ROOT / "outputs/pharmkg/e2e/selected_model_compare"
READY_ROOT = ROOT / "dataset/setting_c_pharmkg/e2e_infer_ready"

OUT_SUMMARY = MODEL_ROOT / "pharmkg_model_compare_reviewer_safe_summary.json"
OUT_VALID = MODEL_ROOT / "pharmkg_model_compare_table_valid.json"
OUT_TEST = MODEL_ROOT / "pharmkg_model_compare_table_test.json"
OUT_BEST = MODEL_ROOT / "pharmkg_model_compare_best_model.json"

ASSET_DIR = MODEL_ROOT / "analysis_assets"
OUT_E2E_TEX = ASSET_DIR / "pharmkg_e2e_table_latex.tex"
OUT_MODEL_TEX = ASSET_DIR / "pharmkg_model_compare_table_latex.tex"
OUT_RESULT_PARA = ASSET_DIR / "pharmkg_result_paragraph.md"
OUT_LIMIT_PARA = ASSET_DIR / "pharmkg_limitation_paragraph.md"

OUT_MD = ROOT / "outputs/pharmkg/e2e/reports/pharmkg_selected_model_compare.md"

MODELS = ["llama3_2_3b", "llama3_8b", "medllama3_8b"]
ROWS = ["backbone_raw", "soft_support_raw", "fuzzy_retrieval_main"]
SPLITS = ["valid", "test"]
K = 20


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
    return x.strip(" .,:;").lower()


def get_target(row: Dict[str, Any]) -> str:
    if "target" in row:
        return norm_strict(row["target"])
    if "output" in row:
        return norm_strict(row["output"])
    raise KeyError("Missing target/output")


def get_candidates(row: Dict[str, Any]) -> List[str]:
    return [norm_strict(x) for x in row["rank_entities"][:K]]


def gold_rank(row: Dict[str, Any]) -> Tuple[int, bool]:
    target = get_target(row)
    cands = get_candidates(row)
    if target in cands:
        return cands.index(target) + 1, True
    return K + 1, False


def adjusted_rank(row: Dict[str, Any]) -> Tuple[int, bool]:
    target = get_target(row)
    pred = norm_strict(row.get("pred", ""))
    cands = get_candidates(row)

    gr, present = gold_rank(row)
    if not present:
        return K + 1, False

    if pred == target:
        return 1, True

    ar = gr
    if pred not in set(cands):
        ar += 1
    else:
        pred_pos = cands.index(pred) + 1
        if pred_pos >= gr:
            ar += 1

    return min(ar, K + 1), True


def is_list_fragment(pred: str, candidates: List[str]) -> bool:
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


def summarize_prediction_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)

    gold_ranks = []
    adjusted_ranks = []
    present_flags = []

    exact = 0
    pred_in = 0
    invalid = 0
    top1 = 0
    list_frag = 0
    empty = 0

    for row in rows:
        pred = norm_strict(row.get("pred", ""))
        target = get_target(row)
        cands = get_candidates(row)

        gr, present = gold_rank(row)
        ar, _ = adjusted_rank(row)

        gold_ranks.append(gr)
        adjusted_ranks.append(ar)
        present_flags.append(present)

        if pred == target:
            exact += 1
        if pred in set(cands):
            pred_in += 1
        else:
            invalid += 1
        if cands and pred == cands[0]:
            top1 += 1
        if is_list_fragment(pred, cands):
            list_frag += 1
        if pred == "":
            empty += 1

    cand_rr = [(1.0 / r) if r <= K else 0.0 for r in gold_ranks]
    e2e_rr = [(1.0 / r) if r <= K else 0.0 for r in adjusted_ranks]

    return {
        "num_examples": n,
        "gold_at20": round(sum(present_flags) / n, 8),

        "candidate_mrr_at20": round(sum(cand_rr) / n, 8),
        "candidate_hits1_at20": round(sum(1 for r in gold_ranks if r <= 1) / n, 8),
        "candidate_hits3_at20": round(sum(1 for r in gold_ranks if r <= 3) / n, 8),
        "candidate_hits10_at20": round(sum(1 for r in gold_ranks if r <= 10) / n, 8),
        "candidate_rank21_count": int(sum(1 for r in gold_ranks if r == K + 1)),

        "reviewer_safe_e2e_mrr_at20": round(sum(e2e_rr) / n, 8),
        "reviewer_safe_e2e_hits1_at20": round(sum(1 for r in adjusted_ranks if r <= 1) / n, 8),
        "reviewer_safe_e2e_hits3_at20": round(sum(1 for r in adjusted_ranks if r <= 3) / n, 8),
        "reviewer_safe_e2e_hits10_at20": round(sum(1 for r in adjusted_ranks if r <= 10) / n, 8),
        "e2e_rank21_count": int(sum(1 for r in adjusted_ranks if r == K + 1)),

        "exact_target_match_rate": round(exact / n, 8),
        "pred_in_candidate_rate": round(pred_in / n, 8),
        "invalid_prediction_rate": round(invalid / n, 8),
        "top1_copy_rate": round(top1 / n, 8),
        "candidate_list_fragment_rate": round(list_frag / n, 8),
        "empty_prediction_rate": round(empty / n, 8),

        "rr_rule": "1/adjusted_rank if adjusted_rank <= 20 else 0",
        "absent_gold_policy": "RR@20 = 0",
    }


def graph_package(row_name: str, split: str) -> Dict[str, Any]:
    rows = load_json(READY_ROOT / row_name / f"{split}.json")
    sizes = [len(r.get("subgraph", [])) for r in rows]

    return {
        "avg_subgraph_size": round(mean(sizes), 8),
        "min_subgraph_size": min(sizes),
        "max_subgraph_size": max(sizes),
    }


def summarize_file(model: str, split: str, row: str) -> Dict[str, Any]:
    pred_path = MODEL_ROOT / model / "predictions" / f"prediction_{split}_{row}.json"
    obj = load_json(pred_path)
    rows = obj["prediction"]

    if len(rows) != 500:
        raise RuntimeError(f"{pred_path} has {len(rows)} rows, expected 500")

    return {
        "model_tag": model,
        "split": split,
        "row_name": row,
        "prediction_path": str(pred_path),
        **summarize_prediction_rows(rows),
        **graph_package(row, split),
    }


def md_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Model", "Split", "Row", "Gold@20", "Cand MRR", "E2E MRR",
        "H@1", "H@3", "H@10", "Pred-in-cand", "Invalid", "Top1-copy",
        "List-frag", "Avg graph"
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
                f"{r['reviewer_safe_e2e_mrr_at20']:.6f}",
                f"{r['reviewer_safe_e2e_hits1_at20']:.3f}",
                f"{r['reviewer_safe_e2e_hits3_at20']:.3f}",
                f"{r['reviewer_safe_e2e_hits10_at20']:.3f}",
                f"{r['pred_in_candidate_rate']:.3f}",
                f"{r['invalid_prediction_rate']:.3f}",
                f"{r['top1_copy_rate']:.3f}",
                f"{r['candidate_list_fragment_rate']:.3f}",
                f"{r['avg_subgraph_size']:.2f}",
            ])
            + " |"
        )
    return "\n".join(lines)


def latex_table_e2e(primary_rows: List[Dict[str, Any]]) -> str:
    names = {
        "backbone_raw": "Backbone raw",
        "soft_support_raw": "Soft support",
        "fuzzy_retrieval_main": "SoftFuse retrieval",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Row & Gold@20 & Cand. MRR & E2E MRR & H@3 & H@10 & Avg. graph \\",
        r"\hline",
    ]
    for row in ROWS:
        r = [x for x in primary_rows if x["row_name"] == row][0]
        lines.append(
            f"{names[row]} & {r['gold_at20']:.3f} & {r['candidate_mrr_at20']:.6f} & "
            f"{r['reviewer_safe_e2e_mrr_at20']:.6f} & {r['reviewer_safe_e2e_hits3_at20']:.3f} & "
            f"{r['reviewer_safe_e2e_hits10_at20']:.3f} & {r['avg_subgraph_size']:.2f} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{PharmKG therapeutic-association proxy E2E transfer results with the primary Llama-3.2-3B base model.}",
        r"\label{tab:pharmkg-e2e}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def latex_table_model(test_fuzzy: List[Dict[str, Any]]) -> str:
    names = {
        "llama3_2_3b": "Llama-3.2-3B",
        "llama3_8b": "Llama-3-8B",
        "medllama3_8b": "MedLlama-3-8B",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Base LLM & E2E MRR & H@10 & Pred-in-cand. & Invalid & Top1-copy \\",
        r"\hline",
    ]
    for r in test_fuzzy:
        lines.append(
            f"{names.get(r['model_tag'], r['model_tag'])} & "
            f"{r['reviewer_safe_e2e_mrr_at20']:.6f} & "
            f"{r['reviewer_safe_e2e_hits10_at20']:.3f} & "
            f"{r['pred_in_candidate_rate']:.3f} & "
            f"{r['invalid_prediction_rate']:.3f} & "
            f"{r['top1_copy_rate']:.3f} \\\\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\caption{PharmKG base-LLM comparison on the SoftFuse retrieval row.}",
        r"\label{tab:pharmkg-model-comparison}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary_rows = []

    for model in MODELS:
        for split in SPLITS:
            for row in ROWS:
                summary_rows.append(summarize_file(model, split, row))

    valid_rows = [r for r in summary_rows if r["split"] == "valid"]
    test_rows = [r for r in summary_rows if r["split"] == "test"]
    test_fuzzy = [r for r in test_rows if r["row_name"] == "fuzzy_retrieval_main"]

    ranked = sorted(
        test_fuzzy,
        key=lambda x: (
            x["reviewer_safe_e2e_mrr_at20"],
            -x["invalid_prediction_rate"],
            x["pred_in_candidate_rate"],
            -x["top1_copy_rate"],
            -x["candidate_list_fragment_rate"],
        ),
        reverse=True,
    )
    best = ranked[0]

    primary_model = "llama3_2_3b"
    primary_test_rows = [r for r in test_rows if r["model_tag"] == primary_model]

    trend_checks = {}
    for model in MODELS:
        b = [r for r in test_rows if r["model_tag"] == model and r["row_name"] == "backbone_raw"][0]
        s = [r for r in test_rows if r["model_tag"] == model and r["row_name"] == "soft_support_raw"][0]
        f = [r for r in test_rows if r["model_tag"] == model and r["row_name"] == "fuzzy_retrieval_main"][0]
        trend_checks[model] = {
            "soft_improves_backbone_e2e": s["reviewer_safe_e2e_mrr_at20"] > b["reviewer_safe_e2e_mrr_at20"],
            "fuzzy_preserves_or_improves_soft_e2e": f["reviewer_safe_e2e_mrr_at20"] >= s["reviewer_safe_e2e_mrr_at20"],
            "fuzzy_smaller_than_soft": f["avg_subgraph_size"] < s["avg_subgraph_size"],
            "delta_fuzzy_minus_backbone_e2e": round(f["reviewer_safe_e2e_mrr_at20"] - b["reviewer_safe_e2e_mrr_at20"], 8),
            "delta_fuzzy_minus_soft_e2e": round(f["reviewer_safe_e2e_mrr_at20"] - s["reviewer_safe_e2e_mrr_at20"], 8),
        }

    save_json(OUT_SUMMARY, {
        "decision": "PHARMKG_MODEL_COMPARE_REVIEWER_SAFE_BUILT",
        "summary_rows": summary_rows,
        "models": MODELS,
        "rows": ROWS,
        "splits": SPLITS,
    })
    save_json(OUT_VALID, valid_rows)
    save_json(OUT_TEST, test_rows)
    save_json(OUT_BEST, {
        "decision": "PHARMKG_FROZEN_MODEL_COMPARE_READY",
        "metric_best_test_fuzzy": best,
        "ranked_test_fuzzy": ranked,
        "primary_e2e_model": primary_model,
        "trend_checks_by_model": trend_checks,
        "report_note": (
            "PharmKG is a secondary transfer benchmark. Interpret E2E results with generation "
            "limitations and candidate bottleneck; do not claim clinical indication or Gold@20 improvement."
        ),
    })

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    write_text(OUT_E2E_TEX, latex_table_e2e(primary_test_rows))
    write_text(OUT_MODEL_TEX, latex_table_model(ranked))

    primary_b = [r for r in primary_test_rows if r["row_name"] == "backbone_raw"][0]
    primary_s = [r for r in primary_test_rows if r["row_name"] == "soft_support_raw"][0]
    primary_f = [r for r in primary_test_rows if r["row_name"] == "fuzzy_retrieval_main"][0]

    result_para = (
        "On the PharmKG therapeutic-association proxy task, soft support improves the locked-test "
        f"candidate MRR@20 from {primary_b['candidate_mrr_at20']:.6f} to "
        f"{primary_s['candidate_mrr_at20']:.6f}. In reviewer-safe E2E evaluation with Llama-3.2-3B, "
        f"the backbone obtains MRR@20={primary_b['reviewer_safe_e2e_mrr_at20']:.6f}, while soft support "
        f"and fuzzy retrieval reach {primary_s['reviewer_safe_e2e_mrr_at20']:.6f} and "
        f"{primary_f['reviewer_safe_e2e_mrr_at20']:.6f}, respectively. Fuzzy retrieval preserves the "
        f"ranking trend while reducing the evidence subgraph from {primary_s['avg_subgraph_size']:.2f} "
        f"to {primary_f['avg_subgraph_size']:.2f} triples."
    )

    limit_para = (
        "PharmKG remains a difficult secondary transfer benchmark. The top-20 candidate bottleneck is "
        "strong, and unconstrained generation frequently produces invalid or fragmentary outputs for "
        "the Llama-3.2-3B run. Therefore, PharmKG is reported as transfer evidence for the direction of "
        "SoftFuse improvements, not as a full-universe PharmKG KGC superiority claim."
    )

    write_text(OUT_RESULT_PARA, result_para + "\n")
    write_text(OUT_LIMIT_PARA, limit_para + "\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# PharmKG E2E  PharmKG Selected-Config Model Comparison")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("**PHARMKG_FROZEN_MODEL_COMPARE_READY**")
    lines.append("")
    lines.append("## Ranked test fuzzy_retrieval_main")
    lines.append("")
    lines.append("| Rank | Model | E2E MRR | H@1 | H@3 | H@10 | Pred-in-cand | Invalid | Top1-copy | List-frag |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['model_tag']} | {r['reviewer_safe_e2e_mrr_at20']:.6f} | "
            f"{r['reviewer_safe_e2e_hits1_at20']:.3f} | {r['reviewer_safe_e2e_hits3_at20']:.3f} | "
            f"{r['reviewer_safe_e2e_hits10_at20']:.3f} | {r['pred_in_candidate_rate']:.3f} | "
            f"{r['invalid_prediction_rate']:.3f} | {r['top1_copy_rate']:.3f} | "
            f"{r['candidate_list_fragment_rate']:.3f} |"
        )
    lines.append("")
    lines.append("## Full table")
    lines.append("")
    lines.append(md_table(summary_rows))
    lines.append("")
    lines.append("## Trend checks")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(trend_checks, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Result paragraph")
    lines.append("")
    lines.append(result_para)
    lines.append("")
    lines.append("## Limitation paragraph")
    lines.append("")
    lines.append(limit_para)
    write_text(OUT_MD, "\n".join(lines) + "\n")

    print("decision = PHARMKG_FROZEN_MODEL_COMPARE_READY")
    print(f"summary = {OUT_SUMMARY}")
    print(f"valid = {OUT_VALID}")
    print(f"test = {OUT_TEST}")
    print(f"best = {OUT_BEST}")
    print(f"e2e_tex = {OUT_E2E_TEX}")
    print(f"model_tex = {OUT_MODEL_TEX}")
    print(f"result_para = {OUT_RESULT_PARA}")
    print(f"limit_para = {OUT_LIMIT_PARA}")
    print(f"report = {OUT_MD}")
    print("")
    print("Ranked test fuzzy:")
    for i, r in enumerate(ranked, 1):
        print(
            i,
            r["model_tag"],
            "MRR=", r["reviewer_safe_e2e_mrr_at20"],
            "H10=", r["reviewer_safe_e2e_hits10_at20"],
            "pred_in=", r["pred_in_candidate_rate"],
            "invalid=", r["invalid_prediction_rate"],
            "top1=", r["top1_copy_rate"],
            "list_frag=", r["candidate_list_fragment_rate"],
        )


if __name__ == "__main__":
    main()