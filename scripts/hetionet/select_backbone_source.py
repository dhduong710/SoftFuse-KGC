#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(".")
RESULT_DIR = ROOT / "outputs" / "hetionet"
REPORT_DIR = ROOT / "outputs" / "hetionet" / "reports"
BASELINE_DIR = ROOT / "dataset" / "setting_d_hetionet" / "baseline_outputs"

MAIN_TABLE = RESULT_DIR / "hetionet_baseline_main_table.json"
OUT_JSON = RESULT_DIR / "day3_hetionet_source_selection.json"
OUT_MD = REPORT_DIR / "day3_hetionet_source_selection.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_brief(m):
    if m is None:
        return None
    return {
        "model_name": m.get("model_name"),
        "split": m.get("split"),
        "gold_present_at20": m.get("gold_present_at20"),
        "mrr_at20": m.get("mrr_at20"),
        "hits1_at20": m.get("hits1_at20"),
        "hits3_at20": m.get("hits3_at20"),
        "hits10_at20": m.get("hits10_at20"),
        "gold_rank_21_count": m.get("gold_rank_21_count"),
        "top1_dominance": m.get("top1_dominance"),
    }


def main() -> None:
    if not MAIN_TABLE.exists():
        raise FileNotFoundError(
            f"Missing baseline main table: {MAIN_TABLE}. "
            "Run KGE and GNN baseline scripts before source selection."
        )

    table = read_json(MAIN_TABLE)
    valid = table.get("valid", [])
    test = table.get("test", [])

    valid_by_name = {x.get("model_name"): x for x in valid}
    test_by_name = {x.get("model_name"): x for x in test}

    expected_models = ["transe", "distmult", "complex", "rotate", "rgcn", "hrgat"]
    missing_models = [
        m for m in expected_models
        if m not in valid_by_name or m not in test_by_name
    ]

    valid_best = valid[0] if valid else None
    test_best = test[0] if test else None

    rgcn_valid = valid_by_name.get("rgcn")
    rgcn_test = test_by_name.get("rgcn")

    rgcn_dir = BASELINE_DIR / "rgcn"
    rgcn_embedding = rgcn_dir / "entity_embeddings_rgcn.pt"
    rgcn_valid_top20 = rgcn_dir / "valid_top20.json"
    rgcn_test_top20 = rgcn_dir / "test_top20.json"

    rgcn_metrics_ready = rgcn_valid is not None and rgcn_test is not None
    rgcn_files_ready = (
        rgcn_valid_top20.exists()
        and rgcn_test_top20.exists()
        and rgcn_embedding.exists()
    )
    all_models_ready = len(missing_models) == 0

    if all_models_ready and rgcn_metrics_ready and rgcn_files_ready:
        decision = "DAY3_HETIONET_SOURCE_RGCN_READY"
    elif rgcn_metrics_ready and rgcn_valid_top20.exists() and rgcn_test_top20.exists():
        decision = "DAY3_HETIONET_SOURCE_RGCN_PARTIAL_READY_MISSING_EMBEDDING_OR_OTHER_BASELINES"
    else:
        decision = "DAY3_HETIONET_SOURCE_NOT_READY"

    source = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_d_hetionet",
        "dataset": "Hetionet v1.0",
        "task": "(?, CtD, disease)",
        "target_relation": "CtD",
        "target_relation_normalized": "compound_treats_disease",
        "prediction_type": "predicted_head",
        "candidate_universe": "Compound",
        "top_k": 20,
        "gold_injection": False,
        "expected_models": expected_models,
        "missing_models": missing_models,
        "all_models_ready": all_models_ready,
        "recommended_primary_source": "rgcn",
        "recommended_primary_source_path": str(rgcn_dir),
        "recommended_valid_top20_path": str(rgcn_valid_top20),
        "recommended_test_top20_path": str(rgcn_test_top20),
        "recommended_embedding_path": str(rgcn_embedding),
        "rgcn_files_ready": {
            "valid_top20": rgcn_valid_top20.exists(),
            "test_top20": rgcn_test_top20.exists(),
            "entity_embeddings_rgcn": rgcn_embedding.exists(),
        },
        "valid_best_by_mrr": metric_brief(valid_best),
        "test_best_by_mrr": metric_brief(test_best),
        "rgcn_valid_metrics": metric_brief(rgcn_valid),
        "rgcn_test_metrics": metric_brief(rgcn_test),
        "reason": (
            "R-GCN is selected as the main DrKGC/SoftFuse source for consistency "
            "with PrimeKG, PharmKG, and the DrKGC-style graph-adapter pipeline. "
            "All six structure baselines should still be reported in the result table. "
            "If another baseline has higher MRR, it should be described as the strongest "
            "structure-only candidate generator, while R-GCN remains the graph-LLM source."
        ),
        "fallback_rule": (
            "If R-GCN has zero Gold@20 or severe top-1 collapse, keep R-GCN for "
            "DrKGC-lineage consistency, but optionally build an additional best-valid "
            "candidate-source row as a diagnostic ablation."
        ),
    }

    write_json(OUT_JSON, source)

    md_lines = []
    md_lines.append("# Hetionet source selection")
    md_lines.append("")
    md_lines.append("## Decision")
    md_lines.append("")
    md_lines.append(f"`{decision}`")
    md_lines.append("")
    md_lines.append("## Recommended source")
    md_lines.append("")
    md_lines.append("- Primary source: `rgcn`")
    md_lines.append(f"- Source folder: `{rgcn_dir}`")
    md_lines.append(f"- Valid top20: `{rgcn_valid_top20}`")
    md_lines.append(f"- Test top20: `{rgcn_test_top20}`")
    md_lines.append(f"- Embedding: `{rgcn_embedding}`")
    md_lines.append("")
    md_lines.append("## Rationale")
    md_lines.append("")
    md_lines.append(
        "R-GCN is selected as the primary SoftFuse/DrKGC-ready source for consistency "
        "with PrimeKG, PharmKG, and the graph-adapter setting. All six baselines remain "
        "in the structure-only comparison table."
    )
    md_lines.append("")
    md_lines.append("## Best validation model by MRR")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps(metric_brief(valid_best), ensure_ascii=False, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Best test model by MRR")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps(metric_brief(test_best), ensure_ascii=False, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## R-GCN validation/test")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps(
        {
            "valid": metric_brief(rgcn_valid),
            "test": metric_brief(rgcn_test),
            "files_ready": source["rgcn_files_ready"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Missing models")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps(missing_models, ensure_ascii=False, indent=2))
    md_lines.append("```")
    md_lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps(source, ensure_ascii=False, indent=2))
    print(f"\nSaved:\n  {OUT_JSON}\n  {OUT_MD}")


if __name__ == "__main__":
    main()
