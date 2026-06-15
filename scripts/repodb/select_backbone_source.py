#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(".")
RESULT_DIR = ROOT / "outputs" / "repodb"
REPORT_DIR = ROOT / "outputs" / "repodb" / "reports"
BASELINE_DIR = ROOT / "dataset" / "setting_f_repodb" / "baseline_outputs"

MAIN_TABLE = RESULT_DIR / "repodb_baseline_main_table.json"
OUT_JSON = RESULT_DIR / "day4_repodb_source_selection.json"
OUT_MD = REPORT_DIR / "day4_repodb_source_selection.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def brief(m):
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


def main():
    if not MAIN_TABLE.exists():
        raise FileNotFoundError(
            f"Missing baseline main table: {MAIN_TABLE}. "
            "Run Day 4 KGE/GNN baselines first."
        )

    table = read_json(MAIN_TABLE)
    valid = table.get("valid", [])
    test = table.get("test", [])

    valid_by_name = {x.get("model_name"): x for x in valid}
    test_by_name = {x.get("model_name"): x for x in test}

    expected = ["transe", "distmult", "complex", "rotate", "rgcn", "hrgat"]
    missing = [m for m in expected if m not in valid_by_name or m not in test_by_name]

    valid_best = valid[0] if valid else None
    test_best = test[0] if test else None

    rgcn_valid = valid_by_name.get("rgcn")
    rgcn_test = test_by_name.get("rgcn")

    rgcn_dir = BASELINE_DIR / "rgcn"
    rgcn_embedding = rgcn_dir / "entity_embeddings_rgcn.pt"

    rgcn_ready = (
        rgcn_valid is not None
        and rgcn_test is not None
        and (rgcn_dir / "valid_top20.json").exists()
        and (rgcn_dir / "test_top20.json").exists()
        and rgcn_embedding.exists()
    )

    rgcn_nontrivial = False
    if rgcn_valid is not None and rgcn_test is not None:
        rgcn_nontrivial = (
            float(rgcn_valid.get("gold_present_at20", 0.0)) >= 0.05
            or float(rgcn_test.get("gold_present_at20", 0.0)) >= 0.05
        )

    best_valid_name = valid_best["model_name"] if valid_best else None

    # Policy:
    # - Prefer R-GCN as main SoftFuse source if non-trivial.
    # - Still report best-valid structure baseline separately.
    if not missing and rgcn_ready and rgcn_nontrivial:
        decision = "DAY4_REPODB_SOURCE_RGCN_READY"
        recommended = "rgcn"
        source_path = str(rgcn_dir)
        embedding_path = str(rgcn_embedding)
    elif not missing:
        decision = "DAY4_REPODB_SOURCE_USE_BEST_VALID_DIAGNOSTIC"
        recommended = best_valid_name
        source_path = str(BASELINE_DIR / recommended) if recommended else None
        embedding_path = str(rgcn_embedding) if rgcn_embedding.exists() else None
    else:
        decision = "DAY4_REPODB_SOURCE_PARTIAL_READY"
        recommended = "rgcn" if rgcn_ready else None
        source_path = str(rgcn_dir) if rgcn_ready else None
        embedding_path = str(rgcn_embedding) if rgcn_embedding.exists() else None

    obj = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_f_repodb",
        "dataset": "repoDB",
        "task": "(?, repoDB_approved_indication, disease)",
        "target_relation": "repoDB::approved_indication::Compound:Disease",
        "target_relation_normalized": "repodb_approved_indication",
        "prediction_type": "predicted_head",
        "candidate_universe": "train_approved_relation_compound_heads",
        "top_k": 20,
        "gold_injection": False,
        "expected_models": expected,
        "missing_models": missing,
        "recommended_primary_source": recommended,
        "recommended_primary_source_path": source_path,
        "recommended_embedding_path": embedding_path,
        "valid_best_by_mrr": brief(valid_best),
        "test_best_by_mrr": brief(test_best),
        "rgcn_valid_metrics": brief(rgcn_valid),
        "rgcn_test_metrics": brief(rgcn_test),
        "source_policy": (
            "Prefer R-GCN as the graph-compatible DrKGC/SoftFuse source if it has non-trivial Gold@20. "
            "Report all six structure baselines. If another model is best by validation MRR, describe it as "
            "the strongest structure-only candidate generator. For repoDB, clinical validation value is more important "
            "than claiming universal baseline superiority."
        ),
    }

    write_json(OUT_JSON, obj)

    lines = []
    lines.append("# repoDB source selection")
    lines.append("")
    lines.append(f"- Decision: `{decision}`")
    lines.append(f"- Recommended primary source: `{recommended}`")
    lines.append(f"- Source path: `{source_path}`")
    lines.append(f"- Embedding path: `{embedding_path}`")
    lines.append("")
    lines.append("## Best validation model")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(brief(valid_best), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Best test model")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(brief(test_best), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## R-GCN metrics")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"valid": brief(rgcn_valid), "test": brief(rgcn_test)}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Source policy")
    lines.append("")
    lines.append(obj["source_policy"])
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(obj, ensure_ascii=False, indent=2))
    print(f"\nSaved:\n  {OUT_JSON}\n  {OUT_MD}")


if __name__ == "__main__":
    main()
