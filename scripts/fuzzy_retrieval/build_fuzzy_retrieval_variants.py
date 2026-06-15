from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

IN_FEATURES = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_path_features.json"
OUT_DIR = ROOT / "dataset/setting_a/fuzzy_retrieval"
SUMMARY_OUT = ROOT / "outputs/fuzzy_retrieval/retrieval_variant_build_summary.json"
REPORT_OUT = ROOT / "outputs/fuzzy_retrieval/reports/fuzzy_retrieval_variant_build.md"

CONFIG_PATHS = [
    ROOT / "configs/fuzzy_retrieval/retrieval_tight.json",
    ROOT / "configs/fuzzy_retrieval/retrieval_directplus.json",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def band_weight(band: str, cfg: Dict[str, Any]) -> float:
    if band == "top_band":
        return float(cfg["top_band_weight"])
    if band == "mid_band":
        return float(cfg["mid_band_weight"])
    return float(cfg["tail_band_weight"])


def score_triple(tr: Dict[str, Any], candidate_bands: List[str], cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    touched_positions = tr.get("touched_candidate_positions", [])
    touched_bands = []
    for pos in touched_positions:
        if isinstance(pos, int) and 0 <= pos < len(candidate_bands):
            touched_bands.append(candidate_bands[pos])

    if touched_bands:
        band_signal = max(band_weight(b, cfg) for b in touched_bands)
        best_band = max(touched_bands, key=lambda b: band_weight(b, cfg))
    else:
        band_signal = 0.0
        best_band = None

    density_hint = float(tr.get("local_density_hint", 0))
    density_norm = min(density_hint, 10.0) / 10.0

    touches_candidate = bool(tr.get("touches_candidate", False))
    touches_query = bool(tr.get("touches_query", False))
    direct_shortcut = bool(tr.get("direct_candidate_query_flag", False))
    contra_flag = bool(tr.get("contra_flag", False))

    score = (
        band_signal
        + float(cfg["touch_candidate"]) * float(touches_candidate)
        + float(cfg["touch_query"]) * float(touches_query)
        + float(cfg["direct_shortcut"]) * float(direct_shortcut)
        + float(cfg["contra"]) * float(contra_flag)
        + float(cfg["density"]) * density_norm
    )

    debug = {
        "band_signal": round(band_signal, 6),
        "best_touched_band": best_band,
        "touch_candidate_term": round(float(cfg["touch_candidate"]) * float(touches_candidate), 6),
        "touch_query_term": round(float(cfg["touch_query"]) * float(touches_query), 6),
        "direct_shortcut_term": round(float(cfg["direct_shortcut"]) * float(direct_shortcut), 6),
        "contra_term": round(float(cfg["contra"]) * float(contra_flag), 6),
        "density_term": round(float(cfg["density"]) * density_norm, 6),
    }
    return round(score, 6), debug


def make_triple_key(tr: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    return (tr["head_id"], tr["relation_id"], tr["tail_id"])


def target_size_from_original(n: int, retain_ratio: float, min_keep: int) -> int:
    if n <= 0:
        return 0
    return min(n, max(min_keep, int(round(n * retain_ratio))))


def build_variant(rows: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []

    total_orig = 0
    total_selected = 0
    total_direct_orig = 0
    total_direct_selected = 0
    total_contra_orig = 0
    total_contra_selected = 0
    coverage_rates = []
    top_band_coverage_rates = []

    for row in rows:
        triple_rows = row.get("triple_feature_rows", [])
        candidate_bands = row.get("candidate_support_bands", [])
        candidate_ids = row.get("candidate_entity_ids", [])
        candidate_names = row.get("candidate_entities", [])
        support_scores = row.get("support_scores", [])
        contra_flags = row.get("contra_flags", [])

        scored_triples = []
        for tr in triple_rows:
            score, debug = score_triple(tr, candidate_bands, cfg)
            scored_triples.append({
                **tr,
                "triple_score": score,
                "score_debug": debug,
            })

        orig_touched_candidates = set()
        orig_touched_top_band = set()
        for tr in scored_triples:
            for pos in tr.get("touched_candidate_positions", []):
                if isinstance(pos, int) and 0 <= pos < len(candidate_ids):
                    orig_touched_candidates.add(pos)
                    if candidate_bands[pos] == "top_band":
                        orig_touched_top_band.add(pos)

        selected_keys = set()
        selected_rows = []

        # stage A: coverage pass
        for cand_pos, _ in enumerate(candidate_ids):
            touching = [tr for tr in scored_triples if cand_pos in tr.get("touched_candidate_positions", [])]
            if not touching:
                continue
            best_tr = max(touching, key=lambda x: (x["triple_score"], -x["triple_index"]))
            key = make_triple_key(best_tr)
            if key not in selected_keys:
                selected_keys.add(key)
                selected_rows.append(best_tr)

        # stage B: global fill
        target_size = target_size_from_original(
            len(scored_triples),
            retain_ratio=float(cfg["retain_ratio"]),
            min_keep=int(cfg["min_keep"]),
        )

        global_sorted = sorted(
            scored_triples,
            key=lambda x: (x["triple_score"], x["touches_top_band_candidate"], -x["triple_index"]),
            reverse=True,
        )

        for tr in global_sorted:
            if len(selected_rows) >= target_size:
                break
            key = make_triple_key(tr)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected_rows.append(tr)

        selected_rows = sorted(
            selected_rows,
            key=lambda x: (x["triple_score"], x["touches_top_band_candidate"], -x["triple_index"]),
            reverse=True,
        )

        selected_subgraph = [[tr["head_id"], tr["relation_id"], tr["tail_id"]] for tr in selected_rows]

        sel_touched_candidates = set()
        sel_touched_top_band = set()
        for tr in selected_rows:
            for pos in tr.get("touched_candidate_positions", []):
                if isinstance(pos, int) and 0 <= pos < len(candidate_ids):
                    sel_touched_candidates.add(pos)
                    if candidate_bands[pos] == "top_band":
                        sel_touched_top_band.add(pos)

        cand_cov = len(sel_touched_candidates) / len(orig_touched_candidates) if orig_touched_candidates else 1.0
        top_cov = len(sel_touched_top_band) / len(orig_touched_top_band) if orig_touched_top_band else 1.0
        coverage_rates.append(cand_cov)
        top_band_coverage_rates.append(top_cov)

        num_direct_orig = sum(int(bool(tr.get("direct_candidate_query_flag", False))) for tr in scored_triples)
        num_direct_sel = sum(int(bool(tr.get("direct_candidate_query_flag", False))) for tr in selected_rows)
        num_contra_orig = sum(int(bool(tr.get("contra_flag", False))) for tr in scored_triples)
        num_contra_sel = sum(int(bool(tr.get("contra_flag", False))) for tr in selected_rows)

        total_orig += len(scored_triples)
        total_selected += len(selected_rows)
        total_direct_orig += num_direct_orig
        total_direct_selected += num_direct_sel
        total_contra_orig += num_contra_orig
        total_contra_selected += num_contra_sel

        out_rows.append({
            "row_index": row["row_index"],
            "row_uid": row["row_uid"],
            "split": row["split"],
            "query_entity": row["query_entity"],
            "query_entity_id": row["query_entity_id"],
            "gold_entity": row["gold_entity"],
            "gold_entity_id": row["gold_entity_id"],
            "variant_name": cfg["variant_name"],
            "candidate_entities": candidate_names,
            "candidate_entity_ids": candidate_ids,
            "support_scores": support_scores,
            "candidate_support_bands": candidate_bands,
            "contra_flags": contra_flags,
            "selected_subgraph": selected_subgraph,
            "triple_score_rows": selected_rows,
            "path_scores": [],
            "subgraph_summary": {
                "original_subgraph_size": len(scored_triples),
                "selected_subgraph_size": len(selected_rows),
                "target_size": target_size,
                "num_original_direct_shortcuts": num_direct_orig,
                "num_selected_direct_shortcuts": num_direct_sel,
                "num_original_contra_triples": num_contra_orig,
                "num_selected_contra_triples": num_contra_sel,
                "original_candidate_touched_count": len(orig_touched_candidates),
                "selected_candidate_touched_count": len(sel_touched_candidates),
                "candidate_coverage_preserved_rate": round(cand_cov, 6),
                "top_band_coverage_preserved_rate": round(top_cov, 6),
            },
        })

    summary = {
        "variant_name": cfg["variant_name"],
        "num_rows": len(out_rows),
        "avg_original_subgraph_size": round(total_orig / len(out_rows), 6) if out_rows else 0.0,
        "avg_selected_subgraph_size": round(total_selected / len(out_rows), 6) if out_rows else 0.0,
        "direct_shortcut_rate_before": round(total_direct_orig / total_orig, 6) if total_orig else 0.0,
        "direct_shortcut_rate_after": round(total_direct_selected / total_selected, 6) if total_selected else 0.0,
        "contra_rate_before": round(total_contra_orig / total_orig, 6) if total_orig else 0.0,
        "contra_rate_after": round(total_contra_selected / total_selected, 6) if total_selected else 0.0,
        "avg_candidate_coverage_preserved_rate": round(sum(coverage_rates) / len(coverage_rates), 6) if coverage_rates else 1.0,
        "avg_top_band_coverage_preserved_rate": round(sum(top_band_coverage_rates) / len(top_band_coverage_rates), 6) if top_band_coverage_rates else 1.0,
        "config": cfg,
    }
    return out_rows, summary


def main() -> None:
    rows = load_json(IN_FEATURES)
    assert isinstance(rows, list) and len(rows) == 500, "valid_path_features.json must contain 500 rows"

    summaries = []

    for config_path in CONFIG_PATHS:
        cfg = load_json(config_path)
        out_rows, summary = build_variant(rows, cfg)

        variant_name = cfg["variant_name"]
        out_json = OUT_DIR / f"valid_{variant_name}.json"
        save_json(out_json, out_rows)

        summaries.append({
            "variant_name": variant_name,
            "output_path": str(out_json.relative_to(ROOT)),
            **summary,
        })

        print("=" * 80)
        print("BUILT VARIANT:", variant_name)
        print("output:", out_json)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("=" * 80)

    save_json(SUMMARY_OUT, {
        "stage": "fuzzy_retrieval_variant_build",
        "variant_summaries": summaries,
    })

    report_lines = [
        "# Fuzzy Retrieval Variant Build",
        "",
        "## Built variants",
    ]
    for s in summaries:
        report_lines.extend([
            f"### {s['variant_name']}",
            f"- output_path: `{s['output_path']}`",
            f"- avg_original_subgraph_size: {s['avg_original_subgraph_size']}",
            f"- avg_selected_subgraph_size: {s['avg_selected_subgraph_size']}",
            f"- direct_shortcut_rate_before: {s['direct_shortcut_rate_before']}",
            f"- direct_shortcut_rate_after: {s['direct_shortcut_rate_after']}",
            f"- contra_rate_before: {s['contra_rate_before']}",
            f"- contra_rate_after: {s['contra_rate_after']}",
            f"- avg_candidate_coverage_preserved_rate: {s['avg_candidate_coverage_preserved_rate']}",
            f"- avg_top_band_coverage_preserved_rate: {s['avg_top_band_coverage_preserved_rate']}",
            "",
        ])

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(report_lines), encoding="utf-8")

    print("SAVED SUMMARY:", SUMMARY_OUT)
    print("SAVED REPORT :", REPORT_OUT)


if __name__ == "__main__":
    main()
