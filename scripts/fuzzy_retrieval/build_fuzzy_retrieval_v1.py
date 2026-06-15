from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

IN_FEATURES = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_path_features.json"
OUT_JSON = ROOT / "dataset/setting_a/fuzzy_retrieval/valid_fuzzy_retrieval_v1.json"
OUT_MANIFEST = ROOT / "dataset/setting_a/fuzzy_retrieval/path_score_manifest.json"
OUT_REPORT = ROOT / "outputs/fuzzy_retrieval/reports/fuzzy_retrieval_v1_build.md"

TOP_BAND_WEIGHT = 1.00
MID_BAND_WEIGHT = 0.60
TAIL_BAND_WEIGHT = 0.25

W_TOUCH_CAND = 0.35
W_TOUCH_QUERY = 0.05
W_DIRECT_SHORTCUT = -0.80
W_CONTRA = -0.10
W_DENSITY = 0.05

RETAIN_RATIO = 0.67
MIN_KEEP = 20
VARIANT_NAME = "soft_support_fuzzy_retrieval_v1"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def band_weight(band: str) -> float:
    if band == "top_band":
        return TOP_BAND_WEIGHT
    if band == "mid_band":
        return MID_BAND_WEIGHT
    return TAIL_BAND_WEIGHT


def score_triple(tr: Dict[str, Any], candidate_bands: List[str]) -> Tuple[float, Dict[str, Any]]:
    touched_positions = tr.get("touched_candidate_positions", [])
    touched_bands = []
    for pos in touched_positions:
        if isinstance(pos, int) and 0 <= pos < len(candidate_bands):
            touched_bands.append(candidate_bands[pos])

    if touched_bands:
        band_signal = max(band_weight(b) for b in touched_bands)
        best_band = max(
            touched_bands,
            key=lambda b: band_weight(b)
        )
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
        + W_TOUCH_CAND * float(touches_candidate)
        + W_TOUCH_QUERY * float(touches_query)
        + W_DIRECT_SHORTCUT * float(direct_shortcut)
        + W_CONTRA * float(contra_flag)
        + W_DENSITY * density_norm
    )

    debug = {
        "band_signal": round(band_signal, 6),
        "best_touched_band": best_band,
        "touch_candidate_term": round(W_TOUCH_CAND * float(touches_candidate), 6),
        "touch_query_term": round(W_TOUCH_QUERY * float(touches_query), 6),
        "direct_shortcut_term": round(W_DIRECT_SHORTCUT * float(direct_shortcut), 6),
        "contra_term": round(W_CONTRA * float(contra_flag), 6),
        "density_term": round(W_DENSITY * density_norm, 6),
    }
    return round(score, 6), debug


def make_triple_key(tr: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    return (tr["head_id"], tr["relation_id"], tr["tail_id"])


def target_size_from_original(n: int) -> int:
    if n <= 0:
        return 0
    return min(n, max(MIN_KEEP, int(round(n * RETAIN_RATIO))))


def main() -> None:
    rows = load_json(IN_FEATURES)
    assert isinstance(rows, list) and len(rows) == 500, "valid_path_features.json must contain 500 rows"

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
            score, debug = score_triple(tr, candidate_bands)
            scored_triples.append({
                **tr,
                "triple_score": score,
                "score_debug": debug,
            })

        # Original coverage
        orig_touched_candidates = set()
        orig_touched_top_band = set()
        for tr in scored_triples:
            for pos in tr.get("touched_candidate_positions", []):
                if isinstance(pos, int) and 0 <= pos < len(candidate_ids):
                    orig_touched_candidates.add(pos)
                    if candidate_bands[pos] == "top_band":
                        orig_touched_top_band.add(pos)

        # Stage A: coverage pass
        selected_keys = set()
        selected_rows = []

        for cand_pos, cand_id in enumerate(candidate_ids):
            touching = [
                tr for tr in scored_triples
                if cand_pos in tr.get("touched_candidate_positions", [])
            ]
            if not touching:
                continue
            best_tr = max(touching, key=lambda x: (x["triple_score"], -x["triple_index"]))
            key = make_triple_key(best_tr)
            if key not in selected_keys:
                selected_keys.add(key)
                selected_rows.append(best_tr)

        # Stage B: global fill
        target_size = target_size_from_original(len(scored_triples))
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

        # Keep deterministic order by score desc after selection
        selected_rows = sorted(
            selected_rows,
            key=lambda x: (x["triple_score"], x["touches_top_band_candidate"], -x["triple_index"]),
            reverse=True,
        )

        selected_subgraph = [
            [tr["head_id"], tr["relation_id"], tr["tail_id"]]
            for tr in selected_rows
        ]

        # Selected coverage
        sel_touched_candidates = set()
        sel_touched_top_band = set()
        for tr in selected_rows:
            for pos in tr.get("touched_candidate_positions", []):
                if isinstance(pos, int) and 0 <= pos < len(candidate_ids):
                    sel_touched_candidates.add(pos)
                    if candidate_bands[pos] == "top_band":
                        sel_touched_top_band.add(pos)

        cand_cov = (
            len(sel_touched_candidates) / len(orig_touched_candidates)
            if orig_touched_candidates else 1.0
        )
        top_cov = (
            len(sel_touched_top_band) / len(orig_touched_top_band)
            if orig_touched_top_band else 1.0
        )
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
            "variant_name": VARIANT_NAME,
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

    manifest = {
        "stage": "fuzzy_retrieval_v1_build",
        "variant_name": VARIANT_NAME,
        "scoring_level": "triple-first",
        "notes": [
            "This v1 scores triples, not full paths yet.",
            "ontology_consistency_flag is disabled in v1 because schema_known_count was zero during path feature build.",
            "Selection policy = coverage pass + global fill.",
            "Candidate order remains fixed from soft_support_raw."
        ],
        "weights": {
            "top_band_weight": TOP_BAND_WEIGHT,
            "mid_band_weight": MID_BAND_WEIGHT,
            "tail_band_weight": TAIL_BAND_WEIGHT,
            "touch_candidate": W_TOUCH_CAND,
            "touch_query": W_TOUCH_QUERY,
            "direct_shortcut": W_DIRECT_SHORTCUT,
            "contra": W_CONTRA,
            "density": W_DENSITY,
        },
        "selection_policy": {
            "stage_a": "coverage_pass_best_triple_per_candidate",
            "stage_b": "global_fill_by_triple_score",
            "retain_ratio": RETAIN_RATIO,
            "min_keep": MIN_KEEP,
        },
        "summary": {
            "num_rows": len(out_rows),
            "avg_original_subgraph_size": round(total_orig / len(out_rows), 6) if out_rows else 0.0,
            "avg_selected_subgraph_size": round(total_selected / len(out_rows), 6) if out_rows else 0.0,
            "direct_shortcut_rate_before": round(total_direct_orig / total_orig, 6) if total_orig else 0.0,
            "direct_shortcut_rate_after": round(total_direct_selected / total_selected, 6) if total_selected else 0.0,
            "contra_rate_before": round(total_contra_orig / total_orig, 6) if total_orig else 0.0,
            "contra_rate_after": round(total_contra_selected / total_selected, 6) if total_selected else 0.0,
            "avg_candidate_coverage_preserved_rate": round(sum(coverage_rates) / len(coverage_rates), 6) if coverage_rates else 1.0,
            "avg_top_band_coverage_preserved_rate": round(sum(top_band_coverage_rates) / len(top_band_coverage_rates), 6) if top_band_coverage_rates else 1.0,
        },
    }

    save_json(OUT_JSON, out_rows)
    save_json(OUT_MANIFEST, manifest)

    report_lines = [
        "# Fuzzy Retrieval v1 Build",
        "",
        f"- variant_name: {VARIANT_NAME}",
        "- scoring_level: triple-first",
        "- ontology consistency disabled in v1",
        "- candidate stage kept fixed from soft_support_raw",
        "",
        "## Selection policy",
        f"- retain_ratio: {RETAIN_RATIO}",
        f"- min_keep: {MIN_KEEP}",
        "- stage_a: best triple per candidate",
        "- stage_b: global fill by triple_score",
        "",
        "## Summary",
        f"- avg_original_subgraph_size: {manifest['summary']['avg_original_subgraph_size']}",
        f"- avg_selected_subgraph_size: {manifest['summary']['avg_selected_subgraph_size']}",
        f"- direct_shortcut_rate_before: {manifest['summary']['direct_shortcut_rate_before']}",
        f"- direct_shortcut_rate_after: {manifest['summary']['direct_shortcut_rate_after']}",
        f"- contra_rate_before: {manifest['summary']['contra_rate_before']}",
        f"- contra_rate_after: {manifest['summary']['contra_rate_after']}",
        f"- avg_candidate_coverage_preserved_rate: {manifest['summary']['avg_candidate_coverage_preserved_rate']}",
        f"- avg_top_band_coverage_preserved_rate: {manifest['summary']['avg_top_band_coverage_preserved_rate']}",
    ]
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print("=" * 80)
    print("FUZZY RETRIEVAL V1 BUILD DONE")
    print("json    :", OUT_JSON)
    print("manifest:", OUT_MANIFEST)
    print("report  :", OUT_REPORT)
    print("=" * 80)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    main()
