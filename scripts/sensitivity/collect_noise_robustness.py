#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect small-noise robustness metrics.

No LLM inference is required.
This evaluates candidate/ranking stability and graph/evidence package stability.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

VARIANT_ROOT = ROOT / "dataset" / "setting_a" / "noise_robustness"
RESULTS_DIR = ROOT / "outputs" / "sensitivity" / "noise_robustness"
REPORTS_DIR = ROOT / "outputs" / "sensitivity" / "reports"

VARIANTS = [
    "N0_no_noise",
    "N1_support_score_noise_seed1",
    "N2_support_score_noise_seed2",
    "N3_support_score_noise_seed3",
    "N4_subgraph_edge_dropout_5_seed1",
    "N5_subgraph_edge_dropout_5_seed2",
    "N6_subgraph_edge_dropout_5_seed3",
]
SPLITS = ["valid", "test"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_len(x: Any) -> int:
    return len(x) if isinstance(x, list) else 0


def norm_text(x: Any) -> str:
    return "" if x is None else str(x).strip()


def get_gold(row: Dict[str, Any]) -> str:
    return norm_text(row.get("gold_entity", row.get("output", "")))


def get_candidates(row: Dict[str, Any]) -> List[str]:
    return [norm_text(x) for x in row.get("rank_entities", [])]


def find_candidate(name: str, cands: List[str]) -> int | None:
    n = name.lower()
    for i, c in enumerate(cands):
        if c.lower() == n:
            return i
    return None


def gold_rank(row: Dict[str, Any]) -> int:
    idx = find_candidate(get_gold(row), get_candidates(row))
    if idx is None or idx >= 20:
        return 21
    return idx + 1


def rr_at20(rank: int) -> float:
    return 1.0 / rank if rank <= 20 else 0.0


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def round6(x: float) -> float:
    return round(float(x), 6)


def edge_key(edge: Any) -> Tuple[int, int, int]:
    if isinstance(edge, dict):
        h = edge.get("h", edge.get("head", edge.get("src", edge.get("source"))))
        r = edge.get("r", edge.get("relation", edge.get("rel")))
        t = edge.get("t", edge.get("tail", edge.get("dst", edge.get("target"))))
        return (int(h), int(r), int(t))
    if isinstance(edge, (list, tuple)) and len(edge) >= 3:
        return (int(edge[0]), int(edge[1]), int(edge[2]))
    raise ValueError(f"Cannot parse edge: {edge}")


def edge_set(edges: List[Any]) -> set:
    out = set()
    for e in edges or []:
        try:
            out.add(edge_key(e))
        except Exception:
            continue
    return out


def graph_jaccard(a: List[Any], b: List[Any]) -> float:
    aa = edge_set(a)
    bb = edge_set(b)
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def node_ids_in_edges(edges: List[Any]) -> set:
    nodes = set()
    for e in edges or []:
        try:
            h, _, t = edge_key(e)
            nodes.add(h)
            nodes.add(t)
        except Exception:
            continue
    return nodes


def coverage_stats(row: Dict[str, Any]) -> Dict[str, float]:
    edges = row.get("subgraph", [])
    nodes = node_ids_in_edges(edges)

    qid = row.get("query_entity_id")
    query_cov = 0.0
    try:
        query_cov = 1.0 if int(qid) in nodes else 0.0
    except Exception:
        query_cov = 0.0

    cids = row.get("rank_entities_id", [])
    total = 0
    covered = 0
    for cid in cids:
        try:
            total += 1
            if int(cid) in nodes:
                covered += 1
        except Exception:
            pass

    return {
        "query_coverage": query_cov,
        "candidate_coverage": covered / total if total else 0.0,
        "num_candidate_covered": covered,
        "num_candidates": total,
    }


def support_shift(row: Dict[str, Any], base_row: Dict[str, Any]) -> float | None:
    # Prefer noisy post-reorder if present; otherwise compare support_scores.
    a = row.get("support_scores")
    b = base_row.get("support_scores")

    if not isinstance(a, list) or not isinstance(b, list):
        return None
    if len(a) != len(b) or len(a) == 0:
        return None

    diffs = []
    for x, y in zip(a, b):
        try:
            diffs.append(abs(float(x) - float(y)))
        except Exception:
            pass

    return mean(diffs) if diffs else None


def validate_alignment(base_rows: List[Dict[str, Any]], rows: List[Dict[str, Any]], variant: str, split: str) -> None:
    if len(base_rows) != len(rows):
        raise RuntimeError(f"{split}/{variant}: row count mismatch.")

    for i, (b, r) in enumerate(zip(base_rows, rows)):
        if b.get("query_entity_id") != r.get("query_entity_id"):
            raise RuntimeError(f"{split}/{variant}: query mismatch at row {i}")
        if get_gold(b).lower() != get_gold(r).lower():
            raise RuntimeError(f"{split}/{variant}: gold mismatch at row {i}")


def summarize_variant(
    split: str,
    variant: str,
    rows: List[Dict[str, Any]],
    base_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    validate_alignment(base_rows, rows, variant, split)

    ranks = [gold_rank(r) for r in rows]
    base_ranks = [gold_rank(r) for r in base_rows]

    sub_sizes = [safe_len(r.get("subgraph", [])) for r in rows]
    covs = [coverage_stats(r) for r in rows]

    same_top1 = []
    same_order = []
    rank_changed = []
    abs_rank_shift = []
    graph_jaccards = []
    support_shifts = []

    for r, b, rank, base_rank in zip(rows, base_rows, ranks, base_ranks):
        cands = r.get("rank_entities_id", [])
        base_cands = b.get("rank_entities_id", [])

        same_top1.append(
            1.0 if cands and base_cands and cands[0] == base_cands[0] else 0.0
        )
        same_order.append(1.0 if cands == base_cands else 0.0)
        rank_changed.append(1.0 if rank != base_rank else 0.0)
        abs_rank_shift.append(float(abs(rank - base_rank)))

        graph_jaccards.append(graph_jaccard(r.get("subgraph", []), b.get("subgraph", [])))

        sh = support_shift(r, b)
        if sh is not None:
            support_shifts.append(sh)

    return {
        "split": split,
        "variant": variant,
        "num_rows": len(rows),
        "gold_at20": round6(mean([1.0 if r <= 20 else 0.0 for r in ranks])),
        "candidate_mrr_at20": round6(mean([rr_at20(r) for r in ranks])),
        "hits1_at20": round6(mean([1.0 if r <= 1 else 0.0 for r in ranks])),
        "hits3_at20": round6(mean([1.0 if r <= 3 else 0.0 for r in ranks])),
        "hits10_at20": round6(mean([1.0 if r <= 10 else 0.0 for r in ranks])),
        "rank21_count": int(sum(1 for r in ranks if r == 21)),
        "avg_gold_rank_with_21": round6(mean([float(r) for r in ranks])),

        "same_top1_rate_vs_N0": round6(mean(same_top1)),
        "same_candidate_order_rate_vs_N0": round6(mean(same_order)),
        "rank_change_rate_vs_N0": round6(mean(rank_changed)),
        "avg_abs_rank_shift_vs_N0": round6(mean(abs_rank_shift)),

        "avg_subgraph_size": round6(mean([float(x) for x in sub_sizes])),
        "min_subgraph_size": int(min(sub_sizes)) if sub_sizes else None,
        "max_subgraph_size": int(max(sub_sizes)) if sub_sizes else None,
        "query_coverage_rate": round6(mean([c["query_coverage"] for c in covs])),
        "candidate_coverage_rate": round6(mean([c["candidate_coverage"] for c in covs])),
        "avg_graph_jaccard_vs_N0": round6(mean(graph_jaccards)),
        "avg_support_score_shift_vs_N0": round6(mean(support_shifts)) if support_shifts else None,
    }


def add_delta_vs_n0(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_split = {}
    for s in summaries:
        by_split.setdefault(s["split"], {})[s["variant"]] = s

    out = []
    for s in summaries:
        n0 = by_split[s["split"]]["N0_no_noise"]
        ss = dict(s)

        for key in [
            "candidate_mrr_at20",
            "hits1_at20",
            "hits3_at20",
            "hits10_at20",
            "gold_at20",
            "avg_subgraph_size",
            "candidate_coverage_rate",
            "query_coverage_rate",
        ]:
            ss[f"delta_{key}_vs_N0"] = round6(ss[key] - n0[key])

        out.append(ss)

    return out


def collect_cases(
    rows_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]],
    max_cases: int = 8,
) -> Dict[str, Any]:
    out = {}

    for split in SPLITS:
        base = rows_by_key[(split, "N0_no_noise")]

        for variant in VARIANTS:
            if variant == "N0_no_noise":
                continue

            rows = rows_by_key[(split, variant)]
            cases = []

            for idx, (r, b) in enumerate(zip(rows, base)):
                rank_r = gold_rank(r)
                rank_b = gold_rank(b)

                cands_r = r.get("rank_entities", [])
                cands_b = b.get("rank_entities", [])

                order_changed = r.get("rank_entities_id", []) != b.get("rank_entities_id", [])
                graph_j = graph_jaccard(r.get("subgraph", []), b.get("subgraph", []))

                if order_changed or rank_r != rank_b or graph_j < 0.98:
                    cases.append({
                        "split": split,
                        "variant": variant,
                        "row_index": idx,
                        "query_entity": r.get("query_entity"),
                        "gold_entity": get_gold(r),
                        "rank_N0": rank_b,
                        "rank_variant": rank_r,
                        "rank_delta": rank_r - rank_b,
                        "top1_N0": cands_b[0] if cands_b else None,
                        "top1_variant": cands_r[0] if cands_r else None,
                        "same_top1": bool(cands_r and cands_b and cands_r[0] == cands_b[0]),
                        "graph_size_N0": safe_len(b.get("subgraph", [])),
                        "graph_size_variant": safe_len(r.get("subgraph", [])),
                        "graph_jaccard_vs_N0": round6(graph_j),
                        "top5_variant": cands_r[:5],
                    })

            cases = sorted(
                cases,
                key=lambda x: (
                    abs(x["rank_delta"]),
                    int(not x["same_top1"]),
                    1.0 - x["graph_jaccard_vs_N0"],
                ),
                reverse=True,
            )

            out[f"{split}_{variant}"] = cases[:max_cases]

    return out


def md_escape(x: Any) -> str:
    s = "" if x is None else str(x)
    return s.replace("|", "\\|").replace("\n", " ")


def md_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = []
    lines.append("| " + " | ".join(md_escape(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(md_escape(v) for v in row) + " |")
    return "\n".join(lines)


def write_report(result: Dict[str, Any], path: Path) -> None:
    lines = []
    lines.append("# Small-Noise Robustness\n")
    lines.append(f"- Decision: **{result['decision']}**")
    lines.append(f"- Created at: `{result['created_at']}`")
    lines.append("- Main row: **retrieval_main / N0_no_noise**")
    lines.append("- Evaluation: candidate/retrieval-level robustness, no LLM E2E required")
    lines.append("- Policy: reviewer-safe RR@20")
    lines.append("")

    for split in SPLITS:
        lines.append(f"## {split.upper()} summary\n")
        rows = []
        for s in result["summaries"]:
            if s["split"] != split:
                continue
            rows.append([
                s["variant"],
                s["gold_at20"],
                s["candidate_mrr_at20"],
                s["hits1_at20"],
                s["hits3_at20"],
                s["hits10_at20"],
                s["same_top1_rate_vs_N0"],
                s["rank_change_rate_vs_N0"],
                s["avg_abs_rank_shift_vs_N0"],
                s["avg_subgraph_size"],
                s["candidate_coverage_rate"],
                s["avg_graph_jaccard_vs_N0"],
                s["delta_candidate_mrr_at20_vs_N0"],
            ])
        lines.append(md_table(
            [
                "Variant",
                "Gold@20",
                "MRR@20",
                "H@1",
                "H@3",
                "H@10",
                "Same top1",
                "Rank-change",
                "Avg rank shift",
                "Avg graph",
                "Cand coverage",
                "Graph Jaccard",
                "� MRR vs N0",
            ],
            rows,
        ))
        lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append(
        "- Support-score noise tests whether small perturbations in soft evidence scores destabilize candidate ordering."
    )
    lines.append(
        "- Subgraph dropout tests whether the selected evidence package remains structurally usable under light edge loss."
    )
    lines.append(
        "- If MRR@20 changes only slightly and coverage remains stable, SoftFuse can be described as robust to small perturbations."
    )
    lines.append(
        "- These results are appendix robustness evidence and must not replace the selected E2E main result."
    )
    lines.append("")

    lines.append("## Case samples\n")
    for key, cases in result["case_samples"].items():
        lines.append(f"### {key}")
        if not cases:
            lines.append("- No changed cases.")
            lines.append("")
            continue

        case_rows = []
        for c in cases:
            case_rows.append([
                c["row_index"],
                c["query_entity"],
                c["gold_entity"],
                c["rank_N0"],
                c["rank_variant"],
                c["rank_delta"],
                c["top1_N0"],
                c["top1_variant"],
                c["graph_size_N0"],
                c["graph_size_variant"],
                c["graph_jaccard_vs_N0"],
            ])
        lines.append(md_table(
            [
                "idx",
                "query",
                "gold",
                "rank N0",
                "rank var",
                "rank delta",
                "top1 N0",
                "top1 var",
                "graph N0",
                "graph var",
                "Jaccard",
            ],
            case_rows,
        ))
        lines.append("")

    lines.append("## Final decision\n")
    lines.append(f"**{result['decision']}**")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fatal_errors = []
    rows_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    for split in SPLITS:
        for variant in VARIANTS:
            p = VARIANT_ROOT / variant / f"{split}.json"
            if not p.exists():
                fatal_errors.append(f"Missing file: {rel(p)}")
                continue
            rows_by_key[(split, variant)] = load_json(p)

    if fatal_errors:
        result = {
            "week": 25,
            "day": 5,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "decision": "NOISE_ROBUSTNESS_BLOCKED",
            "fatal_errors": fatal_errors,
        }
        write_json(result, RESULTS_DIR / "noise_robustness_summary.json")
        print("decision =", result["decision"])
        print("fatal_errors =", fatal_errors)
        return

    summaries = []

    try:
        for split in SPLITS:
            base_rows = rows_by_key[(split, "N0_no_noise")]
            for variant in VARIANTS:
                rows = rows_by_key[(split, variant)]
                summaries.append(summarize_variant(split, variant, rows, base_rows))
    except Exception as e:
        result = {
            "week": 25,
            "day": 5,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "decision": "NOISE_ROBUSTNESS_BLOCKED",
            "fatal_errors": [repr(e)],
        }
        write_json(result, RESULTS_DIR / "noise_robustness_summary.json")
        print("decision =", result["decision"])
        print("fatal_errors =", result["fatal_errors"])
        return

    summaries = add_delta_vs_n0(summaries)
    case_samples = collect_cases(rows_by_key)

    result = {
        "week": 25,
        "day": 5,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "NOISE_ROBUSTNESS_READY",
        "main_row": "retrieval_main",
        "baseline_variant": "N0_no_noise",
        "reviewer_safe_policy": "RR=1/rank if rank<=20 else 0",
        "summaries": summaries,
        "case_samples": case_samples,
        "fatal_errors": [],
        "notes": [
            "No LLM E2E inference is required for Day 5.",
            "This is retrieval/ranking/evidence robustness.",
            "Week24 main result remains unchanged.",
        ],
    }

    valid_only = {
        "week": 25,
        "day": 5,
        "split": "valid",
        "decision": result["decision"],
        "summaries": [s for s in summaries if s["split"] == "valid"],
        "case_samples": {k: v for k, v in case_samples.items() if k.startswith("valid_")},
    }

    test_only = {
        "week": 25,
        "day": 5,
        "split": "test",
        "decision": result["decision"],
        "summaries": [s for s in summaries if s["split"] == "test"],
        "case_samples": {k: v for k, v in case_samples.items() if k.startswith("test_")},
    }

    out_summary = RESULTS_DIR / "noise_robustness_summary.json"
    out_valid = RESULTS_DIR / "noise_robustness_valid.json"
    out_test = RESULTS_DIR / "noise_robustness_test.json"
    out_report = REPORTS_DIR / "day5_noise_robustness.md"

    write_json(result, out_summary)
    write_json(valid_only, out_valid)
    write_json(test_only, out_test)
    write_report(result, out_report)

    print("=" * 100)
    print("decision =", result["decision"])
    print("summary_json =", rel(out_summary))
    print("valid_json =", rel(out_valid))
    print("test_json =", rel(out_test))
    print("report_md =", rel(out_report))
    print("=" * 100)

    for split in SPLITS:
        print(f"[{split}]")
        for s in summaries:
            if s["split"] != split:
                continue
            print(
                s["variant"],
                "Gold@20 =", s["gold_at20"],
                "MRR@20 =", s["candidate_mrr_at20"],
                "H@10 =", s["hits10_at20"],
                "SameTop1 =", s["same_top1_rate_vs_N0"],
                "RankChange =", s["rank_change_rate_vs_N0"],
                "AvgRankShift =", s["avg_abs_rank_shift_vs_N0"],
                "AvgGraph =", s["avg_subgraph_size"],
                "CandCov =", s["candidate_coverage_rate"],
                "GraphJaccard =", s["avg_graph_jaccard_vs_N0"],
                "DeltaMRR =", s["delta_candidate_mrr_at20_vs_N0"],
            )
        print("-" * 100)


if __name__ == "__main__":
    main()