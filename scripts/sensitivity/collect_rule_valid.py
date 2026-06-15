#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect rule sensitivity valid metrics.

This is candidate-level and graph-package analysis only.
E2E inference is Day 3.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

VARIANT_ROOT = ROOT / "dataset" / "setting_a" / "rule_sensitivity"
RESULTS_DIR = ROOT / "outputs" / "sensitivity" / "rule_sensitivity"
REPORTS_DIR = ROOT / "outputs" / "sensitivity" / "reports"

VARIANTS = ["main_rules", "no_rules", "random_rules"]


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


def safe_len(x: Any) -> int:
    return len(x) if isinstance(x, list) else 0


def extract_subgraph(row: Dict[str, Any]) -> List[Any]:
    if isinstance(row.get("subgraph"), list):
        return row["subgraph"]
    if isinstance(row.get("selected_subgraph"), list):
        return row["selected_subgraph"]
    return []


def get_gold(row: Dict[str, Any]) -> str:
    return str(row.get("gold_entity", row.get("output", ""))).strip()


def get_rank_entities(row: Dict[str, Any]) -> List[str]:
    return [str(x).strip() for x in row.get("rank_entities", [])]


def reviewer_safe_rank(row: Dict[str, Any]) -> int:
    gold = get_gold(row)
    candidates = get_rank_entities(row)
    if gold in candidates:
        idx = candidates.index(gold) + 1
        if idx <= 20:
            return idx
    return 21


def rr_at20_from_rank(rank: int) -> float:
    return 1.0 / rank if rank <= 20 else 0.0


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def round6(x: float) -> float:
    return round(float(x), 6)


def coverage_stats(row: Dict[str, Any]) -> Dict[str, float]:
    edges = extract_subgraph(row)
    nodes = node_ids_in_edges(edges)

    query_id = row.get("query_entity_id")
    candidate_ids = row.get("rank_entities_id", [])

    query_covered = False
    if query_id is not None:
        try:
            query_covered = int(query_id) in nodes
        except Exception:
            query_covered = False

    cand_total = 0
    cand_covered = 0
    for cid in candidate_ids:
        try:
            cand_total += 1
            if int(cid) in nodes:
                cand_covered += 1
        except Exception:
            continue

    return {
        "query_covered": 1.0 if query_covered else 0.0,
        "candidate_coverage": cand_covered / cand_total if cand_total else 0.0,
        "num_candidate_covered": cand_covered,
        "num_candidates": cand_total,
    }


def support_score_shift(row: Dict[str, Any], main_row: Dict[str, Any]) -> float | None:
    a = row.get("support_scores")
    b = main_row.get("support_scores")
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
    if not diffs:
        return None
    return mean(diffs)


def validate_alignment(main_rows: List[Dict[str, Any]], rows: List[Dict[str, Any]], variant: str) -> None:
    if len(main_rows) != len(rows):
        raise RuntimeError(f"{variant}: row count mismatch vs main.")

    for i, (m, r) in enumerate(zip(main_rows, rows)):
        if m.get("query_entity_id") != r.get("query_entity_id"):
            raise RuntimeError(f"{variant}: query mismatch at row {i}")
        if get_gold(m) != get_gold(r):
            raise RuntimeError(f"{variant}: gold mismatch at row {i}")


def summarize_variant(
    variant: str,
    rows: List[Dict[str, Any]],
    main_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    validate_alignment(main_rows, rows, variant)

    ranks = [reviewer_safe_rank(r) for r in rows]
    rrs = [rr_at20_from_rank(r) for r in ranks]

    sub_sizes = [safe_len(extract_subgraph(r)) for r in rows]
    covs = [coverage_stats(r) for r in rows]

    same_candidate_order = []
    same_gold_rank = []
    abs_gold_rank_delta = []
    graph_jaccards = []
    support_shifts = []

    for r, m in zip(rows, main_rows):
        same_candidate_order.append(
            r.get("rank_entities_id") == m.get("rank_entities_id")
        )

        rank_r = reviewer_safe_rank(r)
        rank_m = reviewer_safe_rank(m)
        same_gold_rank.append(rank_r == rank_m)
        abs_gold_rank_delta.append(abs(rank_r - rank_m))

        graph_jaccards.append(graph_jaccard(extract_subgraph(r), extract_subgraph(m)))

        shift = support_score_shift(r, m)
        if shift is not None:
            support_shifts.append(shift)

    summary = {
        "variant": variant,
        "num_rows": len(rows),
        "gold_at20": round6(mean([1.0 if x <= 20 else 0.0 for x in ranks])),
        "candidate_mrr_at20": round6(mean(rrs)),
        "hits1_at20": round6(mean([1.0 if x <= 1 else 0.0 for x in ranks])),
        "hits3_at20": round6(mean([1.0 if x <= 3 else 0.0 for x in ranks])),
        "hits10_at20": round6(mean([1.0 if x <= 10 else 0.0 for x in ranks])),
        "rank21_count": int(sum(1 for x in ranks if x == 21)),
        "avg_gold_rank_with_21": round6(mean([float(x) for x in ranks])),
        "avg_subgraph_size": round6(mean([float(x) for x in sub_sizes])),
        "min_subgraph_size": int(min(sub_sizes)) if sub_sizes else None,
        "max_subgraph_size": int(max(sub_sizes)) if sub_sizes else None,
        "query_coverage_rate": round6(mean([c["query_covered"] for c in covs])),
        "candidate_coverage_preserved_rate": round6(mean([c["candidate_coverage"] for c in covs])),
        "same_candidate_order_rate_vs_main": round6(mean([1.0 if x else 0.0 for x in same_candidate_order])),
        "same_gold_rank_rate_vs_main": round6(mean([1.0 if x else 0.0 for x in same_gold_rank])),
        "avg_abs_gold_rank_delta_vs_main": round6(mean([float(x) for x in abs_gold_rank_delta])),
        "avg_graph_jaccard_vs_main": round6(mean([float(x) for x in graph_jaccards])),
        "avg_support_score_shift_vs_main": (
            round6(mean([float(x) for x in support_shifts]))
            if support_shifts
            else None
        ),
    }

    return summary


def collect_case_samples(
    rows_by_variant: Dict[str, List[Dict[str, Any]]],
    max_cases: int = 10,
) -> Dict[str, Any]:
    main_rows = rows_by_variant["main_rules"]
    out: Dict[str, Any] = {}

    for variant, rows in rows_by_variant.items():
        if variant == "main_rules":
            continue

        cases = []
        for idx, (r, m) in enumerate(zip(rows, main_rows)):
            rank_r = reviewer_safe_rank(r)
            rank_m = reviewer_safe_rank(m)

            sg_r = extract_subgraph(r)
            sg_m = extract_subgraph(m)

            cases.append({
                "row_index": idx,
                "query_entity": r.get("query_entity"),
                "gold_entity": get_gold(r),
                "rank_main": rank_m,
                "rank_variant": rank_r,
                "rank_delta_variant_minus_main": rank_r - rank_m,
                "subgraph_size_main": safe_len(sg_m),
                "subgraph_size_variant": safe_len(sg_r),
                "subgraph_size_delta_variant_minus_main": safe_len(sg_r) - safe_len(sg_m),
                "graph_jaccard_vs_main": round6(graph_jaccard(sg_r, sg_m)),
                "candidate_order_same": r.get("rank_entities_id") == m.get("rank_entities_id"),
                "top5_variant": get_rank_entities(r)[:5],
                "top5_main": get_rank_entities(m)[:5],
            })

        # Prioritize cases where graph package changed most.
        cases_sorted = sorted(
            cases,
            key=lambda x: (x["graph_jaccard_vs_main"], -abs(x["subgraph_size_delta_variant_minus_main"])),
        )

        out[variant] = cases_sorted[:max_cases]

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

    lines.append("# Rule Sensitivity Valid Analysis\n")
    lines.append(f"- Decision: **{result['decision']}**")
    lines.append(f"- Created at: `{result['created_at']}`")
    lines.append("- Split: **valid**")
    lines.append("- Main row: **main_rules / retrieval_main**")
    lines.append("")

    lines.append("## Main valid table\n")

    rows = []
    for s in result["valid_summary"]:
        rows.append([
            s["variant"],
            s["gold_at20"],
            s["candidate_mrr_at20"],
            s["hits1_at20"],
            s["hits3_at20"],
            s["hits10_at20"],
            s["rank21_count"],
            s["avg_subgraph_size"],
            s["candidate_coverage_preserved_rate"],
            s["query_coverage_rate"],
            s["same_candidate_order_rate_vs_main"],
            s["avg_graph_jaccard_vs_main"],
        ])

    lines.append(md_table(
        [
            "Variant",
            "Gold@20",
            "Cand MRR@20",
            "H@1",
            "H@3",
            "H@10",
            "Rank21",
            "Avg graph",
            "Cand coverage",
            "Query coverage",
            "Same cand order",
            "Graph Jaccard vs main",
        ],
        rows,
    ))
    lines.append("")

    lines.append("## Interpretation\n")
    lines.append(
        "- If candidate MRR is identical across variants, this is expected because Day 2 keeps "
        "candidate ordering fixed or nearly fixed and mainly changes the graph evidence package."
    )
    lines.append(
        "- `no_rules` measures the effect of removing the Week15/16 confidence-aware fuzzy "
        "rule-selection layer and falling back to the soft-support source graph."
    )
    lines.append(
        "- `random_rules` is a negative control: it preserves query/candidate coverage as much as possible "
        "but breaks confidence-aware edge selection."
    )
    lines.append(
        "- Day 3 E2E inference will test whether these graph-package changes affect LLM predictions."
    )
    lines.append("")

    lines.append("## Case samples with strongest graph differences\n")
    for variant, cases in result["case_samples"].items():
        lines.append(f"### {variant}")
        case_rows = []
        for c in cases[:10]:
            case_rows.append([
                c["row_index"],
                c["query_entity"],
                c["gold_entity"],
                c["rank_main"],
                c["rank_variant"],
                c["subgraph_size_main"],
                c["subgraph_size_variant"],
                c["graph_jaccard_vs_main"],
                ", ".join(c["top5_variant"][:3]),
            ])
        lines.append(md_table(
            [
                "idx",
                "query",
                "gold",
                "rank main",
                "rank var",
                "graph main",
                "graph var",
                "Jaccard",
                "top3 variant",
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

    rows_by_variant = {}
    fatal_errors = []

    for variant in VARIANTS:
        p = VARIANT_ROOT / variant / "valid.json"
        if not p.exists():
            fatal_errors.append(f"Missing valid package: {rel(p)}")
            continue
        rows = load_json(p)
        rows_by_variant[variant] = rows

    if fatal_errors:
        result = {
            "week": 25,
            "day": 2,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "decision": "RULE_SENSITIVITY_VALID_BLOCKED",
            "fatal_errors": fatal_errors,
        }
        write_json(result, RESULTS_DIR / "rule_sensitivity_valid.json")
        print("decision =", result["decision"])
        print("fatal_errors =", fatal_errors)
        return

    main_rows = rows_by_variant["main_rules"]

    summaries = []
    try:
        for variant in VARIANTS:
            summaries.append(summarize_variant(variant, rows_by_variant[variant], main_rows))
    except Exception as e:
        result = {
            "week": 25,
            "day": 2,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "decision": "RULE_SENSITIVITY_VALID_BLOCKED",
            "fatal_errors": [repr(e)],
        }
        write_json(result, RESULTS_DIR / "rule_sensitivity_valid.json")
        print("decision =", result["decision"])
        print("fatal_errors =", result["fatal_errors"])
        return

    case_samples = collect_case_samples(rows_by_variant)

    result = {
        "week": 25,
        "day": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": "RULE_SENSITIVITY_VALID_READY",
        "split": "valid",
        "variant_root": rel(VARIANT_ROOT),
        "valid_summary": summaries,
        "case_samples": case_samples,
        "fatal_errors": [],
        "notes": [
            "Candidate metrics may be unchanged because rule sensitivity variants mainly change graph/subgraph packages.",
            "E2E sensitivity is evaluated on Day 3.",
            "Reviewer-safe RR@20 is used: RR=1/rank if rank<=20 else 0.",
        ],
    }

    out_json = RESULTS_DIR / "rule_sensitivity_valid.json"
    out_report = REPORTS_DIR / "day2_rule_sensitivity_valid.md"

    write_json(result, out_json)
    write_report(result, out_report)

    print("=" * 100)
    print("decision =", result["decision"])
    print("valid_json =", rel(out_json))
    print("valid_report =", rel(out_report))
    print("=" * 100)

    for s in summaries:
        print(
            s["variant"],
            "Gold@20 =", s["gold_at20"],
            "MRR@20 =", s["candidate_mrr_at20"],
            "H@10 =", s["hits10_at20"],
            "avg_graph =", s["avg_subgraph_size"],
            "cand_cov =", s["candidate_coverage_preserved_rate"],
            "jaccard_vs_main =", s["avg_graph_jaccard_vs_main"],
        )


if __name__ == "__main__":
    main()