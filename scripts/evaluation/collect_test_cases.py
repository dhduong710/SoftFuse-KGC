from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(".").resolve()

BACKBONE = ROOT / "dataset/setting_b/eval_test/test_backbone_raw_eval.json"
ONTOLOGY = ROOT / "dataset/setting_b/eval_test/test_ontology_raw_eval.json"
SOFT = ROOT / "dataset/setting_b/eval_test/test_soft_support_raw_eval.json"
RETR = ROOT / "dataset/setting_b/eval_test/test_retrieval_main_eval.json"
RETR_ARTIFACT = ROOT / "dataset/setting_a/fuzzy_retrieval/test_fuzzy_retrieval_main.json"

OUT_JSON = ROOT / "outputs/evaluation/test_case_shortlist.json"
OUT_MD = ROOT / "outputs/evaluation/reports/test_case_review.md"

MAX_CASES_PER_BUCKET = 10

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def key_of(row: Dict[str, Any]) -> Tuple[int, int]:
    return int(row["query_entity_id"]), int(row["gold_entity_id"])

def core_case(
    b: Dict[str, Any],
    o: Dict[str, Any],
    s: Dict[str, Any],
    r: Dict[str, Any],
    ra: Dict[str, Any],
) -> Dict[str, Any]:
    sub = ra.get("subgraph_summary", {})
    return {
        "row_index": r["row_index"],
        "query_entity": r["query_entity"],
        "query_entity_id": r["query_entity_id"],
        "gold_entity": r["gold_entity"],
        "gold_entity_id": r["gold_entity_id"],
        "backbone_gold_present": b["gold_present"],
        "backbone_gold_rank": b["gold_rank"],
        "ontology_gold_present": o["gold_present"],
        "ontology_gold_rank": o["gold_rank"],
        "soft_gold_present": s["gold_present"],
        "soft_gold_rank": s["gold_rank"],
        "retrieval_gold_present": r["gold_present"],
        "retrieval_gold_rank": r["gold_rank"],
        "backbone_top1": b["top1_candidate"],
        "ontology_top1": o["top1_candidate"],
        "soft_top1": s["top1_candidate"],
        "retrieval_top1": r["top1_candidate"],
        "soft_top5": s["top5_candidates"],
        "retrieval_top5": r["top5_candidates"],
        "selected_source_variant": r["stage_specific"].get("selected_source_variant"),
        "avg_triple_score": r["stage_specific"].get("avg_triple_score"),
        "direct_shortcut_path_rate": r["stage_specific"].get("direct_shortcut_path_rate"),
        "contradiction_path_rate": r["stage_specific"].get("contradiction_path_rate"),
        "original_subgraph_size": sub.get("original_subgraph_size"),
        "selected_subgraph_size": sub.get("selected_subgraph_size"),
        "num_original_direct_shortcuts": sub.get("num_original_direct_shortcuts"),
        "num_selected_direct_shortcuts": sub.get("num_selected_direct_shortcuts"),
        "candidate_coverage_preserved_rate": sub.get("candidate_coverage_preserved_rate"),
        "top_band_coverage_preserved_rate": sub.get("top_band_coverage_preserved_rate"),
        "subgraph_shrink": (
            sub.get("original_subgraph_size", 0) - sub.get("selected_subgraph_size", 0)
            if sub.get("original_subgraph_size") is not None and sub.get("selected_subgraph_size") is not None
            else None
        ),
        "direct_shortcut_reduction": (
            sub.get("num_original_direct_shortcuts", 0) - sub.get("num_selected_direct_shortcuts", 0)
            if sub.get("num_original_direct_shortcuts") is not None and sub.get("num_selected_direct_shortcuts") is not None
            else None
        ),
    }

def main():
    backbone = load_json(BACKBONE)
    ontology = load_json(ONTOLOGY)
    soft = load_json(SOFT)
    retrieval = load_json(RETR)
    retr_artifact = load_json(RETR_ARTIFACT)

    assert len(backbone) == len(ontology) == len(soft) == len(retrieval) == len(retr_artifact) == 500

    b_map = {key_of(x): x for x in backbone}
    o_map = {key_of(x): x for x in ontology}
    s_map = {key_of(x): x for x in soft}
    r_map = {key_of(x): x for x in retrieval}
    ra_map = {(int(x["query_entity_id"]), int(x["gold_entity_id"])): x for x in retr_artifact}

    bucket1 = []
    bucket2 = []
    bucket3 = []

    for key in b_map:
        b = b_map[key]
        o = o_map[key]
        s = s_map[key]
        r = r_map[key]
        ra = ra_map[key]

        cc = core_case(b, o, s, r, ra)

        # Bucket 1: backbone -> retrieval improved
        if int(r["gold_rank"]) < int(b["gold_rank"]):
            cc1 = dict(cc)
            cc1["delta_backbone_to_retrieval"] = int(b["gold_rank"]) - int(r["gold_rank"])
            bucket1.append(cc1)

        # Bucket 2: ontology failure -> retrieval success
        if (not o["gold_present"]) and bool(r["gold_present"]):
            cc2 = dict(cc)
            cc2["delta_ontology_to_retrieval"] = int(o["gold_rank"]) - int(r["gold_rank"])
            bucket2.append(cc2)

        # Bucket 3: same rank, cleaner graph
        same_rank = int(r["gold_rank"]) == int(s["gold_rank"])
        shrink = cc["subgraph_shrink"]
        shortcut_red = cc["direct_shortcut_reduction"]
        cov = cc["candidate_coverage_preserved_rate"]

        if (
            same_rank
            and shrink is not None and shrink > 0
            and shortcut_red is not None and shortcut_red > 0
            and cov == 1.0
        ):
            cc3 = dict(cc)
            cc3["same_rank_vs_soft"] = True
            bucket3.append(cc3)

    bucket1 = sorted(
        bucket1,
        key=lambda x: (
            x["delta_backbone_to_retrieval"],
            int(bool(x["retrieval_gold_present"])),
            -int(x["retrieval_gold_rank"]),
        ),
        reverse=True,
    )[:MAX_CASES_PER_BUCKET]

    bucket2 = sorted(
        bucket2,
        key=lambda x: (
            x["delta_ontology_to_retrieval"],
            -int(x["retrieval_gold_rank"]),
        ),
        reverse=True,
    )[:MAX_CASES_PER_BUCKET]

    bucket3 = sorted(
        bucket3,
        key=lambda x: (
            x["subgraph_shrink"] if x["subgraph_shrink"] is not None else -1,
            x["direct_shortcut_reduction"] if x["direct_shortcut_reduction"] is not None else -1,
        ),
        reverse=True,
    )[:MAX_CASES_PER_BUCKET]

    shortlist = {
        "stage": "test_case_review_shortlist",
        "status": "BUILT",
        "bucket_counts": {
            "backbone_to_retrieval_improved_available": len([
                1 for key in b_map if int(r_map[key]["gold_rank"]) < int(b_map[key]["gold_rank"])
            ]),
            "ontology_failure_retrieval_success_available": len([
                1 for key in o_map if (not o_map[key]["gold_present"]) and bool(r_map[key]["gold_present"])
            ]),
            "same_rank_cleaner_graph_available": len([
                1 for key in s_map
                if (
                    int(r_map[key]["gold_rank"]) == int(s_map[key]["gold_rank"])
                    and (ra_map[key].get("subgraph_summary", {}).get("original_subgraph_size", 0)
                         - ra_map[key].get("subgraph_summary", {}).get("selected_subgraph_size", 0)) > 0
                    and (ra_map[key].get("subgraph_summary", {}).get("num_original_direct_shortcuts", 0)
                         - ra_map[key].get("subgraph_summary", {}).get("num_selected_direct_shortcuts", 0)) > 0
                    and ra_map[key].get("subgraph_summary", {}).get("candidate_coverage_preserved_rate", 0.0) == 1.0
                )
            ]),
        },
        "shortlist": {
            "backbone_to_retrieval_improved": bucket1,
            "ontology_failure_retrieval_success": bucket2,
            "same_rank_cleaner_graph": bucket3,
        },
    }

    save_json(OUT_JSON, shortlist)

    md = []
    md.append("# Test Case Review Shortlist")
    md.append("")
    md.append(f"- status: **{shortlist['status']}**")
    md.append("")
    md.append("## 1. Bucket counts")
    for k, v in shortlist["bucket_counts"].items():
        md.append(f"- {k}: `{v}`")
    md.append("")
    md.append("## 2. backbone_to_retrieval_improved")
    for x in bucket1[:5]:
        md.append(
            f"- row={x['row_index']} | query=`{x['query_entity']}` | gold=`{x['gold_entity']}` | "
            f"backbone_rank={x['backbone_gold_rank']} -> retrieval_rank={x['retrieval_gold_rank']}"
        )
    md.append("")
    md.append("## 3. ontology_failure_retrieval_success")
    for x in bucket2[:5]:
        md.append(
            f"- row={x['row_index']} | query=`{x['query_entity']}` | gold=`{x['gold_entity']}` | "
            f"ontology_rank={x['ontology_gold_rank']} -> retrieval_rank={x['retrieval_gold_rank']}"
        )
    md.append("")
    md.append("## 4. same_rank_cleaner_graph")
    for x in bucket3[:5]:
        md.append(
            f"- row={x['row_index']} | query=`{x['query_entity']}` | gold=`{x['gold_entity']}` | "
            f"soft_rank={x['soft_gold_rank']} = retrieval_rank={x['retrieval_gold_rank']} | "
            f"subgraph_shrink={x['subgraph_shrink']} | shortcut_reduction={x['direct_shortcut_reduction']}"
        )
    md.append("")
    md.append("## 5. Conclusion")
    md.append(
        "Built test-side shortlist for interpretation. The shortlist separates ranking improvement cases, ontology failure recovery cases, "
        "and same-rank cleaner-graph cases."
    )

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps(shortlist, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
