from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(".").resolve()

# ---------- shared train source ----------
# If this file is missing, rebuild the backbone-ready PrimeKG package first.
# The E2E packages reuse this train split and replace valid/test candidate order
# with backbone, soft-support, or fuzzy-retrieval variants.
TRAIN_PLACEHOLDER = ROOT / "dataset/setting_a/backbone_ready/train.json"

# ---------- canonical aligned evidence base ----------
ALIGNED = {
    "valid": ROOT / "dataset/setting_a/aligned_evidence/valid_aligned_evidence.json",
    "test": ROOT / "dataset/setting_a/aligned_evidence/test_aligned_evidence.json",
}

# ---------- candidate-order sources ----------
RAW = {
    "valid": ROOT / "dataset/setting_a/backbone_candidates/valid_top20_raw.json",
    "test": ROOT / "dataset/setting_a/backbone_candidates/test_top20_raw.json",
}
SOFT = {
    "valid": ROOT / "dataset/setting_a/soft_support_ranked_candidates/valid_top20_soft_support_main.json",
    "test": ROOT / "dataset/setting_a/soft_support_ranked_candidates/test_top20_soft_support_main.json",
}
RETR = {
    "valid": ROOT / "dataset/setting_a/fuzzy_retrieval/valid_fuzzy_retrieval_main.json",
    "test": ROOT / "dataset/setting_a/fuzzy_retrieval/test_fuzzy_retrieval_main.json",
}

OUT_ROOT = ROOT / "dataset/setting_a/e2e_infer_ready"
ROW_DIRS = {
    "backbone_raw": OUT_ROOT / "backbone_raw",
    "soft_support_raw": OUT_ROOT / "soft_support_raw",
    "retrieval_main": OUT_ROOT / "retrieval_main",
}

SUMMARY_PATH = ROOT / "outputs/e2e/infer_ready_build_summary.json"
REPORT_PATH = ROOT / "outputs/e2e/reports/build_infer_ready.md"

QUESTION_RE = re.compile(r"\n\nQuestion:\s*(.*?)\nAnswer:\s*$", re.DOTALL)

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def require_paths(paths: List[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required E2E inputs:\n"
            + "\n".join(f"- {path}" for path in missing)
            + "\nRebuild the upstream data steps before running this script."
        )

def key_of(row: Dict[str, Any]) -> Tuple[int, int]:
    return int(row["query_entity_id"]), int(row["gold_entity_id"])

def extract_question(input_text: str, query_entity: str) -> str:
    m = QUESTION_RE.search(input_text)
    if m:
        return m.group(1).strip()
    return f"What is related to {query_entity}?"

def build_prompt(query_entity: str, candidate_entities: List[str], question_text: str) -> str:
    answer_options = "(" + ", ".join([f"'{name}'" for name in candidate_entities]) + ")"
    refer_parts = [f"'{query_entity}': [QUERY]"] + [f"'{name}': [ENTITY]" for name in candidate_entities]
    refer_str = ", ".join(refer_parts)

    return (
        "You are a biomedical scientist. The task is to predict the answer based on the given question, "
        "and you only need to answer one entity. The answer must be in "
        + answer_options
        + ".\nYou can refer to the entity embeddings: "
        + refer_str
        + ".\n\nQuestion: "
        + question_text
        + "\nAnswer: "
    )

def compute_rank(candidate_entities: List[str], gold_entity: str) -> int:
    return candidate_entities.index(gold_entity) + 1 if gold_entity in candidate_entities else 21

def build_backbone_or_soft_split(
    split: str,
    row_name: str,
    order_rows: List[Dict[str, Any]],
    aligned_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aligned_map = {key_of(x): x for x in aligned_rows}
    out = []

    for i, order_row in enumerate(order_rows):
        key = key_of(order_row)
        base = aligned_map[key]

        candidate_entities = order_row["candidate_entities"]
        candidate_entity_ids = order_row["candidate_entity_ids"]
        gold_entity = order_row["gold_entity"]
        question_text = extract_question(base["input"], order_row["query_entity"])

        row = dict(base)
        row["split"] = split
        row["infer_row_name"] = row_name
        row["query_entity"] = order_row["query_entity"]
        row["query_entity_id"] = order_row["query_entity_id"]
        row["gold_entity"] = order_row["gold_entity"]
        row["gold_entity_id"] = order_row["gold_entity_id"]
        row["rank_entities"] = candidate_entities
        row["rank_entities_id"] = candidate_entity_ids
        row["rank"] = compute_rank(candidate_entities, gold_entity)
        row["input"] = build_prompt(order_row["query_entity"], candidate_entities, question_text)
        row["output"] = gold_entity
        row["candidate_entities"] = candidate_entities
        row["candidate_entity_ids"] = candidate_entity_ids
        row["question_text"] = question_text

        if row_name == "backbone_raw":
            row["source_variant"] = "backbone_raw"
            row["gold_in_topk_raw"] = order_row.get("gold_in_topk_raw")
            row["gold_rank_in_full_universe"] = order_row.get("gold_rank_in_full_universe")
            # subgraph stays canonical aligned-evidence subgraph
        else:
            row["variant_name"] = order_row.get("variant_name")
            row["support_scores"] = order_row.get("support_scores", [])
            row["support_rank_order"] = order_row.get("support_rank_order", [])
            row["candidate_debug_rows"] = order_row.get("candidate_debug_rows", [])
            # subgraph also stays canonical aligned-evidence subgraph

        out.append(row)

    return out

def build_retrieval_split(
    split: str,
    retr_rows: List[Dict[str, Any]],
    aligned_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    aligned_map = {key_of(x): x for x in aligned_rows}
    out = []

    for i, retr_row in enumerate(retr_rows):
        key = key_of(retr_row)
        base = aligned_map[key]

        candidate_entities = retr_row["candidate_entities"]
        candidate_entity_ids = retr_row["candidate_entity_ids"]
        gold_entity = retr_row["gold_entity"]
        question_text = extract_question(base["input"], retr_row["query_entity"])

        row = dict(base)
        row["split"] = split
        row["infer_row_name"] = "retrieval_main"
        row["query_entity"] = retr_row["query_entity"]
        row["query_entity_id"] = retr_row["query_entity_id"]
        row["gold_entity"] = retr_row["gold_entity"]
        row["gold_entity_id"] = retr_row["gold_entity_id"]
        row["rank_entities"] = candidate_entities
        row["rank_entities_id"] = candidate_entity_ids
        row["rank"] = compute_rank(candidate_entities, gold_entity)
        row["input"] = build_prompt(retr_row["query_entity"], candidate_entities, question_text)
        row["output"] = gold_entity
        row["candidate_entities"] = candidate_entities
        row["candidate_entity_ids"] = candidate_entity_ids
        row["question_text"] = question_text

        row["variant_name"] = retr_row.get("variant_name")
        row["selected_source_variant"] = retr_row.get("selected_source_variant")
        row["support_scores"] = retr_row.get("support_scores", [])
        row["candidate_support_bands"] = retr_row.get("candidate_support_bands", [])
        row["contra_flags"] = retr_row.get("contra_flags", [])
        row["subgraph"] = retr_row.get("selected_subgraph", [])
        row["selected_subgraph"] = retr_row.get("selected_subgraph", [])
        row["triple_score_rows"] = retr_row.get("triple_score_rows", [])
        row["path_scores"] = retr_row.get("path_scores", [])
        row["subgraph_summary"] = retr_row.get("subgraph_summary", [])

        out.append(row)

    return out

def write_package(row_name: str, train_rows: List[Dict[str, Any]], valid_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]]):
    out_dir = ROW_DIRS[row_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "train.json", train_rows)
    save_json(out_dir / "valid.json", valid_rows)
    save_json(out_dir / "test.json", test_rows)

    manifest = {
        "row_name": row_name,
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "contract_fields_required": ["input", "output", "query_entity_id", "rank_entities_id", "subgraph"],
        "sample_test_top_keys": list(test_rows[0].keys())[:30] if test_rows else [],
    }
    save_json(out_dir / "prep_manifest.json", manifest)
    return manifest

def main():
    require_paths(
        [
            TRAIN_PLACEHOLDER,
            ALIGNED["valid"],
            ALIGNED["test"],
            RAW["valid"],
            RAW["test"],
            SOFT["valid"],
            SOFT["test"],
            RETR["valid"],
            RETR["test"],
        ]
    )

    train_placeholder_rows = load_json(TRAIN_PLACEHOLDER)

    aligned_valid = load_json(ALIGNED["valid"])
    aligned_test = load_json(ALIGNED["test"])

    raw_valid = load_json(RAW["valid"])
    raw_test = load_json(RAW["test"])

    soft_valid = load_json(SOFT["valid"])
    soft_test = load_json(SOFT["test"])

    retr_valid = load_json(RETR["valid"])
    retr_test = load_json(RETR["test"])

    # Build splits
    backbone_valid = build_backbone_or_soft_split("valid", "backbone_raw", raw_valid, aligned_valid)
    backbone_test = build_backbone_or_soft_split("test", "backbone_raw", raw_test, aligned_test)

    soft_valid_rows = build_backbone_or_soft_split("valid", "soft_support_raw", soft_valid, aligned_valid)
    soft_test_rows = build_backbone_or_soft_split("test", "soft_support_raw", soft_test, aligned_test)

    retr_valid_rows = build_retrieval_split("valid", retr_valid, aligned_valid)
    retr_test_rows = build_retrieval_split("test", retr_test, aligned_test)

    manifests = {}
    manifests["backbone_raw"] = write_package("backbone_raw", train_placeholder_rows, backbone_valid, backbone_test)
    manifests["soft_support_raw"] = write_package("soft_support_raw", train_placeholder_rows, soft_valid_rows, soft_test_rows)
    manifests["retrieval_main"] = write_package("retrieval_main", train_placeholder_rows, retr_valid_rows, retr_test_rows)

    def pkg_summary(row_name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "num_rows": len(rows),
            "sample_rank": rows[0]["rank"],
            "sample_num_candidates": len(rows[0]["rank_entities"]),
            "sample_subgraph_size": len(rows[0]["subgraph"]),
            "has_required_fields": all(
                k in rows[0] for k in ["input", "output", "query_entity_id", "rank_entities_id", "subgraph", "rank", "rank_entities"]
            ) if rows else False,
        }

    summary = {
        "stage": "build_infer_ready",
        "status": "BUILT",
        "checkpoint_dir": "outputs/e2e/e2e_primary_checkpoint/checkpoint-final",
        "model_name_or_path": "meta-llama/Llama-3.2-3B",
        "kge_embedding_path": "dataset/setting_a/backbone_ready/entity_embeddings_rgcn.pt",
        "packages": {
            "backbone_raw": {
                "path": str(ROW_DIRS["backbone_raw"]),
                "manifest": manifests["backbone_raw"],
                "valid_summary": pkg_summary("backbone_raw", backbone_valid),
                "test_summary": pkg_summary("backbone_raw", backbone_test),
            },
            "soft_support_raw": {
                "path": str(ROW_DIRS["soft_support_raw"]),
                "manifest": manifests["soft_support_raw"],
                "valid_summary": pkg_summary("soft_support_raw", soft_valid_rows),
                "test_summary": pkg_summary("soft_support_raw", soft_test_rows),
            },
            "retrieval_main": {
                "path": str(ROW_DIRS["retrieval_main"]),
                "manifest": manifests["retrieval_main"],
                "valid_summary": pkg_summary("retrieval_main", retr_valid_rows),
                "test_summary": pkg_summary("retrieval_main", retr_test_rows),
                "selected_source_variant_set_test": sorted(set(x.get("selected_source_variant") for x in retr_test_rows)),
            },
        },
    }

    save_json(SUMMARY_PATH, summary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("# Build Infer-Ready Packages")
    md.append("")
    md.append(f"- status: **{summary['status']}**")
    md.append(f"- checkpoint_dir: `{summary['checkpoint_dir']}`")
    md.append(f"- model_name_or_path: `{summary['model_name_or_path']}`")
    md.append(f"- kge_embedding_path: `{summary['kge_embedding_path']}`")
    md.append("")
    md.append("## 1. Package summaries")
    for row_name, pkg in summary["packages"].items():
        md.append(f"### {row_name}")
        md.append(f"- path: `{pkg['path']}`")
        md.append(f"- valid_summary: `{pkg['valid_summary']}`")
        md.append(f"- test_summary: `{pkg['test_summary']}`")
        if row_name == "retrieval_main":
            md.append(f"- selected_source_variant_set_test: `{pkg['selected_source_variant_set_test']}`")
        md.append("")
    md.append("## 2. Conclusion")
    md.append(
        "Built infer-ready packages for backbone_raw, soft_support_raw, and retrieval_main. "
        "Each package now contains train/valid/test JSONs that satisfy the current infer.py contract."
    )

    REPORT_PATH.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
