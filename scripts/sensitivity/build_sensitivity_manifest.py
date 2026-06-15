#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the sensitivity manifest used by the robustness scripts.

This script does not run new experiments. It audits the selected E2E inputs and
writes:
- outputs/sensitivity/protocol/sensitivity_manifest.json
- outputs/sensitivity/reports/sensitivity_manifest_audit.md
"""

from __future__ import annotations

import json
import hashlib
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = ROOT / "outputs" / "sensitivity"
REPORTS_DIR = ROOT / "outputs" / "sensitivity" / "reports"
PROTOCOL_DIR = RESULTS_DIR / "protocol"

RULE_DIR = RESULTS_DIR / "rule_sensitivity"
TEMPLATE_DIR = RESULTS_DIR / "template_sensitivity"
NOISE_DIR = RESULTS_DIR / "noise_robustness"

E2E_READY_ROOT = ROOT / "dataset" / "setting_a" / "e2e_infer_ready"

E2E_ROWS = {
    "backbone_raw": E2E_READY_ROOT / "backbone_raw",
    "soft_support_raw": E2E_READY_ROOT / "soft_support_raw",
    "retrieval_main": E2E_READY_ROOT / "retrieval_main",
}

E2E_ARTIFACT_PATHS = {
    "selected_decode_test_dir": ROOT / "outputs" / "e2e" / "selected_decode_test",
    "model_compare_dir": ROOT / "outputs" / "e2e" / "model_compare",
}

CHECKPOINT_CANDIDATE_ROOTS = [
    ROOT / "outputs" / "e2e" / "e2e_primary_checkpoint" / "checkpoint-final",
    ROOT / "outputs" / "e2e" / "e2e_primary_checkpoint",
]


SELECTED_DECODING = {
    "config_name": "cfg01_mnt16_rp100_ng0",
    "selected_on": "valid_only",
    "max_new_tokens": 16,
    "min_new_tokens": 1,
    "do_sample": False,
    "num_beams": 1,
    "temperature": 1.0,
    "repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
}

E2E_REFERENCE = {
    "primekg_locked_test_llama3_2_3b": {
        "backbone_raw": {
            "gold_at20": 0.240,
            "candidate_mrr_at20": 0.064563,
            "e2e_mrr_at20": 0.047636,
            "hits3_at20": 0.048,
            "hits10_at20": 0.178,
            "pred_in_candidate_rate": 0.980,
            "invalid_rate": 0.020,
            "top1_copy_rate": 0.038,
            "avg_graph_size": 59.93,
        },
        "soft_support_raw": {
            "gold_at20": 0.240,
            "candidate_mrr_at20": 0.125326,
            "e2e_mrr_at20": 0.074676,
            "hits3_at20": 0.132,
            "hits10_at20": 0.218,
            "pred_in_candidate_rate": 0.980,
            "invalid_rate": 0.020,
            "top1_copy_rate": 0.008,
            "avg_graph_size": 59.93,
        },
        "retrieval_main": {
            "gold_at20": 0.240,
            "candidate_mrr_at20": 0.125326,
            "e2e_mrr_at20": 0.074687,
            "hits3_at20": 0.132,
            "hits10_at20": 0.218,
            "pred_in_candidate_rate": 0.998,
            "invalid_rate": 0.002,
            "top1_copy_rate": 0.010,
            "avg_graph_size": 32.34,
        },
    }
}

REQUIRED_FIELDS_COMMON = [
    "triple",
    "triple_id",
    "type",
    "query_entity",
    "query_entity_id",
    "rank_entities",
    "rank_entities_id",
    "rank",
    "input",
    "output",
    "subgraph",
]

REQUIRED_FIELDS_EVAL = [
    "gold_entity",
    "gold_entity_id",
    "gold_in_topk_raw",
    "gold_rank_in_full_universe",
    "split",
    "candidate_entities",
    "candidate_entity_ids",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    for p in [
        RESULTS_DIR,
        REPORTS_DIR,
        PROTOCOL_DIR,
        RULE_DIR,
        TEMPLATE_DIR,
        NOISE_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, max_mb: int = 64) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        return f"SKIPPED_LARGE_FILE_{size_mb:.2f}MB"

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(float(statistics.mean(values)), 6)


def min_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(min(values))


def max_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(max(values))


def extract_question_from_input(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    marker = "Question:"
    if marker not in text:
        return None
    part = text.split(marker, 1)[1]
    if "Answer:" in part:
        part = part.split("Answer:", 1)[0]
    return part.strip()


def summarize_json_rows(path: Path, split: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "path": rel(path),
        "exists": path.exists(),
        "is_file": path.is_file() if path.exists() else False,
        "num_rows": None,
        "top_keys": [],
        "sha256": None,
        "required_missing_field_rows": None,
        "bad_candidate_len_rows": None,
        "bad_rank_id_len_rows": None,
        "bad_subgraph_rows": None,
        "prompt_missing_query_token_rows": None,
        "prompt_missing_entity_token_rows": None,
        "prompt_missing_question_rows": None,
        "prompt_missing_answer_rows": None,
        "candidate_len": None,
        "subgraph_size": None,
        "sample_question": None,
        "sample_output": None,
    }

    if not path.exists() or not path.is_file():
        return out

    out["sha256"] = sha256_file(path)

    try:
        data = load_json(path)
    except Exception as e:
        out["load_error"] = repr(e)
        return out

    if not isinstance(data, list):
        out["json_kind"] = type(data).__name__
        out["load_error"] = "Expected a list of rows."
        return out

    out["json_kind"] = "list"
    out["num_rows"] = len(data)

    if not data:
        return out

    first = data[0]
    if isinstance(first, dict):
        out["top_keys"] = list(first.keys())
        out["sample_question"] = first.get("question_text") or extract_question_from_input(first.get("input", ""))
        out["sample_output"] = first.get("output")

    required = list(REQUIRED_FIELDS_COMMON)
    if split in {"valid", "test"}:
        required += REQUIRED_FIELDS_EVAL

    missing_required = 0
    bad_candidate_len = 0
    bad_rank_id_len = 0
    bad_subgraph = 0
    missing_query_token = 0
    missing_entity_token = 0
    missing_question = 0
    missing_answer = 0

    candidate_lens: List[int] = []
    rank_id_lens: List[int] = []
    subgraph_sizes: List[int] = []

    for row in data:
        if not isinstance(row, dict):
            missing_required += 1
            continue

        if any(k not in row for k in required):
            missing_required += 1

        rank_entities = row.get("rank_entities", [])
        rank_entities_id = row.get("rank_entities_id", [])
        subgraph = row.get("subgraph", [])
        input_text = row.get("input", "")

        if isinstance(rank_entities, list):
            candidate_lens.append(len(rank_entities))
            if split in {"valid", "test"} and len(rank_entities) != 20:
                bad_candidate_len += 1
        else:
            bad_candidate_len += 1

        if isinstance(rank_entities_id, list):
            rank_id_lens.append(len(rank_entities_id))
            if isinstance(rank_entities, list) and len(rank_entities_id) != len(rank_entities):
                bad_rank_id_len += 1
        else:
            bad_rank_id_len += 1

        if isinstance(subgraph, list):
            subgraph_sizes.append(len(subgraph))
            if len(subgraph) == 0:
                bad_subgraph += 1
        else:
            bad_subgraph += 1

        if not isinstance(input_text, str) or "[QUERY]" not in input_text:
            missing_query_token += 1
        if not isinstance(input_text, str) or "[ENTITY]" not in input_text:
            missing_entity_token += 1
        if not isinstance(input_text, str) or "Question:" not in input_text:
            missing_question += 1
        if not isinstance(input_text, str) or "Answer:" not in input_text:
            missing_answer += 1

    out["required_missing_field_rows"] = missing_required
    out["bad_candidate_len_rows"] = bad_candidate_len
    out["bad_rank_id_len_rows"] = bad_rank_id_len
    out["bad_subgraph_rows"] = bad_subgraph
    out["prompt_missing_query_token_rows"] = missing_query_token
    out["prompt_missing_entity_token_rows"] = missing_entity_token
    out["prompt_missing_question_rows"] = missing_question
    out["prompt_missing_answer_rows"] = missing_answer

    out["candidate_len"] = {
        "min": min_or_none(candidate_lens),
        "max": max_or_none(candidate_lens),
        "avg": mean_or_none(candidate_lens),
    }
    out["rank_id_len"] = {
        "min": min_or_none(rank_id_lens),
        "max": max_or_none(rank_id_lens),
        "avg": mean_or_none(rank_id_lens),
    }
    out["subgraph_size"] = {
        "min": min_or_none(subgraph_sizes),
        "max": max_or_none(subgraph_sizes),
        "avg": mean_or_none(subgraph_sizes),
    }

    return out


def audit_e2e_rows() -> Dict[str, Any]:
    audit: Dict[str, Any] = {}

    for row_name, row_dir in E2E_ROWS.items():
        row_out: Dict[str, Any] = {
            "row_dir": rel(row_dir),
            "exists": row_dir.exists(),
            "splits": {},
            "map_files": {},
            "prep_manifest": {},
        }

        for split in ["train", "valid", "test"]:
            row_out["splits"][split] = summarize_json_rows(row_dir / f"{split}.json", split)

        for fname in ["entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl"]:
            p = row_dir / fname
            row_out["map_files"][fname] = {
                "path": rel(p),
                "exists": p.exists(),
                "size_bytes": p.stat().st_size if p.exists() and p.is_file() else None,
            }

        manifest_path = row_dir / "prep_manifest.json"
        row_out["prep_manifest"] = {
            "path": rel(manifest_path),
            "exists": manifest_path.exists(),
        }
        if manifest_path.exists():
            try:
                manifest = load_json(manifest_path)
                row_out["prep_manifest"]["top_keys"] = list(manifest.keys()) if isinstance(manifest, dict) else []
            except Exception as e:
                row_out["prep_manifest"]["load_error"] = repr(e)

        audit[row_name] = row_out

    return audit


def find_checkpoint() -> Dict[str, Any]:
    candidates = list(CHECKPOINT_CANDIDATE_ROOTS)

    checkpoint_root = ROOT / "outputs" / "e2e" / "e2e_primary_checkpoint"
    if checkpoint_root.exists():
        candidates.extend(sorted(checkpoint_root.glob("checkpoint-*")))

    seen = set()
    unique_candidates = []
    for p in candidates:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            seen.add(rp)
            unique_candidates.append(p)

    checked = []
    selected = None

    for p in unique_candidates:
        adapter_config = p / "adapter_config.json"
        graph_model = p / "graph_model.bin"
        adapter_bin = p / "adapter_model.bin"
        adapter_safetensors = p / "adapter_model.safetensors"

        info = {
            "path": rel(p),
            "exists": p.exists(),
            "adapter_config": adapter_config.exists(),
            "graph_model": graph_model.exists(),
            "adapter_model_bin": adapter_bin.exists(),
            "adapter_model_safetensors": adapter_safetensors.exists(),
            "is_valid_checkpoint": False,
        }

        has_adapter_weights = adapter_bin.exists() or adapter_safetensors.exists()
        info["is_valid_checkpoint"] = bool(
            p.exists() and adapter_config.exists() and graph_model.exists() and has_adapter_weights
        )

        checked.append(info)

        if info["is_valid_checkpoint"] and selected is None:
            selected = info

    return {
        "selected_checkpoint": selected,
        "checked_candidates": checked,
    }


def audit_e2e_artifact_paths() -> Dict[str, Any]:
    out = {}
    for name, path in E2E_ARTIFACT_PATHS.items():
        out[name] = {
            "path": rel(path),
            "exists": path.exists(),
            "is_file": path.is_file() if path.exists() else False,
            "is_dir": path.is_dir() if path.exists() else False,
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        }

        if path.exists() and path.is_file() and path.suffix == ".json":
            try:
                data = load_json(path)
                out[name]["top_keys"] = list(data.keys()) if isinstance(data, dict) else []
            except Exception as e:
                out[name]["load_error"] = repr(e)

    return out


def build_sensitivity_groups() -> Dict[str, Any]:
    return {
        "rule_sensitivity": {
            "role": "appendix sensitivity / negative-control analysis",
            "main_row": "retrieval_main",
            "variants": [
                {
                    "name": "main_rules",
                    "description": "Current SoftFuse retrieval_main with selected E2E rules/support/retrieval behavior.",
                    "may_change_main_result": False,
                },
                {
                    "name": "no_rules",
                    "description": "Disable hard-coded rule contribution while keeping candidate source and evaluation policy fixed.",
                    "may_change_main_result": False,
                },
                {
                    "name": "random_rules",
                    "description": "Negative control: randomize/shuffle rule support mapping under comparable budget.",
                    "may_change_main_result": False,
                },
            ],
            "splits": ["valid", "test"],
            "metrics": [
                "Gold@20",
                "candidate_mrr_at20",
                "hits1_at20",
                "hits3_at20",
                "hits10_at20",
                "avg_subgraph_size",
                "candidate_coverage_preserved_rate",
                "same_candidate_order_rate_vs_main",
                "same_rank_rate_vs_main",
                "support_score_shift",
                "e2e_mrr_at20",
                "pred_in_candidate_rate",
                "invalid_rate",
                "top1_copy_rate",
            ],
        },
        "template_sensitivity": {
            "role": "appendix prompt wording sensitivity",
            "main_row": "retrieval_main",
            "variants": [
                {
                    "name": "T0_canonical",
                    "template": "What drug is indicated for {}?",
                },
                {
                    "name": "T1_treatment",
                    "template": "Which drug is used to treat {}?",
                },
                {
                    "name": "T2_medication",
                    "template": "Which medication is indicated for {}?",
                },
                {
                    "name": "T3_association_neutral",
                    "template": "What drug is therapeutically associated with {}?",
                },
            ],
            "splits": ["valid", "test"],
            "metrics": [
                "E2E MRR@20",
                "H@3",
                "H@10",
                "pred_in_candidate_rate",
                "invalid_rate",
                "list_fragment_rate",
                "top1_copy_rate",
            ],
        },
        "small_noise_robustness": {
            "role": "appendix robustness under small perturbations",
            "main_row": "retrieval_main",
            "variants": [
                {"name": "N0_no_noise"},
                {"name": "N1_support_score_noise_seed1"},
                {"name": "N2_support_score_noise_seed2"},
                {"name": "N3_support_score_noise_seed3"},
                {"name": "N4_subgraph_edge_dropout_5_seed1_optional"},
                {"name": "N5_subgraph_edge_dropout_5_seed2_optional"},
                {"name": "N6_subgraph_edge_dropout_5_seed3_optional"},
            ],
            "preferred_if_time_limited": [
                "N0_no_noise",
                "N1_support_score_noise_seed1",
                "N2_support_score_noise_seed2",
                "N3_support_score_noise_seed3",
            ],
            "metrics": [
                "candidate_mrr_at20",
                "rank_change_rate",
                "same_top1_rate",
                "avg_abs_rank_shift",
                "avg_subgraph_size",
                "coverage_preserved_rate",
            ],
        },
    }


def collect_fatal_errors(
    e2e_audit: Dict[str, Any],
    checkpoint_audit: Dict[str, Any],
    e2e_artifact_audit: Dict[str, Any],
) -> List[str]:
    errors: List[str] = []

    for row_name, row_info in e2e_audit.items():
        if not row_info["exists"]:
            errors.append(f"Missing E2E row directory: {row_info['row_dir']}")
            continue

        for split in ["train", "valid", "test"]:
            split_info = row_info["splits"][split]
            if not split_info["exists"]:
                errors.append(f"Missing {row_name}/{split}.json")
                continue

            if split_info.get("load_error"):
                errors.append(f"Cannot load {row_name}/{split}.json: {split_info['load_error']}")
                continue

            if split_info.get("required_missing_field_rows", 0) != 0:
                errors.append(
                    f"{row_name}/{split}.json has rows missing required fields: "
                    f"{split_info.get('required_missing_field_rows')}"
                )

            if split in {"valid", "test"} and split_info.get("bad_candidate_len_rows", 0) != 0:
                errors.append(
                    f"{row_name}/{split}.json has non-20 candidate rows: "
                    f"{split_info.get('bad_candidate_len_rows')}"
                )

            if split_info.get("prompt_missing_query_token_rows", 0) != 0:
                errors.append(
                    f"{row_name}/{split}.json has prompts missing [QUERY]: "
                    f"{split_info.get('prompt_missing_query_token_rows')}"
                )

            if split_info.get("prompt_missing_entity_token_rows", 0) != 0:
                errors.append(
                    f"{row_name}/{split}.json has prompts missing [ENTITY]: "
                    f"{split_info.get('prompt_missing_entity_token_rows')}"
                )

    if checkpoint_audit.get("selected_checkpoint") is None:
        errors.append(
            "No valid E2E checkpoint found with adapter_config + adapter weights + graph_model.bin."
        )

    return errors


def collect_warnings(
    e2e_audit: Dict[str, Any],
    e2e_artifact_audit: Dict[str, Any],
) -> List[str]:
    warnings: List[str] = []

    for name, info in e2e_artifact_audit.items():
        if not info.get("exists"):
            warnings.append(f"E2E artifact not found: {info.get('path')}")

    for row_name, row_info in e2e_audit.items():
        for split in ["train", "valid", "test"]:
            split_info = row_info["splits"][split]
            if split_info.get("bad_subgraph_rows", 0):
                warnings.append(
                    f"{row_name}/{split}.json has bad or empty subgraph rows: "
                    f"{split_info.get('bad_subgraph_rows')}"
                )
            if split_info.get("prompt_missing_question_rows", 0):
                warnings.append(
                    f"{row_name}/{split}.json has prompts missing Question marker: "
                    f"{split_info.get('prompt_missing_question_rows')}"
                )
            if split_info.get("prompt_missing_answer_rows", 0):
                warnings.append(
                    f"{row_name}/{split}.json has prompts missing Answer marker: "
                    f"{split_info.get('prompt_missing_answer_rows')}"
                )

    return warnings


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


def write_markdown_report(protocol: Dict[str, Any], out_path: Path) -> None:
    e2e_audit = protocol["input_audit"]["e2e_rows"]
    e2e_artifact_audit = protocol["input_audit"]["e2e_artifacts"]
    checkpoint = protocol["input_audit"]["checkpoint"]

    lines: List[str] = []

    lines.append("# Sensitivity Manifest Audit\n")
    lines.append(f"- Decision: **{protocol['decision']}**")
    lines.append(f"- Created at: `{protocol['created_at']}`")
    lines.append(f"- Primary dataset: **{protocol['primary_dataset']}**")
    lines.append(f"- Primary model: **{protocol['primary_model']}**")
    lines.append(f"- Main row: **{protocol['main_row']}**")
    lines.append(f"- Selected decoding config: **{protocol['selected_decoding']['config_name']}**")
    lines.append("")

    lines.append("## Selected decoding\n")
    lines.append("```json")
    lines.append(json.dumps(protocol["selected_decoding"], indent=2, ensure_ascii=False))
    lines.append("```\n")

    lines.append("## E2E reference table\n")
    ref_rows = []
    ref = protocol["e2e_reference"]["primekg_locked_test_llama3_2_3b"]
    for row_name, m in ref.items():
        ref_rows.append([
            row_name,
            m["gold_at20"],
            m["candidate_mrr_at20"],
            m["e2e_mrr_at20"],
            m["hits3_at20"],
            m["hits10_at20"],
            m["pred_in_candidate_rate"],
            m["invalid_rate"],
            m["top1_copy_rate"],
            m["avg_graph_size"],
        ])
    lines.append(md_table(
        [
            "Row",
            "Gold@20",
            "Cand MRR@20",
            "E2E MRR@20",
            "H@3",
            "H@10",
            "Pred-in-cand",
            "Invalid",
            "Top1-copy",
            "Avg graph",
        ],
        ref_rows,
    ))
    lines.append("")

    lines.append("## E2E-ready row audit\n")
    row_table = []
    for row_name, row_info in e2e_audit.items():
        for split in ["train", "valid", "test"]:
            s = row_info["splits"][split]
            row_table.append([
                row_name,
                split,
                s.get("exists"),
                s.get("num_rows"),
                s.get("candidate_len", {}).get("avg") if s.get("candidate_len") else None,
                s.get("subgraph_size", {}).get("avg") if s.get("subgraph_size") else None,
                s.get("required_missing_field_rows"),
                s.get("bad_candidate_len_rows"),
                s.get("bad_subgraph_rows"),
            ])
    lines.append(md_table(
        [
            "Row",
            "Split",
            "Exists",
            "Rows",
            "Avg cand len",
            "Avg subgraph",
            "Missing field rows",
            "Bad cand len rows",
            "Bad subgraph rows",
        ],
        row_table,
    ))
    lines.append("")

    lines.append("## Sample questions\n")
    sample_rows = []
    for row_name, row_info in e2e_audit.items():
        for split in ["valid", "test"]:
            s = row_info["splits"][split]
            sample_rows.append([
                row_name,
                split,
                s.get("sample_question"),
                s.get("sample_output"),
            ])
    lines.append(md_table(["Row", "Split", "Sample question", "Sample output"], sample_rows))
    lines.append("")

    lines.append("## Checkpoint audit\n")
    selected = checkpoint.get("selected_checkpoint")
    if selected:
        lines.append(f"- Selected checkpoint: `{selected['path']}`")
    else:
        lines.append("- Selected checkpoint: **MISSING**")
    lines.append("")
    checkpoint_rows = []
    for c in checkpoint.get("checked_candidates", []):
        checkpoint_rows.append([
            c["path"],
            c["exists"],
            c["adapter_config"],
            c["adapter_model_bin"] or c["adapter_model_safetensors"],
            c["graph_model"],
            c["is_valid_checkpoint"],
        ])
    lines.append(md_table(
        ["Path", "Exists", "Adapter config", "Adapter weights", "Graph model", "Valid"],
        checkpoint_rows,
    ))
    lines.append("")

    lines.append("## E2E artifact audit\n")
    artifact_rows = []
    for name, info in e2e_artifact_audit.items():
        artifact_rows.append([
            name,
            info["path"],
            info["exists"],
            "file" if info.get("is_file") else ("dir" if info.get("is_dir") else "missing"),
        ])
    lines.append(md_table(["Name", "Path", "Exists", "Kind"], artifact_rows))
    lines.append("")

    lines.append("## Sensitivity groups\n")
    for group_name, group in protocol["sensitivity_groups"].items():
        lines.append(f"### {group_name}")
        lines.append(f"- Role: {group['role']}")
        lines.append(f"- Main row: `{group['main_row']}`")
        variant_names = [v["name"] for v in group["variants"]]
        lines.append(f"- Variants: {', '.join(variant_names)}")
        lines.append("")

    lines.append("## Fatal errors\n")
    if protocol["fatal_errors"]:
        for e in protocol["fatal_errors"]:
            lines.append(f"- {e}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Warnings\n")
    if protocol["warnings"]:
        for w in protocol["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Final rule\n")
    lines.append(
        "Sensitivity outputs are supporting evidence only. "
        "They must not replace the locked main E2E result, and no test-based tuning is allowed."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    e2e_audit = audit_e2e_rows()
    checkpoint_audit = find_checkpoint()
    e2e_artifact_audit = audit_e2e_artifact_paths()

    fatal_errors = collect_fatal_errors(e2e_audit, checkpoint_audit, e2e_artifact_audit)
    warnings = collect_warnings(e2e_audit, e2e_artifact_audit)

    decision = (
        "SENSITIVITY_MANIFEST_READY"
        if len(fatal_errors) == 0
        else "SENSITIVITY_MANIFEST_BLOCKED"
    )

    protocol = {
        "title": "Sensitivity manifest audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "primary_dataset": "PrimeKG Setting A",
        "primary_task": "(?, indication, disease), drug-only head prediction",
        "primary_model": "Llama-3.2-3B",
        "main_row": "retrieval_main",
        "graph_num_rels": 4,
        "selected_decoding": SELECTED_DECODING,
        "metric_policy": {
            "reviewer_safe_rr_at20": "RR=1/rank if rank<=20 else 0",
            "rank21_policy": "rank21 is descriptive only; reciprocal rank is zero outside top-20",
            "raw_infer_py_mrr": "audit_only_not_reviewer_safe_metric",
            "gold_injection": False,
            "test_tuning": False,
        },
        "e2e_reference": E2E_REFERENCE,
        "sensitivity_groups": build_sensitivity_groups(),
        "input_audit": {
            "e2e_rows": e2e_audit,
            "checkpoint": checkpoint_audit,
            "e2e_artifacts": e2e_artifact_audit,
        },
        "fatal_errors": fatal_errors,
        "warnings": warnings,
        "do_not_change": [
            "Do not overwrite outputs/e2e artifacts.",
            "Do not change dataset/setting_a/e2e_infer_ready source packages.",
            "Do not tune decoding/template/rules on test.",
            "Do not promote sensitivity variants as a new contribution.",
            "Do not use raw infer.py MRR as the reviewer-safe metric.",
        ],
        "outputs": {
            "manifest_json": "outputs/sensitivity/protocol/sensitivity_manifest.json",
            "audit_report_md": "outputs/sensitivity/reports/sensitivity_manifest_audit.md",
        },
    }

    protocol_path = PROTOCOL_DIR / "sensitivity_manifest.json"
    report_path = REPORTS_DIR / "sensitivity_manifest_audit.md"

    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(protocol, report_path)

    print("=" * 100)
    print(f"decision = {decision}")
    print(f"manifest_json = {rel(protocol_path)}")
    print(f"audit_report_md = {rel(report_path)}")
    print("=" * 100)

    if fatal_errors:
        print("FATAL ERRORS:")
        for e in fatal_errors:
            print(f"- {e}")
    else:
        print("No fatal errors.")

    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"- {w}")

    print("=" * 100)


if __name__ == "__main__":
    main()
