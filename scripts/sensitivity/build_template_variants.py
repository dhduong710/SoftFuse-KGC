#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build question-template sensitivity variants.

Only the question wording is changed.
Candidate list, rank, output/gold, subgraph, and support fields are preserved.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[2]

PROTOCOL_PATH = ROOT / "outputs" / "sensitivity" / "protocol" / "sensitivity_manifest.json"

SRC_MAIN = ROOT / "dataset" / "setting_a" / "e2e_infer_ready" / "retrieval_main"
OUT_ROOT = ROOT / "dataset" / "setting_a" / "template_sensitivity"
RESULTS_DIR = ROOT / "outputs" / "sensitivity" / "template_sensitivity"
REPORTS_DIR = ROOT / "outputs" / "sensitivity" / "reports"

SPLITS = ["train", "valid", "test"]

TEMPLATES = {
    "T0_canonical": "What drug is indicated for {query_entity}?",
    "T1_treatment": "Which drug is used to treat {query_entity}?",
    "T2_medication": "Which medication is indicated for {query_entity}?",
    "T3_association_neutral": "What drug is therapeutically associated with {query_entity}?",
}


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


def replace_question_in_prompt(input_text: str, new_question: str) -> str:
    if not isinstance(input_text, str):
        raise ValueError("input_text is not a string")

    pattern = r"Question:\s*.*?\nAnswer:"
    repl = f"Question: {new_question}\nAnswer:"

    new_text, n = re.subn(pattern, repl, input_text, count=1, flags=re.DOTALL)
    if n == 1:
        return new_text

    # Fallback if Answer appears without newline.
    pattern2 = r"Question:\s*.*?Answer:"
    repl2 = f"Question: {new_question}\nAnswer:"
    new_text, n = re.subn(pattern2, repl2, input_text, count=1, flags=re.DOTALL)
    if n == 1:
        return new_text

    raise ValueError("Cannot find Question/Answer block in prompt")


def make_question(template: str, row: Dict[str, Any]) -> str:
    query_entity = row.get("query_entity")
    if query_entity is None:
        raise ValueError("Missing query_entity")
    return template.format(query_entity=query_entity)


def copy_sidecars(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    for p in src_dir.iterdir():
        if p.name in {"train.json", "valid.json", "test.json", "prep_manifest.json"}:
            continue
        if p.is_file():
            shutil.copy2(p, dst_dir / p.name)

    for fname in ["entity2id.pkl", "id2entity.pkl", "relation2id.pkl", "id2relation.pkl"]:
        src = src_dir / fname
        if src.exists() and src.is_file():
            shutil.copy2(src, dst_dir / fname)


def summarize_rows(rows: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
    cand_lens = [len(r.get("rank_entities", [])) for r in rows]
    sub_sizes = [len(r.get("subgraph", [])) for r in rows]

    return {
        "split": split,
        "num_rows": len(rows),
        "avg_candidate_len": round(sum(cand_lens) / len(cand_lens), 6) if cand_lens else None,
        "min_candidate_len": min(cand_lens) if cand_lens else None,
        "max_candidate_len": max(cand_lens) if cand_lens else None,
        "avg_subgraph_size": round(sum(sub_sizes) / len(sub_sizes), 6) if sub_sizes else None,
        "min_subgraph_size": min(sub_sizes) if sub_sizes else None,
        "max_subgraph_size": max(sub_sizes) if sub_sizes else None,
        "sample_question": rows[0].get("question_text") if rows else None,
        "sample_output": rows[0].get("output") if rows else None,
    }


def build_variant_rows(rows: List[Dict[str, Any]], split: str, variant: str, template: str) -> List[Dict[str, Any]]:
    out = []

    for row in rows:
        r = copy.deepcopy(row)
        q = make_question(template, r)

        r["question_text"] = q
        r["input"] = replace_question_in_prompt(r["input"], q)

        r["template_sensitivity_variant"] = variant
        r["template_sensitivity_string"] = template
        r["sensitivity_split"] = split
        r["infer_row_name"] = f"template_sensitivity_{variant}"

        out.append(r)

    return out


def main() -> None:
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(f"Missing sensitivity manifest: {rel(PROTOCOL_PATH)}")

    protocol = load_json(PROTOCOL_PATH)
    if protocol.get("decision") != "SENSITIVITY_MANIFEST_READY":
        raise RuntimeError(f"sensitivity manifest is not READY: {protocol.get('decision')}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "week": 25,
        "day": 4,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "decision": None,
        "source_row": rel(SRC_MAIN),
        "output_root": rel(OUT_ROOT),
        "main_row": "retrieval_main",
        "selected_decoding": "cfg01_mnt16_rp100_ng0",
        "primary_model": "Llama-3.2-3B",
        "gold_injection": False,
        "template_variants": {},
        "notes": [
            "Only question wording is changed.",
            "Candidate list/order, output/gold, rank, subgraph, and support fields are preserved.",
            "T0_canonical remains the main prompt.",
            "T3_association_neutral is robustness-only and should not redefine the PrimeKG indication task.",
        ],
    }

    fatal_errors = []

    for variant, template in TEMPLATES.items():
        variant_dir = OUT_ROOT / variant
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)

        copy_sidecars(SRC_MAIN, variant_dir)

        split_summaries = {}

        for split in SPLITS:
            src_path = SRC_MAIN / f"{split}.json"
            rows = load_json(src_path)
            built = build_variant_rows(rows, split, variant, template)

            out_path = variant_dir / f"{split}.json"
            write_json(built, out_path)

            split_summaries[split] = summarize_rows(built, split)

        v_manifest = {
            "variant": variant,
            "template": template,
            "definition": "Question-template sensitivity variant",
            "source_row": rel(SRC_MAIN),
            "splits": split_summaries,
            "may_replace_main_prompt": False,
        }
        write_json(v_manifest, variant_dir / "manifest.json")

        manifest["template_variants"][variant] = {
            "template": template,
            "dir": rel(variant_dir),
            "manifest": rel(variant_dir / "manifest.json"),
            "splits": split_summaries,
        }

    # Sanity checks.
    for variant in TEMPLATES:
        for split in SPLITS:
            p = OUT_ROOT / variant / f"{split}.json"
            if not p.exists():
                fatal_errors.append(f"Missing output: {rel(p)}")
                continue
            rows = load_json(p)
            if not isinstance(rows, list) or not rows:
                fatal_errors.append(f"Bad rows: {rel(p)}")
                continue

            bad_question = sum(1 for r in rows if not r.get("question_text"))
            bad_prompt = sum(1 for r in rows if "Question:" not in r.get("input", "") or "Answer:" not in r.get("input", ""))
            bad_candidate = sum(1 for r in rows if split in {"valid", "test"} and len(r.get("rank_entities", [])) != 20)

            if bad_question:
                fatal_errors.append(f"{rel(p)} has rows missing question_text: {bad_question}")
            if bad_prompt:
                fatal_errors.append(f"{rel(p)} has bad prompt rows: {bad_prompt}")
            if bad_candidate:
                fatal_errors.append(f"{rel(p)} has non-20 candidate rows: {bad_candidate}")

    manifest["fatal_errors"] = fatal_errors
    manifest["decision"] = "QUESTION_TEMPLATE_VARIANTS_READY" if not fatal_errors else "QUESTION_TEMPLATE_VARIANTS_BLOCKED"

    out_manifest = RESULTS_DIR / "template_variants.json"
    write_json(manifest, out_manifest)

    print("=" * 100)
    print("decision =", manifest["decision"])
    print("template_variants_json =", rel(out_manifest))
    print("output_root =", rel(OUT_ROOT))
    print("=" * 100)

    for variant, info in manifest["template_variants"].items():
        print(f"[{variant}] template = {info['template']}")
        for split, s in info["splits"].items():
            print(
                split,
                "rows =", s["num_rows"],
                "avg_graph =", s["avg_subgraph_size"],
                "sample_question =", s["sample_question"],
            )
        print("-" * 100)

    if fatal_errors:
        print("FATAL ERRORS:")
        for e in fatal_errors:
            print("-", e)


if __name__ == "__main__":
    main()