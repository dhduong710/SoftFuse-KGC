#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Week 22 Day 2: Raw inventory for PharmKG / Dataset 2.

This script:
1. Downloads PharmKG-8k split files if missing.
2. Optionally downloads the official Zenodo raw archive.
3. Inventories local raw files.
4. Inspects TSV/CSV/TXT files.
5. Detects triple-file candidates.
6. Detects relation/entity/type-file candidates.
7. Writes JSON + Markdown reports.

It does NOT select the final target relation.
Target relation/schema selection is Day 3.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except Exception as e:
    pd = None
    print(f"[WARN] pandas is not available: {e}", file=sys.stderr)


ROOT = Path(".")
RAW_ROOT = ROOT / "data" / "raw" / "pharmkg"
PHARMKG8K_DIR = RAW_ROOT / "PharmKG-8k"
ZENODO_DIR = RAW_ROOT / "zenodo"

RESULT_DIR = ROOT / "outputs" / "pharmkg"
REPORT_DIR = ROOT / "outputs" / "pharmkg" / "reports"
INV_DIR = ROOT / "dataset" / "setting_c_pharmkg" / "raw_inventory"

RAW_INVENTORY_PATH = RESULT_DIR / "dataset2_raw_inventory.json"
REPORT_PATH = REPORT_DIR / "day2_raw_inventory.md"
FILE_MANIFEST_PATH = INV_DIR / "file_manifest.json"
DOWNLOAD_STATUS_PATH = INV_DIR / "download_status.json"
TRIPLE_CANDIDATES_PATH = INV_DIR / "triple_file_candidates.json"
RELATION_HINTS_PATH = INV_DIR / "relation_hint_candidates.json"

URLS = {
    "train.tsv": "https://raw.githubusercontent.com/biomed-AI/PharmKG/master/data/PharmKG-8k/train.tsv",
    "valid.tsv": "https://raw.githubusercontent.com/biomed-AI/PharmKG/master/data/PharmKG-8k/valid.tsv",
    "test.tsv": "https://raw.githubusercontent.com/biomed-AI/PharmKG/master/data/PharmKG-8k/test.tsv",
    "entity2vec.txt": "https://raw.githubusercontent.com/biomed-AI/PharmKG/master/data/PharmKG-8k/entity2vec.txt",
    "relation2vec.txt": "https://raw.githubusercontent.com/biomed-AI/PharmKG/master/data/PharmKG-8k/relation2vec.txt",
}

README_URLS = {
    "MindRank-Biotech_README.md": "https://raw.githubusercontent.com/MindRank-Biotech/PharmKG/master/README.md",
    "biomed_AI_README.md": "https://raw.githubusercontent.com/biomed-AI/PharmKG/master/README.md",
}

ZENODO_URL = "https://zenodo.org/records/4077338/files/raw_PharmKG-180k.zip?download=1"

THERAPEUTIC_KEYWORDS = [
    "indication",
    "indicated",
    "treat",
    "treatment",
    "therapy",
    "therapeutic",
    "drug",
    "disease",
    "chemical",
    "compound",
    "association",
    "associated",
    "alleviate",
    "effect",
    "contra",
    "side",
    "symptom",
]

TYPE_KEYWORDS = [
    "drug",
    "disease",
    "chemical",
    "gene",
    "protein",
    "compound",
    "entity",
    "type",
    "node",
]


def ensure_dirs() -> None:
    for p in [RAW_ROOT, PHARMKG8K_DIR, ZENODO_DIR, RESULT_DIR, REPORT_DIR, INV_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_download(url: str, out_path: Path, force: bool = False, timeout: int = 60) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    status = {
        "url": url,
        "path": str(out_path),
        "exists_before": out_path.exists(),
        "downloaded": False,
        "ok": False,
        "error": None,
        "size_bytes": None,
    }

    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        status["ok"] = True
        status["size_bytes"] = out_path.stat().st_size
        return status

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 SoftFuse Week22 Dataset Inventory"
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
        out_path.write_bytes(data)
        status["downloaded"] = True
        status["ok"] = out_path.exists() and out_path.stat().st_size > 0
        status["size_bytes"] = out_path.stat().st_size if out_path.exists() else None
    except Exception as e:
        status["error"] = repr(e)
        status["ok"] = False

    return status


def file_md5(path: Path, max_bytes: int | None = None) -> str | None:
    try:
        h = hashlib.md5()
        with path.open("rb") as f:
            if max_bytes is None:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            else:
                h.update(f.read(max_bytes))
        return h.hexdigest()
    except Exception:
        return None


def count_lines(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None


def is_text_like(path: Path) -> bool:
    return path.suffix.lower() in {".tsv", ".csv", ".txt", ".md", ".json"}


def sniff_delimiter(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return "\t"
    if suffix == ".csv":
        return ","

    try:
        sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,; ")
        return dialect.delimiter
    except Exception:
        if suffix == ".txt":
            return "\t"
        return "\t"


def read_table_sample(path: Path, nrows: int = 5) -> dict:
    result = {
        "readable": False,
        "delimiter": None,
        "num_columns_sample": None,
        "sample_rows": [],
        "error": None,
    }

    if pd is None:
        result["error"] = "pandas_not_available"
        return result

    try:
        delim = sniff_delimiter(path)
        result["delimiter"] = "\\t" if delim == "\t" else delim

        df = pd.read_csv(
            path,
            sep=delim,
            header=None,
            nrows=nrows,
            dtype=str,
            engine="python",
            on_bad_lines="skip",
        )
        result["readable"] = True
        result["num_columns_sample"] = int(df.shape[1])
        result["sample_rows"] = df.fillna("").values.tolist()
    except Exception as e:
        result["error"] = repr(e)

    return result


def inspect_possible_triples(path: Path, max_relation_count_rows: int | None = None) -> dict:
    """
    Treat a readable >=3-column TSV/CSV/TXT as possible h-r-t triples.
    For split files, PharmKG-8k is expected to be 3 columns.
    """
    out = {
        "is_possible_triple_file": False,
        "num_rows": count_lines(path),
        "num_columns_sample": None,
        "relation_col_index_assumed": 1,
        "head_col_index_assumed": 0,
        "tail_col_index_assumed": 2,
        "relation_counts_top30": [],
        "num_unique_heads": None,
        "num_unique_tails": None,
        "num_unique_entities_head_tail": None,
        "num_unique_relations": None,
        "relation_keyword_hits": [],
        "error": None,
    }

    if pd is None:
        out["error"] = "pandas_not_available"
        return out

    sample = read_table_sample(path, nrows=5)
    out["num_columns_sample"] = sample.get("num_columns_sample")

    if not sample["readable"] or not sample.get("num_columns_sample") or sample["num_columns_sample"] < 3:
        return out

    lower_name = path.name.lower()
    likely_by_name = any(k in lower_name for k in ["train", "valid", "test", "triple", "kg", "edge"])
    if not likely_by_name:
        return out

    try:
        delim = sniff_delimiter(path)
        nrows = max_relation_count_rows
        df = pd.read_csv(
            path,
            sep=delim,
            header=None,
            nrows=nrows,
            dtype=str,
            engine="python",
            on_bad_lines="skip",
        )
        if df.shape[1] < 3:
            return out

        df3 = df.iloc[:, :3].fillna("")
        rels = df3.iloc[:, 1].astype(str)
        heads = df3.iloc[:, 0].astype(str)
        tails = df3.iloc[:, 2].astype(str)

        rel_counter = Counter(rels.tolist())
        out["is_possible_triple_file"] = True
        out["relation_counts_top30"] = [
            {"relation": str(k), "count": int(v)}
            for k, v in rel_counter.most_common(30)
        ]
        out["num_unique_heads"] = int(heads.nunique())
        out["num_unique_tails"] = int(tails.nunique())
        out["num_unique_entities_head_tail"] = int(pd.concat([heads, tails]).nunique())
        out["num_unique_relations"] = int(rels.nunique())

        relation_values_joined = " ".join(map(str, rel_counter.keys())).lower()
        hits = sorted({kw for kw in THERAPEUTIC_KEYWORDS if kw in relation_values_joined})
        out["relation_keyword_hits"] = hits
    except Exception as e:
        out["error"] = repr(e)

    return out


def build_manifest() -> list[dict]:
    rows = []
    for path in sorted(RAW_ROOT.rglob("*")):
        if path.is_dir():
            continue

        stat = path.stat()
        item = {
            "path": str(path),
            "relative_path": str(path.relative_to(RAW_ROOT)),
            "file_name": path.name,
            "suffix": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 4),
            "num_lines": count_lines(path) if is_text_like(path) else None,
            "md5_first_1mb": file_md5(path, max_bytes=1024 * 1024),
            "is_zip": zipfile.is_zipfile(path),
            "is_text_like": is_text_like(path),
        }

        if item["is_text_like"] and stat.st_size < 200 * 1024 * 1024:
            item["table_sample"] = read_table_sample(path, nrows=5)
        else:
            item["table_sample"] = None

        if item["is_zip"]:
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    item["zip_num_members"] = len(zf.namelist())
                    item["zip_first_members"] = zf.namelist()[:50]
            except Exception as e:
                item["zip_error"] = repr(e)

        rows.append(item)
    return rows


def detect_file_candidates(manifest: list[dict]) -> dict:
    triple_candidates = []
    entity_candidates = []
    relation_candidates = []
    type_candidates = []

    for item in manifest:
        path = Path(item["path"])
        name = item["file_name"].lower()
        relpath = item["relative_path"].lower()

        if item["is_text_like"]:
            if any(k in name for k in ["train", "valid", "test", "triple", "kg", "edge"]):
                triple_info = inspect_possible_triples(path)
                if triple_info["is_possible_triple_file"]:
                    triple_candidates.append({
                        "path": item["path"],
                        "relative_path": item["relative_path"],
                        "file_name": item["file_name"],
                        "size_mb": item["size_mb"],
                        **triple_info,
                    })

            if any(k in relpath for k in ["entity", "node"]):
                entity_candidates.append({
                    "path": item["path"],
                    "relative_path": item["relative_path"],
                    "file_name": item["file_name"],
                    "num_lines": item["num_lines"],
                    "sample": item.get("table_sample"),
                })

            if "relation" in relpath or "rel" in name:
                relation_candidates.append({
                    "path": item["path"],
                    "relative_path": item["relative_path"],
                    "file_name": item["file_name"],
                    "num_lines": item["num_lines"],
                    "sample": item.get("table_sample"),
                })

            if "type" in relpath or any(k in name for k in ["drug", "disease", "chemical", "gene", "protein"]):
                type_candidates.append({
                    "path": item["path"],
                    "relative_path": item["relative_path"],
                    "file_name": item["file_name"],
                    "num_lines": item["num_lines"],
                    "sample": item.get("table_sample"),
                })

    return {
        "triple_file_candidates": triple_candidates,
        "entity_file_candidates": entity_candidates,
        "relation_file_candidates": relation_candidates,
        "type_file_candidates": type_candidates,
    }


def build_relation_hints(triple_candidates: list[dict]) -> dict:
    all_relations = Counter()
    keyword_hits = []

    for cand in triple_candidates:
        for row in cand.get("relation_counts_top30", []):
            rel = str(row["relation"])
            cnt = int(row["count"])
            all_relations[rel] += cnt

    for rel, cnt in all_relations.most_common():
        rel_l = rel.lower()
        hits = [kw for kw in THERAPEUTIC_KEYWORDS if kw in rel_l]
        if hits:
            keyword_hits.append({
                "relation": rel,
                "count_in_top_counts": int(cnt),
                "keyword_hits": hits,
                "note": "candidate_hint_only_not_final_selection",
            })

    return {
        "num_relations_seen_in_top_counts": len(all_relations),
        "relation_counts_merged_top50": [
            {"relation": str(k), "count": int(v)}
            for k, v in all_relations.most_common(50)
        ],
        "therapeutic_keyword_relation_hints": keyword_hits[:50],
        "warning": (
            "These are only Day 2 hints. Final target/support/excluded relation "
            "selection must be done on Day 3 after schema inspection."
        ),
    }


def infer_day2_decision(candidates: dict, relation_hints: dict) -> tuple[str, list[str]]:
    notes = []

    triple_files = candidates["triple_file_candidates"]
    entity_files = candidates["entity_file_candidates"]
    relation_files = candidates["relation_file_candidates"]
    type_files = candidates["type_file_candidates"]

    if not triple_files:
        return "RAW_DOWNLOAD_FAILED_OR_NO_TRIPLE_FILES_FOUND", [
            "No readable triple-like train/valid/test/kg/edge file was found."
        ]

    if not relation_files:
        notes.append("No explicit relation metadata file found beyond relation2vec or relation column values.")

    if not entity_files:
        notes.append("No explicit entity metadata file found beyond entity2vec or triple columns.")

    if not type_files:
        notes.append("No explicit entity type map was found. Day 3 may need raw Zenodo archive or heuristic type mapping.")

    if not relation_hints["therapeutic_keyword_relation_hints"]:
        notes.append("No obvious therapeutic relation keyword found in visible relation strings. Relation IDs may need mapping.")

    if not type_files:
        return "PARTIAL_READY_NEEDS_ENTITY_TYPE_MAP", notes

    if not relation_hints["therapeutic_keyword_relation_hints"]:
        return "PARTIAL_READY_NEEDS_RELATION_LABEL_MAP", notes

    return "RAW_INVENTORY_READY", notes


def write_report(inventory: dict) -> None:
    decision = inventory["decision"]
    notes = inventory["decision_notes"]
    manifest = inventory["file_manifest"]
    candidates = inventory["file_candidates"]
    relation_hints = inventory["relation_hints"]

    triple_rows = candidates["triple_file_candidates"]
    entity_rows = candidates["entity_file_candidates"]
    relation_rows = candidates["relation_file_candidates"]
    type_rows = candidates["type_file_candidates"]

    def bullet_file_rows(rows, max_n=20):
        if not rows:
            return "- None found\n"
        lines = []
        for r in rows[:max_n]:
            lines.append(
                f"- `{r.get('relative_path', r.get('path'))}` "
                f"(lines={r.get('num_rows', r.get('num_lines'))}, "
                f"cols={r.get('num_columns_sample', 'NA')})"
            )
        if len(rows) > max_n:
            lines.append(f"- ... {len(rows) - max_n} more")
        return "\n".join(lines) + "\n"

    rel_top = relation_hints["relation_counts_merged_top50"][:20]
    rel_lines = "\n".join([f"- `{x['relation']}`: {x['count']}" for x in rel_top]) or "- None"

    hint_top = relation_hints["therapeutic_keyword_relation_hints"][:20]
    hint_lines = "\n".join([
        f"- `{x['relation']}`: count={x['count_in_top_counts']}, hits={x['keyword_hits']}"
        for x in hint_top
    ]) or "- None"

    md = f"""# Week 22 Day 2 — Raw Inventory for PharmKG / Dataset 2

## Decision

`{decision}`

## Decision notes

{chr(10).join(f"- {n}" for n in notes) if notes else "- No blocking notes."}

## Download status

See:

- `{DOWNLOAD_STATUS_PATH}`

## Raw files found

Number of files: **{len(manifest)}**

Manifest:

- `{FILE_MANIFEST_PATH}`

## Triple file candidates

{bullet_file_rows(triple_rows)}

## Entity file candidates

{bullet_file_rows(entity_rows)}

## Relation file candidates

{bullet_file_rows(relation_rows)}

## Entity/type file candidates

{bullet_file_rows(type_rows)}

## Top relation values seen in triple-like files

{rel_lines}

## Therapeutic relation keyword hints

{hint_lines}

## Day 2 answers

1. Dataset has files? **{'yes' if manifest else 'no'}**
2. Candidate triple files? **{'yes' if triple_rows else 'no'}**
3. Candidate entity files? **{'yes' if entity_rows else 'no'}**
4. Candidate relation files? **{'yes' if relation_rows else 'no'}**
5. Explicit entity type map? **{'yes' if type_rows else 'not confirmed'}**
6. Relation like indication/treatment/therapeutic association? **{'hint found' if hint_top else 'not confirmed'}**
7. Enough Drug/Disease entities? **not confirmed on Day 2**
8. Enough drug-disease triples for split? **not confirmed until Day 3 relation/type selection**

## Important warning

Day 2 only inventories the raw files. It does **not** select the final target relation.
Do not report that PharmKG has an indication relation unless Day 3 confirms the semantics.

## Files written

- `{RAW_INVENTORY_PATH}`
- `{FILE_MANIFEST_PATH}`
- `{TRIPLE_CANDIDATES_PATH}`
- `{RELATION_HINTS_PATH}`
- `{REPORT_PATH}`

## Next step: Day 3

Select:

- target relation
- relation direction
- drug entity type
- disease entity type
- support relations
- excluded relations

Expected Day 3 decision:

`GO_TASK_SELECTED` or `PARTIAL_READY_NEEDS_MANUAL_SCHEMA_CHECK`
"""
    REPORT_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-zenodo", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    download_status = {
        "created_at": now_str(),
        "skip_download": bool(args.skip_download),
        "force_download": bool(args.force_download),
        "download_zenodo": bool(args.download_zenodo),
        "pharmkg8k": {},
        "readmes": {},
        "zenodo": None,
    }

    if not args.skip_download:
        print("[download] PharmKG-8k split files")
        for fname, url in URLS.items():
            out_path = PHARMKG8K_DIR / fname
            st = safe_download(url, out_path, force=args.force_download)
            download_status["pharmkg8k"][fname] = st
            print(f"  {fname}: ok={st['ok']} size={st['size_bytes']} error={st['error']}")

        print("[download] README files")
        for fname, url in README_URLS.items():
            out_path = RAW_ROOT / fname
            st = safe_download(url, out_path, force=args.force_download)
            download_status["readmes"][fname] = st
            print(f"  {fname}: ok={st['ok']} size={st['size_bytes']} error={st['error']}")

        if args.download_zenodo:
            print("[download] Zenodo raw archive; this may be large")
            out_path = ZENODO_DIR / "raw_PharmKG-180k.zip"
            st = safe_download(ZENODO_URL, out_path, force=args.force_download, timeout=300)
            download_status["zenodo"] = st
            print(f"  raw_PharmKG-180k.zip: ok={st['ok']} size={st['size_bytes']} error={st['error']}")

    write_json(download_status, DOWNLOAD_STATUS_PATH)

    print("[inventory] Building file manifest")
    manifest = build_manifest()
    candidates = detect_file_candidates(manifest)
    relation_hints = build_relation_hints(candidates["triple_file_candidates"])
    decision, notes = infer_day2_decision(candidates, relation_hints)

    inventory = {
        "week": 22,
        "day": 2,
        "created_at": now_str(),
        "decision": decision,
        "decision_notes": notes,
        "dataset_2_name": "PharmKG candidate",
        "raw_root": str(RAW_ROOT),
        "download_status_path": str(DOWNLOAD_STATUS_PATH),
        "file_manifest": manifest,
        "file_candidates": candidates,
        "relation_hints": relation_hints,
        "next_day": {
            "day": 3,
            "task": "select task, relation, schema, and entity types",
            "must_not_assume": [
                "Do not assume a relation is clinical indication unless confirmed.",
                "Do not assume entity type from string names only if metadata exists.",
                "Do not build train/valid/test before relation and type schema are selected."
            ],
        },
    }

    write_json(inventory, RAW_INVENTORY_PATH)
    write_json(manifest, FILE_MANIFEST_PATH)
    write_json(candidates["triple_file_candidates"], TRIPLE_CANDIDATES_PATH)
    write_json(relation_hints, RELATION_HINTS_PATH)
    write_report(inventory)

    print("\nSaved:")
    print(f"  {RAW_INVENTORY_PATH}")
    print(f"  {FILE_MANIFEST_PATH}")
    print(f"  {TRIPLE_CANDIDATES_PATH}")
    print(f"  {RELATION_HINTS_PATH}")
    print(f"  {REPORT_PATH}")

    print("\nDecision:", decision)
    if notes:
        print("Notes:")
        for n in notes:
            print(" -", n)


if __name__ == "__main__":
    main()