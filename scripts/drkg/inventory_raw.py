#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import tarfile
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DRKG_URLS = [
    "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz",
]

EXPECTED_TOP_FILES = [
    "drkg.tsv",
    "entity2src.tsv",
    "relation_glossary.tsv",
    "embed/DRKG_TransE_l2_entity.npy",
    "embed/DRKG_TransE_l2_relation.npy",
    "embed/entities.tsv",
    "embed/relations.tsv",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 3)


def download_with_progress(urls: list[str], out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] existing file: {out_path} ({file_size_mb(out_path)} MB)")
        return

    last_error = None
    for url in urls:
        try:
            print(f"[download] {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "SoftFuse-KGC-DRKG-Downloader/1.0"},
            )

            with urllib.request.urlopen(req, timeout=90) as response:
                total = response.headers.get("Content-Length")
                total_int = int(total) if total else None

                tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                downloaded = 0
                t0 = time.time()

                with tmp.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_int:
                            pct = downloaded / total_int * 100
                            print(
                                f"\r  {downloaded / 1e6:.1f} MB / {total_int / 1e6:.1f} MB ({pct:.1f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)

                print()
                tmp.replace(out_path)
                print(f"[done] downloaded to {out_path} in {time.time() - t0:.1f}s")
                return

        except Exception as e:
            last_error = e
            print(f"[warn] failed URL: {url}")
            print(f"       reason: {repr(e)}")

    raise RuntimeError(f"All DRKG download URLs failed. Last error: {repr(last_error)}")


def safe_extract_tar(tar_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extract_done"
    if marker.exists():
        print(f"[skip] already extracted: {extract_dir}")
        return

    mkdir(extract_dir)
    print(f"[extract] {tar_path} -> {extract_dir}")

    # Safe tar extraction against path traversal.
    with tarfile.open(tar_path, "r:gz") as tf:
        base = extract_dir.resolve()
        for member in tf.getmembers():
            target = (extract_dir / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"Unsafe tar member path: {member.name}")
        tf.extractall(extract_dir)

    marker.write_text(now_iso(), encoding="utf-8")
    print("[done] extraction complete")


def find_file(root: Path, filename: str) -> Path | None:
    hits = sorted([p for p in root.rglob(filename) if p.is_file()])
    return hits[0] if hits else None


def list_files(root: Path, max_files: int = 300) -> list[dict[str, Any]]:
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root)
            out.append({
                "relative_path": str(rel),
                "absolute_path": str(p),
                "size_mb": file_size_mb(p),
            })
    return out[:max_files]


def open_text_auto(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def infer_entity_type(entity: str) -> str:
    # DRKG entities typically look like Compound::DB00001, Disease::MESH:D000xxx, Gene::xxxx.
    if "::" in entity:
        return entity.split("::", 1)[0]
    if ":" in entity:
        return entity.split(":", 1)[0]
    return "UNKNOWN"


def parse_drkg_tsv(path: Path) -> Iterable[tuple[str, str, str]]:
    with open_text_auto(path) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            h, r, t = row[0].strip(), row[1].strip(), row[2].strip()
            if not h or not r or not t:
                continue
            # DRKG tsv normally has no header, but keep this guard.
            if h.lower() in {"head", "h"} and r.lower() in {"relation", "r"}:
                continue
            yield h, r, t


def read_table_sample(path: Path, max_rows: int = 10) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"found": False}

    with open_text_auto(path) as f:
        sample = f.read(4096)

    delim = "\t" if "\t" in sample else ","

    rows = []
    num_rows = 0
    fieldnames = None

    with open_text_auto(path) as f:
        reader = csv.reader(f, delimiter=delim)
        first = next(reader, None)
        if first is not None:
            fieldnames = first
            rows.append(first)
            num_rows = 1
        for row in reader:
            num_rows += 1
            if len(rows) < max_rows:
                rows.append(row)

    return {
        "found": True,
        "path": str(path),
        "size_mb": file_size_mb(path),
        "num_rows_including_first": num_rows,
        "first_row_or_header": fieldnames,
        "sample_rows": rows,
    }


def inventory_embeddings(extract_dir: Path) -> dict[str, Any]:
    embed_dir = None
    for p in sorted(extract_dir.rglob("embed")):
        if p.is_dir():
            embed_dir = p
            break

    if embed_dir is None:
        return {"found": False}

    files = []
    for p in sorted(embed_dir.rglob("*")):
        if p.is_file():
            files.append({
                "relative_path": str(p.relative_to(embed_dir)),
                "absolute_path": str(p),
                "size_mb": file_size_mb(p),
            })

    return {
        "found": True,
        "path": str(embed_dir),
        "files": files,
    }


def scan_drkg(
    drkg_path: Path,
    max_samples_per_relation: int = 5,
    max_samples_per_pair: int = 5,
) -> dict[str, Any]:
    print(f"[scan] DRKG triples: {drkg_path}")

    relation_counts = Counter()
    head_type_counts = Counter()
    tail_type_counts = Counter()
    entity_type_counts = Counter()
    entity_pair_counts = Counter()
    relation_pair_counts = Counter()

    entity_seen_by_type: dict[str, set[str]] = defaultdict(set)
    relation_samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    pair_samples: dict[str, list[dict[str, str]]] = defaultdict(list)

    compound_disease_relation_counts = Counter()
    compound_disease_direction_counts = Counter()
    compound_disease_samples: dict[str, list[dict[str, str]]] = defaultdict(list)
    compound_disease_entities: dict[str, dict[str, set[str]]] = defaultdict(lambda: {"compound": set(), "disease": set()})

    num_triples = 0

    t0 = time.time()
    for h, r, t in parse_drkg_tsv(drkg_path):
        num_triples += 1
        ht = infer_entity_type(h)
        tt = infer_entity_type(t)

        relation_counts[r] += 1
        head_type_counts[ht] += 1
        tail_type_counts[tt] += 1
        entity_type_counts[ht] += 1
        entity_type_counts[tt] += 1

        entity_seen_by_type[ht].add(h)
        entity_seen_by_type[tt].add(t)

        pair_key = f"{ht}->{tt}"
        entity_pair_counts[pair_key] += 1
        relation_pair_counts[f"{r} || {pair_key}"] += 1

        if len(relation_samples[r]) < max_samples_per_relation:
            relation_samples[r].append({"head": h, "relation": r, "tail": t, "head_type": ht, "tail_type": tt})

        if len(pair_samples[pair_key]) < max_samples_per_pair:
            pair_samples[pair_key].append({"head": h, "relation": r, "tail": t, "head_type": ht, "tail_type": tt})

        is_cd = ht == "Compound" and tt == "Disease"
        is_dc = ht == "Disease" and tt == "Compound"

        if is_cd or is_dc:
            direction = "Compound->Disease" if is_cd else "Disease->Compound"
            key = f"{r} || {direction}"
            compound_disease_relation_counts[key] += 1
            compound_disease_direction_counts[direction] += 1

            if is_cd:
                compound = h
                disease = t
            else:
                compound = t
                disease = h

            compound_disease_entities[key]["compound"].add(compound)
            compound_disease_entities[key]["disease"].add(disease)

            if len(compound_disease_samples[key]) < 10:
                compound_disease_samples[key].append({
                    "head": h,
                    "relation": r,
                    "tail": t,
                    "head_type": ht,
                    "tail_type": tt,
                    "normalized_compound": compound,
                    "normalized_disease": disease,
                    "direction": direction,
                })

        if num_triples % 1_000_000 == 0:
            print(f"  scanned {num_triples:,} triples in {time.time() - t0:.1f}s")

    unique_entity_type_counts = {
        k: len(v) for k, v in sorted(entity_seen_by_type.items(), key=lambda x: x[0])
    }

    cd_candidates = []
    for key, count in compound_disease_relation_counts.most_common():
        rel, direction = key.split(" || ", 1)
        source_hint = rel.split("::", 1)[0] if "::" in rel else "UNKNOWN"
        rel_lower = rel.lower()
        semantic_hint = {
            "contains_treat": "treat" in rel_lower,
            "contains_palliate": "palliat" in rel_lower,
            "contains_indication": "indication" in rel_lower or "indicat" in rel_lower,
            "contains_association": "associat" in rel_lower,
            "contains_therapeutic": "therapeut" in rel_lower,
            "contains_contra": "contra" in rel_lower,
            "contains_side_effect": "side" in rel_lower or "effect" in rel_lower,
        }

        cd_candidates.append({
            "relation": rel,
            "direction": direction,
            "count": int(count),
            "source_hint": source_hint,
            "num_unique_compounds": len(compound_disease_entities[key]["compound"]),
            "num_unique_diseases": len(compound_disease_entities[key]["disease"]),
            "semantic_hint": semantic_hint,
            "sample_rows": compound_disease_samples[key],
        })

    out = {
        "num_triples": int(num_triples),
        "num_relations": int(len(relation_counts)),
        "relation_counts": dict(relation_counts.most_common()),
        "head_type_counts": dict(head_type_counts.most_common()),
        "tail_type_counts": dict(tail_type_counts.most_common()),
        "unique_entity_type_counts": unique_entity_type_counts,
        "entity_pair_counts": dict(entity_pair_counts.most_common()),
        "relation_pair_counts_top200": dict(relation_pair_counts.most_common(200)),
        "compound_disease_direction_counts": dict(compound_disease_direction_counts.most_common()),
        "compound_disease_relation_candidates": cd_candidates,
        "relation_samples_top100": {
            r: relation_samples[r] for r, _ in relation_counts.most_common(100)
        },
        "pair_samples_top50": {
            p: pair_samples[p] for p, _ in entity_pair_counts.most_common(50)
        },
    }

    return out


def write_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    inv = summary["inventory"]
    scan = summary["scan"]
    cd = scan["compound_disease_relation_candidates"]

    lines = []
    lines.append("# DRKG raw inventory")
    lines.append("")
    lines.append(f"- Created at: `{now_iso()}`")
    lines.append("- Setting: `setting_e_drkg`")
    lines.append("- Dataset: `DRKG`")
    lines.append("")
    lines.append("## Download / raw files")
    lines.append("")
    lines.append(f"- Archive path: `{inv['archive']['path']}`")
    lines.append(f"- Archive size MB: `{inv['archive']['size_mb']}`")
    lines.append(f"- `drkg.tsv`: `{inv['detected_paths'].get('drkg_tsv')}`")
    lines.append(f"- `entity2src.tsv`: `{inv['detected_paths'].get('entity2src_tsv')}`")
    lines.append(f"- `relation_glossary.tsv`: `{inv['detected_paths'].get('relation_glossary_tsv')}`")
    lines.append(f"- Embeddings found: `{inv['embed_inventory'].get('found')}`")
    lines.append("")
    lines.append("## Graph inventory")
    lines.append("")
    lines.append(f"- Triples: `{scan['num_triples']}`")
    lines.append(f"- Relations: `{scan['num_relations']}`")
    lines.append("")
    lines.append("## Unique entity type counts")
    lines.append("")
    for k, v in scan["unique_entity_type_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Top entity-pair counts")
    lines.append("")
    for k, v in list(scan["entity_pair_counts"].items())[:20]:
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## CompoundDisease relation candidates")
    lines.append("")
    lines.append("| Relation | Direction | Count | Compounds | Diseases | Source | Treat hint | Association hint | Contra hint |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|")
    for item in cd[:40]:
        hint = item["semantic_hint"]
        lines.append(
            f"| `{item['relation']}` | `{item['direction']}` | {item['count']} | "
            f"{item['num_unique_compounds']} | {item['num_unique_diseases']} | `{item['source_hint']}` | "
            f"{hint['contains_treat']} | {hint['contains_association']} | {hint['contains_contra']} |"
        )
    lines.append("")
    lines.append("## Day 2 recommendation")
    lines.append("")
    lines.append("Choose one target relation based on:")
    lines.append("")
    lines.append("1. Clear Compound�Disease or reversible Disease�Compound semantics.")
    lines.append("2. Enough edges after coverage-safe split.")
    lines.append("3. No overclaiming: use treatment-like / therapeutic-association proxy wording unless relation semantics are explicit.")
    lines.append("4. Candidate universe must remain Compound-only.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    repo = Path(args.root).resolve()
    setting_dir = repo / "dataset" / "setting_e_drkg"
    raw_dir = mkdir(setting_dir / "raw_inventory")
    task_dir = mkdir(setting_dir / "task_spec")
    result_dir = mkdir(repo / "outputs" / "drkg")
    report_dir = mkdir(repo / "outputs" / "drkg" / "reports")

    archive_path = raw_dir / "drkg.tar.gz"
    extract_dir = raw_dir / "extracted"

    if not args.skip_download:
        download_with_progress(DRKG_URLS, archive_path)
    else:
        print("[skip] download")

    if not args.skip_extract:
        safe_extract_tar(archive_path, extract_dir)
    else:
        print("[skip] extract")

    drkg_tsv = find_file(extract_dir, "drkg.tsv")
    entity2src = find_file(extract_dir, "entity2src.tsv")
    relation_glossary = find_file(extract_dir, "relation_glossary.tsv")

    if drkg_tsv is None:
        raise FileNotFoundError("Cannot find drkg.tsv after extraction.")

    manifest = {
        "created_at": now_iso(),
        "setting": "setting_e_drkg",
        "dataset": "DRKG",
        "archive": {
            "path": str(archive_path),
            "size_mb": file_size_mb(archive_path) if archive_path.exists() else None,
        },
        "detected_paths": {
            "drkg_tsv": str(drkg_tsv) if drkg_tsv else None,
            "entity2src_tsv": str(entity2src) if entity2src else None,
            "relation_glossary_tsv": str(relation_glossary) if relation_glossary else None,
        },
        "raw_files_sample": list_files(extract_dir),
        "expected_top_files": EXPECTED_TOP_FILES,
        "entity2src_sample": read_table_sample(entity2src) if entity2src else {"found": False},
        "relation_glossary_sample": read_table_sample(relation_glossary) if relation_glossary else {"found": False},
        "embed_inventory": inventory_embeddings(extract_dir),
    }

    write_json(manifest, raw_dir / "raw_file_manifest.json")

    scan = scan_drkg(drkg_tsv)

    write_json(scan["relation_counts"], raw_dir / "relation_counts.json")
    write_json(scan["unique_entity_type_counts"], raw_dir / "entity_type_counts.json")
    write_json(scan["entity_pair_counts"], raw_dir / "entity_pair_counts.json")
    write_json(scan["compound_disease_relation_candidates"], raw_dir / "compound_disease_relation_candidates.json")

    task_candidates = {
        "created_at": now_iso(),
        "setting": "setting_e_drkg",
        "dataset": "DRKG",
        "status": "DAY1_CANDIDATES_ONLY_NOT_FROZEN",
        "candidate_task_family": "Compound-Disease head prediction",
        "preferred_task_form": "(?, relation, disease)",
        "prediction_type": "predicted_head",
        "candidate_universe": "Compound",
        "query_entity_type": "Disease",
        "target_relation_to_select_on_day2": None,
        "wording_policy": [
            "Use treatment-like relation only if relation semantics explicitly indicate treatment.",
            "Otherwise use therapeutic-association proxy or drug-disease association proxy.",
            "Do not call the task clinical indication unless relation semantics justify it."
        ],
        "compound_disease_relation_candidates": scan["compound_disease_relation_candidates"],
    }
    write_json(task_candidates, task_dir / "task_schema_candidates_day1.json")

    summary = {
        "created_at": now_iso(),
        "decision": "DAY1_DRKG_RAW_INVENTORY_READY",
        "inventory": manifest,
        "scan": scan,
        "next_step": "Select target relation and task schema.",
    }

    write_json(summary, raw_dir / "raw_inventory.json")
    write_json(summary, result_dir / "day1_drkg_inventory_summary.json")
    write_markdown_report(report_dir / "day1_drkg_inventory.md", summary)

    print("\n[DONE] Day 1 DRKG inventory")
    print(json.dumps({
        "decision": summary["decision"],
        "drkg_tsv": str(drkg_tsv),
        "num_triples": scan["num_triples"],
        "num_relations": scan["num_relations"],
        "unique_entity_type_counts": scan["unique_entity_type_counts"],
        "top_compound_disease_candidates": scan["compound_disease_relation_candidates"][:10],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
