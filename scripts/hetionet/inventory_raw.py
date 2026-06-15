from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ZENODO_URLS = [
    # Main Zenodo records endpoint. Filename contains "/", so it is URL-encoded.
    "https://zenodo.org/records/268568/files/dhimmel%2Fhetionet-v1.0.0.zip?download=1",
    # Older Zenodo endpoint kept as fallback.
    "https://zenodo.org/record/268568/files/dhimmel%2Fhetionet-v1.0.0.zip?download=1",
]

EXPECTED_MD5 = "acce866650fc8bc27ac274b7dddf003e"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(urls: List[str], out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[skip] zip already exists: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
        return

    last_error = None
    for url in urls:
        try:
            print(f"[download] {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "SoftFuse-KGC-Hetionet-Downloader/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                total = response.headers.get("Content-Length")
                total_int = int(total) if total is not None else None

                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                downloaded = 0
                t0 = time.time()

                with tmp_path.open("wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_int:
                            pct = downloaded / total_int * 100
                            sys.stdout.write(
                                f"\r  {downloaded / 1e6:.1f} MB / {total_int / 1e6:.1f} MB ({pct:.1f}%)"
                            )
                        else:
                            sys.stdout.write(f"\r  {downloaded / 1e6:.1f} MB")
                        sys.stdout.flush()

                print()
                tmp_path.replace(out_path)
                dt = time.time() - t0
                print(f"[done] downloaded to {out_path} in {dt:.1f}s")
                return

        except Exception as e:
            last_error = e
            print(f"[warn] failed URL: {url}")
            print(f"       reason: {repr(e)}")

    raise RuntimeError(f"All download URLs failed. Last error: {repr(last_error)}")


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    marker = extract_dir / ".extract_done"
    if marker.exists():
        print(f"[skip] already extracted: {extract_dir}")
        return

    mkdir(extract_dir)
    print(f"[extract] {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    marker.write_text(now_iso(), encoding="utf-8")
    print("[done] extraction complete")


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 3)


def find_first(root: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(root.glob(pat))
        hits = [p for p in hits if p.is_file()]
        if hits:
            return hits[0]
    return None


def open_text_auto(path: Path):
    name = path.name.lower()
    if name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def sniff_delimiter(path: Path) -> str:
    with open_text_auto(path) as f:
        sample = f.read(4096)
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    return "\t"


def read_tsv_dict_sample(path: Path, max_rows: int = 5) -> Tuple[List[str], List[Dict[str, str]]]:
    delim = sniff_delimiter(path)
    with open_text_auto(path) as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(dict(row))
        return list(reader.fieldnames or []), rows


def infer_node_kind_from_id(node_id: str) -> str:
    # Hetionet node IDs often look like "Compound::DB00014" or "Disease::DOID:0050741".
    if "::" in node_id:
        return node_id.split("::", 1)[0]
    if ":" in node_id:
        return node_id.split(":", 1)[0]
    return "UNKNOWN"


def inventory_nodes(nodes_path: Optional[Path]) -> Dict:
    if nodes_path is None:
        return {"found": False}

    print(f"[inventory] nodes: {nodes_path}")
    delim = sniff_delimiter(nodes_path)
    kind_counts = Counter()
    samples = []
    num_rows = 0

    with open_text_auto(nodes_path) as f:
        reader = csv.DictReader(f, delimiter=delim)
        fieldnames = list(reader.fieldnames or [])

        for row in reader:
            num_rows += 1
            if len(samples) < 5:
                samples.append(dict(row))

            kind = (
                row.get("kind")
                or row.get("metanode")
                or row.get("label")
                or row.get("type")
                or row.get("category")
            )

            if not kind:
                node_id = row.get("id") or row.get("identifier") or row.get("node_id") or ""
                kind = infer_node_kind_from_id(node_id)

            kind_counts[kind] += 1

    return {
        "found": True,
        "path": str(nodes_path),
        "size_mb": file_size_mb(nodes_path),
        "num_rows": num_rows,
        "fieldnames": fieldnames,
        "kind_counts": dict(kind_counts.most_common()),
        "sample_rows": samples,
    }


def parse_sif_edges(edges_path: Path):
    with open_text_auto(edges_path) as f:
        reader = csv.reader(f, delimiter="\t")
        first = next(reader, None)

        if first is None:
            return

        # Some files may have a header. SIF normally has no header:
        # source \t metaedge \t target
        lower = [x.lower() for x in first]
        has_header = (
            len(lower) >= 3
            and ("source" in lower[0] or "subject" in lower[0] or "head" in lower[0])
            and ("target" in lower[2] or "object" in lower[2] or "tail" in lower[2])
        )

        if not has_header:
            if len(first) >= 3:
                yield first[0], first[1], first[2]
        for row in reader:
            if len(row) < 3:
                continue
            yield row[0], row[1], row[2]


def inventory_edges(edges_path: Optional[Path]) -> Dict:
    if edges_path is None:
        return {"found": False}

    print(f"[inventory] edges: {edges_path}")
    metaedge_counts = Counter()
    src_kind_counts = Counter()
    dst_kind_counts = Counter()
    sample_edges = []
    task_edges = defaultdict(list)
    task_src = defaultdict(set)
    task_dst = defaultdict(set)

    target_metaedges = {
        "CtD": "Compound-treats-Disease",
        "CpD": "Compound-palliates-Disease",
    }

    num_rows = 0
    for src, rel, dst in parse_sif_edges(edges_path):
        num_rows += 1
        metaedge_counts[rel] += 1
        src_kind_counts[infer_node_kind_from_id(src)] += 1
        dst_kind_counts[infer_node_kind_from_id(dst)] += 1

        if len(sample_edges) < 10:
            sample_edges.append({"source": src, "metaedge": rel, "target": dst})

        if rel in target_metaedges:
            task_src[rel].add(src)
            task_dst[rel].add(dst)
            if len(task_edges[rel]) < 10:
                task_edges[rel].append({"source": src, "metaedge": rel, "target": dst})

    task_summary = {}
    for rel, name in target_metaedges.items():
        task_summary[rel] = {
            "name": name,
            "num_edges": int(metaedge_counts.get(rel, 0)),
            "num_unique_compounds": len(task_src[rel]),
            "num_unique_diseases": len(task_dst[rel]),
            "sample_edges": task_edges[rel],
        }

    return {
        "found": True,
        "path": str(edges_path),
        "size_mb": file_size_mb(edges_path),
        "num_rows": num_rows,
        "metaedge_counts": dict(metaedge_counts.most_common()),
        "source_kind_counts": dict(src_kind_counts.most_common()),
        "target_kind_counts": dict(dst_kind_counts.most_common()),
        "sample_edges": sample_edges,
        "task_candidate_summary": task_summary,
    }


def inventory_meta_table(path: Optional[Path], name: str) -> Dict:
    if path is None:
        return {"found": False}

    print(f"[inventory] {name}: {path}")
    fields, sample = read_tsv_dict_sample(path, max_rows=20)
    num_rows = 0
    delim = sniff_delimiter(path)
    with open_text_auto(path) as f:
        reader = csv.DictReader(f, delimiter=delim)
        for _ in reader:
            num_rows += 1

    return {
        "found": True,
        "path": str(path),
        "size_mb": file_size_mb(path),
        "num_rows": num_rows,
        "fieldnames": fields,
        "sample_rows": sample,
    }


def list_raw_files(extract_dir: Path, max_files: int = 300) -> List[Dict]:
    files = []
    for p in sorted(extract_dir.rglob("*")):
        if p.is_file():
            files.append({
                "path": str(p),
                "size_mb": file_size_mb(p),
                "suffixes": "".join(p.suffixes),
            })
    return files[:max_files]


def write_json(path: Path, obj) -> None:
    mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_markdown_report(path: Path, inventory: Dict, task_spec: Dict) -> None:
    mkdir(path.parent)

    edge_inv = inventory.get("edges", {})
    node_inv = inventory.get("nodes", {})
    task_summary = edge_inv.get("task_candidate_summary", {})

    lines = []
    lines.append("# Hetionet raw inventory\n")
    lines.append(f"- Created at: `{now_iso()}`")
    lines.append(f"- Setting: `setting_d_hetionet`")
    lines.append(f"- Download source: Zenodo Hetionet v1.0.0")
    lines.append("")

    lines.append("## Raw files detected\n")
    lines.append(f"- Nodes found: `{node_inv.get('found')}`")
    lines.append(f"- Edges found: `{edge_inv.get('found')}`")
    lines.append(f"- Metaedges found: `{inventory.get('metaedges', {}).get('found')}`")
    lines.append(f"- Metanodes found: `{inventory.get('metanodes', {}).get('found')}`")
    lines.append("")

    if node_inv.get("found"):
        lines.append("## Node inventory\n")
        lines.append(f"- Number of node rows: `{node_inv.get('num_rows')}`")
        lines.append("- Node type counts:")
        for k, v in list(node_inv.get("kind_counts", {}).items())[:30]:
            lines.append(f"  - `{k}`: {v}")
        lines.append("")

    if edge_inv.get("found"):
        lines.append("## Edge inventory\n")
        lines.append(f"- Number of edge rows: `{edge_inv.get('num_rows')}`")
        lines.append("- Top metaedge counts:")
        for k, v in list(edge_inv.get("metaedge_counts", {}).items())[:40]:
            lines.append(f"  - `{k}`: {v}")
        lines.append("")

    lines.append("## Candidate task summary\n")
    for rel, s in task_summary.items():
        lines.append(f"### `{rel}` — {s.get('name')}")
        lines.append(f"- Edges: `{s.get('num_edges')}`")
        lines.append(f"- Unique compounds: `{s.get('num_unique_compounds')}`")
        lines.append(f"- Unique diseases: `{s.get('num_unique_diseases')}`")
        lines.append("")

    lines.append("## Draft task decision\n")
    lines.append("```json")
    lines.append(json.dumps(task_spec["recommended_task"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repo root.")
    parser.add_argument("--force-md5", action="store_true", help="Fail if MD5 does not match expected Zenodo MD5.")
    args = parser.parse_args()

    repo = Path(args.root).resolve()

    setting_dir = repo / "dataset" / "setting_d_hetionet"
    raw_dir = mkdir(setting_dir / "raw_inventory")
    task_dir = mkdir(setting_dir / "task_spec")
    report_dir = mkdir(repo / "outputs" / "hetionet" / "reports")
    result_dir = mkdir(repo / "outputs" / "hetionet")
    script_dir = mkdir(repo / "scripts" / "hetionet")

    zip_path = raw_dir / "hetionet-v1.0.0.zip"
    extract_dir = raw_dir / "extracted"

    protocol = {
        "week": 26,
        "day": 1,
        "setting": "setting_d_hetionet",
        "dataset": "Hetionet v1.0",
        "decision_stage": "RAW_INVENTORY_AND_TASK_SCHEMA_DRAFT",
        "reviewer_safe_policy": {
            "top_k": 20,
            "gold_injection": False,
            "absent_gold_rank_sentinel": 21,
            "rr_definition": "RR = 1/rank if rank <= 20 else 0",
            "split_policy": "To be built on Day 2 with train-only graph and exact-leak checks.",
        },
        "created_at": now_iso(),
    }
    write_json(result_dir / "day1_hetionet_schema_manifest.json", protocol)

    download_with_progress(ZENODO_URLS, zip_path)

    got_md5 = md5_file(zip_path)
    print(f"[md5] {got_md5}")
    if got_md5 != EXPECTED_MD5:
        msg = f"MD5 mismatch: got={got_md5}, expected={EXPECTED_MD5}"
        if args.force_md5:
            raise RuntimeError(msg)
        print(f"[warn] {msg}")
    else:
        print("[ok] MD5 matches expected Zenodo value")

    extract_zip(zip_path, extract_dir)

    nodes_path = find_first(extract_dir, [
        "**/hetionet-v1.0-nodes.tsv",
        "**/*nodes.tsv",
    ])
    edges_path = find_first(extract_dir, [
        "**/hetionet-v1.0-edges.sif.gz",
        "**/hetionet-v1.0-edges.sif",
        "**/*edges.sif.gz",
        "**/*edges.sif",
        "**/*edges.tsv",
    ])
    metaedges_path = find_first(extract_dir, [
        "**/metaedges.tsv",
        "**/*metaedges*.tsv",
    ])
    metanodes_path = find_first(extract_dir, [
        "**/metanodes.tsv",
        "**/*metanodes*.tsv",
    ])
    json_path = find_first(extract_dir, [
        "**/hetionet-v1.0.json.bz2",
        "**/*hetionet*.json.bz2",
        "**/*hetionet*.json",
    ])

    inventory = {
        "created_at": now_iso(),
        "setting": "setting_d_hetionet",
        "download": {
            "zip_path": str(zip_path),
            "zip_size_mb": file_size_mb(zip_path),
            "md5": got_md5,
            "expected_md5": EXPECTED_MD5,
            "md5_match": got_md5 == EXPECTED_MD5,
        },
        "detected_paths": {
            "nodes_path": str(nodes_path) if nodes_path else None,
            "edges_path": str(edges_path) if edges_path else None,
            "metaedges_path": str(metaedges_path) if metaedges_path else None,
            "metanodes_path": str(metanodes_path) if metanodes_path else None,
            "json_path": str(json_path) if json_path else None,
        },
        "raw_files_sample": list_raw_files(extract_dir),
        "nodes": inventory_nodes(nodes_path),
        "edges": inventory_edges(edges_path),
        "metaedges": inventory_meta_table(metaedges_path, "metaedges"),
        "metanodes": inventory_meta_table(metanodes_path, "metanodes"),
    }

    write_json(raw_dir / "raw_inventory.json", inventory)

    edge_summary = inventory.get("edges", {}).get("task_candidate_summary", {})
    write_json(raw_dir / "hetionet_task_candidate_summary.json", edge_summary)

    task_spec = {
        "created_at": now_iso(),
        "setting": "setting_d_hetionet",
        "dataset": "Hetionet v1.0",
        "status": "DRAFT_DAY1",
        "recommended_task": {
            "task_name": "hetionet_compound_treats_disease_head_prediction",
            "task_form": "(?, CtD, disease)",
            "prediction_type": "predicted_head",
            "target_relation_code": "CtD",
            "target_relation_name": "Compound-treats-Disease",
            "query_entity_type": "Disease",
            "missing_entity_type": "Compound",
            "candidate_universe": "Compound nodes; final universe selected after coverage checks.",
            "top_k": 20,
            "gold_injection": False,
            "absent_gold_rank_sentinel": 21,
            "claim_scope": "Classic drug-repurposing / compound-disease treatment external validation, not clinical validation.",
        },
        "optional_auxiliary_relation": {
            "relation_code": "CpD",
            "relation_name": "Compound-palliates-Disease",
            "use": "Optional auxiliary/support relation, not primary target unless CtD is too small after split coverage.",
        },
        "day2_required_checks": [
            "Confirm CtD edge count and unique compounds/diseases.",
            "Build coverage-safe train/valid/test splits.",
            "Ensure valid/test compounds and diseases appear in train graph.",
            "Build entity2id/id2entity/relation2id/id2relation.",
            "Build train_enriched.tsv from train-only evidence graph.",
            "Run exact leakage check: valid/test target triples must not appear in retrieved subgraphs.",
        ],
        "task_candidate_summary_from_day1": edge_summary,
    }

    write_json(task_dir / "task_spec_draft.json", task_spec)
    write_markdown_report(report_dir / "day1_hetionet_inventory.md", inventory, task_spec)

    print("\n[DONE] Day 1 artifacts:")
    print(f"  - {raw_dir / 'raw_inventory.json'}")
    print(f"  - {raw_dir / 'hetionet_task_candidate_summary.json'}")
    print(f"  - {task_dir / 'task_spec_draft.json'}")
    print(f"  - {result_dir / 'day1_hetionet_schema_manifest.json'}")
    print(f"  - {report_dir / 'day1_hetionet_inventory.md'}")

    if not edges_path:
        print("\n[WARN] Could not find edge SIF/TSV file. Inspect extracted raw files manually.")
    if "CtD" not in edge_summary or edge_summary.get("CtD", {}).get("num_edges", 0) == 0:
        print("\n[WARN] CtD relation not detected or has zero edges. Check metaedge naming in raw files.")


if __name__ == "__main__":
    main()