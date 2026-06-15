#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
SETTING_DIR = ROOT / "dataset" / "setting_f_repodb"
RAW_DIR = SETTING_DIR / "raw_inventory"
DOWNLOAD_DIR = RAW_DIR / "downloads"
EXPORT_DIR = RAW_DIR / "exported_tables"
MANUAL_DIR = RAW_DIR / "manual"

RESULT_DIR = ROOT / "outputs" / "repodb"
REPORT_DIR = ROOT / "outputs" / "repodb" / "reports"

RDATA_URLS = [
    "https://raw.githubusercontent.com/adam-sam-brown/repoDB/master/Shiny_Application/data/shiny.RData",
    "https://github.com/adam-sam-brown/repoDB/raw/master/Shiny_Application/data/shiny.RData",
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
    if not path.exists():
        return 0.0
    return round(path.stat().st_size / (1024 * 1024), 4)


def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
    return name.strip("_") or "table"


def download_file(urls: list[str], out_path: Path, force: bool = False) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return {
            "status": "exists",
            "path": str(out_path),
            "size_mb": file_size_mb(out_path),
            "url_used": None,
        }

    last_error = None

    for url in urls:
        try:
            print(f"[download] {url}")
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "SoftFuse-KGC-repoDB-Day1/1.0"},
            )
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")

            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
                shutil.copyfileobj(resp, f)

            tmp.replace(out_path)

            if out_path.stat().st_size <= 0:
                raise RuntimeError("downloaded file is empty")

            return {
                "status": "downloaded",
                "path": str(out_path),
                "size_mb": file_size_mb(out_path),
                "url_used": url,
            }

        except Exception as e:
            last_error = repr(e)
            print(f"[warn] failed: {url}")
            print(f"       {last_error}")

    return {
        "status": "failed",
        "path": str(out_path),
        "size_mb": file_size_mb(out_path),
        "url_used": None,
        "error": last_error,
    }


def export_with_pyreadr(rdata_path: Path, export_dir: Path) -> dict[str, Any]:
    try:
        import pyreadr  # type: ignore
    except Exception as e:
        return {
            "method": "pyreadr",
            "status": "unavailable",
            "error": repr(e),
            "exported_files": [],
        }

    try:
        result = pyreadr.read_r(str(rdata_path))
        exported = []

        for name, obj in result.items():
            if isinstance(obj, pd.DataFrame):
                out = export_dir / f"pyreadr_{safe_name(name)}.tsv"
                obj.to_csv(out, sep="\t", index=False)
                exported.append({
                    "object_name": name,
                    "path": str(out),
                    "rows": int(len(obj)),
                    "cols": int(len(obj.columns)),
                })

        return {
            "method": "pyreadr",
            "status": "success" if exported else "no_dataframes_exported",
            "exported_files": exported,
        }

    except Exception as e:
        return {
            "method": "pyreadr",
            "status": "failed",
            "error": repr(e),
            "exported_files": [],
        }


def export_with_rscript(rdata_path: Path, export_dir: Path) -> dict[str, Any]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return {
            "method": "Rscript",
            "status": "unavailable",
            "error": "Rscript not found in PATH",
            "exported_files": [],
        }

    export_dir.mkdir(parents=True, exist_ok=True)

    r_code = f'''
args <- commandArgs(trailingOnly=TRUE)
rdata_path <- args[1]
out_dir <- args[2]
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)

load(rdata_path)
objs <- ls()

sink(file.path(out_dir, "rdata_objects_str.txt"))
for (nm in objs) {{
  cat("\\n============================================================\\n")
  cat("OBJECT:", nm, "\\n")
  obj <- get(nm)
  print(class(obj))
  print(dim(obj))
  try(str(obj, max.level=2), silent=TRUE)
}}
sink()

export_one <- function(obj, nm, prefix="rscript") {{
  if (is.data.frame(obj)) {{
    path <- file.path(out_dir, paste0(prefix, "_", gsub("[^A-Za-z0-9_.-]+", "_", nm), ".tsv"))
    utils::write.table(obj, file=path, sep="\\t", row.names=FALSE, quote=TRUE, na="")
    return(path)
  }}
  if (is.matrix(obj)) {{
    path <- file.path(out_dir, paste0(prefix, "_", gsub("[^A-Za-z0-9_.-]+", "_", nm), ".tsv"))
    utils::write.table(as.data.frame(obj), file=path, sep="\\t", row.names=FALSE, quote=TRUE, na="")
    return(path)
  }}
  return(NA)
}}

exported <- c()

for (nm in objs) {{
  obj <- get(nm)
  p <- export_one(obj, nm)
  if (!is.na(p)) {{
    exported <- c(exported, p)
  }} else if (is.list(obj)) {{
    for (subnm in names(obj)) {{
      subobj <- obj[[subnm]]
      p2 <- export_one(subobj, paste0(nm, "__", subnm))
      if (!is.na(p2)) exported <- c(exported, p2)
    }}
  }}
}}

writeLines(exported, file.path(out_dir, "rscript_exported_files.txt"))
'''

    helper_path = export_dir / "export_repodb_rdata.R"
    helper_path.write_text(r_code, encoding="utf-8")

    try:
        proc = subprocess.run(
            [rscript, str(helper_path), str(rdata_path), str(export_dir)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        exported = []
        exported_list = export_dir / "rscript_exported_files.txt"

        if exported_list.exists():
            for line in exported_list.read_text(encoding="utf-8").splitlines():
                p = Path(line.strip())
                if p.exists():
                    exported.append({
                        "path": str(p),
                        "size_mb": file_size_mb(p),
                    })

        return {
            "method": "Rscript",
            "status": "success" if exported else "no_dataframes_exported",
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
            "exported_files": exported,
        }

    except Exception as e:
        return {
            "method": "Rscript",
            "status": "failed",
            "error": repr(e),
            "exported_files": [],
        }


def sniff_read_table(path: Path, nrows: int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", dtype=str, nrows=nrows, keep_default_na=False)
    if suffix == ".csv":
        return pd.read_csv(path, sep=",", dtype=str, nrows=nrows, keep_default_na=False)

    # Try TSV first, then CSV.
    try:
        return pd.read_csv(path, sep="\t", dtype=str, nrows=nrows, keep_default_na=False)
    except Exception:
        return pd.read_csv(path, sep=",", dtype=str, nrows=nrows, keep_default_na=False)


def normalized_colname(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9]+", "_", c)
    return c.strip("_")


def find_columns(columns: list[str]) -> dict[str, list[str]]:
    norm_map = {c: normalized_colname(c) for c in columns}

    groups = {
        "status": [],
        "phase": [],
        "drug_name": [],
        "drugbank_id": [],
        "drugcentral_id": [],
        "disease_name": [],
        "umls": [],
        "indication": [],
        "nct": [],
    }

    for c, n in norm_map.items():
        if "status" in n:
            groups["status"].append(c)
        if "phase" in n:
            groups["phase"].append(c)
        if "drugbank" in n or n in {"drugbank_id", "drug_bank_id"}:
            groups["drugbank_id"].append(c)
        if "drugcentral" in n or "drug_central" in n:
            groups["drugcentral_id"].append(c)
        if "drug" in n or "compound" in n:
            groups["drug_name"].append(c)
        if "disease" in n:
            groups["disease_name"].append(c)
        if "indication" in n:
            groups["indication"].append(c)
        if "umls" in n or n in {"cui", "cuis"}:
            groups["umls"].append(c)
        if "nct" in n or "clinicaltrial" in n or "clinical_trial" in n:
            groups["nct"].append(c)

    return groups


def value_counts_safe(df: pd.DataFrame, col: str, topn: int = 30) -> dict[str, int]:
    if col not in df.columns:
        return {}
    vc = Counter(str(x).strip() for x in df[col].fillna("").tolist())
    return dict(vc.most_common(topn))


def status_summary(df: pd.DataFrame, status_cols: list[str]) -> dict[str, Any]:
    out = {}
    for col in status_cols:
        vc = value_counts_safe(df, col, topn=50)
        approved = sum(v for k, v in vc.items() if "approved" in k.lower())
        failed = sum(
            v for k, v in vc.items()
            if any(tok in k.lower() for tok in ["terminated", "withdrawn", "suspended", "failed"])
        )
        out[col] = {
            "value_counts_top50": vc,
            "approved_like_count": int(approved),
            "failed_like_count": int(failed),
        }
    return out


def infer_table_score(inv: dict[str, Any]) -> float:
    rows = inv.get("num_rows", 0)
    cols = inv.get("num_cols", 0)
    col_groups = inv.get("column_groups", {})

    score = 0.0
    score += min(rows / 1000.0, 20.0)
    score += min(cols / 10.0, 5.0)

    if col_groups.get("status"):
        score += 10
    if col_groups.get("drugbank_id"):
        score += 8
    if col_groups.get("drugcentral_id"):
        score += 4
    if col_groups.get("drug_name"):
        score += 4
    if col_groups.get("umls"):
        score += 8
    if col_groups.get("disease_name") or col_groups.get("indication"):
        score += 4
    if col_groups.get("phase"):
        score += 2

    return score


def inventory_tables() -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    candidates = []

    for base in [EXPORT_DIR, MANUAL_DIR]:
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".tsv", ".csv", ".txt"}:
                candidates.append(p)

    inventories = []

    for path in candidates:
        try:
            df_sample = sniff_read_table(path, nrows=5000)
            df_full = sniff_read_table(path, nrows=None)

            cols = list(df_full.columns)
            col_groups = find_columns(cols)

            inv = {
                "path": str(path),
                "relative_path": str(path.relative_to(RAW_DIR)) if path.is_relative_to(RAW_DIR) else str(path),
                "size_mb": file_size_mb(path),
                "num_rows": int(len(df_full)),
                "num_cols": int(len(cols)),
                "columns": cols,
                "column_groups": col_groups,
                "status_summary": status_summary(df_full, col_groups["status"]),
                "sample_rows": df_full.head(5).to_dict(orient="records"),
            }
            inv["table_score"] = infer_table_score(inv)
            inventories.append(inv)

        except Exception as e:
            inventories.append({
                "path": str(path),
                "size_mb": file_size_mb(path),
                "error": repr(e),
                "table_score": -1,
            })

    inventories = sorted(inventories, key=lambda x: x.get("table_score", -1), reverse=True)
    best = inventories[0] if inventories and inventories[0].get("table_score", -1) >= 0 else None

    return inventories, best


def copy_best_table(best: dict[str, Any] | None) -> dict[str, Any]:
    if not best:
        return {
            "status": "no_best_table",
            "path": None,
        }

    src = Path(best["path"])
    dst = RAW_DIR / "repodb_best_table_raw.tsv"

    try:
        df = sniff_read_table(src, nrows=None)
        df.to_csv(dst, sep="\t", index=False)
        return {
            "status": "copied",
            "source_path": str(src),
            "path": str(dst),
            "num_rows": int(len(df)),
            "num_cols": int(len(df.columns)),
            "columns": list(df.columns),
        }
    except Exception as e:
        return {
            "status": "failed",
            "source_path": str(src),
            "path": str(dst),
            "error": repr(e),
        }


def write_report(summary: dict[str, Any]) -> None:
    path = REPORT_DIR / "day1_repodb_inventory.md"
    invs = summary["table_inventory"]
    best = summary["best_table"]

    lines = []
    lines.append("# repoDB raw inventory")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append(f"- Created at: `{summary['created_at']}`")
    lines.append("- Setting: `setting_f_repodb`")
    lines.append("")
    lines.append("## Source notes")
    lines.append("")
    lines.append("- repoDB is used as a clinical drug-repositioning external validation benchmark.")
    lines.append("- The official web app describes repoDB as containing repositioning successes and failures.")
    lines.append("- Current repoDB web data links drugs to DrugCentral/DrugBank IDs and diseases to UMLS terms.")
    lines.append("")
    lines.append("## Download/export status")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["download_status"], ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## RData export status")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary["export_status"], ensure_ascii=False, indent=2)[:6000])
    lines.append("```")
    lines.append("")
    lines.append("## Tables found")
    lines.append("")
    lines.append("| Rank | Rows | Cols | Score | Path | Status cols | DrugBank cols | UMLS cols |")
    lines.append("|---:|---:|---:|---:|---|---|---|---|")

    for i, item in enumerate(invs[:20], start=1):
        cg = item.get("column_groups", {})
        lines.append(
            f"| {i} | {item.get('num_rows')} | {item.get('num_cols')} | "
            f"{item.get('table_score'):.2f} | `{item.get('relative_path', item.get('path'))}` | "
            f"`{cg.get('status', [])}` | `{cg.get('drugbank_id', [])}` | `{cg.get('umls', [])}` |"
        )

    lines.append("")
    lines.append("## Best table probe")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(best, ensure_ascii=False, indent=2)[:8000])
    lines.append("```")
    lines.append("")
    lines.append("## Day 2 next step")
    lines.append("")
    lines.append("Day 2 should normalize the best table into canonical fields:")
    lines.append("")
    lines.append("```text")
    lines.append("drug_name")
    lines.append("drugbank_id")
    lines.append("drugcentral_id")
    lines.append("disease_name / indication")
    lines.append("umls_cui")
    lines.append("status")
    lines.append("phase")
    lines.append("label: approved vs failed-like")
    lines.append("```")
    lines.append("")
    lines.append("If automatic RData export failed, manually download the full repoDB dataset from the repoDB web app and place it under `dataset/setting_f_repodb/raw_inventory/manual/`, then rerun Day 1 with `--skip-download`.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--manual-only", action="store_true")
    args = parser.parse_args()

    for p in [DOWNLOAD_DIR, EXPORT_DIR, MANUAL_DIR, RESULT_DIR, REPORT_DIR]:
        mkdir(p)

    rdata_path = DOWNLOAD_DIR / "shiny.RData"

    if args.manual_only:
        download_status = {
            "status": "manual_only",
            "path": None,
            "note": "Skipping RData download and using files under manual/ only.",
        }
    elif args.skip_download:
        download_status = {
            "status": "skipped",
            "path": str(rdata_path),
            "exists": rdata_path.exists(),
            "size_mb": file_size_mb(rdata_path),
        }
    else:
        download_status = download_file(RDATA_URLS, rdata_path, force=args.force_download)

    export_status = {
        "pyreadr": None,
        "Rscript": None,
    }

    if rdata_path.exists() and rdata_path.stat().st_size > 0 and not args.manual_only:
        export_status["pyreadr"] = export_with_pyreadr(rdata_path, EXPORT_DIR)

        # Run Rscript if pyreadr unavailable or exported no tables.
        py_status = export_status["pyreadr"].get("status")
        if py_status not in {"success"}:
            export_status["Rscript"] = export_with_rscript(rdata_path, EXPORT_DIR)
        else:
            export_status["Rscript"] = {
                "method": "Rscript",
                "status": "skipped_pyreadr_success",
                "exported_files": [],
            }
    else:
        export_status["pyreadr"] = {
            "method": "pyreadr",
            "status": "skipped_no_rdata",
            "exported_files": [],
        }
        export_status["Rscript"] = {
            "method": "Rscript",
            "status": "skipped_no_rdata",
            "exported_files": [],
        }

    table_inventory, best_table = inventory_tables()
    best_copy = copy_best_table(best_table)

    decision = "DAY1_REPODB_RAW_INVENTORY_READY"
    if not best_table:
        decision = "DAY1_REPODB_NEEDS_MANUAL_DOWNLOAD_OR_RDATA_EXPORT"

    manifest = {
        "created_at": now_iso(),
        "setting": "setting_f_repodb",
        "dataset": "repoDB",
        "download_status": download_status,
        "export_status": export_status,
        "manual_dir": str(MANUAL_DIR),
        "export_dir": str(EXPORT_DIR),
        "files": [
            {
                "path": str(p),
                "size_mb": file_size_mb(p),
            }
            for p in sorted(RAW_DIR.rglob("*"))
            if p.is_file()
        ],
    }

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "setting": "setting_f_repodb",
        "dataset": "repoDB",
        "download_status": download_status,
        "export_status": export_status,
        "table_inventory": table_inventory,
        "best_table": best_table,
        "best_table_copy": best_copy,
        "next_step": "Normalize repoDB status/drug/disease fields and select task schema.",
    }

    write_json(manifest, RAW_DIR / "raw_file_manifest.json")
    write_json(table_inventory, RAW_DIR / "table_inventory.json")
    write_json(best_table, RAW_DIR / "best_table_probe.json")
    write_json(summary, RAW_DIR / "raw_inventory_summary.json")
    write_json(summary, RESULT_DIR / "day1_repodb_inventory_summary.json")
    write_report(summary)

    print(json.dumps({
        "decision": decision,
        "download_status": download_status,
        "export_status": export_status,
        "num_tables_found": len(table_inventory),
        "best_table_copy": best_copy,
        "best_table_brief": None if not best_table else {
            "path": best_table.get("path"),
            "num_rows": best_table.get("num_rows"),
            "num_cols": best_table.get("num_cols"),
            "columns": best_table.get("columns"),
            "column_groups": best_table.get("column_groups"),
            "status_summary": best_table.get("status_summary"),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
