#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(".")
SETTING_DIR = ROOT / "dataset" / "setting_f_repodb"

RAW_DIR = SETTING_DIR / "raw_inventory"
TASK_DIR = SETTING_DIR / "task_spec"

RESULT_DIR = ROOT / "outputs" / "repodb"
REPORT_DIR = ROOT / "outputs" / "repodb" / "reports"

DRKG_GRAPH_DIR = ROOT / "dataset" / "setting_e_drkg" / "graph"

TARGET_RELATION = "repoDB::approved_indication::Compound:Disease"
FAILED_RELATION = "repoDB::failed_or_suspended::Compound:Disease"

TOP_K = 20
ABSENT_RANK = 21


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_text(x: Any) -> str:
    s = "" if x is None else str(x)
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_drugbank_id(x: Any) -> str:
    s = clean_text(x).upper()
    m = re.search(r"\bDB\d{5}\b", s)
    return m.group(0) if m else ""


def normalize_cui(x: Any) -> str:
    s = clean_text(x).upper()
    m = re.search(r"\bC\d{7}\b", s)
    return m.group(0) if m else ""


def canonical_compound_id(drug_id: str) -> str:
    return f"Compound::{drug_id}"


def canonical_disease_id(cui: str) -> str:
    return f"Disease::UMLS:{cui}"


def normalize_status(status: Any, trial_status: Any = "", detailed_status: Any = "", phase: Any = "") -> dict[str, Any]:
    status_s = clean_text(status)
    trial_s = clean_text(trial_status)
    detailed_s = clean_text(detailed_status)
    phase_s = clean_text(phase)

    combined = " ".join([status_s, trial_s, detailed_s, phase_s]).lower()

    if status_s.lower() == "approved" or "approved" in trial_s.lower():
        label = "approved"
        relation = TARGET_RELATION
        is_positive = True
        is_failed_like = False
    elif any(tok in combined for tok in ["terminated", "withdrawn", "suspended", "failed"]):
        label = "failed_like"
        relation = FAILED_RELATION
        is_positive = False
        is_failed_like = True
    else:
        label = "unknown_or_other"
        relation = None
        is_positive = False
        is_failed_like = False

    return {
        "status_raw": status_s,
        "trial_status_raw": trial_s,
        "detailed_status_raw": detailed_s,
        "phase_raw": phase_s,
        "label": label,
        "relation": relation,
        "is_positive": is_positive,
        "is_failed_like": is_failed_like,
    }


def load_raw_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def normalize_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []

    for i, r in df.iterrows():
        drug_id = normalize_drugbank_id(r.get("drug_id", "")) or normalize_drugbank_id(r.get("Drug", ""))
        cui = normalize_cui(r.get("ind_id", "")) or normalize_cui(r.get("Indication", ""))

        drug_name = clean_text(r.get("drug_name", ""))
        if not drug_name:
            drug_name = clean_text(r.get("Drug", ""))
            drug_name = re.sub(r"\(DBID:\s*DB\d{5}\)", "", drug_name).strip()

        disease_name = clean_text(r.get("ind_name", ""))
        if not disease_name:
            disease_name = clean_text(r.get("Indication", ""))
            disease_name = re.sub(r"\(CUI:\s*C\d{7}\)", "", disease_name).strip()

        sem_type = clean_text(r.get("sem_type", ""))

        st = normalize_status(
            status=r.get("status", ""),
            trial_status=r.get("TrialStatus", ""),
            detailed_status=r.get("DetailedStatus", ""),
            phase=r.get("phase", ""),
        )

        valid_ids = bool(drug_id and cui)

        compound_entity = canonical_compound_id(drug_id) if drug_id else ""
        disease_entity = canonical_disease_id(cui) if cui else ""

        row = {
            "raw_row_index": int(i),
            "drug_name": drug_name,
            "drugbank_id": drug_id,
            "compound_entity": compound_entity,
            "disease_name": disease_name,
            "umls_cui": cui,
            "disease_entity": disease_entity,
            "sem_type": sem_type,
            **st,
            "valid_drugbank_and_cui": valid_ids,
            "source_dataset": "repoDB",
            "setting": "setting_f_repodb",
        }
        rows.append(row)

    return rows


def write_rows_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def dedupe_pairs(rows: list[dict[str, Any]], label: str, relation: str) -> list[dict[str, Any]]:
    pair_map = {}

    for r in rows:
        if r["label"] != label:
            continue
        if not r["valid_drugbank_and_cui"]:
            continue

        key = (r["compound_entity"], r["disease_entity"])

        if key not in pair_map:
            pair_map[key] = {
                "compound_entity": r["compound_entity"],
                "drugbank_id": r["drugbank_id"],
                "drug_name": r["drug_name"],
                "relation": relation,
                "disease_entity": r["disease_entity"],
                "umls_cui": r["umls_cui"],
                "disease_name": r["disease_name"],
                "sem_type": r["sem_type"],
                "label": label,
                "num_raw_rows": 0,
                "raw_row_indices": [],
                "trial_status_values": Counter(),
                "phase_values": Counter(),
                "detailed_status_values": Counter(),
            }

        obj = pair_map[key]
        obj["num_raw_rows"] += 1
        obj["raw_row_indices"].append(int(r["raw_row_index"]))
        obj["trial_status_values"][r["trial_status_raw"]] += 1
        obj["phase_values"][r["phase_raw"]] += 1
        obj["detailed_status_values"][r["detailed_status_raw"]] += 1

    out = []
    for obj in pair_map.values():
        obj["trial_status_values"] = dict(obj["trial_status_values"].most_common(20))
        obj["phase_values"] = dict(obj["phase_values"].most_common(20))
        obj["detailed_status_values"] = dict(obj["detailed_status_values"].most_common(20))
        out.append(obj)

    out = sorted(out, key=lambda x: (x["disease_entity"], x["compound_entity"]))
    return out


def write_edge_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        for r in rows:
            writer.writerow([r["compound_entity"], r["relation"], r["disease_entity"]])


def write_pair_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "compound_entity", "drugbank_id", "drug_name",
        "relation",
        "disease_entity", "umls_cui", "disease_name", "sem_type",
        "label", "num_raw_rows",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def find_conflicts(approved: list[dict[str, Any]], failed: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    approved_keys = {(r["compound_entity"], r["disease_entity"]) for r in approved}
    failed_keys = {(r["compound_entity"], r["disease_entity"]) for r in failed}
    conflict_keys = approved_keys & failed_keys

    conflict_rows = []
    for key in sorted(conflict_keys):
        a = next(r for r in approved if (r["compound_entity"], r["disease_entity"]) == key)
        f = next(r for r in failed if (r["compound_entity"], r["disease_entity"]) == key)

        conflict_rows.append({
            "compound_entity": key[0],
            "disease_entity": key[1],
            "drugbank_id": a["drugbank_id"],
            "drug_name": a["drug_name"],
            "umls_cui": a["umls_cui"],
            "disease_name": a["disease_name"],
            "approved_raw_count": a["num_raw_rows"],
            "failed_raw_count": f["num_raw_rows"],
        })

    failed_clean = [
        r for r in failed
        if (r["compound_entity"], r["disease_entity"]) not in conflict_keys
    ]

    return conflict_rows, failed_clean


def try_coverage_split(edges: list[tuple[str, str, str]], valid_size: int, test_size: int, seed: int, max_attempts: int = 300) -> dict[str, Any]:
    target_holdout = valid_size + test_size
    unique_edges = sorted(set(edges))

    best_holdout = []
    best_attempt = None

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        shuffled = unique_edges[:]
        rng.shuffle(shuffled)

        train_set = set(unique_edges)
        h_count = Counter(h for h, _, _ in train_set)
        t_count = Counter(t for _, _, t in train_set)

        heldout = []

        for e in shuffled:
            if len(heldout) >= target_holdout:
                break

            h, rel, t = e
            if h_count[h] <= 1:
                continue
            if t_count[t] <= 1:
                continue

            train_set.remove(e)
            h_count[h] -= 1
            t_count[t] -= 1
            heldout.append(e)

        if len(heldout) > len(best_holdout):
            best_holdout = heldout
            best_attempt = attempt

        if len(heldout) >= target_holdout:
            return {
                "requested_valid_size": valid_size,
                "requested_test_size": test_size,
                "requested_holdout": target_holdout,
                "feasible": True,
                "attempt_used": attempt,
                "actual_holdout_possible": len(heldout),
                "projected_train_size": len(unique_edges) - target_holdout,
                "projected_valid_size": valid_size,
                "projected_test_size": test_size,
                "coverage_policy": "valid/test drugs and diseases remain in train approved target relation",
            }

    return {
        "requested_valid_size": valid_size,
        "requested_test_size": test_size,
        "requested_holdout": target_holdout,
        "feasible": False,
        "best_attempt": best_attempt,
        "best_holdout_found": len(best_holdout),
        "projected_train_size_if_best": len(unique_edges) - len(best_holdout),
        "coverage_policy": "valid/test drugs and diseases remain in train approved target relation",
    }


def choose_split(edges: list[tuple[str, str, str]], seed: int) -> dict[str, Any]:
    candidates = [
        (500, 500),
        (400, 400),
        (300, 300),
        (250, 250),
        (200, 200),
        (100, 100),
    ]

    attempts = []

    for valid_size, test_size in candidates:
        res = try_coverage_split(edges, valid_size, test_size, seed=seed)
        attempts.append(res)
        if res["feasible"]:
            return {
                "decision": "SPLIT_SIZE_FEASIBLE",
                "recommended_valid_size": valid_size,
                "recommended_test_size": test_size,
                "recommended_train_size": len(edges) - valid_size - test_size,
                "attempts": attempts,
            }

    return {
        "decision": "NO_CANDIDATE_SPLIT_SIZE_FEASIBLE",
        "recommended_valid_size": None,
        "recommended_test_size": None,
        "recommended_train_size": None,
        "attempts": attempts,
    }


def load_drkg_entities() -> set[str]:
    p = DRKG_GRAPH_DIR / "entity2id.json"
    if not p.exists():
        return set()
    return set(read_json(p).keys())


def mapping_probe(approved: list[dict[str, Any]], failed_clean: list[dict[str, Any]]) -> dict[str, Any]:
    drkg_entities = load_drkg_entities()

    all_rows = approved + failed_clean
    compounds = sorted({r["compound_entity"] for r in all_rows})
    diseases = sorted({r["disease_entity"] for r in all_rows})

    compound_exact = [x for x in compounds if x in drkg_entities]

    # Try multiple disease exact forms because DRKG may use Disease::MESH, Disease::OMIM, or other namespaces.
    disease_exact_umls = [x for x in diseases if x in drkg_entities]

    cui_to_forms = {}
    for d in diseases:
        cui = d.split("Disease::UMLS:", 1)[-1]
        forms = [
            f"Disease::UMLS:{cui}",
            f"Disease::{cui}",
            cui,
        ]
        cui_to_forms[d] = forms

    disease_any_exact = []
    disease_form_hits = {}

    for d, forms in cui_to_forms.items():
        hits = [f for f in forms if f in drkg_entities]
        if hits:
            disease_any_exact.append(d)
            disease_form_hits[d] = hits

    approved_compounds = sorted({r["compound_entity"] for r in approved})
    approved_diseases = sorted({r["disease_entity"] for r in approved})

    return {
        "drkg_entity_map_found": bool(drkg_entities),
        "drkg_num_entities": len(drkg_entities),
        "num_unique_compounds_repodb": len(compounds),
        "num_unique_diseases_repodb": len(diseases),
        "num_unique_approved_compounds": len(approved_compounds),
        "num_unique_approved_diseases": len(approved_diseases),
        "compound_exact_hits_in_drkg": len(compound_exact),
        "compound_exact_hit_rate": len(compound_exact) / len(compounds) if compounds else 0.0,
        "approved_compound_exact_hits_in_drkg": len([x for x in approved_compounds if x in drkg_entities]),
        "approved_compound_exact_hit_rate": (
            len([x for x in approved_compounds if x in drkg_entities]) / len(approved_compounds)
            if approved_compounds else 0.0
        ),
        "disease_exact_umls_hits_in_drkg": len(disease_exact_umls),
        "disease_any_exact_hits_in_drkg": len(disease_any_exact),
        "disease_any_exact_hit_rate": len(disease_any_exact) / len(diseases) if diseases else 0.0,
        "sample_compound_hits": compound_exact[:20],
        "sample_compound_misses": [x for x in compounds if x not in drkg_entities][:20],
        "sample_disease_hits": disease_any_exact[:20],
        "sample_disease_misses": [x for x in diseases if x not in disease_form_hits][:20],
        "disease_form_hits_sample": dict(list(disease_form_hits.items())[:20]),
    }


def write_report(summary: dict[str, Any]) -> None:
    path = REPORT_DIR / "day2_repodb_task_schema.md"

    norm = summary["normalization_summary"]
    split = summary["target_relation_feasibility"]["recommended_split"]
    mp = summary["mapping_probe"]

    lines = []
    lines.append("# repoDB normalization and task schema selection")
    lines.append("")
    lines.append(f"- Decision: `{summary['decision']}`")
    lines.append(f"- Task: `{summary['task_spec']['task_form']}`")
    lines.append(f"- Target relation: `{TARGET_RELATION}`")
    lines.append("")
    lines.append("## Normalization summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(norm, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Split feasibility")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(split, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Mapping probe to DRKG")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(mp, ensure_ascii=False, indent=2)[:5000])
    lines.append("```")
    lines.append("")
    lines.append("## Protocol decision")
    lines.append("")
    lines.append("```text")
    lines.append("Positive target = approved repoDB drug-indication pairs.")
    lines.append("Failed-like pairs are reserved as diagnostic/negative evidence, not positive labels.")
    lines.append("Candidate universe = approved-training DrugBank compounds.")
    lines.append("Entity format = Compound::DBxxxxx and Disease::UMLS:Cxxxxxxx.")
    lines.append("No valid/test gold injection.")
    lines.append("```")
    lines.append("")
    lines.append("## Day 3 next step")
    lines.append("")
    lines.append("Day 3 should build the actual coverage-safe split and construct a repoDB evidence graph. If disease CUI mapping to DRKG is weak, use repoDB-local disease nodes and reuse DRKG evidence mainly on the drug side.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args()

    for p in [TASK_DIR, RESULT_DIR, REPORT_DIR]:
        mkdir(p)

    raw_path = RAW_DIR / "repodb_best_table_raw.tsv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing Day 1 best table: {raw_path}")

    df = load_raw_table(raw_path)
    rows = normalize_rows(df)

    approved = dedupe_pairs(rows, label="approved", relation=TARGET_RELATION)
    failed = dedupe_pairs(rows, label="failed_like", relation=FAILED_RELATION)
    conflicts, failed_clean = find_conflicts(approved, failed)

    target_edges = [
        (r["compound_entity"], TARGET_RELATION, r["disease_entity"])
        for r in approved
    ]

    failed_edges = [
        (r["compound_entity"], FAILED_RELATION, r["disease_entity"])
        for r in failed_clean
    ]

    split_rec = choose_split(target_edges, seed=args.seed)
    mp = mapping_probe(approved, failed_clean)

    # Write normalized files.
    write_json(rows, TASK_DIR / "normalized_repodb_rows.json")
    write_rows_tsv(rows, TASK_DIR / "normalized_repodb_rows.tsv")

    write_json(approved, TASK_DIR / "approved_pairs_unique.json")
    write_pair_tsv(approved, TASK_DIR / "approved_pairs_unique.tsv")
    write_edge_tsv(approved, TASK_DIR / "target_edges_unique.tsv")

    write_json(failed_clean, TASK_DIR / "failed_pairs_unique.json")
    write_pair_tsv(failed_clean, TASK_DIR / "failed_pairs_unique.tsv")
    write_edge_tsv(failed_clean, TASK_DIR / "failed_diagnostic_edges_unique.tsv")

    write_json(conflicts, TASK_DIR / "conflict_pairs.json")
    if conflicts:
        write_pair_tsv(conflicts, TASK_DIR / "conflict_pairs.tsv")
    else:
        (TASK_DIR / "conflict_pairs.tsv").write_text("", encoding="utf-8")

    status_counts = Counter(r["label"] for r in rows)
    valid_id_counts = Counter(r["valid_drugbank_and_cui"] for r in rows)

    normalization_summary = {
        "created_at": now_iso(),
        "raw_rows": int(len(rows)),
        "raw_columns": list(df.columns),
        "label_counts_raw": dict(status_counts),
        "valid_drugbank_and_cui_counts": {str(k): int(v) for k, v in valid_id_counts.items()},
        "approved_unique_pairs": len(approved),
        "failed_like_unique_pairs_before_conflict_removal": len(failed),
        "failed_like_unique_pairs_after_conflict_removal": len(failed_clean),
        "conflict_pair_count": len(conflicts),
        "unique_approved_drugs": len({r["compound_entity"] for r in approved}),
        "unique_approved_diseases": len({r["disease_entity"] for r in approved}),
        "unique_failed_drugs": len({r["compound_entity"] for r in failed_clean}),
        "unique_failed_diseases": len({r["disease_entity"] for r in failed_clean}),
        "target_relation": TARGET_RELATION,
        "failed_diagnostic_relation": FAILED_RELATION,
    }

    write_json(normalization_summary, TASK_DIR / "normalization_summary.json")
    write_json(mp, TASK_DIR / "mapping_probe_drkg.json")

    feasibility = {
        "created_at": now_iso(),
        "target_relation": TARGET_RELATION,
        "num_unique_target_edges": len(target_edges),
        "num_unique_compounds": len({h for h, _, _ in target_edges}),
        "num_unique_diseases": len({t for _, _, t in target_edges}),
        "recommended_split": split_rec,
    }
    write_json(feasibility, TASK_DIR / "target_relation_feasibility.json")

    task_spec = {
        "created_at": now_iso(),
        "setting": "setting_f_repodb",
        "dataset": "repoDB",
        "status": "FROZEN_DAY2",
        "task_name": "repodb_approved_indication_head_prediction",
        "task_form": "(?, repoDB_approved_indication, disease)",
        "prediction_type": "predicted_head",
        "target_relation": TARGET_RELATION,
        "target_relation_normalized": "repodb_approved_indication",
        "failed_diagnostic_relation": FAILED_RELATION,
        "target_relation_direction": "Compound->Disease",
        "query_entity_type": "Disease",
        "query_entity_namespace": "UMLS CUI",
        "missing_entity_type": "Compound",
        "compound_namespace": "DrugBank ID",
        "entity_format": {
            "compound": "Compound::DBxxxxx",
            "disease": "Disease::UMLS:Cxxxxxxx",
        },
        "candidate_universe_policy": "train_approved_relation_compound_heads",
        "top_k": TOP_K,
        "gold_injection": False,
        "absent_gold_rank_sentinel": ABSENT_RANK,
        "claim_scope": "Clinical drug-repositioning external validation using approved repoDB pairs; not a claim of prospective clinical validation.",
        "recommended_wording": "repoDB approved drug-indication prediction / clinical drug-repositioning external validation",
        "target_edges_path": str(TASK_DIR / "target_edges_unique.tsv"),
        "failed_diagnostic_edges_path": str(TASK_DIR / "failed_diagnostic_edges_unique.tsv"),
    }
    write_json(task_spec, TASK_DIR / "task_spec.json")

    schema_manifest = {
        "created_at": now_iso(),
        "setting": "setting_f_repodb",
        "dataset": "repoDB",
        "reviewer_safe_policy": {
            "top_k": TOP_K,
            "gold_injection": False,
            "absent_gold_rank_sentinel": ABSENT_RANK,
            "rr_definition": "RR = 1/rank if rank <= 20 else 0",
        },
        "split_policy": {
            "coverage_safe": True,
            "valid_test_gold_drugs_in_train_candidate_universe": True,
            "valid_test_query_diseases_seen_in_train_target": True,
            "recommended_split": split_rec,
        },
        "label_policy": {
            "positive_target": "Approved repoDB pairs only",
            "failed_like_pairs": "Reserved for diagnostic/negative evidence; not used as positive labels",
            "conflict_policy": "If a pair is both approved and failed-like, keep as approved positive and remove from failed-like diagnostic set.",
        },
        "mapping_policy": {
            "compound_mapping_to_drkg": "Exact Compound::DBxxxxx mapping where possible",
            "disease_mapping_to_drkg": "Probe exact CUI forms; if weak, use repoDB-local UMLS disease nodes",
            "day3_graph_policy": "Build repoDB target graph and optionally attach DRKG evidence for mapped compounds/diseases.",
        },
    }
    write_json(schema_manifest, TASK_DIR / "schema_manifest.json")

    decision = "DAY2_REPODB_TASK_SCHEMA_READY"
    if split_rec["decision"] != "SPLIT_SIZE_FEASIBLE":
        decision = "DAY2_REPODB_TASK_SCHEMA_NEEDS_SMALLER_SPLIT"
    if len(approved) < 500:
        decision = "DAY2_REPODB_TOO_FEW_APPROVED_TARGET_EDGES"

    summary = {
        "created_at": now_iso(),
        "decision": decision,
        "normalization_summary": normalization_summary,
        "target_relation_feasibility": feasibility,
        "mapping_probe": mp,
        "task_spec": task_spec,
        "schema_manifest": schema_manifest,
        "next_step": "Day 3 build actual split and evidence graph.",
    }

    write_json(summary, RESULT_DIR / "day2_repodb_task_schema_summary.json")
    write_report(summary)

    print(json.dumps({
        "decision": decision,
        "normalization_summary": normalization_summary,
        "recommended_split": split_rec,
        "mapping_probe_brief": {
            "compound_exact_hit_rate": mp["compound_exact_hit_rate"],
            "approved_compound_exact_hit_rate": mp["approved_compound_exact_hit_rate"],
            "disease_any_exact_hit_rate": mp["disease_any_exact_hit_rate"],
            "compound_hits": mp["compound_exact_hits_in_drkg"],
            "disease_hits": mp["disease_any_exact_hits_in_drkg"],
        },
        "task_spec": task_spec,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
