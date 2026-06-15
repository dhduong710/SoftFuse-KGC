import json
import shutil
from pathlib import Path
from statistics import mean

import torch


BASE = Path("dataset/setting_c_pharmkg")
READY06 = BASE / "softfuse_ready"
OUT_ROOT = BASE / "e2e_infer_ready"

SOFT_DIR = BASE / "soft_support"
FUZZY_DIR = BASE / "fuzzy_retrieval"
RGCN_STATE = BASE / "baseline_outputs" / "rgcn" / "model_state.pt"

ROWS = ["backbone_raw", "soft_support_raw", "fuzzy_retrieval_main"]
SPLITS = ["valid", "test"]

EXPECTED = {
    "num_entities": 7247,
    "num_rels": 28,
    "candidate_size": 20,
    "train_rows": 28960,
    "valid_rows": 500,
    "test_rows": 500,
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def export_rgcn_embedding():
    if not RGCN_STATE.exists():
        raise FileNotFoundError(f"Missing R-GCN model state: {RGCN_STATE}")

    state = torch.load(RGCN_STATE, map_location="cpu")

    emb = None
    tried = []

    if isinstance(state, dict):
        for key in ["base.weight", "model.base.weight", "module.base.weight"]:
            tried.append(key)
            if key in state:
                emb = state[key]
                break

        if emb is None:
            for outer_key in ["state_dict", "model_state_dict", "model"]:
                if outer_key in state and isinstance(state[outer_key], dict):
                    inner = state[outer_key]
                    for key in ["base.weight", "model.base.weight", "module.base.weight"]:
                        tried.append(f"{outer_key}.{key}")
                        if key in inner:
                            emb = inner[key]
                            break
                    if emb is not None:
                        break

    if emb is None:
        keys = list(state.keys())[:30] if isinstance(state, dict) else []
        raise RuntimeError(
            "Could not find R-GCN entity embedding key. "
            f"Tried: {tried}. Top-level keys: {keys}"
        )

    emb = emb.detach().cpu().float()

    if tuple(emb.shape) != (EXPECTED["num_entities"], 128):
        raise RuntimeError(
            f"Unexpected embedding shape: {tuple(emb.shape)}. "
            f"Expected ({EXPECTED['num_entities']}, 128)."
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / "entity_embeddings_rgcn.pt"
    torch.save(emb, out_path)

    print(f"[OK] exported embedding: {out_path}")
    print(f"     shape = {tuple(emb.shape)}")

    return out_path, tuple(emb.shape)


def rebuild_prompt(query_entity, rank_entities):
    answer_options = "(" + ", ".join([f"'{name}'" for name in rank_entities]) + ")"

    refer_parts = [f"'{query_entity}': [QUERY]"]
    for name in rank_entities:
        refer_parts.append(f"'{name}': [ENTITY]")

    refer_str = ", ".join(refer_parts)
    question = f"What drug is therapeutically associated with {query_entity}?"

    prompt = (
        "You are a biomedical scientist. The task is to predict the answer based on the given question, "
        "and you only need to answer one entity. The answer must be in "
        + answer_options
        + ".\nYou can refer to the entity embeddings: "
        + refer_str
        + ".\n\nQuestion: "
        + question
        + "\nAnswer: "
    )
    return prompt


def compute_rank(output_name, rank_entities):
    try:
        return int(rank_entities.index(output_name) + 1)
    except ValueError:
        return 21


def get_variant_rows(split, row_name):
    if row_name == "backbone_raw":
        return None

    if row_name == "soft_support_raw":
        path = SOFT_DIR / f"{split}_top20_soft_support_main.json"
    elif row_name == "fuzzy_retrieval_main":
        path = FUZZY_DIR / f"{split}_fuzzy_retrieval_main.json"
    else:
        raise ValueError(row_name)

    if not path.exists():
        raise FileNotFoundError(f"Missing variant source: {path}")

    rows = load_json(path)
    by_row_index = {}
    for r in rows:
        if "row_index" not in r:
            raise KeyError(f"Missing row_index in {path}")
        by_row_index[int(r["row_index"])] = r

    return by_row_index


def candidate_names_and_ids(row):
    names = row.get("rank_entities", None)
    ids = row.get("rank_entities_id", None)

    if names is None:
        names = row.get("candidate_entities", None)
    if ids is None:
        ids = row.get("candidate_entity_ids", None)

    if names is None or ids is None:
        raise KeyError("Cannot find candidate/rank entity fields.")

    if len(names) != EXPECTED["candidate_size"]:
        raise RuntimeError(f"Expected 20 candidates, got {len(names)}")

    if len(ids) != EXPECTED["candidate_size"]:
        raise RuntimeError(f"Expected 20 candidate ids, got {len(ids)}")

    return list(names), [int(x) for x in ids]


def selected_subgraph_for(row_name, base_row, variant_row):
    if row_name == "backbone_raw":
        return base_row["subgraph"]

    if row_name == "soft_support_raw":
        return base_row["subgraph"]

    if row_name == "fuzzy_retrieval_main":
        sg = variant_row.get("selected_subgraph")
        if sg is None:
            sg = variant_row.get("subgraph")
        if sg is None:
            raise KeyError("Missing selected_subgraph/subgraph for fuzzy_retrieval_main")
        return sg

    raise ValueError(row_name)


def build_split(row_name, split):
    base_rows = load_json(READY06 / f"{split}.json")
    variant_by_idx = get_variant_rows(split, row_name)

    out_rows = []
    for base_row in base_rows:
        new_row = dict(base_row)
        row_index = int(base_row.get("row_index", len(out_rows)))

        variant_row = None
        if variant_by_idx is not None:
            if row_index not in variant_by_idx:
                raise KeyError(f"row_index={row_index} missing in {row_name}/{split}")
            variant_row = variant_by_idx[row_index]
            rank_entities, rank_entities_id = candidate_names_and_ids(variant_row)
        else:
            rank_entities, rank_entities_id = candidate_names_and_ids(base_row)

        subgraph = selected_subgraph_for(row_name, base_row, variant_row)

        output_name = new_row["output"]
        rank = compute_rank(output_name, rank_entities)

        new_row["rank_entities"] = rank_entities
        new_row["rank_entities_id"] = rank_entities_id
        new_row["candidate_entities"] = rank_entities
        new_row["candidate_entity_ids"] = rank_entities_id
        new_row["rank"] = rank
        new_row["gold_rank_in_top20_or_21"] = rank
        new_row["gold_present"] = bool(rank <= 20)
        new_row["gold_in_topk_e2e"] = bool(rank <= 20)
        new_row["subgraph"] = subgraph
        new_row["input"] = rebuild_prompt(new_row["query_entity"], rank_entities)

        new_row["e2e_row_name"] = row_name
        new_row["e2e_prepare_source"] = {
            "base_ready": str(READY06),
            "variant_source": None if variant_row is None else row_name,
            "split": split,
        }

        if variant_row is not None:
            for key in [
                "support_scores",
                "support_rank_order",
                "candidate_debug_rows",
                "candidate_support_bands",
                "contra_flags",
                "triple_score_rows",
                "path_scores",
                "subgraph_summary",
                "selected_source_variant",
                "variant_name",
            ]:
                if key in variant_row:
                    new_row[key] = variant_row[key]

            if row_name == "fuzzy_retrieval_main":
                new_row["original_subgraph"] = variant_row.get("original_subgraph", base_row.get("subgraph"))
                new_row["selected_subgraph"] = subgraph

        out_rows.append(new_row)

    return out_rows


def copy_static_files(dst: Path):
    static_names = [
        "entity2id.pkl",
        "id2entity.pkl",
        "relation2id.pkl",
        "id2relation.pkl",
        "entity2id.json",
        "id2entity.json",
        "relation2id.json",
        "id2relation.json",
        "prompt_lexicon.json",
        "rules.json",
        "support_schema.json",
    ]

    for name in static_names:
        src = READY06 / name
        if src.exists():
            shutil.copy2(src, dst / name)


def validate_rows(rows, split, row_name, emb_shape):
    errors = []
    subgraph_sizes = []
    max_rel = -1
    max_ent = -1

    for i, r in enumerate(rows):
        prefix = f"{row_name}/{split}/row={i}"

        if "[QUERY]" not in r["input"]:
            errors.append(f"{prefix}: missing [QUERY] in input")

        n_entity_tokens = r["input"].count("[ENTITY]")
        if n_entity_tokens != EXPECTED["candidate_size"]:
            errors.append(f"{prefix}: [ENTITY] count {n_entity_tokens} != 20")

        if len(r["rank_entities"]) != EXPECTED["candidate_size"]:
            errors.append(f"{prefix}: rank_entities length != 20")

        if len(r["rank_entities_id"]) != EXPECTED["candidate_size"]:
            errors.append(f"{prefix}: rank_entities_id length != 20")

        expected_rank = compute_rank(r["output"], r["rank_entities"])
        if int(r["rank"]) != expected_rank:
            errors.append(f"{prefix}: rank={r['rank']} expected={expected_rank}")

        for ent_id in [r["query_entity_id"]] + list(r["rank_entities_id"]):
            ent_id = int(ent_id)
            max_ent = max(max_ent, ent_id)
            if ent_id < 0 or ent_id >= emb_shape[0]:
                errors.append(f"{prefix}: entity id out of range: {ent_id}")

        sg = r.get("subgraph", [])
        subgraph_sizes.append(len(sg))

        for edge in sg:
            if len(edge) != 3:
                errors.append(f"{prefix}: bad edge format: {edge}")
                continue
            h, rel, t = int(edge[0]), int(edge[1]), int(edge[2])
            max_ent = max(max_ent, h, t)
            max_rel = max(max_rel, rel)

            if h < 0 or h >= emb_shape[0] or t < 0 or t >= emb_shape[0]:
                errors.append(f"{prefix}: subgraph entity id out of range: {edge}")

            if rel < 0 or rel >= EXPECTED["num_rels"]:
                errors.append(f"{prefix}: relation id out of range for PharmKG num_rels=28: {edge}")

    summary = {
        "num_rows": len(rows),
        "avg_subgraph_size": round(mean(subgraph_sizes), 6) if subgraph_sizes else 0.0,
        "min_subgraph_size": min(subgraph_sizes) if subgraph_sizes else 0,
        "max_subgraph_size": max(subgraph_sizes) if subgraph_sizes else 0,
        "gold_present_rate": round(mean([1.0 if int(r["rank"]) <= 20 else 0.0 for r in rows]), 6),
        "rank21_count": int(sum(1 for r in rows if int(r["rank"]) == 21)),
        "max_relation_id_seen": int(max_rel),
        "max_entity_id_seen": int(max_ent),
        "errors": errors[:50],
        "num_errors": len(errors),
    }

    if errors:
        raise RuntimeError(
            f"Validation failed for {row_name}/{split}. "
            f"num_errors={len(errors)}. First errors: {errors[:5]}"
        )

    return summary


def build_one_row(row_name, emb_shape):
    dst = OUT_ROOT / row_name
    dst.mkdir(parents=True, exist_ok=True)

    copy_static_files(dst)

    train_src = READY06 / "train.json"
    train_dst = dst / "train.json"
    if not train_src.exists():
        raise FileNotFoundError(
            f"Missing {train_src}. Regenerate it with: "
            "python scripts/pharmkg/build_softfuse_ready_package.py"
        )
    shutil.copy2(train_src, train_dst)

    split_summaries = {}

    train_rows = load_json(train_dst)
    split_summaries["train"] = {
        "num_rows": len(train_rows),
        "note": "Copied from softfuse_ready/train.json for supervised E2E training compatibility.",
    }

    for split in SPLITS:
        rows = build_split(row_name, split)
        save_json(rows, dst / f"{split}.json")
        split_summaries[split] = validate_rows(rows, split, row_name, emb_shape)

    manifest = {
        "decision": "PHARMKG_E2E_READY_PACKAGE_BUILT",
        "row_name": row_name,
        "output_dir": str(dst),
        "source_ready_package": str(READY06),
        "embedding_path": str(OUT_ROOT / "entity_embeddings_rgcn.pt"),
        "graph_num_rels": EXPECTED["num_rels"],
        "target_relation": "T",
        "target_relation_id": 23,
        "target_relation_normalized": "therapeutic_association_proxy",
        "splits": split_summaries,
        "important_notes": [
            "Use --graph_num_rels 28 in main.py and infer.py.",
            "Use reviewer-safe metric: RR=1/rank if rank<=20 else 0.",
            "Do not call PharmKG relation T clinical indication; use therapeutic_association_proxy.",
            "backbone_raw and soft_support_raw keep original 100-edge subgraphs.",
            "fuzzy_retrieval_main uses selected_subgraph from 11_fuzzy_retrieval.",
        ],
    }

    save_json(manifest, dst / "prep_manifest.json")
    print(f"[OK] built {row_name}: {dst}")
    return manifest


def write_report(root_manifest):
    lines = []
    lines.append("# Week 23 E2E PharmKG Day 1 Prepare Report")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append("PHARMKG_E2E_READY_PACKAGE_BUILT")
    lines.append("")
    lines.append("## Embedding")
    lines.append("")
    lines.append(f"- Path: `{root_manifest['embedding_path']}`")
    lines.append(f"- Shape: `{root_manifest['embedding_shape']}`")
    lines.append("- Required E2E argument: `--graph_num_rels 28`")
    lines.append("")
    lines.append("## Ready rows")
    lines.append("")

    for row_name, manifest in root_manifest["rows"].items():
        lines.append(f"### {row_name}")
        lines.append("")
        lines.append(f"- Output dir: `{manifest['output_dir']}`")
        for split in ["train", "valid", "test"]:
            s = manifest["splits"][split]
            if split == "train":
                lines.append(f"- {split}: {s['num_rows']} rows")
            else:
                lines.append(
                    f"- {split}: rows={s['num_rows']}, "
                    f"Gold@20={s['gold_present_rate']}, "
                    f"Rank21={s['rank21_count']}, "
                    f"avg_subgraph_size={s['avg_subgraph_size']}, "
                    f"max_rel_id={s['max_relation_id_seen']}"
                )
        lines.append("")

    lines.append("## Day 2 command requirements")
    lines.append("")
    lines.append("Use the exported embedding and graph relation count:")
    lines.append("")
    lines.append("```bash")
    lines.append("--kge_embedding_path dataset/setting_c_pharmkg/e2e_infer_ready/entity_embeddings_rgcn.pt \\")
    lines.append("--graph_num_rels 28")
    lines.append("```")
    lines.append("")

    report_path = OUT_ROOT / "day1_prepare_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote report: {report_path}")


def main():
    if not READY06.exists():
        raise FileNotFoundError(f"Missing ready package: {READY06}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    emb_path, emb_shape = export_rgcn_embedding()

    root_manifest = {
        "decision": "PHARMKG_E2E_DAY1_READY",
        "output_root": str(OUT_ROOT),
        "embedding_path": str(emb_path),
        "embedding_shape": list(emb_shape),
        "graph_num_rels": EXPECTED["num_rels"],
        "rows": {},
    }

    for row_name in ROWS:
        manifest = build_one_row(row_name, emb_shape)
        root_manifest["rows"][row_name] = manifest

    save_json(root_manifest, OUT_ROOT / "day1_prepare_manifest.json")
    write_report(root_manifest)

    print("\n[DONE] Day 1 PharmKG E2E preparation complete.")
    print(f"Output root: {OUT_ROOT}")
    print(f"Embedding: {emb_path}")
    print("Use --graph_num_rels 28 for train/infer.")


if __name__ == "__main__":
    main()
