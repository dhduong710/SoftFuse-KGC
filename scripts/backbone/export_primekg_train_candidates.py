from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("dataset/setting_a/aligned_evidence/train_aligned_evidence.json")
DEFAULT_OUTPUT = Path("dataset/setting_a/backbone_candidates/train_top20_raw.json")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def to_raw_candidate_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    candidate_entities = row.get("rank_entities")
    candidate_entity_ids = row.get("rank_entities_id")
    if not isinstance(candidate_entities, list) or not isinstance(candidate_entity_ids, list):
        raise ValueError("Aligned row is missing rank_entities/rank_entities_id lists.")
    if len(candidate_entities) != len(candidate_entity_ids):
        raise ValueError("Aligned row has mismatched candidate name/id list lengths.")

    return {
        "split": split,
        "query_entity": row["query_entity"],
        "query_entity_id": row["query_entity_id"],
        "gold_entity": row["gold_entity"],
        "gold_entity_id": row["gold_entity_id"],
        "candidate_entities": candidate_entities,
        "candidate_entity_ids": candidate_entity_ids,
        "gold_rank_in_full_universe": row.get("gold_rank_in_full_universe", row.get("rank")),
        "gold_in_topk_raw": bool(row.get("gold_in_topk_raw", False)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the optional PrimeKG train_top20_raw.json artifact from the "
            "checked-in train_aligned_evidence.json file."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize without writing the output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aligned_rows = load_json(args.input)
    if not isinstance(aligned_rows, list):
        raise TypeError(f"Expected a list of rows in {args.input}")

    raw_rows = [to_raw_candidate_row(row, args.split) for row in aligned_rows]
    candidate_lengths = sorted({len(row["candidate_entities"]) for row in raw_rows})
    gold_in_topk = sum(1 for row in raw_rows if row["gold_in_topk_raw"])

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "num_rows": len(raw_rows),
        "candidate_lengths": candidate_lengths,
        "gold_in_topk_rate": round(gold_in_topk / max(len(raw_rows), 1), 6),
        "dry_run": args.dry_run,
    }

    if not args.dry_run:
        write_json(raw_rows, args.output)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
