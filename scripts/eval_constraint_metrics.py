from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_type_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path, sep="\t")
    type_col = "final_type" if "final_type" in df.columns else "type"
    return {
        str(row["entity"]).strip(): str(row[type_col]).strip()
        for _, row in df.iterrows()
    }


def relation_is_valid(
    head_type: str,
    relation: str,
    tail_type: str,
    schema_rules: dict[str, list[str]],
) -> bool:
    allowed = schema_rules.get(relation)
    return allowed == [head_type, tail_type]


def evaluate_prediction_mode(
    samples: list[dict],
    type_map: dict[str, str],
    schema_rules: dict[str, list[str]],
    ks: list[int],
) -> dict:
    results = {
        "num_queries": len(samples),
        "metrics": {},
    }

    for k in ks:
        total_predictions = 0
        total_violations = 0
        query_has_violation = 0

        for sample in samples:
            relation = str(sample.get("setting_a_relation", "indication")).strip()
            query_disease = str(sample.get("query_disease", "")).strip()
            candidate_drugs = [str(x).strip() for x in sample.get("candidate_drugs", [])[:k]]

            tail_type = type_map.get(query_disease, "MISSING")
            local_violation = False

            for drug in candidate_drugs:
                head_type = type_map.get(drug, "MISSING")
                total_predictions += 1

                if not relation_is_valid(head_type, relation, tail_type, schema_rules):
                    total_violations += 1
                    local_violation = True

            query_has_violation += int(local_violation)

        denom_queries = max(len(samples), 1)
        denom_preds = max(total_predictions, 1) # total_predictions = samples * k, but just in case some samples have no candidates, we avoid div by zero

        results["metrics"][f"ConstraintViolationRate@{k}"] = total_violations / denom_preds
        results["metrics"][f"QueryHasConstraintViolation@{k}"] = query_has_violation / denom_queries

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input annotation JSON")
    parser.add_argument("--type-map", type=str, required=True, help="Path to type_map.tsv")
    parser.add_argument("--schema", type=str, required=True, help="Path to schema_rules.json")
    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="List of K values",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation/constraint_eval_summary.json",
        help="Output JSON summary path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    type_map_path = Path(args.type_map)
    schema_path = Path(args.schema)
    out_path = Path(args.output)

    samples = read_json(input_path)
    if not isinstance(samples, list):
        raise ValueError("Input annotation JSON must be a list of samples.")

    type_map = load_type_map(type_map_path)
    schema_rules = read_json(schema_path)

    summary = evaluate_prediction_mode(samples, type_map, schema_rules, args.ks)
    summary["input_path"] = str(input_path)
    summary["type_map_path"] = str(type_map_path)
    summary["schema_path"] = str(schema_path)
    summary["ks"] = args.ks

    write_json(out_path, summary)

    print("Saved:", out_path)
    print("num_queries =", summary["num_queries"])
    for k, v in summary["metrics"].items():
        print(f"{k} = {v}")


if __name__ == "__main__":
    main()