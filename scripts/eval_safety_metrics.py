from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safety_violation_at_k_from_flags(contra_flags: list[int], k: int) -> int:
    return int(any(int(x) == 1 for x in contra_flags[:k]))


def contra_at_k_from_flags(contra_flags: list[int], k: int) -> int:
    return sum(int(x) == 1 for x in contra_flags[:k])


def normalize_mock_sample(sample: dict) -> dict:
    topk = [str(x).strip() for x in sample.get("topk", [])]
    contra_set = {str(x).strip() for x in sample.get("contra_set", [])}
    contra_flags = [1 if d in contra_set else 0 for d in topk]

    return {
        "query": str(sample.get("query", "")).strip(),
        "ranked_drugs": topk,
        "contra_flags": contra_flags,
    }


def normalize_annotation_sample(sample: dict) -> dict:
    ranked_drugs = [str(x).strip() for x in sample.get("candidate_drugs", [])]
    contra_flags = [int(x) for x in sample.get("contra_flags", [])]

    if len(ranked_drugs) != len(contra_flags):
        raise ValueError(
            f"candidate_drugs and contra_flags length mismatch: "
            f"{len(ranked_drugs)} vs {len(contra_flags)}"
        )

    return {
        "query": str(sample.get("query_disease", "")).strip(),
        "ranked_drugs": ranked_drugs,
        "contra_flags": contra_flags,
    }


def normalize_samples(data: list[dict]) -> list[dict]:
    normed = []
    for sample in data:
        if "candidate_drugs" in sample and "contra_flags" in sample:
            normed.append(normalize_annotation_sample(sample))
        elif "topk" in sample and "contra_set" in sample:
            normed.append(normalize_mock_sample(sample))
        else:
            raise ValueError(
                "Unsupported sample format. Need either "
                "{candidate_drugs, contra_flags} or {topk, contra_set}."
            )
    return normed


def evaluate(samples: list[dict], ks: list[int]) -> dict:
    results = {
        "num_queries": len(samples),
        "metrics": {},
    }

    for k in ks:
        violation_vals = []
        contra_vals = []

        for sample in samples:
            flags = sample["contra_flags"]
            violation_vals.append(safety_violation_at_k_from_flags(flags, k))
            contra_vals.append(contra_at_k_from_flags(flags, k))

        n = max(len(samples), 1)
        results["metrics"][f"SafetyViolation@{k}"] = sum(violation_vals) / n
        results["metrics"][f"Contra@{k}"] = sum(contra_vals) / n

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSON path")
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
        default="outputs/evaluation/safety_eval_summary.json",
        help="Output JSON summary path",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    data = read_json(in_path)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples.")

    samples = normalize_samples(data)
    summary = evaluate(samples, args.ks)
    summary["input_path"] = str(in_path)
    summary["ks"] = args.ks

    write_json(out_path, summary)

    print("Saved:", out_path)
    print("num_queries =", summary["num_queries"])
    for k, v in summary["metrics"].items():
        print(f"{k} = {v}")
    

if __name__ == "__main__":
    main()