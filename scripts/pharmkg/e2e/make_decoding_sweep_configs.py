from pathlib import Path
import json

OUT = Path("outputs/pharmkg/e2e/decoding_sweep_valid/configs.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

configs = [
    {"config_id": "cfg01_mnt16_rp100_ng0", "max_new_tokens": 16, "repetition_penalty": 1.00, "no_repeat_ngram_size": 0},
    {"config_id": "cfg02_mnt16_rp105_ng0", "max_new_tokens": 16, "repetition_penalty": 1.05, "no_repeat_ngram_size": 0},
    {"config_id": "cfg03_mnt16_rp110_ng0", "max_new_tokens": 16, "repetition_penalty": 1.10, "no_repeat_ngram_size": 0},
    {"config_id": "cfg04_mnt16_rp110_ng3", "max_new_tokens": 16, "repetition_penalty": 1.10, "no_repeat_ngram_size": 3},
    {"config_id": "cfg05_mnt24_rp100_ng0", "max_new_tokens": 24, "repetition_penalty": 1.00, "no_repeat_ngram_size": 0},
    {"config_id": "cfg06_mnt24_rp105_ng0", "max_new_tokens": 24, "repetition_penalty": 1.05, "no_repeat_ngram_size": 0},
    {"config_id": "cfg07_mnt24_rp110_ng0", "max_new_tokens": 24, "repetition_penalty": 1.10, "no_repeat_ngram_size": 0},
    {"config_id": "cfg08_mnt24_rp110_ng3", "max_new_tokens": 24, "repetition_penalty": 1.10, "no_repeat_ngram_size": 3},
]

payload = {
    "decision": "PHARMKG_VALID_DECODING_SWEEP_CONFIGS_READY",
    "dataset": "PharmKG therapeutic_association_proxy task",
    "selection_split": "valid",
    "selection_row": "fuzzy_retrieval_main",
    "graph_num_rels": 28,
    "top1_copy_collapse_threshold": 0.90,
    "selection_rule": [
        "maximize fuzzy_retrieval_main valid reviewer_safe_mrr_at20",
        "avoid top1-copy collapse if possible",
        "then minimize invalid_prediction_rate",
        "then maximize pred_in_candidate_rate",
        "then minimize candidate_list_fragment_rate",
        "then minimize top1_copy_rate",
        "then maximize average valid MRR across rows",
    ],
    "configs": configs,
}

with OUT.open("w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"[OK] wrote {OUT}")
for c in configs:
    print(c)