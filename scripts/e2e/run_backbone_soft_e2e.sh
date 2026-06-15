#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

CKPT="${CKPT:-outputs/e2e/e2e_primary_checkpoint/checkpoint-final}"
MODEL="meta-llama/Llama-3.2-3B"
KGE="dataset/setting_a/backbone_ready/entity_embeddings_rgcn.pt"

BACKBONE_DS="dataset/setting_a/e2e_infer_ready/backbone_raw"
SOFT_DS="dataset/setting_a/e2e_infer_ready/soft_support_raw"

mkdir -p outputs/e2e

echo "============================================================"
echo "[1/2] Running backbone_raw E2E infer on test"
echo "============================================================"
python infer.py \
  --model_name_or_path "$MODEL" \
  --model_type llama \
  --dataset_path "$BACKBONE_DS" \
  --kge_embedding_path "$KGE" \
  --checkpoint_dir "$CKPT" \
  --eval_split test \
  --source_max_len 768 \
  --target_max_len 64 \
  --candidate_k 20 \
  --subgraph_tau 100 \
  --max_new_tokens 16 \
  --output_suffix backbone_raw_e2e \
  2>&1 | tee outputs/e2e/backbone_raw_e2e.log

echo "============================================================"
echo "[2/2] Running soft_support_raw E2E infer on test"
echo "============================================================"
python infer.py \
  --model_name_or_path "$MODEL" \
  --model_type llama \
  --dataset_path "$SOFT_DS" \
  --kge_embedding_path "$KGE" \
  --checkpoint_dir "$CKPT" \
  --eval_split test \
  --source_max_len 768 \
  --target_max_len 64 \
  --candidate_k 20 \
  --subgraph_tau 100 \
  --max_new_tokens 16 \
  --output_suffix soft_support_raw_e2e \
  2>&1 | tee outputs/e2e/soft_support_raw_e2e.log

echo "backbone_raw and soft_support_raw infer runs finished."
