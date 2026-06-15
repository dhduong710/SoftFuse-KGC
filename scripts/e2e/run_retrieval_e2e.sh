#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

CKPT="${CKPT:-outputs/e2e/e2e_primary_checkpoint/checkpoint-final}"
MODEL="meta-llama/Llama-3.2-3B"
KGE="dataset/setting_a/backbone_ready/entity_embeddings_rgcn.pt"
RETR_DS="dataset/setting_a/e2e_infer_ready/retrieval_main"

mkdir -p outputs/e2e

echo "============================================================"
echo "Running retrieval_main E2E infer on test"
echo "============================================================"

python infer.py \
  --model_name_or_path "$MODEL" \
  --model_type llama \
  --dataset_path "$RETR_DS" \
  --kge_embedding_path "$KGE" \
  --checkpoint_dir "$CKPT" \
  --eval_split test \
  --source_max_len 768 \
  --target_max_len 64 \
  --candidate_k 20 \
  --subgraph_tau 100 \
  --max_new_tokens 16 \
  --output_suffix retrieval_main_e2e \
  2>&1 | tee outputs/e2e/retrieval_main_e2e.log

echo "retrieval_main infer finished."
