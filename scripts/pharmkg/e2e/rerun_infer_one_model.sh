#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME="${MODEL_NAME:?Please set MODEL_NAME}"
MODEL_TAG="${MODEL_TAG:?Please set MODEL_TAG}"
MODEL_TYPE="${MODEL_TYPE:-mistral}"

ROOT="dataset/setting_c_pharmkg/e2e_infer_ready"
EMB="${ROOT}/entity_embeddings_rgcn.pt"

CKPT="outputs/pharmkg/e2e/model_compare/${MODEL_TAG}/main_checkpoint/checkpoint-final"

echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "MODEL_TYPE=${MODEL_TYPE}"
echo "CKPT=${CKPT}"

test -f "${CKPT}/graph_model.bin"
test -f "${CKPT}/adapter_config.json"
if [ ! -f "${CKPT}/adapter_model.bin" ] && [ ! -f "${CKPT}/adapter_model.safetensors" ]; then
  echo "[ERROR] Missing adapter weights in ${CKPT}"
  exit 1
fi

for ROW in backbone_raw soft_support_raw fuzzy_retrieval_main; do
  for SPLIT in valid test; do
    echo ""
    echo "============================================================"
    echo "Infer MODEL=${MODEL_TAG} ROW=${ROW} SPLIT=${SPLIT}"
    echo "============================================================"

    python infer.py \
      --dataset_path "${ROOT}/${ROW}" \
      --model_name_or_path "${MODEL_NAME}" \
      --model_type "${MODEL_TYPE}" \
      --kge_embedding_path "${EMB}" \
      --graph_num_rels 28 \
      --checkpoint_dir "${CKPT}" \
      --eval_split "${SPLIT}" \
      --output_suffix "${ROW}" \
      --max_new_tokens 32 \
      --min_new_tokens 1 \
      --do_sample False \
      --num_beams 1 \
      --temperature 1.0
  done
done

echo "[DONE] rerun inference complete for ${MODEL_TAG}"