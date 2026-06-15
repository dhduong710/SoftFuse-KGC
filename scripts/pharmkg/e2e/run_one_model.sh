#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME="${MODEL_NAME:?Please set MODEL_NAME}"
MODEL_TAG="${MODEL_TAG:?Please set MODEL_TAG}"
MODEL_TYPE="${MODEL_TYPE:-llama}"
USE_QUANT="${USE_QUANT:-False}"

ROOT="dataset/setting_c_pharmkg/e2e_infer_ready"
EMB="${ROOT}/entity_embeddings_rgcn.pt"

OUT_ROOT="outputs/pharmkg/e2e/model_compare"
TRAIN_OUT="${OUT_ROOT}/${MODEL_TAG}/main_checkpoint"
CKPT="${TRAIN_OUT}/checkpoint-final"

mkdir -p "${TRAIN_OUT}"
mkdir -p "outputs/pharmkg/reports/model_compare"

echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_TAG=${MODEL_TAG}"
echo "MODEL_TYPE=${MODEL_TYPE}"
echo "USE_QUANT=${USE_QUANT}"
echo "TRAIN_OUT=${TRAIN_OUT}"

rm -rf "${TRAIN_OUT}"

python main.py \
  --dataset_path "${ROOT}/backbone_raw" \
  --model_name_or_path "${MODEL_NAME}" \
  --model_type "${MODEL_TYPE}" \
  --kge_embedding_path "${EMB}" \
  --graph_num_rels 28 \
  --output_dir "${TRAIN_OUT}" \
  --source_max_len 768 \
  --target_max_len 64 \
  --use_quant "${USE_QUANT}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --dataloader_num_workers 4 \
  --learning_rate 0.0002 \
  --lr_scheduler_type constant \
  --warmup_ratio 0.03 \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --save_strategy no \
  --logging_steps 50 \
  --report_to none \
  --bf16 True

echo "[OK] Training done for ${MODEL_TAG}"

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

echo "[DONE] ${MODEL_TAG}"