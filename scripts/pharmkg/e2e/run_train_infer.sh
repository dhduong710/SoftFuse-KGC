#!/usr/bin/env bash
set -euo pipefail

# =========================
# User config
# =========================
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.2-3B}"
ROOT="dataset/setting_c_pharmkg/e2e_infer_ready"
EMB="${ROOT}/entity_embeddings_rgcn.pt"

OUT_ROOT="outputs/pharmkg/e2e/main"
TRAIN_OUT="${OUT_ROOT}/main_checkpoint"
CKPT="${TRAIN_OUT}/checkpoint-final"

mkdir -p "${OUT_ROOT}"

echo "MODEL_NAME=${MODEL_NAME}"
echo "ROOT=${ROOT}"
echo "EMB=${EMB}"
echo "TRAIN_OUT=${TRAIN_OUT}"
echo "CKPT=${CKPT}"

# =========================
# Train one fixed E2E checkpoint
# =========================
python main.py \
  --dataset_path "${ROOT}/backbone_raw" \
  --model_name_or_path "${MODEL_NAME}" \
  --model_type llama \
  --kge_embedding_path "${EMB}" \
  --graph_num_rels 28 \
  --output_dir "${TRAIN_OUT}" \
  --source_max_len 768 \
  --target_max_len 64 \
  --use_quant False \
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

echo "[OK] Training done."

# =========================
# Verify checkpoint
# =========================
test -f "${CKPT}/graph_model.bin"
test -f "${CKPT}/adapter_config.json"
if [ ! -f "${CKPT}/adapter_model.bin" ] && [ ! -f "${CKPT}/adapter_model.safetensors" ]; then
  echo "[ERROR] Missing adapter weights in ${CKPT}"
  exit 1
fi

echo "[OK] Checkpoint verified: ${CKPT}"
ls -lh "${CKPT}"

# =========================
# Infer all rows
# =========================
for ROW in backbone_raw soft_support_raw fuzzy_retrieval_main; do
  for SPLIT in valid test; do
    echo ""
    echo "============================================================"
    echo "Infer ROW=${ROW} SPLIT=${SPLIT}"
    echo "============================================================"

    python infer.py \
      --dataset_path "${ROOT}/${ROW}" \
      --model_name_or_path "${MODEL_NAME}" \
      --model_type llama \
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

echo ""
echo "[DONE] Day 2 train + infer complete."
echo "Outputs saved under: ${TRAIN_OUT}"