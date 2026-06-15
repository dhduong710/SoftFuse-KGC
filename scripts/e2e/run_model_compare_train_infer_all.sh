#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ============================================================
# User config
# ============================================================
ROOT="dataset/setting_a/e2e_infer_ready"
KGE="dataset/setting_a/backbone_ready/entity_embeddings_rgcn.pt"
OUT_ROOT="outputs/e2e/model_compare"
LOG_ROOT="outputs/e2e/reports/model_compare"

BEST_JSON="outputs/e2e/decoding_sweep_valid/decoding_sweep_best_config.json"

# Existing Llama-3.2-3B checkpoint. This path should directly contain
# adapter_config.json, adapter_model.*, and graph_model.bin.
LLAMA32_CKPT="${LLAMA32_CKPT:-outputs/e2e/e2e_primary_checkpoint/checkpoint-final}"

# Set these paths/names before running if needed.
LLAMA32_MODEL="${LLAMA32_MODEL:-meta-llama/Llama-3.2-3B}"
LLAMA3_8B_MODEL="${LLAMA3_8B_MODEL:-meta-llama/Meta-Llama-3-8B}"

# You should set MEDLLAMA3_8B_MODEL explicitly, e.g.
# export MEDLLAMA3_8B_MODEL=/path/to/medllama3-8b
MEDLLAMA3_8B_MODEL="${MEDLLAMA3_8B_MODEL:?Please set MEDLLAMA3_8B_MODEL to your MedLlama-3-8B HF name or local path}"

# By default, reuse the existing Llama-3.2-3B checkpoint.
# Set RETRAIN_LLAMA32=1 if you want to train it again under outputs/e2e/model_compare.
RETRAIN_LLAMA32="${RETRAIN_LLAMA32:-0}"

# Continue to next model if one model fails.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

test -f "${BEST_JSON}"
test -f "${KGE}"

# Read selected decoding config from Day 2.
read MAX_NEW REP_PEN NGRAM CONFIG_ID < <(python - <<'PY'
import json
p = "outputs/e2e/decoding_sweep_valid/decoding_sweep_best_config.json"
obj = json.load(open(p, encoding="utf-8"))
b = obj["best_config"]
print(b["max_new_tokens"], b["repetition_penalty"], b["no_repeat_ngram_size"], b["config_id"])
PY
)

echo "============================================================"
echo "PrimeKG model comparison train/infer"
echo "============================================================"
echo "ROOT=${ROOT}"
echo "KGE=${KGE}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "CONFIG_ID=${CONFIG_ID}"
echo "MAX_NEW=${MAX_NEW}"
echo "REP_PEN=${REP_PEN}"
echo "NGRAM=${NGRAM}"
echo "RETRAIN_LLAMA32=${RETRAIN_LLAMA32}"
echo "CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

verify_ckpt() {
  local CKPT="$1"
  test -f "${CKPT}/graph_model.bin"
  test -f "${CKPT}/adapter_config.json"
  if [ ! -f "${CKPT}/adapter_model.bin" ] && [ ! -f "${CKPT}/adapter_model.safetensors" ]; then
    echo "[ERROR] Missing adapter weights in ${CKPT}"
    return 1
  fi
}

train_model() {
  local TAG="$1"
  local MODEL="$2"
  local MODEL_TYPE="$3"
  local TRAIN_OUT="${OUT_ROOT}/${TAG}/main_checkpoint"

  echo ""
  echo "============================================================"
  echo "[TRAIN] TAG=${TAG}"
  echo "MODEL=${MODEL}"
  echo "MODEL_TYPE=${MODEL_TYPE}"
  echo "TRAIN_OUT=${TRAIN_OUT}"
  echo "============================================================"

  rm -rf "${TRAIN_OUT}"
  mkdir -p "${TRAIN_OUT}"

  python main.py \
    --dataset_path "${ROOT}/backbone_raw" \
    --model_name_or_path "${MODEL}" \
    --model_type "${MODEL_TYPE}" \
    --kge_embedding_path "${KGE}" \
    --graph_num_rels 4 \
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
    --bf16 True \
    2>&1 | tee "${LOG_ROOT}/day4_${TAG}_train.log"

  verify_ckpt "${TRAIN_OUT}/checkpoint-final"
  echo "[OK] checkpoint verified: ${TRAIN_OUT}/checkpoint-final"
}

infer_model() {
  local TAG="$1"
  local MODEL="$2"
  local MODEL_TYPE="$3"
  local CKPT="$4"

  local MODEL_DIR="${OUT_ROOT}/${TAG}"
  local PRED_OUT="${MODEL_DIR}/predictions"
  mkdir -p "${PRED_OUT}"

  verify_ckpt "${CKPT}"

  echo ""
  echo "============================================================"
  echo "[INFER] TAG=${TAG}"
  echo "MODEL=${MODEL}"
  echo "MODEL_TYPE=${MODEL_TYPE}"
  echo "CKPT=${CKPT}"
  echo "============================================================"

  local CKPT_PARENT
  CKPT_PARENT="$(dirname "${CKPT}")"

  for ROW in backbone_raw soft_support_raw retrieval_main; do
    for SPLIT in valid test; do
      echo ""
      echo "------------------------------------------------------------"
      echo "Infer TAG=${TAG} ROW=${ROW} SPLIT=${SPLIT}"
      echo "------------------------------------------------------------"

      local SUFFIX="w24mc_${TAG}_${ROW}"

      python infer.py \
        --model_name_or_path "${MODEL}" \
        --model_type "${MODEL_TYPE}" \
        --dataset_path "${ROOT}/${ROW}" \
        --kge_embedding_path "${KGE}" \
        --graph_num_rels 4 \
        --checkpoint_dir "${CKPT}" \
        --eval_split "${SPLIT}" \
        --source_max_len 768 \
        --target_max_len 64 \
        --max_new_tokens "${MAX_NEW}" \
        --min_new_tokens 1 \
        --do_sample False \
        --num_beams 1 \
        --temperature 1.0 \
        --repetition_penalty "${REP_PEN}" \
        --no_repeat_ngram_size "${NGRAM}" \
        --output_suffix "${SUFFIX}" \
        2>&1 | tee "${LOG_ROOT}/day4_${TAG}_infer_${SPLIT}_${ROW}.log"

      local SRC_PRED="${CKPT_PARENT}/prediction_${SPLIT}_${SUFFIX}.json"
      local SRC_METRICS="${CKPT_PARENT}/ranking_metrics_${SPLIT}_${SUFFIX}.json"
      local SRC_TXT="${CKPT_PARENT}/metrics_${SPLIT}_${SUFFIX}.txt"

      test -f "${SRC_PRED}"
      cp "${SRC_PRED}" "${PRED_OUT}/prediction_${SPLIT}_${ROW}.json"

      if [ -f "${SRC_METRICS}" ]; then
        cp "${SRC_METRICS}" "${PRED_OUT}/raw_ranking_metrics_${SPLIT}_${ROW}.json"
      fi
      if [ -f "${SRC_TXT}" ]; then
        cp "${SRC_TXT}" "${PRED_OUT}/raw_metrics_${SPLIT}_${ROW}.txt"
      fi
    done
  done

  cat > "${MODEL_DIR}/model_run_manifest.json" <<EOF
{
  "model_tag": "${TAG}",
  "model_name_or_path": "${MODEL}",
  "model_type": "${MODEL_TYPE}",
  "checkpoint_dir": "${CKPT}",
  "dataset_root": "${ROOT}",
  "kge_embedding_path": "${KGE}",
  "graph_num_rels": 4,
  "selected_config_id": "${CONFIG_ID}",
  "max_new_tokens": ${MAX_NEW},
  "repetition_penalty": ${REP_PEN},
  "no_repeat_ngram_size": ${NGRAM},
  "splits": ["valid", "test"],
  "rows": ["backbone_raw", "soft_support_raw", "retrieval_main"],
  "reviewer_safe_metric_note": "Raw infer.py metrics are audit-only. The collect script recomputes reviewer-safe metrics."
}
EOF

  echo "[OK] inference complete for ${TAG}"
}

run_one_model() {
  local TAG="$1"
  local MODEL="$2"
  local MODEL_TYPE="$3"
  local MODE="$4"

  echo ""
  echo "################################################################################"
  echo "START MODEL TAG=${TAG} MODE=${MODE}"
  echo "################################################################################"

  local CKPT=""

  if [ "${MODE}" = "reuse" ]; then
    CKPT="${LLAMA32_CKPT}"
    verify_ckpt "${CKPT}"
    mkdir -p "${OUT_ROOT}/${TAG}"
    echo "[REUSE] ${TAG} checkpoint = ${CKPT}"
  else
    train_model "${TAG}" "${MODEL}" "${MODEL_TYPE}"
    CKPT="${OUT_ROOT}/${TAG}/main_checkpoint/checkpoint-final"
  fi

  infer_model "${TAG}" "${MODEL}" "${MODEL_TYPE}" "${CKPT}"

  echo "################################################################################"
  echo "DONE MODEL TAG=${TAG}"
  echo "################################################################################"
}

run_with_error_policy() {
  local TAG="$1"
  local MODEL="$2"
  local MODEL_TYPE="$3"
  local MODE="$4"

  if run_one_model "${TAG}" "${MODEL}" "${MODEL_TYPE}" "${MODE}"; then
    echo "[OK] ${TAG}" | tee -a "${LOG_ROOT}/day4_model_status.tsv"
  else
    echo "[FAILED] ${TAG}" | tee -a "${LOG_ROOT}/day4_model_status.tsv"
    if [ "${CONTINUE_ON_ERROR}" = "1" ]; then
      echo "[WARN] Continuing to next model because CONTINUE_ON_ERROR=1"
    else
      exit 1
    fi
  fi
}

# Reset status file
echo -e "status\tmodel_tag" > "${LOG_ROOT}/day4_model_status.tsv"

# 1) Llama-3.2-3B
if [ "${RETRAIN_LLAMA32}" = "1" ]; then
  run_with_error_policy "llama3_2_3b" "${LLAMA32_MODEL}" "llama" "train"
else
  run_with_error_policy "llama3_2_3b" "${LLAMA32_MODEL}" "llama" "reuse"
fi

# 2) Llama-3-8B
run_with_error_policy "llama3_8b" "${LLAMA3_8B_MODEL}" "llama" "train"

# 3) MedLlama-3-8B
run_with_error_policy "medllama3_8b" "${MEDLLAMA3_8B_MODEL}" "llama" "train"

echo ""
echo "============================================================"
echo "[DONE] Day 4 model comparison train/infer script finished."
echo "Status:"
cat "${LOG_ROOT}/day4_model_status.tsv"
echo "Outputs under: ${OUT_ROOT}"
echo "Logs under: ${LOG_ROOT}"
echo "============================================================"
