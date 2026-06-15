#!/usr/bin/env bash
set -euo pipefail

# Question-template sensitivity E2E inference.
# Run from repository root.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-meta-llama/Llama-3.2-3B}"
MODEL_TYPE="${MODEL_TYPE:-llama}"

SRC_CKPT="${SRC_CKPT:-outputs/e2e/e2e_primary_checkpoint/checkpoint-final}"
DST_CKPT="outputs/sensitivity/template_sensitivity/e2e_checkpoint"

KGE_PATH="dataset/setting_a/backbone_ready/entity_embeddings_rgcn.pt"
OUT_DIR="outputs/sensitivity/template_sensitivity"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${DST_CKPT}"

echo "================================================================================"
echo "Week25 Day4 Template Sensitivity E2E"
echo "MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "SRC_CKPT=${SRC_CKPT}"
echo "DST_CKPT=${DST_CKPT}"
echo "================================================================================"

for f in adapter_config.json adapter_model.bin adapter_model.safetensors graph_model.bin README.md; do
  if [[ -f "${SRC_CKPT}/${f}" ]]; then
    cp -f "${SRC_CKPT}/${f}" "${DST_CKPT}/${f}"
  fi
done

if [[ ! -f "${DST_CKPT}/adapter_config.json" ]]; then
  echo "Missing ${DST_CKPT}/adapter_config.json"
  exit 1
fi

if [[ ! -f "${DST_CKPT}/graph_model.bin" ]]; then
  echo "Missing ${DST_CKPT}/graph_model.bin"
  exit 1
fi

if [[ ! -f "${DST_CKPT}/adapter_model.bin" && ! -f "${DST_CKPT}/adapter_model.safetensors" ]]; then
  echo "Missing adapter weights in ${DST_CKPT}"
  exit 1
fi

if [[ ! -f "${KGE_PATH}" ]]; then
  echo "Missing KGE embedding path: ${KGE_PATH}"
  exit 1
fi

VARIANTS=("T0_canonical" "T1_treatment" "T2_medication" "T3_association_neutral")
SPLITS=("valid" "test")

for variant in "${VARIANTS[@]}"; do
  DATASET_PATH="dataset/setting_a/template_sensitivity/${variant}"

  if [[ ! -f "${DATASET_PATH}/valid.json" || ! -f "${DATASET_PATH}/test.json" || ! -f "${DATASET_PATH}/train.json" ]]; then
    echo "Missing train/valid/test in ${DATASET_PATH}"
    exit 1
  fi

  for split in "${SPLITS[@]}"; do
    suffix="template_sensitivity_${variant}"
    log_file="${LOG_DIR}/${split}_${variant}.log"

    echo "================================================================================"
    echo "Running template=${variant} split=${split}"
    echo "dataset=${DATASET_PATH}"
    echo "log=${log_file}"
    echo "================================================================================"

    python infer.py \
      --dataset_path "${DATASET_PATH}" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --model_type "${MODEL_TYPE}" \
      --kge_embedding_path "${KGE_PATH}" \
      --graph_num_rels 4 \
      --checkpoint_dir "${DST_CKPT}" \
      --eval_split "${split}" \
      --output_suffix "${suffix}" \
      --max_new_tokens 16 \
      --min_new_tokens 1 \
      --do_sample false \
      --num_beams 1 \
      --temperature 1.0 \
      --repetition_penalty 1.0 \
      --no_repeat_ngram_size 0 \
      2>&1 | tee "${log_file}"

    echo "Done template=${variant} split=${split}"
  done
done

echo "================================================================================"
echo "All Week25 template sensitivity E2E runs finished."
echo "Outputs should be under: ${OUT_DIR}"
echo "================================================================================"
