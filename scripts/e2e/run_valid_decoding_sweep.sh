#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-meta-llama/Llama-3.2-3B}"
MODEL_TYPE="${MODEL_TYPE:-llama}"

ROOT="dataset/setting_a/e2e_infer_ready"
KGE="dataset/setting_a/backbone_ready/entity_embeddings_rgcn.pt"

# This directory must contain adapter_config.json, adapter_model.*, and graph_model.bin.
CKPT="${CKPT:-outputs/e2e/e2e_primary_checkpoint/checkpoint-final}"

CONFIG_JSON="outputs/e2e/decoding_sweep_valid/configs.json"
OUT_ROOT="outputs/e2e/decoding_sweep_valid"
PRED_OUT="${OUT_ROOT}/decoding_sweep_predictions"

mkdir -p "${OUT_ROOT}" "${PRED_OUT}" "outputs/e2e/reports"

echo "MODEL=${MODEL}"
echo "MODEL_TYPE=${MODEL_TYPE}"
echo "CKPT=${CKPT}"
echo "KGE=${KGE}"
echo "CONFIG_JSON=${CONFIG_JSON}"

test -f "${CONFIG_JSON}"
test -f "${KGE}"
test -f "${CKPT}/graph_model.bin"
test -f "${CKPT}/adapter_config.json"

if [ ! -f "${CKPT}/adapter_model.bin" ] && [ ! -f "${CKPT}/adapter_model.safetensors" ]; then
  echo "[ERROR] Missing adapter weights in ${CKPT}"
  exit 1
fi

CKPT_PARENT="$(dirname "${CKPT}")"

python - <<'PY' > outputs/e2e/decoding_sweep_valid/config_lines.tsv
import json
p = "outputs/e2e/decoding_sweep_valid/configs.json"
obj = json.load(open(p, encoding="utf-8"))
for c in obj["configs"]:
    print(
        c["config_id"],
        c["max_new_tokens"],
        c["repetition_penalty"],
        c["no_repeat_ngram_size"],
        sep="\t"
    )
PY

while IFS=$'\t' read -r CFG MAX_NEW REP_PEN NGRAM; do
  echo ""
  echo "################################################################################"
  echo "Running config ${CFG}: max_new_tokens=${MAX_NEW}, repetition_penalty=${REP_PEN}, no_repeat_ngram_size=${NGRAM}"
  echo "################################################################################"

  mkdir -p "${PRED_OUT}/${CFG}"

  for ROW in backbone_raw soft_support_raw retrieval_main; do
    echo ""
    echo "============================================================"
    echo "VALID SWEEP CFG=${CFG} ROW=${ROW}"
    echo "============================================================"

    SUFFIX="w24valid_${CFG}_${ROW}"

    python infer.py \
      --model_name_or_path "${MODEL}" \
      --model_type "${MODEL_TYPE}" \
      --dataset_path "${ROOT}/${ROW}" \
      --kge_embedding_path "${KGE}" \
      --graph_num_rels 4 \
      --checkpoint_dir "${CKPT}" \
      --eval_split valid \
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
      2>&1 | tee "${OUT_ROOT}/infer_${SUFFIX}.log"

    SRC_PRED="${CKPT_PARENT}/prediction_valid_${SUFFIX}.json"
    SRC_METRICS="${CKPT_PARENT}/ranking_metrics_valid_${SUFFIX}.json"
    SRC_TXT="${CKPT_PARENT}/metrics_valid_${SUFFIX}.txt"

    test -f "${SRC_PRED}"
    cp "${SRC_PRED}" "${PRED_OUT}/${CFG}/prediction_valid_${ROW}.json"

    if [ -f "${SRC_METRICS}" ]; then
      cp "${SRC_METRICS}" "${PRED_OUT}/${CFG}/raw_ranking_metrics_valid_${ROW}.json"
    fi
    if [ -f "${SRC_TXT}" ]; then
      cp "${SRC_TXT}" "${PRED_OUT}/${CFG}/raw_metrics_valid_${ROW}.txt"
    fi
  done
done < outputs/e2e/decoding_sweep_valid/config_lines.tsv

echo ""
echo "[DONE] Day 2 valid decoding sweep inference complete."
echo "Predictions: ${PRED_OUT}"
