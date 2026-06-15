#!/usr/bin/env bash
set -euo pipefail

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-meta-llama/Llama-3.2-3B}"
MODEL_TYPE="${MODEL_TYPE:-llama}"

ROOT="dataset/setting_a/e2e_infer_ready"
KGE="dataset/setting_a/backbone_ready/entity_embeddings_rgcn.pt"

CKPT="${CKPT:-outputs/e2e/e2e_primary_checkpoint/checkpoint-final}"

BEST_JSON="outputs/e2e/decoding_sweep_valid/decoding_sweep_best_config.json"
OUT_ROOT="outputs/e2e/selected_decode_test"
PRED_OUT="${OUT_ROOT}/predictions"

mkdir -p "${OUT_ROOT}" "${PRED_OUT}" "outputs/e2e/reports"

test -f "${BEST_JSON}"
test -f "${KGE}"
test -f "${CKPT}/graph_model.bin"
test -f "${CKPT}/adapter_config.json"

if [ ! -f "${CKPT}/adapter_model.bin" ] && [ ! -f "${CKPT}/adapter_model.safetensors" ]; then
  echo "[ERROR] Missing adapter weights in ${CKPT}"
  exit 1
fi

read MAX_NEW REP_PEN NGRAM CONFIG_ID < <(python - <<'PY'
import json
p = "outputs/e2e/decoding_sweep_valid/decoding_sweep_best_config.json"
obj = json.load(open(p, encoding="utf-8"))
b = obj["best_config"]
print(b["max_new_tokens"], b["repetition_penalty"], b["no_repeat_ngram_size"], b["config_id"])
PY
)

echo "MODEL=${MODEL}"
echo "MODEL_TYPE=${MODEL_TYPE}"
echo "CKPT=${CKPT}"
echo "CONFIG_ID=${CONFIG_ID}"
echo "MAX_NEW=${MAX_NEW}"
echo "REP_PEN=${REP_PEN}"
echo "NGRAM=${NGRAM}"

CKPT_PARENT="$(dirname "${CKPT}")"

for ROW in backbone_raw soft_support_raw retrieval_main; do
  for SPLIT in valid test; do
    echo ""
    echo "============================================================"
    echo "SELECTED DECODE ROW=${ROW} SPLIT=${SPLIT} CONFIG=${CONFIG_ID}"
    echo "============================================================"

    SUFFIX="selected_decode_${ROW}"

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
      2>&1 | tee "${OUT_ROOT}/infer_${SPLIT}_${ROW}.log"

    SRC_PRED="${CKPT_PARENT}/prediction_${SPLIT}_${SUFFIX}.json"
    SRC_METRICS="${CKPT_PARENT}/ranking_metrics_${SPLIT}_${SUFFIX}.json"
    SRC_TXT="${CKPT_PARENT}/metrics_${SPLIT}_${SUFFIX}.txt"

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

cp "${BEST_JSON}" "${OUT_ROOT}/selected_decode_config.json"

echo ""
echo "[DONE] Frozen decode valid/test inference complete."
echo "Predictions: ${PRED_OUT}"
