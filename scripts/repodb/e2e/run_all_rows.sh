#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs/repodb/logs outputs/repodb/e2e_repodb_rgcn

if [[ -z "${MODEL_NAME_OR_PATH:-}" ]]; then
  echo "[ERROR] Please set MODEL_NAME_OR_PATH first."
  echo "Example:"
  echo "  export MODEL_NAME_OR_PATH=/path/to/Llama-3.2-3B-Instruct"
  exit 1
fi

MODEL_TYPE="${MODEL_TYPE:-llama}"

GRAPH_NUM_RELS=63
SOURCE_MAX_LEN="${SOURCE_MAX_LEN:-768}"
TARGET_MAX_LEN="${TARGET_MAX_LEN:-64}"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
SEED="${SEED:-2028}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"

run_row () {
  local row_name="$1"
  local dataset_path="$2"

  local out_dir="outputs/repodb/e2e_repodb_rgcn/${row_name}"
  local ckpt="${out_dir}/checkpoint-final"
  local kge="${dataset_path}/entity_embeddings_rgcn.pt"

  echo "================================================================================"
  echo "[ROW] ${row_name}"
  echo "dataset_path   = ${dataset_path}"
  echo "output_dir     = ${out_dir}"
  echo "checkpoint     = ${ckpt}"
  echo "kge            = ${kge}"
  echo "graph_num_rels = ${GRAPH_NUM_RELS}"
  echo "================================================================================"

  if [[ ! -f "${kge}" ]]; then
    echo "[ERROR] Missing KGE embedding: ${kge}"
    exit 1
  fi
  for required_split in train valid test; do
    if [[ ! -f "${dataset_path}/${required_split}.json" ]]; then
      echo "[ERROR] Missing ${dataset_path}/${required_split}.json"
      echo "Regenerate this package with the build script listed in ${dataset_path}/README.md."
      exit 1
    fi
  done

  mkdir -p "${out_dir}"

  if [[ -f "${ckpt}/adapter_config.json" && -f "${ckpt}/graph_model.bin" ]]; then
    echo "[skip train] checkpoint-final already exists for ${row_name}"
  else
    echo "[train] ${row_name}"

    python main.py \
      --dataset_path "${dataset_path}" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --model_type "${MODEL_TYPE}" \
      --kge_embedding_path "${kge}" \
      --graph_num_rels "${GRAPH_NUM_RELS}" \
      --source_max_len "${SOURCE_MAX_LEN}" \
      --target_max_len "${TARGET_MAX_LEN}" \
      --output_dir "${out_dir}" \
      --num_train_epochs "${EPOCHS}" \
      --per_device_train_batch_size "${BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRAD_ACCUM}" \
      --learning_rate "${LR}" \
      --lr_scheduler_type constant \
      --warmup_ratio 0.03 \
      --optim paged_adamw_32bit \
      --lora_r 32 \
      --lora_alpha 32 \
      --lora_dropout 0.1 \
      --remove_unused_columns False \
      --dataloader_num_workers 4 \
      --save_steps 500 \
      --logging_steps 10 \
      --bf16 True \
      --tf32 True \
      --report_to none \
      --seed "${SEED}" \
      2>&1 | tee "outputs/repodb/logs/day8_train_${row_name}.log"
  fi

  if [[ ! -f "${ckpt}/adapter_config.json" || ! -f "${ckpt}/graph_model.bin" ]]; then
    echo "[ERROR] Missing checkpoint files after training: ${ckpt}"
    ls -lh "${ckpt}" || true
    exit 1
  fi

  for split in valid test; do
    echo "[infer] ${row_name} ${split}"

    python infer.py \
      --dataset_path "${dataset_path}" \
      --model_name_or_path "${MODEL_NAME_OR_PATH}" \
      --model_type "${MODEL_TYPE}" \
      --kge_embedding_path "${kge}" \
      --graph_num_rels "${GRAPH_NUM_RELS}" \
      --checkpoint_dir "${ckpt}" \
      --eval_split "${split}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --min_new_tokens 1 \
      --do_sample False \
      --num_beams 1 \
      --repetition_penalty 1.0 \
      --no_repeat_ngram_size 0 \
      2>&1 | tee "outputs/repodb/logs/day8_infer_${row_name}_${split}.log"
  done
}

run_row "backbone_raw" "dataset/setting_f_repodb/e2e_soft_support_ready/rgcn_raw_display_control"
run_row "soft_support_sweep" "dataset/setting_f_repodb/e2e_soft_support_ready/rgcn_sweep_selected"
run_row "fuzzy_retrieval_main" "dataset/setting_f_repodb/e2e_fuzzy_retrieval_ready/rgcn"

echo "[DONE] repoDB E2E train/infer all rows"
