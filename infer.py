import os
import json
import numpy as np
import argparse
from pathlib import Path

import bitsandbytes as bnb  # noqa: F401
import torch

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    HfArgumentParser,
    GenerationConfig,
    set_seed,
)
from peft import PeftModel

from arguments import Arguments, GenerationArguments
from data import DataModule
from model import GraphEnhancer, DrKGC
from torch.cuda.amp import autocast


torch.cuda.empty_cache()


class Evaluator:
    def __init__(self, args, tokenizer, model, data_module, generation_config):
        self.args = args
        self.generation_config = generation_config
        self.tokenizer = tokenizer
        self.model = model
        self.data_module = data_module

        self.output_dir = os.path.dirname(args.checkpoint_dir)

    @torch.no_grad()
    def ranking_metrics(self, dataset):
        self.model.eval()

        preds = []
        ranks = []

        generated = []
        for ex_idx, ex in enumerate(dataset):
            prompt = ex["input"]

            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.cuda()
            self.generation_config.eos_token_id = self.tokenizer.eos_token_id

            subgraph = [ex["subgraph"]] if "subgraph" in ex else None

            output = self.model.generate(
                input_ids=input_ids,
                query_ids=torch.LongTensor([ex["query_entity_id"]]).to(input_ids.device),
                entity_ids=torch.LongTensor([ex["rank_entities_id"]]).to(input_ids.device),
                subgraph=subgraph,
                generation_config=self.generation_config,
            )
            generated.append(output.sequences[0].cpu().numpy().tolist())

        batch_preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        for ex_idx, ex in enumerate(dataset):
            target = ex["output"]
            rank = ex["rank"]
            pred = str(batch_preds[ex_idx]).strip()

            topk_names = ex["rank_entities"]

            # Keep original example untouched
            out_ex = dict(ex)

            if target == pred:
                pred_rank = 1
            else:
                pred_rank = rank
                if pred not in set(topk_names) or topk_names.index(pred) >= rank:
                    pred_rank += 1

            out_ex["target"] = target
            out_ex["pred_rank"] = pred_rank
            out_ex["pred"] = pred
            preds.append(out_ex)
            ranks.append(pred_rank)

        ranks = np.array(ranks, dtype=np.float64)

        metrics = {
            "mrr": float(np.mean(1.0 / ranks)),
            "hits1": float(np.mean(ranks <= 1)),
            "hits3": float(np.mean(ranks <= 3)),
            "hits10": float(np.mean(ranks <= 10)),
            "num_examples": int(len(ranks)),
        }
        metrics = {k: (round(v, 8) if isinstance(v, float) else v) for k, v in metrics.items()}

        print("ranking metrics:")
        print(metrics)

        return preds, metrics


def choose_dataset(data_module, eval_split: str):
    eval_split = eval_split.lower().strip()
    if eval_split == "valid":
        return data_module.eval_ds
    if eval_split == "test":
        return data_module.test_ds
    raise ValueError(f"Unsupported eval_split={eval_split}. Use 'valid' or 'test'.")


def build_out_name(prefix: str, split: str, suffix: str) -> str:
    suffix = suffix.strip()
    if suffix:
        return f"{prefix}_{split}_{suffix}.json" if prefix != "metrics" else f"{prefix}_{split}_{suffix}.txt"
    return f"{prefix}_{split}.json" if prefix != "metrics" else f"{prefix}_{split}.txt"


if __name__ == "__main__":
    set_seed(3407)

    hfparser = HfArgumentParser((Arguments, GenerationArguments))
    (data_args, generation_args, _) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(data_args))

    eval_split = args.eval_split.lower().strip()
    if eval_split not in {"valid", "test"}:
        raise ValueError("--eval_split must be one of: valid, test")

    print(f"Load LLM: {args.model_name_or_path}")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Eval split: {eval_split}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["[QUERY]", "[ENTITY]", "[RELATION]"])

    generation_config.bos_token_id = tokenizer.bos_token_id

    if args.model_type == "llama":
        pretrained_model_class = LlamaForCausalLM
    elif args.model_type == "mistral":
        pretrained_model_class = AutoModelForCausalLM
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    # For E2E inference on a single RTX 4090 24GB, place the full model on GPU 0.
    # This avoids accelerate meta/offload tensors and prevents DrKGC wrapper .cuda() errors.
    model = pretrained_model_class.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    model = model.half()

    kge_embedding = torch.load(args.kge_embedding_path, map_location="cpu")
    kge_embedding_dim = kge_embedding.shape[1]
    llm_config = model.config

    embed_model = GraphEnhancer(
        kge_embedding,
        kge_embedding_dim,
        args.graph_num_rels,
        128,
        1,
        1024,
        llm_config.hidden_size,
        llm_config.hidden_act,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    state = torch.load(ckpt_dir / "graph_model.bin", map_location="cpu")
    embed_model.load_state_dict(state)

    # Move only the graph enhancer to GPU. The LLM/PEFT model is already on GPU
    # through device_map={"": 0}. Do not call .cuda() on the whole DrKGC wrapper.
    embed_model = embed_model.half().cuda()

    model = DrKGC(tokenizer, model, embed_model)
    model.eval()

    data_module = DataModule(args, tokenizer)
    dataset = choose_dataset(data_module, eval_split)

    evaluator = Evaluator(args, tokenizer, model, data_module, generation_config)

    with autocast():
        preds, metrics = evaluator.ranking_metrics(dataset)

    output_dir = os.path.dirname(args.checkpoint_dir)
    suffix = args.output_suffix.strip()

    pred_path = os.path.join(
        output_dir,
        f"prediction_{eval_split}" + (f"_{suffix}" if suffix else "") + ".json"
    )
    metrics_json_path = os.path.join(
        output_dir,
        f"ranking_metrics_{eval_split}" + (f"_{suffix}" if suffix else "") + ".json"
    )
    metrics_txt_path = os.path.join(
        output_dir,
        f"metrics_{eval_split}" + (f"_{suffix}" if suffix else "") + ".txt"
    )

    output = {
        "args": vars(args),
        "generation_config": vars(generation_config),
        "prediction": preds,
    }

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write(f"ranking metrics: {metrics}\n")

    print(f"Saved prediction to: {pred_path}")
    print(f"Saved ranking metrics to: {metrics_json_path}")
    print(f"Saved text metrics to: {metrics_txt_path}")