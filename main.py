import os
import argparse
from pathlib import Path

import bitsandbytes as bnb  # noqa: F401
import torch

import transformers
from transformers import (
    AutoConfig,
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    Seq2SeqTrainer,
    HfArgumentParser,
    set_seed,
    BitsAndBytesConfig,
)

from peft.tuners.lora import LoraLayer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from arguments import Arguments, FinetuningArguments, GenerationArguments
from data import make_data_module
from model import GraphEnhancer, DrKGC


def get_accelerate_model(args, config, pretrained_model_class):
    device_map = (
        "auto"
        if os.environ.get("LOCAL_RANK") is None
        else {"": int(os.environ.get("LOCAL_RANK", "0"))}
    )

    if args.use_quant:
        compute_dtype = torch.bfloat16
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            torch_dtype=torch.bfloat16,
        )
    else:
        model = pretrained_model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            low_cpu_mem_usage=True,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )

    if args.use_quant:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    else:
        model.gradient_checkpointing_enable()

    if args.model_type == "llama":
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
    elif args.model_type == "mistral":
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
        )
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    model = get_peft_model(model, peft_config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight") and module.weight.dtype == torch.float32:
                module = module.to(torch.bfloat16)

    return model


class SavePeftModelCallback(transformers.TrainerCallback):
    KEEP_FILES = {
        "adapter_model.bin",
        "adapter_model.safetensors",
        "adapter_config.json",
        "graph_model.bin",
        "README.md",
    }

    @staticmethod
    def _verify_checkpoint(checkpoint_folder: str):
        p = Path(checkpoint_folder)

        has_adapter = (
            (p / "adapter_model.bin").exists()
            or (p / "adapter_model.safetensors").exists()
        )
        has_config = (p / "adapter_config.json").exists()
        has_graph = (p / "graph_model.bin").exists()

        print(f"[checkpoint verify] folder={p}")
        print(f"  has_adapter = {has_adapter}")
        print(f"  has_config  = {has_config}")
        print(f"  has_graph   = {has_graph}")

        if not has_adapter:
            raise RuntimeError(
                f"Checkpoint {p} is missing adapter weights "
                f"(adapter_model.bin or adapter_model.safetensors)."
            )
        if not has_config:
            raise RuntimeError(f"Checkpoint {p} is missing adapter_config.json.")
        if not has_graph:
            raise RuntimeError(f"Checkpoint {p} is missing graph_model.bin.")

    def _save_and_cleanup(self, checkpoint_folder: str, model):
        os.makedirs(checkpoint_folder, exist_ok=True)

        # DrKGC.save_pretrained() should save:
        # - adapter weights/config through llm_model.save_pretrained()
        # - graph_model.bin through torch.save(...)
        model.save_pretrained(checkpoint_folder)

        # Keep only the files needed for reload / inference
        for file_name in os.listdir(checkpoint_folder):
            full_path = os.path.join(checkpoint_folder, file_name)
            if os.path.isfile(full_path) and file_name not in self.KEEP_FILES:
                os.remove(full_path)

        self._verify_checkpoint(checkpoint_folder)

    def on_save(self, args, state, control, **kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = state.best_model_checkpoint
            print(f"Saving the best checkpoint to: {checkpoint_folder}")
        else:
            checkpoint_folder = os.path.join(
                args.output_dir,
                f"checkpoint-{state.global_step}",
            )
            print(f"Saving checkpoint at step {state.global_step} to: {checkpoint_folder}")

        self._save_and_cleanup(checkpoint_folder, kwargs["model"])

    def on_train_end(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, "checkpoint-final")
        print(f"Saving the final checkpoint to: {checkpoint_folder}")
        self._save_and_cleanup(checkpoint_folder, kwargs["model"])


def train():
    hfparser = HfArgumentParser(
        (Arguments, FinetuningArguments, GenerationArguments)
    )
    (data_args, training_args, generation_args, _) = hfparser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    set_seed(training_args.seed)
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(data_args), **vars(training_args))

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Seed: {training_args.seed}")
    print(f"Load LLM: {args.model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        data_args.model_name_or_path,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["[QUERY]", "[ENTITY]", "[RELATION]"])

    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    if args.model_type == "llama":
        model = get_accelerate_model(args, model_config, LlamaForCausalLM)
    elif args.model_type == "mistral":
        model = get_accelerate_model(args, model_config, AutoModelForCausalLM)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    model.config.use_cache = False

    kge_embedding = torch.load(args.kge_embedding_path, map_location="cpu")
    kge_embedding_dim = kge_embedding.shape[1]
    llm_config = model.config

    embed_model = GraphEnhancer(
        kge_embedding,
        kge_embedding_dim,
        args.graph_num_rels,      # num_rels
        128,    # gnn_hidden_dim
        1,      # gnn_num_hidden_layers
        1024,   # adapter_size
        llm_config.hidden_size,
        llm_config.hidden_act,
    ).to(torch.bfloat16)

    model = DrKGC(tokenizer, model, embed_model)

    model.llm_model.base_model.model.lm_head.weight = torch.nn.Parameter(
        model.llm_model.base_model.model.lm_head.weight.clone()
    )

    data_module = make_data_module(args, tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    trainer.add_callback(SavePeftModelCallback)

    train_result = trainer.train()
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    train()