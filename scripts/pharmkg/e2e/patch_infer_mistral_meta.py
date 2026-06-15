from pathlib import Path
import shutil

path = Path("infer.py")
bak = Path("infer.py.bak_before_mistral_meta_fix")
if not bak.exists():
    shutil.copy2(path, bak)
    print(f"[BACKUP] {path} -> {bak}")

text = path.read_text(encoding="utf-8")

# 1) Add AutoModelForCausalLM import if needed
if "AutoModelForCausalLM" not in text:
    text = text.replace(
        "LlamaForCausalLM,\n    HfArgumentParser,",
        "LlamaForCausalLM,\n    AutoModelForCausalLM,\n    HfArgumentParser,",
    )
    print("[OK] added AutoModelForCausalLM import")
else:
    print("[SKIP] AutoModelForCausalLM already imported")

# 2) Replace base model loading block
old_load = '''    model = LlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.checkpoint_dir)
    model = model.half()
'''

new_load = '''    if args.model_type == "llama":
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
'''

if old_load not in text:
    raise RuntimeError("Could not find the old LlamaForCausalLM loading block in infer.py")

text = text.replace(old_load, new_load, 1)
print("[OK] patched model loading block")

# 3) Replace DrKGC wrapper cuda block
old_wrap = '''    model = DrKGC(tokenizer, model, embed_model)
    model = model.half()
    model.cuda()
    model.eval()
'''

new_wrap = '''    # Move only the graph enhancer to GPU. The LLM/PEFT model is already on GPU
    # through device_map={"": 0}. Do not call .cuda() on the whole DrKGC wrapper.
    embed_model = embed_model.half().cuda()

    model = DrKGC(tokenizer, model, embed_model)
    model.eval()
'''

if old_wrap not in text:
    raise RuntimeError("Could not find the old DrKGC .cuda() wrapper block in infer.py")

text = text.replace(old_wrap, new_wrap, 1)
print("[OK] patched DrKGC wrapper block")

path.write_text(text, encoding="utf-8")
print("[DONE] infer.py patched for Mistral + no meta tensor inference")