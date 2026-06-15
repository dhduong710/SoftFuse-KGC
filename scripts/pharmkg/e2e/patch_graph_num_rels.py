from pathlib import Path
import re
import shutil


def backup(path: Path):
    bak = path.with_suffix(path.suffix + ".bak_before_pharmkg_e2e")
    if not bak.exists():
        shutil.copy2(path, bak)
    return bak


def patch_arguments(path: Path):
    text = path.read_text(encoding="utf-8")
    if "graph_num_rels" in text:
        print(f"[SKIP] {path}: graph_num_rels already exists")
        return

    lines = text.splitlines()
    out = []
    inserted = False

    for line in lines:
        out.append(line)
        if line.strip().startswith("kge_embedding_path:"):
            out.append(
                '    graph_num_rels: int = field(default=4, metadata={"help": "Number of graph relation types for GraphEnhancer/R-GCN"})'
            )
            inserted = True

    if not inserted:
        raise RuntimeError("Could not find kge_embedding_path field in arguments.py")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[OK] patched {path}: added graph_num_rels")


def patch_graphenhancer_num_rels(path: Path, use_args_name: str = "args.graph_num_rels"):
    text = path.read_text(encoding="utf-8")
    if use_args_name in text:
        print(f"[SKIP] {path}: already uses {use_args_name}")
        return

    pattern = (
        r"(GraphEnhancer\(\s*\n"
        r"\s*kge_embedding,\s*\n"
        r"\s*kge_embedding_dim,\s*\n"
        r"\s*)4(\s*,)"
    )

    new_text, n = re.subn(pattern, rf"\1{use_args_name}\2", text, count=1)

    if n != 1:
        raise RuntimeError(
            f"Could not patch GraphEnhancer num_rels in {path}. "
            "Please inspect the GraphEnhancer(...) block manually."
        )

    path.write_text(new_text, encoding="utf-8")
    print(f"[OK] patched {path}: GraphEnhancer num_rels -> {use_args_name}")


def main():
    files = {
        "arguments": Path("arguments.py"),
        "main": Path("main.py"),
        "infer": Path("infer.py"),
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
        bak = backup(path)
        print(f"[BACKUP] {path} -> {bak}")

    patch_arguments(files["arguments"])
    patch_graphenhancer_num_rels(files["main"])
    patch_graphenhancer_num_rels(files["infer"])

    print("\n[DONE] PharmKG E2E graph_num_rels patch complete.")
    print("Use --graph_num_rels 28 for PharmKG.")


if __name__ == "__main__":
    main()