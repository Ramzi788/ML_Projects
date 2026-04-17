import argparse
import re
import zipfile
from pathlib import Path

import torch


BASE_CONFIG = {
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 4,
    "d_ff": 512,
    "block_size": 128,
    "dropout": 0.1,
}

REQUIRED_CODE_FILES = [
    "create_dataset.py",
    "create_submission.py",
    "generate.py",
    "gpt.py",
    "improved_gpt.py",
    "model.py",
    "train.py",
]
REQUIRED_FILES = REQUIRED_CODE_FILES + ["gpt_model.pt"]
OPTIONAL_FILES = ["improved_gpt_model.pt", "improvement_notes.txt", "requirements.txt"]


def normalize_name(name):
    name = re.sub(r"\s+", " ", name.strip())
    if not name:
        raise ValueError("student_name must not be empty")
    return name.replace(" ", "_")


def verify_required_files():
    missing = [file_name for file_name in REQUIRED_FILES if not Path(file_name).is_file()]
    if missing:
        raise FileNotFoundError("Missing required file(s): " + ", ".join(missing))


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if "state" not in checkpoint or "config" not in checkpoint:
        raise KeyError(f"{path} must contain 'state' and 'config'")
    return checkpoint


def verify_base_checkpoint():
    checkpoint = load_checkpoint("gpt_model.pt")
    cfg = checkpoint["config"]

    for key, value in BASE_CONFIG.items():
        if cfg.get(key) != value:
            raise ValueError(f"gpt_model.pt must use the required base config: {key}={value}")

    state = checkpoint["state"]
    vocab_size = checkpoint.get("vocab_size", state["tok_embed.weight"].shape[0])
    if state["tok_embed.weight"].shape[0] != vocab_size:
        raise RuntimeError("tok_embed.weight does not match vocab_size")
    if state["tok_embed.weight"].shape[1] != cfg["d_model"]:
        raise RuntimeError("tok_embed.weight does not match d_model")
    if "lm_head.weight" in state and tuple(state["lm_head.weight"].shape) != (vocab_size, cfg["d_model"]):
        raise RuntimeError("lm_head.weight has the wrong shape")

    print("[OK] required base checkpoint found.")
    if "best_val_ppl" in checkpoint:
        print(f"     best validation perplexity : {checkpoint['best_val_ppl']:.2f}")
    if "test_ppl" in checkpoint:
        print(f"     saved test perplexity      : {checkpoint['test_ppl']:.2f}")


def verify_improvement_files():
    model_path = Path("improved_gpt_model.pt")
    notes_path = Path("improvement_notes.txt")

    if not model_path.is_file():
        if notes_path.is_file():
            print("[WARN] improvement_notes.txt was found, but improved_gpt_model.pt is missing.")
        return False

    if not notes_path.is_file():
        raise FileNotFoundError("improved_gpt_model.pt was found, but improvement_notes.txt is missing")

    from model import GPTModel

    checkpoint = load_checkpoint("improved_gpt_model.pt")
    cfg = checkpoint["config"]
    state = checkpoint["state"]
    vocab_size = checkpoint.get("vocab_size", state["tok_embed.weight"].shape[0])

    model = GPTModel(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        d_ff=cfg["d_ff"],
        max_len=cfg["block_size"],
        dropout=cfg.get("dropout", 0.1),
    )
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        x = torch.randint(0, vocab_size, (1, min(8, cfg["block_size"])))
        logits = model(x)

    expected = (1, min(8, cfg["block_size"]), vocab_size)
    if tuple(logits.shape) != expected:
        raise RuntimeError(
            f"improved checkpoint sanity check failed: got {tuple(logits.shape)}, expected {expected}."
        )

    print("[OK] optional improved checkpoint loaded successfully.")
    if "best_val_ppl" in checkpoint:
        print(f"     best validation perplexity : {checkpoint['best_val_ppl']:.2f}")
    if "test_ppl" in checkpoint:
        print(f"     saved test perplexity      : {checkpoint['test_ppl']:.2f}")
    return True


def collect_files():
    files = sorted(path.name for path in Path(".").glob("*.py"))
    files.append("gpt_model.pt")
    for file_name in OPTIONAL_FILES:
        if Path(file_name).is_file():
            files.append(file_name)
    return list(dict.fromkeys(files))


def write_zip(output_name, files):
    with zipfile.ZipFile(output_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_name in files:
            zf.write(file_name, arcname=file_name)

    print(f"Submission archive written to {output_name}")
    for file_name in files:
        print(f"  - {file_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_name", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    student_name = normalize_name(args.student_name)
    output_name = args.output or f"{student_name}_submission.zip"

    verify_required_files()
    verify_base_checkpoint()
    verify_improvement_files()
    files = collect_files()
    write_zip(output_name, files)


if __name__ == "__main__":
    main()
