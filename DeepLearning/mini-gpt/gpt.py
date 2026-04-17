import argparse
import random

import torch
import torch.nn as nn

from create_dataset import (
    load_wikitext2,
    get_tokenizer,
    tokenize_dataset,
    create_dataloaders,
    build_word_vocab,
    sanity_check_lm_dataset,
    sanity_check_build_word_vocab,
)
from model import GPTModel
from train import train, evaluate
from generate import generate_text


BASE_CONFIG = {
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 4,
    "d_ff": 512,
    "block_size": 128,
    "dropout": 0.1,
}

BASE_TRAIN_OPTS = {
    "lr": 3e-4,
    "num_epochs": 15,
    "batch_size": 64,
    "weight_decay": 0.01,
}

REFERENCE_BASE_VAL_PPL = 267.55
REFERENCE_BASE_PUBLIC_TEST_PPL = 269.63
PUBLIC_TEST_GOOD = 275

PROMPTS = [
    "The history of artificial intelligence",
    "In the beginning of the 20th century",
    "Scientists have recently discovered",
]


def set_seed(seed=7150):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def sanity_check_model(model, tokenizer, device):
    model.eval()
    x = torch.randint(0, tokenizer.vocab_size, (2, 16), device=device)
    logits = model(x)
    expected = torch.Size([2, 16, tokenizer.vocab_size])
    assert logits.shape == expected, f"model output shape {logits.shape} != expected {expected}."
    model.train()
    print("  [OK] model forward-pass shape is correct.")


def main(args):
    set_seed(args.seed)

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    print(f"Using device: {device}")

    print("\n=== Loading and tokenizing WikiText-2 ===")
    dataset = load_wikitext2()
    tokenizer = get_tokenizer()

    train_ids = tokenize_dataset(dataset["train"], tokenizer)
    val_ids = tokenize_dataset(dataset["validation"], tokenizer)
    test_ids = tokenize_dataset(dataset["test"], tokenizer)

    print(f"  GPT-2 BPE vocab size : {tokenizer.vocab_size:,}")
    print(f"  Train tokens         : {len(train_ids):,}")
    print(f"  Validation tokens    : {len(val_ids):,}")
    print(f"  Test tokens          : {len(test_ids):,}")

    print("\n=== Tokenizer comparison: word-level vs. GPT-2 BPE ===")
    raw_train_text = " ".join(line for line in dataset["train"]["text"] if line.strip())
    word2idx, _ = build_word_vocab(raw_train_text)

    word_vocab_size = len(word2idx)
    bpe_vocab_size = tokenizer.vocab_size
    print(f"  Word-level vocab size : {word_vocab_size:,}")
    print(f"  GPT-2 BPE vocab size  : {bpe_vocab_size:,}")

    ratio = word_vocab_size / bpe_vocab_size
    if word_vocab_size > bpe_vocab_size:
        print(f"  -> The word-level vocabulary is {ratio:.1f}x larger than BPE.")
    else:
        print(f"  -> The word-level vocabulary is {ratio:.1f}x the size of BPE on this corpus.")

    train_dl, val_dl, test_dl = create_dataloaders(
        train_ids,
        val_ids,
        test_ids,
        block_size=BASE_CONFIG["block_size"],
        batch_size=BASE_TRAIN_OPTS["batch_size"],
    )

    print(f"\n  Training batches     : {len(train_dl):,}")
    print(f"  Validation batches   : {len(val_dl):,}")
    print(f"  Test batches         : {len(test_dl):,}")

    print("\n=== Building GPT model (base) ===")
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=BASE_CONFIG["d_model"],
        num_heads=BASE_CONFIG["num_heads"],
        num_layers=BASE_CONFIG["num_layers"],
        d_ff=BASE_CONFIG["d_ff"],
        max_len=BASE_CONFIG["block_size"],
        dropout=BASE_CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {n_params / 1e6:.2f}M")

    print("\n=== Running sanity checks ===")
    sanity_check_build_word_vocab()
    sanity_check_lm_dataset(dataset["train"], tokenizer, block_size=BASE_CONFIG["block_size"])
    sanity_check_model(model, tokenizer, device)
    print("  all checks passed.\n")

    print("=== Training ===")
    print("  Run type           : required base model")
    print("  Checkpoint         : gpt_model.pt")
    print("  Local feedback     : public WikiText-2 test perplexity")
    print(
        f"  Reference run      : val_ppl ~= {REFERENCE_BASE_VAL_PPL:.2f}, "
        f"public_test_ppl ~= {REFERENCE_BASE_PUBLIC_TEST_PPL:.2f}\n"
    )

    _, _, val_ppls = train(model, train_dl, val_dl, BASE_TRAIN_OPTS, device)

    criterion = nn.CrossEntropyLoss()
    _, test_ppl = evaluate(model, test_dl, criterion, device)
    best_val_ppl = min(val_ppls)

    print(f"\n{'=' * 50}")
    print(f"  Best validation perplexity : {best_val_ppl:.2f}")
    print(f"  Public test perplexity     : {test_ppl:.2f}")
    if test_ppl < PUBLIC_TEST_GOOD:
        print("  [OK] Local public-test result is in the expected range.")
    else:
        print("  [!] Local public-test result is weaker than expected - keep tuning.")
    print("  [Note] Final grading uses a private held-out corpus.")
    print(f"{'=' * 50}\n")

    torch.save(
        {
            "state": model.state_dict(),
            "config": BASE_CONFIG,
            "train_opts": BASE_TRAIN_OPTS,
            "mode": "base",
            "vocab_size": tokenizer.vocab_size,
            "best_val_ppl": best_val_ppl,
            "test_ppl": test_ppl,
            "public_test_ppl": test_ppl,
        },
        "gpt_model.pt",
    )
    print("Checkpoint saved to gpt_model.pt")

    print("\n=== Sample generations ===")
    for prompt in PROMPTS:
        print(f"\n[Prompt] {prompt!r}")
        output = generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=80,
            temperature=0.8,
            top_k=50,
            device=str(device),
        )
        print(f"[Output] {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="base", choices=["base"])
    parser.add_argument("--seed", type=int, default=7150)
    args = parser.parse_args()
    main(args)
