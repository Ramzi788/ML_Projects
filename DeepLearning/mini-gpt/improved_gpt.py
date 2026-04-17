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


IMPROVED_CONFIG = {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 6,
    "d_ff": 1024,
    "block_size": 256,
    "dropout": 0.2,
}

IMPROVED_TRAIN_OPTS = {
    "lr": 5e-4,
    "num_epochs": 25,
    "batch_size": 32,
    "weight_decay": 0.1,
}

BASE_REFERENCE_PUBLIC_TEST_PPL = 269.63

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


def main(seed=7150):
    set_seed(seed)

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
        block_size=IMPROVED_CONFIG["block_size"],
        batch_size=IMPROVED_TRAIN_OPTS["batch_size"],
    )

    print(f"\n  Training batches     : {len(train_dl):,}")
    print(f"  Validation batches   : {len(val_dl):,}")
    print(f"  Test batches         : {len(test_dl):,}")

    print("\n=== Building GPT model (improved) ===")
    model = GPTModel(
        vocab_size=tokenizer.vocab_size,
        d_model=IMPROVED_CONFIG["d_model"],
        num_heads=IMPROVED_CONFIG["num_heads"],
        num_layers=IMPROVED_CONFIG["num_layers"],
        d_ff=IMPROVED_CONFIG["d_ff"],
        max_len=IMPROVED_CONFIG["block_size"],
        dropout=IMPROVED_CONFIG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters : {n_params / 1e6:.2f}M")

    print("\n=== Running sanity checks ===")
    sanity_check_build_word_vocab()
    sanity_check_lm_dataset(dataset["train"], tokenizer, block_size=IMPROVED_CONFIG["block_size"])
    sanity_check_model(model, tokenizer, device)
    print("  all checks passed.\n")

    print("=== Training ===")
    print("  Run type           : optional improved model")
    print("  Checkpoint         : improved_gpt_model.pt")
    print("  Tuning signal      : validation perplexity and public WikiText-2 test perplexity")
    print(f"  Base reference     : public_test_ppl ~= {BASE_REFERENCE_PUBLIC_TEST_PPL:.2f}")
    print("  Final grading      : TA-only private held-out corpus\n")

    _, _, val_ppls = train(model, train_dl, val_dl, IMPROVED_TRAIN_OPTS, device)

    criterion = nn.CrossEntropyLoss()
    _, test_ppl = evaluate(model, test_dl, criterion, device)
    best_val_ppl = min(val_ppls)

    print(f"\n{'=' * 50}")
    print(f"  Best validation perplexity : {best_val_ppl:.2f}")
    print(f"  Public test perplexity     : {test_ppl:.2f}")
    if test_ppl < BASE_REFERENCE_PUBLIC_TEST_PPL:
        print("  [OK] Public test perplexity improved over the base reference run.")
    else:
        print("  [!] Keep tuning if you want a stronger improved model.")
    print("  [Note] Final bonus grading uses a private held-out corpus.")
    print(f"{'=' * 50}\n")

    torch.save(
        {
            "state": model.state_dict(),
            "config": IMPROVED_CONFIG,
            "train_opts": IMPROVED_TRAIN_OPTS,
            "mode": "improved",
            "vocab_size": tokenizer.vocab_size,
            "best_val_ppl": best_val_ppl,
            "test_ppl": test_ppl,
            "public_test_ppl": test_ppl,
        },
        "improved_gpt_model.pt",
    )
    print("Checkpoint saved to improved_gpt_model.pt")

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
    main()
