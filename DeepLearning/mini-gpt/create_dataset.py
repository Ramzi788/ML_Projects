import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def load_wikitext2():
    return load_dataset("wikitext", "wikitext-2-raw-v1")


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = 10 ** 9
    return tokenizer


def tokenize_dataset(dataset_split, tokenizer):
    text = " ".join(line for line in dataset_split["text"] if line.strip())
    encodings = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encodings["input_ids"].squeeze(0)


class LMDataset(Dataset):
    def __init__(self, token_ids, block_size):
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self):
        return (len(self.token_ids) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.token_ids[start : start + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def create_dataloaders(train_ids, val_ids, test_ids, block_size, batch_size):
    train_ds = LMDataset(train_ids, block_size)
    val_ds = LMDataset(val_ids, block_size)
    test_ds = LMDataset(test_ids, block_size)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    return train_dl, val_dl, test_dl


def build_word_vocab(text):
    unique = sorted(set(text.split()))
    word2idx = {w: i for i, w in enumerate(unique)}
    idx2word = {i: w for i, w in enumerate(unique)}
    return word2idx, idx2word


def sanity_check_lm_dataset(dataset_split, tokenizer, block_size=128):
    token_ids = tokenize_dataset(dataset_split, tokenizer)
    ds = LMDataset(token_ids, block_size)
    expected_len = (len(token_ids) - 1) // block_size

    assert len(ds) == expected_len, (
        f"LMDataset.__len__ returned {len(ds)}, expected {expected_len}."
    )

    x0, y0 = ds[0]
    assert x0.shape == torch.Size([block_size]), (
        f"x.shape should be ({block_size},), got {x0.shape}."
    )
    assert y0.shape == torch.Size([block_size]), (
        f"y.shape should be ({block_size},), got {y0.shape}."
    )
    assert (y0 == token_ids[1 : block_size + 1]).all(), (
        "y should equal token_ids[1 : block_size + 1]."
    )

    if len(ds) > 1:
        x1, _ = ds[1]
        assert (x1 == token_ids[block_size : 2 * block_size]).all(), (
            "the second example should start at token_ids[block_size]."
        )

    print("  [OK] LMDataset sanity check passed.")


def sanity_check_build_word_vocab():
    sample = "the cat sat on the mat"
    word2idx, idx2word = build_word_vocab(sample)
    unique = sorted(set(sample.split()))

    assert len(word2idx) == len(unique), "word2idx has the wrong size."
    for word, idx in word2idx.items():
        assert idx2word[idx] == word, "idx2word should invert word2idx."

    print("  [OK] build_word_vocab sanity check passed.")
