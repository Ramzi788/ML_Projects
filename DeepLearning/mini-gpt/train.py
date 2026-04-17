import math
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm.auto import tqdm

matplotlib.use("Agg")


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    progress = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{num_epochs}", leave=False)

    for x, y in progress:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def train(model, train_dl, val_dl, train_opts, device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_opts["lr"],
        weight_decay=train_opts.get("weight_decay", 0.01),
    )

    total_steps = train_opts["num_epochs"] * len(train_dl)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []
    val_ppls = []

    for epoch in range(1, train_opts["num_epochs"] + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_dl, optimizer, scheduler, criterion, device, epoch, train_opts["num_epochs"])
        val_loss, val_ppl = evaluate(model, val_dl, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ppls.append(val_ppl)

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[{epoch:3d}/{train_opts['num_epochs']}]  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_ppl={val_ppl:7.2f}  lr={lr_now:.2e}  ({elapsed:.1f}s)"
        )

    plot(train_losses, val_losses, val_ppls)
    return train_losses, val_losses, val_ppls


def plot(train_losses, val_losses, val_ppls):
    epochs = list(range(1, len(train_losses) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss")
    ax1.plot(epochs, val_losses, "r-o", markersize=3, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()

    ax2.plot(epochs, val_ppls, "g-o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Validation Perplexity")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved to training_curves.png")
