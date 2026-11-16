import argparse
import csv
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (
    CLASSES,
    MODELS,
    RESULTS,
    SEED,
    SurfaceDataset,
    build_model,
    build_transforms,
    evaluate,
    set_seed,
    set_train_mode,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["cnn", "resnet18", "mobilenet_v3_small"], default="cnn")
    p.add_argument("--name", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=200)
    p.add_argument("--aug", choices=["none", "basic"], default="basic")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, arch, freeze):
    set_train_mode(model, arch, freeze)
    total_loss = 0.0
    correct = 0
    count = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        count += x.size(0)
    return total_loss / count, correct / count


def save_curves(history, path, title):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiet bi:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    train_tf, eval_tf = build_transforms(args.img_size, args.aug)
    train_ds = SurfaceDataset("train", train_tf)
    val_ds = SurfaceDataset("val", eval_tf)
    gen = torch.Generator()
    gen.manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        generator=gen,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Train: {len(train_ds)} anh | Val: {len(val_ds)} anh")

    model = build_model(args.arch, args.pretrained, args.freeze, args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Tham so: {total_params:,} | Huan luyen: {sum(p.numel() for p in trainable):,}")

    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    MODELS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    ckpt_path = MODELS / f"{args.name}.pt"

    history = []
    best_acc = -1.0
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, train_criterion, optimizer, device, args.arch, args.freeze
        )
        va_loss, va_acc, _, _ = evaluate(model, val_loader, eval_criterion, device)
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 4),
                "train_acc": round(tr_acc, 4),
                "val_loss": round(va_loss, 4),
                "val_acc": round(va_acc, 4),
            }
        )
        improved = va_acc > best_acc or (va_acc == best_acc and va_loss < best_loss)
        mark = ""
        if improved:
            best_acc, best_loss, best_epoch, wait = va_acc, va_loss, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": args.arch,
                    "img_size": args.img_size,
                    "classes": CLASSES,
                    "config": vars(args),
                    "epoch": epoch,
                    "val_acc": va_acc,
                },
                ckpt_path,
            )
            mark = "  <- tot nhat"
        else:
            wait += 1
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time() - t0:.1f}s{mark}"
        )
        if wait >= args.patience:
            print(f"Dung som: val khong cai thien sau {args.patience} epoch")
            break

    elapsed = time.time() - start
    with open(RESULTS / f"{args.name}_history.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "best_val_loss": best_loss,
        "train_seconds": round(elapsed, 1),
        "total_params": total_params,
    }
    with open(RESULTS / f"{args.name}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    save_curves(history, RESULTS / f"{args.name}_curves.png", args.name)
    print(f"Xong sau {elapsed / 60:.1f} phut | tot nhat epoch {best_epoch}, val acc {best_acc:.4f}")
    print("Trong so:", ckpt_path)


if __name__ == "__main__":
    main()