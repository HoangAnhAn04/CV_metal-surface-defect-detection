import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from common import CLASSES, MEAN, MODELS, RESULTS, SEED, STD, SurfaceDataset, build_model

PERTURBATIONS = ["none", "blur", "noise", "dark", "bright", "lowcontrast"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--perturb", choices=PERTURBATIONS, default="none")
    p.add_argument("--batch", type=int, default=32)
    return p.parse_args()


def make_perturb(kind):
    rng = np.random.default_rng(SEED)

    def apply(img):
        if kind == "blur":
            return img.filter(ImageFilter.GaussianBlur(radius=2.0))
        if kind == "noise":
            arr = np.asarray(img, dtype=np.float32)
            noise = rng.normal(0, 25, arr.shape[:2])[:, :, None]
            return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
        if kind == "dark":
            return ImageEnhance.Brightness(img).enhance(0.4)
        if kind == "bright":
            return ImageEnhance.Brightness(img).enhance(1.8)
        if kind == "lowcontrast":
            return ImageEnhance.Contrast(img).enhance(0.4)
        return img

    return apply


def build_eval_tf(img_size, kind):
    ops = []
    if kind != "none":
        ops.append(transforms.Lambda(make_perturb(kind)))
    ops += [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
    return transforms.Compose(ops)


def load_model(name, device):
    ckpt = torch.load(MODELS / f"{name}.pt", map_location=device, weights_only=False)
    model = build_model(ckpt["arch"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict(model, loader, device):
    probs = []
    labels = []
    for x, y in loader:
        out = model(x.to(device))
        probs.append(torch.softmax(out, 1).cpu().numpy())
        labels.extend(y.tolist())
    return np.concatenate(probs), np.array(labels)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def save_confusion(cm, path, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Du doan")
    ax.set_ylabel("Thuc te")
    ax.set_title(title)
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR = RESULTS if (args.split == "test" and args.perturb == "none") else RESULTS / "robustness"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    label_ids = list(range(len(CLASSES)))
    rows = []
    wrong_lines = []

    for name in args.models:
        model, ckpt = load_model(name, device)
        tf = build_eval_tf(ckpt["img_size"], args.perturb)
        ds = SurfaceDataset(args.split, tf)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
        probs, labels = predict(model, loader, device)
        preds = probs.argmax(1)

        tag = f"{name}_{args.split}_{args.perturb}"
        text = classification_report(
            labels, preds, labels=label_ids, target_names=CLASSES, digits=4, zero_division=0
        )
        rep = classification_report(
            labels,
            preds,
            labels=label_ids,
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(labels, preds, labels=label_ids)
        correct = int((preds == labels).sum())
        n = len(labels)
        low, high = wilson(correct, n)
        macro = rep["macro avg"]

        save_confusion(cm, OUT_DIR / f"{tag}_confusion.png", f"{name} | {args.split} | {args.perturb}")
        header = f"Mo hinh: {name} | tap: {args.split} | bien doi: {args.perturb}\n"
        header += f"Dung {correct}/{n} | acc {correct / n:.4f} | CI95 [{low:.4f}, {high:.4f}]\n\n"
        (OUT_DIR / f"{tag}_report.txt").write_text(
            header + text + "\nConfusion matrix:\n" + str(cm) + "\n", encoding="utf-8"
        )

        wrong = np.where(preds != labels)[0]
        for i in wrong:
            wrong_lines.append(
                f"  [{name}] {ds.samples[i][0].name}: that {CLASSES[labels[i]]}, "
                f"doan {CLASSES[preds[i]]} ({probs[i].max():.3f})"
            )
        metrics = {
            "model": name,
            "split": args.split,
            "perturb": args.perturb,
            "n": n,
            "correct": correct,
            "accuracy": correct / n,
            "ci95_low": float(low),
            "ci95_high": float(high),
            "macro_precision": float(macro["precision"]),
            "macro_recall": float(macro["recall"]),
            "macro_f1": float(macro["f1-score"]),
            "wrong": int(len(wrong)),
        }
        (OUT_DIR / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        rows.append(metrics)

    print(f"TONG HOP | tap {args.split} | bien doi {args.perturb}")
    print(f"{'mo hinh':<16}{'acc':>8}{'CI95 duoi':>11}{'macroP':>8}{'macroR':>8}{'macroF1':>9}{'sai':>5}")
    for r in rows:
        print(
            f"{r['model']:<16}{r['accuracy']:>8.4f}{r['ci95_low']:>11.4f}"
            f"{r['macro_precision']:>8.4f}{r['macro_recall']:>8.4f}{r['macro_f1']:>9.4f}{r['wrong']:>5}"
        )
    print("ANH SAI:")
    print("\n".join(wrong_lines) if wrong_lines else "  khong co")


if __name__ == "__main__":
    main()