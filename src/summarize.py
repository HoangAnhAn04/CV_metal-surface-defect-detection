import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS

ROB = RESULTS / "robustness"
OUT = RESULTS / "summary"
MODELS = ["cnn_baseline", "cnn_noaug", "cnn_strong", "resnet18_ft", "resnet18_strong"]
NAMES = {
    "cnn_baseline": "CNN (augmentation cơ bản)",
    "cnn_noaug": "CNN (không augmentation)",
    "cnn_strong": "CNN (augmentation mạnh)",
    "resnet18_ft": "ResNet18 (augmentation cơ bản)",
    "resnet18_strong": "ResNet18 (augmentation mạnh)",
}
PERTURBS = ["none", "blur", "noise", "dark", "bright", "lowcontrast"]
PERTURB_VN = {
    "none": "Ảnh gốc",
    "blur": "Làm mờ",
    "noise": "Nhiễu",
    "dark": "Tối",
    "bright": "Sáng",
    "lowcontrast": "Tương phản thấp",
}

plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 12


def get(model, split, perturb, folder):
    path = folder / f"{model}_{split}_{perturb}_metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def show(title, header, rows):
    print(title)
    print("\t".join(header))
    for row in rows:
        print("\t".join(str(c) for c in row))
    print()


def table_main(split, folder, filename, title):
    header = ["Mô hình", "Đúng", "Tổng", "Accuracy (%)", "CI95 dưới (%)", "CI95 trên (%)", "Macro F1 (%)", "Sai"]
    rows = []
    for m in MODELS:
        d = get(m, split, "none", folder)
        if d:
            rows.append(
                [
                    NAMES[m],
                    d["correct"],
                    d["n"],
                    round(d["accuracy"] * 100, 2),
                    round(d["ci95_low"] * 100, 2),
                    round(d["ci95_high"] * 100, 2),
                    round(d["macro_f1"] * 100, 2),
                    d["wrong"],
                ]
            )
    write_csv(OUT / filename, header, rows)
    show(title, header, rows)


def table_robust():
    header = ["Mô hình"] + [PERTURB_VN[p] for p in PERTURBS]
    rows = []
    for m in MODELS:
        row = [NAMES[m]]
        for p in PERTURBS:
            d = get(m, "val", p, ROB)
            row.append(round(d["accuracy"] * 100, 2) if d else "")
        rows.append(row)
    write_csv(OUT / "bang_do_ben.csv", header, rows)
    show("ĐỘ BỀN TRÊN TẬP VALIDATION BỊ LÀM XẤU (accuracy %)", header, rows)


def fig_robust():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    width = 0.16
    x = np.arange(len(PERTURBS))
    hatches = ["", "//", "..", "xx", "\\\\"]
    for i, m in enumerate(MODELS):
        vals = []
        for p in PERTURBS:
            d = get(m, "val", p, ROB)
            vals.append(d["accuracy"] * 100 if d else np.nan)
        ax.bar(
            x + (i - 2) * width,
            vals,
            width,
            label=NAMES[m],
            hatch=hatches[i],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([PERTURB_VN[p] for p in PERTURBS])
    ax.set_ylabel("Độ chính xác trên tập validation (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "hinh_do_ben.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_test_webcam():
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(MODELS))
    width = 0.36
    series = [
        ("test", RESULTS, "Tập test (ảnh NEU)", ""),
        ("webcam", ROB, "Ảnh webcam", "//"),
    ]
    for j, (split, folder, label, hatch) in enumerate(series):
        vals = []
        lows = []
        highs = []
        for m in MODELS:
            d = get(m, split, "none", folder)
            if d:
                a = d["accuracy"] * 100
                vals.append(a)
                lows.append(a - d["ci95_low"] * 100)
                highs.append(d["ci95_high"] * 100 - a)
            else:
                vals.append(np.nan)
                lows.append(0)
                highs.append(0)
        pos = x + (j - 0.5) * width
        ax.bar(
            pos,
            vals,
            width,
            yerr=[lows, highs],
            capsize=3,
            label=label,
            hatch=hatch,
            edgecolor="black",
            linewidth=0.5,
        )
        for px, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(px, 2, f"{v:.1f}", ha="center", va="bottom", fontsize=10, color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[m].replace(" (", "\n(") for m in MODELS], fontsize=10)
    ax.set_ylabel("Độ chính xác (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "hinh_test_webcam.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table_main("test", RESULTS, "bang_test.csv", "KẾT QUẢ TRÊN TẬP TEST")
    table_robust()
    table_main("webcam", ROB, "bang_webcam.csv", "KẾT QUẢ TRÊN ẢNH WEBCAM")
    fig_robust()
    fig_test_webcam()
    print("Đã lưu vào:", OUT)


if __name__ == "__main__":
    main()