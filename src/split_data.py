import csv
import hashlib
import random
import shutil
from pathlib import Path

import cv2

SEED = 223332813
N_VAL = 45
N_TEST = 45
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SOURCE_SPLITS = ["train", "validation"]
SPLIT_NAMES = ["train", "val", "test"]

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "NEU-DET"
DST = ROOT / "data" / "processed"
REPORT = ROOT / "results" / "split_report.txt"
MANIFEST = ROOT / "results" / "split_manifest.csv"


def collect():
    items = []
    for split in SOURCE_SPLITS:
        base = SRC / split / "images"
        for cls_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in EXTS:
                    items.append((cls_dir.name, f))
    return items


def pixel_hash(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return hashlib.md5(img.tobytes() + str(img.shape).encode()).hexdigest()


def remove_duplicates(items, log):
    seen = {}
    kept = []
    for cls, f in items:
        h = pixel_hash(f)
        if h in seen:
            log(f"Loai anh trung: {f.relative_to(ROOT)} (giong {seen[h].relative_to(ROOT)})")
            continue
        seen[h] = f
        kept.append((cls, f))
    return kept


def split_class(files, rng):
    files = list(files)
    rng.shuffle(files)
    test = files[:N_TEST]
    val = files[N_TEST:N_TEST + N_VAL]
    train = files[N_TEST + N_VAL:]
    return {"train": train, "val": val, "test": test}


def main():
    lines = []

    def log(text=""):
        print(text)
        lines.append(str(text))

    items = collect()
    log("Tong anh goc: " + str(len(items)))
    items = remove_duplicates(items, log)
    log("Sau khi loai trung: " + str(len(items)))

    by_class = {}
    for cls, f in items:
        by_class.setdefault(cls, []).append(f)

    for name in SPLIT_NAMES:
        target = DST / name
        if target.exists():
            shutil.rmtree(target)

    rng = random.Random(SEED)
    rows = []
    counts = {}
    for cls in sorted(by_class):
        parts = split_class(by_class[cls], rng)
        for name in SPLIT_NAMES:
            out_dir = DST / name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in parts[name]:
                shutil.copy2(f, out_dir / f.name)
                rows.append([f.name, cls, name, str(f.relative_to(ROOT))])
            counts[(cls, name)] = len(parts[name])

    log("")
    log(f"{'lop':<18}{'train':>7}{'val':>7}{'test':>7}{'tong':>7}")
    totals = {name: 0 for name in SPLIT_NAMES}
    for cls in sorted(by_class):
        row = [counts[(cls, name)] for name in SPLIT_NAMES]
        for name, n in zip(SPLIT_NAMES, row):
            totals[name] += n
        log(f"{cls:<18}{row[0]:>7}{row[1]:>7}{row[2]:>7}{sum(row):>7}")
    log(
        f"{'TONG':<18}{totals['train']:>7}{totals['val']:>7}"
        f"{totals['test']:>7}{sum(totals.values()):>7}"
    )
    log("Seed: " + str(SEED))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "class", "split", "source"])
        writer.writerows(rows)


if __name__ == "__main__":
    main()