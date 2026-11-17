from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results"
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SMALL = 40
TOP = 8


def load_split(split):
    paths = []
    for cls_dir in sorted(p for p in (DATA / split).iterdir() if p.is_dir()):
        paths += [f for f in sorted(cls_dir.iterdir()) if f.suffix.lower() in EXTS]
    return paths


def to_vector(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (SMALL, SMALL), interpolation=cv2.INTER_AREA)
    vec = small.astype(np.float32).ravel()
    vec -= vec.mean()
    return vec / (np.linalg.norm(vec) + 1e-8)


def vectors(paths):
    return np.stack([to_vector(p) for p in paths])


def describe(name, sims):
    p50, p90, p99 = np.percentile(sims, [50, 90, 99])
    print(f"{name}: trung vi {p50:.3f} | P90 {p90:.3f} | P99 {p99:.3f} | max {sims.max():.3f}")


def make_tile(query, neighbor, sim, split):
    a = cv2.resize(cv2.imread(str(query), cv2.IMREAD_GRAYSCALE), (200, 200))
    b = cv2.resize(cv2.imread(str(neighbor), cv2.IMREAD_GRAYSCALE), (200, 200))
    pair = cv2.cvtColor(np.hstack([a, b]), cv2.COLOR_GRAY2BGR)
    cv2.putText(pair, f"{split} {sim:.3f}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    return pair


def main():
    train_paths = load_split("train")
    train_vec = vectors(train_paths)
    self_sim = train_vec @ train_vec.T
    np.fill_diagonal(self_sim, -1.0)
    describe("train - train", self_sim.max(axis=1))

    candidates = []
    for split in ["val", "test"]:
        paths = load_split(split)
        sim = vectors(paths) @ train_vec.T
        best = sim.argmax(axis=1)
        best_sim = sim.max(axis=1)
        describe(f"{split} - train", best_sim)
        same = np.mean(
            [paths[i].parent.name == train_paths[best[i]].parent.name for i in range(len(paths))]
        )
        print(f"  1-NN cung lop (pixel 40x40): {same:.3f}")
        for i in range(len(paths)):
            candidates.append((best_sim[i], split, paths[i], train_paths[best[i]]))

    candidates.sort(key=lambda c: -c[0])
    print("Cac cap giong nhat:")
    for sim, split, q, n in candidates[:TOP]:
        print(f"  {sim:.3f} {split}/{q.parent.name}/{q.name} <-> train/{n.parent.name}/{n.name}")

    tiles = [make_tile(q, n, s, sp) for s, sp, q, n in candidates[:TOP]]
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    OUT.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT / "leakage_pairs.png"), np.vstack(rows))
    print("Da luu:", OUT / "leakage_pairs.png")


if __name__ == "__main__":
    main()