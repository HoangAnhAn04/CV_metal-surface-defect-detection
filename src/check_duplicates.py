from pathlib import Path
import hashlib
from collections import defaultdict
from PIL import Image
import imagehash

ROOT = Path("data/NEU-DET")
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
THRESHOLD = 5

items = []
for split in ["train", "validation"]:
    img_dir = ROOT / split / "images"
    for p in sorted(img_dir.rglob("*")):
        if p.suffix.lower() in EXTS:
            items.append((split, p.parent.name, p))

print("Tong so anh:", len(items))
for split in ["train", "validation"]:
    print(" ", split, ":", sum(1 for s, _, _ in items if s == split))

md5_map = defaultdict(list)
hashes = []
for split, label, p in items:
    md5_map[hashlib.md5(p.read_bytes()).hexdigest()].append((split, label, p))
    ph = int(str(imagehash.phash(Image.open(p))), 16)
    hashes.append((split, label, p, ph))

exact = [g for g in md5_map.values() if len(g) > 1]

near = []
for i in range(len(hashes)):
    for j in range(i + 1, len(hashes)):
        d = bin(hashes[i][3] ^ hashes[j][3]).count("1")
        if d <= THRESHOLD:
            near.append((d, hashes[i][:3], hashes[j][:3]))

cross_split = [n for n in near if n[1][0] != n[2][0]]
cross_label = [n for n in near if n[1][1] != n[2][1]]

print("Nhom trung y het (MD5):", len(exact))
print("Cap gan giong (pHash <=", THRESHOLD, "):", len(near))
print("  trong do nam o ca train va validation:", len(cross_split))
print("  trong do khac nhan:", len(cross_label))

with open("duplicates_report.txt", "w", encoding="utf-8") as f:
    f.write("TRUNG Y HET\n")
    for g in exact:
        f.write(" | ".join(str(x[2]) for x in g) + "\n")
    f.write("\nGAN GIONG\n")
    for d, a, b in near:
        f.write(f"{d}: {a[2]} <-> {b[2]}\n")
print("Da ghi chi tiet vao duplicates_report.txt")