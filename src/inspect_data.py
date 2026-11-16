from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "NEU-DET"
OUT = ROOT / "results" / "dataset_report.txt"
SPLITS = ["train", "validation"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
lines = []


def log(text=""):
    print(text)
    lines.append(str(text))


def list_images(split):
    result = {}
    base = DATA / split / "images"
    for cls_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        result[cls_dir.name] = sorted(
            f for f in cls_dir.iterdir() if f.suffix.lower() in EXTS
        )
    return result


def index_xml(split):
    base = DATA / split / "annotations"
    if not base.exists():
        return {}
    return {p.stem: p for p in base.rglob("*.xml")}


def parse_xml(path):
    root = ET.parse(path).getroot()
    size = root.find("size")
    w = int(float(size.find("width").text))
    h = int(float(size.find("height").text))
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        boxes.append(
            (
                obj.find("name").text.strip(),
                float(bb.find("xmin").text),
                float(bb.find("ymin").text),
                float(bb.find("xmax").text),
                float(bb.find("ymax").text),
            )
        )
    return w, h, boxes


def report_boxes(images, xmls):
    log("Thong ke bbox (ty le dien tich bbox / dien tich anh):")
    for cls, files in images.items():
        ratios = []
        counts = Counter()
        names = Counter()
        for f in files:
            if f.stem not in xmls:
                continue
            w, h, boxes = parse_xml(xmls[f.stem])
            counts[len(boxes)] += 1
            for name, x1, y1, x2, y2 in boxes:
                names[name] += 1
                ratios.append(max(0.0, (x2 - x1) * (y2 - y1)) / (w * h))
        if not ratios:
            continue
        big = sum(r > 0.8 for r in ratios) / len(ratios) * 100
        log(f"  {cls}: so box/anh {dict(sorted(counts.items()))}, nhan XML {dict(names)}")
        log(
            f"     ty le dien tich: min {min(ratios):.2f}, "
            f"tb {sum(ratios) / len(ratios):.2f}, max {max(ratios):.2f}, "
            f"box >80% anh: {big:.0f}%"
        )


def main():
    stems_by_split = {}
    for split in SPLITS:
        log("=" * 60)
        log("TAP: " + split)
        images = list_images(split)
        xmls = index_xml(split)
        log("So file XML: " + str(len(xmls)))
        shapes = Counter()
        unreadable = []
        missing_xml = []
        stems = set()
        total = 0
        for cls, files in images.items():
            log(f"  {cls}: {len(files)} anh")
            total += len(files)
            for f in files:
                stems.add(f.stem)
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is None:
                    unreadable.append(str(f))
                    continue
                shapes[tuple(img.shape)] += 1
                if xmls and f.stem not in xmls:
                    missing_xml.append(f.stem)
        stems_by_split[split] = stems
        log("Tong so anh: " + str(total))
        log("Kich thuoc anh (h, w, kenh): " + str(dict(shapes)))
        log("Anh khong doc duoc: " + str(len(unreadable)))
        if xmls:
            log("Anh thieu XML: " + str(len(missing_xml)))
            log("XML khong co anh: " + str(len(set(xmls) - stems)))
            report_boxes(images, xmls)
        else:
            log("Khong tim thay annotation XML")
    common = stems_by_split["train"] & stems_by_split["validation"]
    log("=" * 60)
    log("Ten file trung giua train va validation: " + str(len(common)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()