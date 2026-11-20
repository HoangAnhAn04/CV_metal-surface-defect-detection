import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from common import MEAN, MODELS, RESULTS, STD, build_model

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
FONT = cv2.FONT_HERSHEY_SIMPLEX
OK_COLOR = (80, 200, 80)
WARN_COLOR = (0, 165, 255)
PANEL_BG = (35, 35, 35)
VIEW = 400
PANEL = 320


class Classifier:
    def __init__(self, name="resnet18_strong", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(MODELS / f"{name}.pt", map_location=self.device, weights_only=False)
        self.model = build_model(ckpt["arch"])
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        self.classes = ckpt["classes"]
        size = ckpt["img_size"]
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    @torch.no_grad()
    def predict(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        x = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        return torch.softmax(self.model(x), dim=1)[0].cpu().numpy()


def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image(path, image):
    ok, buf = cv2.imencode(".png", image)
    if ok:
        buf.tofile(str(path))


def list_images(source):
    source = Path(source)
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in EXTS)


def draw_result(bgr, probs, classes, threshold):
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    sure = conf >= threshold
    color = OK_COLOR if sure else WARN_COLOR

    view = cv2.resize(bgr, (VIEW, VIEW), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(view, (0, 0), (VIEW, 44), PANEL_BG, -1)
    cv2.putText(view, f"{classes[idx]}  {conf * 100:.1f}%", (10, 30), FONT, 0.8, color, 2, cv2.LINE_AA)
    cv2.rectangle(view, (0, 0), (VIEW - 1, VIEW - 1), color, 3)

    panel = np.full((VIEW, PANEL, 3), PANEL_BG, dtype=np.uint8)
    status = "CONFIDENT" if sure else "UNCERTAIN"
    cv2.putText(panel, status, (12, 30), FONT, 0.7, color, 2, cv2.LINE_AA)
    for row, k in enumerate(np.argsort(-probs)):
        y = 70 + row * 52
        bar = int(probs[k] * (PANEL - 24))
        bar_color = color if k == idx else (150, 150, 150)
        cv2.putText(panel, classes[k], (12, y), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{probs[k] * 100:.1f}%", (PANEL - 80, y), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (12, y + 8), (12 + bar, y + 26), bar_color, -1)
    return np.hstack([view, panel])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--model", default="resnet18_strong")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--out", default=str(RESULTS / "predictions"))
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    clf = Classifier(args.model)
    files = list_images(args.source)
    if not files:
        print("Khong tim thay anh trong:", args.source)
        return
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    known = 0
    correct = 0
    for path in files:
        bgr = read_image(path)
        if bgr is None:
            print("Khong doc duoc:", path)
            continue
        probs = clf.predict(bgr)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        pred = clf.classes[idx]
        true = path.parent.name if path.parent.name in clf.classes else ""
        uncertain = conf < args.threshold
        note = ""
        if true:
            known += 1
            correct += int(true == pred)
            if true != pred:
                note += " [SAI, that: " + true + "]"
        if uncertain:
            note += " [KHONG CHAC]"
        print(f"{path.name}: {pred} ({conf * 100:.1f}%){note}")
        rows.append([path.name, pred, round(conf, 4), uncertain, true])

        canvas = draw_result(bgr, probs, clf.classes, args.threshold)
        save_image(out_dir / f"{path.parent.name}_{path.stem}_pred.png", canvas)
        if args.show:
            cv2.imshow("Phan loai loi be mat kim loai (nhan q de thoat)", canvas)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()

    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "predicted", "confidence", "uncertain", "true"])
        writer.writerows(rows)

    print(f"Tong: {len(rows)} anh | khong chac chan (< {args.threshold}): {sum(r[3] for r in rows)}")
    if known:
        print(f"Do chinh xac tren {known} anh co nhan: {correct}/{known} = {correct / known:.4f}")


if __name__ == "__main__":
    main()