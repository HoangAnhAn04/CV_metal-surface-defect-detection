import numpy as np

from common import DATA
from infer import Classifier, list_images, read_image

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]


def main():
    clf = Classifier("resnet18_strong")
    rows = []
    for path in list_images(DATA / "webcam"):
        probs = clf.predict(read_image(path))
        idx = int(np.argmax(probs))
        rows.append((float(probs[idx]), clf.classes[idx] == path.parent.name, path.name))
    conf = np.array([r[0] for r in rows])
    ok = np.array([r[1] for r in rows])
    print(f"Tong {len(rows)} anh | dung {int(ok.sum())}")
    print(f"Do tin cay trung binh: anh dung {conf[ok].mean():.3f} | anh sai {conf[~ok].mean():.3f}")
    print(f"{'nguong':>7}{'giu lai':>9}{'ty le giu':>11}{'acc giu lai':>13}{'sai con lai':>13}")
    for t in THRESHOLDS:
        keep = conf >= t
        n = int(keep.sum())
        acc = float(ok[keep].mean()) if n else 0.0
        print(f"{t:>7.2f}{n:>9}{n / len(rows):>11.3f}{acc:>13.3f}{int((~ok[keep]).sum()):>13}")
    print("Anh sai:")
    for c, good, name in sorted(rows, reverse=True):
        if not good:
            print(f"  {name}: do tin cay {c:.3f}")


if __name__ == "__main__":
    main()