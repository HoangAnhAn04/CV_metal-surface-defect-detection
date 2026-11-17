import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from common import CLASSES, DATA, MODELS, RESULTS
from infer import Classifier, draw_result, list_images, read_image, save_image

CAM_W = 480
ROI_RATIO = 0.6
SMOOTH = 0.6
INTERVAL_MS = 30


def to_photo(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Phân loại lỗi bề mặt kim loại - NEU")
        self.models = sorted(p.stem for p in MODELS.glob("*.pt"))
        default = "resnet18_strong" if "resnet18_strong" in self.models else self.models[0]
        self.clf = Classifier(default)
        self.files = []
        self.index = 0
        self.cap = None
        self.smoothed = None
        self.last_bgr = None
        self.last_roi = None
        self.canvas = None
        self.last_time = time.time()
        self.fps = 0.0
        self.build_ui(default)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def build_ui(self, default):
        bar1 = ttk.Frame(self.root, padding=6)
        bar1.pack(fill="x")
        ttk.Button(bar1, text="Chọn ảnh", command=self.open_image).pack(side="left", padx=2)
        ttk.Button(bar1, text="Chọn thư mục", command=self.open_folder).pack(side="left", padx=2)
        ttk.Button(bar1, text="◀ Trước", command=lambda: self.step(-1)).pack(side="left", padx=2)
        ttk.Button(bar1, text="Sau ▶", command=lambda: self.step(1)).pack(side="left", padx=2)
        self.cam_btn = ttk.Button(bar1, text="Bật webcam", command=self.toggle_camera)
        self.cam_btn.pack(side="left", padx=2)
        ttk.Button(bar1, text="Lưu kết quả", command=self.save_result).pack(side="left", padx=2)

        bar2 = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        bar2.pack(fill="x")
        ttk.Label(bar2, text="Mô hình:").pack(side="left")
        self.model_var = tk.StringVar(value=default)
        box = ttk.Combobox(
            bar2, textvariable=self.model_var, values=self.models, state="readonly", width=18
        )
        box.pack(side="left", padx=4)
        box.bind("<<ComboboxSelected>>", self.change_model)
        ttk.Label(bar2, text="Ngưỡng:").pack(side="left", padx=(12, 0))
        self.thr_var = tk.DoubleVar(value=0.7)
        ttk.Scale(
            bar2, from_=0.3, to=0.99, variable=self.thr_var, length=140, command=self.on_threshold
        ).pack(side="left", padx=4)
        self.thr_text = ttk.Label(bar2, text="0.70", width=5)
        self.thr_text.pack(side="left")
        ttk.Label(bar2, text="Nhãn thật:").pack(side="left", padx=(12, 0))
        self.label_var = tk.StringVar()
        ttk.Combobox(
            bar2, textvariable=self.label_var, values=CLASSES, state="readonly", width=16
        ).pack(side="left", padx=4)
        ttk.Button(bar2, text="Lưu ROI webcam", command=self.save_roi).pack(side="left", padx=2)

        self.view = ttk.Frame(self.root)
        self.view.pack(padx=6)
        self.cam_label = ttk.Label(self.view)
        self.result_label = ttk.Label(self.view)
        self.result_label.pack(side="left")
        self.status = ttk.Label(
            self.root, text="Chọn ảnh, chọn thư mục hoặc bật webcam", padding=6
        )
        self.status.pack(fill="x")

    def threshold(self):
        return round(float(self.thr_var.get()), 2)

    def set_label(self, label, bgr):
        photo = to_photo(bgr)
        label.configure(image=photo)
        label.image = photo

    def render_image(self):
        probs = self.clf.predict(self.last_bgr)
        self.canvas = draw_result(self.last_bgr, probs, self.clf.classes, self.threshold())
        self.set_label(self.result_label, self.canvas)

    def on_threshold(self, _):
        self.thr_text.config(text=f"{self.threshold():.2f}")
        if self.cap is None and self.last_bgr is not None:
            self.render_image()

    def change_model(self, _):
        self.clf = Classifier(self.model_var.get())
        self.smoothed = None
        if self.cap is None and self.last_bgr is not None:
            self.render_image()

    def begin_files(self, files):
        self.stop_camera()
        self.files = files
        self.index = 0
        self.show_current()

    def show_current(self):
        path = self.files[self.index]
        bgr = read_image(path)
        if bgr is None:
            messagebox.showerror("Lỗi", f"Không đọc được ảnh: {path}")
            return
        self.last_bgr = bgr
        self.render_image()
        self.status.config(text=f"{path.name}  ({self.index + 1}/{len(self.files)})")

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            self.begin_files([Path(path)])

    def open_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        files = list_images(folder)
        if not files:
            messagebox.showinfo("Thông báo", "Thư mục không có ảnh")
            return
        self.begin_files(files)

    def step(self, delta):
        if self.cap is not None or not self.files:
            return
        self.index = (self.index + delta) % len(self.files)
        self.show_current()

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cam_label.pack_forget()
        self.cam_btn.config(text="Bật webcam")
        self.smoothed = None

    def toggle_camera(self):
        if self.cap is not None:
            self.stop_camera()
            self.status.config(text="Đã tắt webcam")
            return
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            messagebox.showerror("Lỗi", "Không mở được webcam")
            return
        self.cap = cap
        self.smoothed = None
        self.cam_label.pack(side="left", before=self.result_label)
        self.cam_btn.config(text="Tắt webcam")
        self.status.config(text="Đưa ảnh vào khung vàng")
        self.loop()

    def loop(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.stop_camera()
            self.status.config(text="Mất tín hiệu webcam")
            return
        h, w = frame.shape[:2]
        side = int(min(h, w) * ROI_RATIO)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        roi = frame[y0:y0 + side, x0:x0 + side].copy()
        self.last_roi = roi
        probs = self.clf.predict(roi)
        if self.smoothed is None:
            self.smoothed = probs
        else:
            self.smoothed = SMOOTH * self.smoothed + (1 - SMOOTH) * probs
        now = time.time()
        self.fps = 0.9 * self.fps + 0.1 / max(now - self.last_time, 1e-6)
        self.last_time = now
        live = frame.copy()
        cv2.rectangle(live, (x0, y0), (x0 + side, y0 + side), (0, 255, 255), 2)
        live = cv2.resize(live, (CAM_W, int(h * CAM_W / w)))
        cv2.putText(
            live, f"{self.fps:.0f} FPS", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )
        self.canvas = draw_result(roi, self.smoothed, self.clf.classes, self.threshold())
        self.set_label(self.cam_label, live)
        self.set_label(self.result_label, self.canvas)
        self.root.after(INTERVAL_MS, self.loop)

    def save_result(self):
        if self.canvas is None:
            messagebox.showinfo("Thông báo", "Chưa có kết quả để lưu")
            return
        out = RESULTS / "screens"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"result_{time.strftime('%Y%m%d_%H%M%S')}.png"
        save_image(path, self.canvas)
        self.status.config(text=f"Đã lưu: {path}")

    def save_roi(self):
        if self.cap is None or self.last_roi is None:
            messagebox.showinfo("Thông báo", "Bật webcam trước khi lưu ROI")
            return
        label = self.label_var.get()
        if not label:
            messagebox.showinfo("Thông báo", "Chọn nhãn thật trước khi lưu")
            return
        out = DATA / "webcam" / label
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = out / f"cam_{stamp}_{int(time.time() * 1000) % 1000:03d}.png"
        gray = cv2.cvtColor(cv2.cvtColor(self.last_roi, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        save_image(path, gray)
        count = len(list(out.glob("*.png")))
        self.status.config(text=f"Đã lưu ROI vào {label} (tổng {count} ảnh)")

    def close(self):
        self.stop_camera()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()