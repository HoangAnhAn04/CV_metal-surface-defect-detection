# Phân loại lỗi bề mặt kim loại bằng thị giác máy và học sâu (NEU Surface Defect)

Bài tập lớn cuối kỳ môn **Thị giác máy trong công nghiệp và giao thông**.

- Sinh viên: [Họ và tên] - MSV: 223332813
- Ngôn ngữ và thư viện: Python 3.11, PyTorch, OpenCV, scikit-learn, Tkinter
- Dữ liệu: NEU Surface Defect Database (thép cán nóng, ảnh xám 200x200, 6 loại lỗi)

---

## 1. Dự án làm gì?

Chương trình nhận một ảnh bề mặt thép (từ file hoặc từ webcam) và cho biết ảnh đó thuộc **loại lỗi nào** trong 6 loại, kèm **độ tin cậy** (xác suất dự đoán). Nếu độ tin cậy thấp hơn ngưỡng thì chương trình gắn nhãn "UNCERTAIN" (không chắc chắn).

> **Lưu ý quan trọng:** đây là bài toán **phân loại ảnh (Image Classification)**. Mô hình gán một nhãn cho **toàn bộ ảnh**, **không** xác định vị trí vùng lỗi, nên không phải phát hiện đối tượng (Object Detection) hay phân đoạn (Segmentation).

6 loại lỗi (tên trong code / tiếng Việt):

| Tên trong code  | Tiếng Việt       |
| --------------- | ---------------- |
| crazing         | Nứt rạn          |
| inclusion       | Lẫn tạp chất     |
| patches         | Vết loang (mảng) |
| pitted_surface  | Rỗ bề mặt        |
| rolled-in_scale | Vảy cán          |
| scratches       | Trầy xước        |

Năm mô hình được huấn luyện và so sánh:

| Tên             | Kiến trúc                                                            | Augmentation |
| --------------- | -------------------------------------------------------------------- | ------------ |
| cnn_baseline    | CNN tự thiết kế (390 nghìn tham số)                                  | cơ bản       |
| cnn_noaug       | CNN tự thiết kế                                                      | không có     |
| cnn_strong      | CNN tự thiết kế                                                      | mạnh         |
| resnet18_ft     | ResNet18 pretrained ImageNet, fine-tune toàn bộ (11,2 triệu tham số) | cơ bản       |
| resnet18_strong | ResNet18 pretrained, fine-tune toàn bộ                               | mạnh         |

Mô hình được chọn cho ứng dụng: **resnet18_strong**.

---

## 2. Kết quả chính

Dữ liệu sau khi loại 1 ảnh trùng: 1799 ảnh, chia 1259 train / 270 validation / 270 test (theo từng lớp, seed = 223332813). Khoảng tin cậy 95% (CI95) tính bằng phương pháp Wilson.

**Tập test (ảnh NEU gốc):**

| Mô hình                       | Đúng/Tổng | Accuracy | CI95           |
| ----------------------------- | --------- | -------- | -------------- |
| CNN, augmentation cơ bản      | 267/270   | 98,89%   | 96,78 - 99,62% |
| CNN, không augmentation       | 265/270   | 98,15%   | 95,74 - 99,21% |
| CNN, augmentation mạnh        | 258/270   | 95,56%   | 92,39 - 97,44% |
| ResNet18, augmentation cơ bản | 270/270   | 100%     | 98,60 - 100%   |
| ResNet18, augmentation mạnh   | 270/270   | 100%     | 98,60 - 100%   |

Trên ảnh NEU sạch, các mô hình chênh nhau chỉ vài ảnh, nằm trong sai số thống kê nên chưa đủ để kết luận mô hình nào tốt hơn. Phần phân biệt được các mô hình là độ bền với ảnh xấu.

**Độ bền trên tập validation bị làm xấu (accuracy %):**

| Mô hình              | Gốc   | Mờ    | Nhiễu | Tối   | Sáng  | Tương phản thấp |
| -------------------- | ----- | ----- | ----- | ----- | ----- | --------------- |
| CNN, aug cơ bản      | 99,63 | 29,63 | 16,67 | 80,00 | 62,96 | 35,19           |
| CNN, không aug       | 100   | 29,26 | 16,67 | 73,33 | 57,78 | 29,63           |
| CNN, aug mạnh        | 96,30 | 72,22 | 92,59 | 75,93 | 69,63 | 48,89           |
| ResNet18, aug cơ bản | 100   | 64,07 | 57,78 | 96,67 | 79,63 | 94,44           |
| ResNet18, aug mạnh   | 100   | 98,52 | 98,52 | 100   | 96,30 | 98,52           |

Các phép làm xấu: làm mờ Gauss bán kính 2; nhiễu Gauss độ lệch chuẩn 25; độ sáng x0,4 (tối) và x1,8 (sáng); tương phản x0,4.

**Ảnh chụp bằng webcam (120 khung hình, từ 60 ảnh nguồn lấy từ tập test, hiển thị trên màn hình điện thoại):**

| Mô hình                | Đúng/Tổng   | Accuracy   | CI95               |
| ---------------------- | ----------- | ---------- | ------------------ |
| CNN, aug cơ bản        | 30/120      | 25,00%     | 18,11 - 33,44%     |
| CNN, aug mạnh          | 58/120      | 48,33%     | 39,58 - 57,19%     |
| ResNet18, aug cơ bản   | 53/120      | 44,17%     | 35,60 - 53,10%     |
| **ResNet18, aug mạnh** | **116/120** | **96,67%** | **91,74 - 98,70%** |

Điều kiện chụp: [điền: loại đèn môi trường, độ sáng màn hình điện thoại, cầm tay hay giá đỡ, khoảng cách tới webcam].

Lưu ý: 120 khung đến từ 60 ảnh nguồn (mỗi ảnh chụp 2 khung) nên không phải 120 phép thử độc lập. Mô hình cnn_noaug chưa đo trên webcam.

**Ngưỡng "không chắc chắn"** (resnet18_strong, đo trên bộ webcam):

| Ngưỡng                    | Ảnh giữ lại | Accuracy phần giữ lại | Ảnh sai còn lọt |
| ------------------------- | ----------- | --------------------- | --------------- |
| 0,50                      | 120         | 96,7%                 | 4               |
| 0,70                      | 116         | 98,3%                 | 2               |
| **0,90 (dùng trong app)** | 110         | 99,1%                 | 1               |
| 0,95                      | 100         | 100%                  | 0               |

Ngưỡng được chọn trên chính bộ webcam dùng để đo nên con số 99,1% hơi lạc quan.

**Tốc độ suy luận một ảnh** (máy Dell Precision 7530, Quadro P1000, chưa tính tiền xử lý):

| Mô hình  | Tham số    | Dung lượng | CPU             | GPU           |
| -------- | ---------- | ---------- | --------------- | ------------- |
| CNN      | 390 nghìn  | 1,6 MB     | khoảng 13 ms    | khoảng 2,0 ms |
| ResNet18 | 11,2 triệu | 44,8 MB    | khoảng 30-32 ms | khoảng 5,8 ms |

Ứng dụng webcam chạy khoảng 17-18 khung hình/giây. Con số này gồm cả đọc camera, tiền xử lý, vẽ kết quả và giao diện, không phải chỉ riêng mô hình.

---

## 3. Yêu cầu hệ thống

- Windows 10/11, PowerShell
- Python 3.11 (đã thử với 3.11.5)
- Khoảng 6 GB ổ đĩa trống (PyTorch bản GPU nặng khoảng 3 GB)
- Khuyến nghị có card NVIDIA. Đã chạy với Quadro P1000 (4 GB VRAM). Không có card vẫn chạy được trên CPU nhưng huấn luyện chậm hơn nhiều (ResNet18 khoảng 2 phút mỗi epoch thay vì 35-40 giây)
- Webcam nếu muốn dùng chế độ webcam

---

## 4. Cài đặt từng bước

Mở **PowerShell** (hoặc terminal trong VS Code) và làm lần lượt.

**Bước 4.1. Lấy mã nguồn và vào thư mục dự án**

```powershell
git clone <đường-dẫn-repo-của-bạn> Cuoi_Ky
cd Cuoi_Ky
```

Nếu nhận mã nguồn dạng file nén thì giải nén ra thư mục `Cuoi_Ky` rồi `cd` vào đó.

**Bước 4.2. Tạo môi trường ảo** (một "hộp" Python riêng cho dự án, không ảnh hưởng các dự án khác)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
```

Thành công khi đầu dòng lệnh có `(.venv)` và `python --version` in ra `Python 3.11.x`. Nếu báo lỗi về execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Mỗi lần mở terminal mới phải kích hoạt lại môi trường ảo bằng dòng `.venv\Scripts\Activate.ps1`.

**Bước 4.3. Cài PyTorch**

Có card NVIDIA (đã thử với CUDA 12.6):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Không có card NVIDIA (chạy CPU):

```powershell
pip install torch torchvision
```

**Bước 4.4. Cài các thư viện còn lại**

```powershell
pip install -r requirements.txt
```

**Bước 4.5. Kiểm tra PyTorch có dùng được GPU không**

```powershell
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

In ra `True` là dùng được GPU. In ra `False` thì chương trình vẫn chạy nhưng bằng CPU (xem mục 10 nếu bạn có card NVIDIA mà vẫn ra `False`).

---

## 5. Chuẩn bị dữ liệu

**Bước 5.1. Tải dataset** từ Kaggle (cần tài khoản Kaggle):
https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

Giải nén sao cho có đúng cấu trúc sau (tên thư mục `NEU-DET`):

```
data/
└── NEU-DET/
    ├── train/
    │   ├── annotations/
    │   └── images/
    │       ├── crazing/
    │       ├── inclusion/
    │       ├── patches/
    │       ├── pitted_surface/
    │       ├── rolled-in_scale/
    │       └── scratches/
    └── validation/
        ├── annotations/
        └── images/ (cùng 6 thư mục lớp)
```

Bộ này có 1800 ảnh (1440 train + 360 validation, mỗi lớp 300 ảnh), kích thước 200x200, kèm nhãn vị trí dạng XML. Dự án chỉ dùng nhãn lớp (tên thư mục), không dùng nhãn vị trí.

**Bước 5.2. Kiểm tra dữ liệu** (chỉ đọc, không sửa dữ liệu; kết quả lưu ở `results\dataset_report.txt`)

```powershell
python src\inspect_data.py
python src\check_duplicates.py
```

Kết quả mong đợi: 1800 ảnh, 6 lớp x 300, tất cả đọc được, và đúng 1 cặp ảnh trùng nhau là `patches_101` và `patches_105`.

**Bước 5.3. Chia dữ liệu**

```powershell
python src\split_data.py
```

Chương trình gộp train và validation gốc, loại ảnh trùng y hệt, rồi chia lại theo từng lớp thành train / val / test (210 / 45 / 45 ảnh mỗi lớp, riêng patches có 209 train vì thiếu 1 ảnh trùng). Seed cố định là MSV nên kết quả chia luôn giống nhau. Ảnh được chép vào `data\processed\train|val|test\<tên lớp>`. File `results\split_manifest.csv` ghi lại ảnh nào nằm ở tập nào.

**Bước 5.4. (Tùy chọn) Kiểm tra rò rỉ dữ liệu giữa các tập**

```powershell
python src\check_leakage.py
```

So sánh độ giống nhau giữa ảnh val/test với ảnh train. Kết quả: độ giống không cao hơn mức giống nhau trong nội bộ tập train, nên không có rò rỉ bất thường.

---

## 6. Huấn luyện

Mỗi lệnh huấn luyện một mô hình, lưu trọng số tốt nhất (theo accuracy trên tập val) vào `models\<tên>.pt`, và lưu lịch sử, đường cong loss/accuracy vào `results\`. Tập test **không** được dùng ở bước này.

```powershell
python src\train.py --arch cnn --name cnn_baseline
python src\train.py --arch cnn --name cnn_noaug --aug none
python src\train.py --arch cnn --name cnn_strong --aug strong
python src\train.py --arch resnet18 --pretrained --name resnet18_ft --lr 3e-4 --epochs 25
python src\train.py --arch resnet18 --pretrained --name resnet18_strong --lr 3e-4 --epochs 25 --aug strong
```

Thời gian trên Quadro P1000: CNN khoảng 20 phút, ResNet18 khoảng 16 phút. Lần đầu chạy ResNet18 với `--pretrained`, PyTorch tải trọng số ImageNet (khoảng 45 MB) nên cần có mạng. Trước khi chạy nên đóng các ứng dụng nặng (trình duyệt, Zalo, Discord...) để dành bộ nhớ card.

Các tham số quan trọng (xem đầy đủ bằng `python src\train.py --help`):

| Tham số        | Mặc định   | Ý nghĩa                                                    |
| -------------- | ---------- | ---------------------------------------------------------- |
| `--arch`       | cnn        | `cnn`, `resnet18` hoặc `mobilenet_v3_small`                |
| `--name`       | (bắt buộc) | tên mô hình, cũng là tên file `.pt`                        |
| `--epochs`     | 40         | số lượt học qua toàn bộ tập train                          |
| `--batch`      | 32         | số ảnh mỗi lần cập nhật. Hết bộ nhớ card thì giảm xuống 16 |
| `--lr`         | 1e-3       | learning rate (bước cập nhật trọng số)                     |
| `--aug`        | basic      | `none`, `basic` hoặc `strong`                              |
| `--pretrained` | tắt        | dùng trọng số ImageNet                                     |
| `--freeze`     | tắt        | khóa backbone, chỉ học lớp cuối                            |
| `--patience`   | 10         | dừng sớm nếu val không cải thiện sau chừng này epoch       |

Cố định: bộ tối ưu AdamW (weight decay 1e-4), learning rate giảm dần theo hàm cosine, dropout 0,3 (CNN), seed = 223332813.

---

## 7. Đánh giá

**Tập test, ảnh gốc** (số liệu chính thức; lưu ở `results\`):

```powershell
python src\evaluate.py --models cnn_baseline cnn_noaug cnn_strong resnet18_ft resnet18_strong
```

Với mỗi mô hình sinh ra báo cáo từng lớp (Precision, Recall, F1), ma trận nhầm lẫn (confusion matrix), khoảng tin cậy 95% và danh sách ảnh bị đoán sai.

**Độ bền với ảnh bị làm xấu** (đo trên tập val; lưu ở `results\robustness\`):

```powershell
foreach ($p in "none","blur","noise","dark","bright","lowcontrast") { python src\evaluate.py --models cnn_baseline cnn_noaug cnn_strong resnet18_ft resnet18_strong --split val --perturb $p | Where-Object { $_ -notmatch '^\s+\[' } }
```

**Ảnh webcam.** Bộ ảnh này không có sẵn trong mã nguồn, cần tự chụp:

1. Chép 10 ảnh đầu mỗi lớp của tập test sang thư mục riêng, gửi sang điện thoại:

```powershell
foreach ($c in "crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches") { New-Item -ItemType Directory -Force "data\webcam_src\$c" | Out-Null; Get-ChildItem "data\processed\test\$c" | Select-Object -First 10 | Copy-Item -Destination "data\webcam_src\$c" }
```

2. Mở `python src\app.py`, bật webcam. Với mỗi ảnh: chọn **nhãn thật** ở ô "Nhãn thật", mở ảnh trên điện thoại, đưa ảnh vào khung vàng rồi bấm "Lưu ROI webcam" (hoặc phím F2). Chụp mỗi ảnh 2 khung với vị trí hơi khác nhau. Ảnh lưu vào `data\processed\webcam\<tên lớp>\`.
3. Đánh giá (phải có đủ 6 thư mục lớp):

```powershell
python src\evaluate.py --models cnn_baseline cnn_strong resnet18_ft resnet18_strong --split webcam
```

Ảnh webcam chỉ dùng để đánh giá, không dùng để huấn luyện hay chia vào tập test.

**Phân tích ngưỡng, tốc độ và tổng hợp bảng biểu:**

```powershell
python src\threshold_analysis.py
python src\benchmark.py
python src\summarize.py
```

`summarize.py` đọc các file kết quả có sẵn và xuất 3 bảng CSV cùng 2 biểu đồ vào `results\summary\` (dùng cho báo cáo).

---

## 8. Chạy ứng dụng

**Giao diện đồ họa:**

```powershell
python src\app.py
```

| Nút / ô                    | Chức năng                                                                                |
| -------------------------- | ---------------------------------------------------------------------------------------- |
| Chọn ảnh                   | Mở một file ảnh và nhận diện                                                             |
| Chọn thư mục               | Mở cả thư mục ảnh, dùng nút Trước / Sau để duyệt                                         |
| Bật webcam / Tắt webcam    | Nhận diện trực tiếp. Đưa ảnh vào **khung vàng**, mô hình chỉ xem phần ảnh trong khung đó |
| Lưu kết quả                | Lưu ảnh kết quả đang hiển thị vào `results\screens\`                                     |
| Mô hình                    | Chọn mô hình (mặc định resnet18_strong)                                                  |
| Ngưỡng                     | Kéo để đổi ngưỡng "không chắc chắn" (mặc định 0,90)                                      |
| Nhãn thật + Lưu ROI webcam | Lưu khung hình webcam vào `data\processed\webcam\<nhãn>` để xây bộ đánh giá              |

Cách đọc kết quả: dòng trên cùng là **tên lớp và độ tin cậy**. Viền và chữ màu xanh (CONFIDENT) khi độ tin cậy từ ngưỡng trở lên, màu cam (UNCERTAIN) khi thấp hơn. Bảng bên phải là xác suất của cả 6 lớp. Ở chế độ webcam, kết quả được làm mượt qua nhiều khung hình liên tiếp để không nhảy loạn.

**Dòng lệnh (chạy trên file hoặc thư mục, lưu ảnh kết quả):**

```powershell
python src\infer.py --source data\processed\test
python src\infer.py --source data\processed\test\scratches --show
python src\infer.py --source duong_dan_den_anh.jpg --model resnet18_strong --threshold 0.9
```

- `--show`: hiện từng ảnh trong cửa sổ (bấm phím bất kỳ để sang ảnh sau, bấm `q` để thoát)
- Kết quả lưu ở `results\predictions\` (ảnh vẽ sẵn và file `predictions.csv`)
- Nếu ảnh nằm trong thư mục có tên đúng bằng một lớp thì chương trình coi tên thư mục là nhãn thật và in độ chính xác

Chạy `infer.py` trên toàn bộ `data\processed\test` với `resnet18_strong` phải cho **270/270**.

---

## 9. Cấu trúc thư mục và chức năng từng file

```
Cuoi_Ky/
├── data/
│   ├── NEU-DET/           dữ liệu gốc từ Kaggle, không sửa
│   ├── processed/         dữ liệu đã chia: train, val, test (và webcam nếu có)
│   └── webcam_src/        60 ảnh nguồn dùng để chụp bộ ảnh webcam
├── models/                trọng số đã huấn luyện (*.pt)
├── results/               báo cáo, biểu đồ, kết quả đánh giá
│   ├── robustness/        kết quả đo độ bền (ảnh xấu, webcam)
│   ├── summary/           bảng và biểu đồ tổng hợp cho báo cáo
│   ├── predictions/       ảnh kết quả của infer.py
│   └── screens/           ảnh kết quả lưu từ app
├── src/                   toàn bộ mã nguồn
├── requirements.txt
└── README.md
```

| File trong `src\`       | Chức năng                                                                                                       |
| ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| `inspect_data.py`       | Thống kê dataset gốc: số ảnh mỗi lớp, kích thước, số file XML, kích thước nhãn vị trí                           |
| `check_duplicates.py`   | Tìm ảnh trùng y hệt (MD5) và gần giống (pHash) trong dataset gốc                                                |
| `split_data.py`         | Loại ảnh trùng, chia train/val/test theo từng lớp với seed cố định                                              |
| `common.py`             | Phần dùng chung: hằng số, đọc dữ liệu (Dataset), biến đổi ảnh và augmentation, định nghĩa mô hình, hàm đánh giá |
| `train.py`              | Huấn luyện một mô hình, chọn bản tốt nhất theo val, dừng sớm, lưu trọng số và đường cong                        |
| `evaluate.py`           | Đánh giá mô hình trên val/test/webcam, có thể làm xấu ảnh (mờ, nhiễu, tối, sáng, tương phản thấp)               |
| `check_leakage.py`      | Kiểm tra ảnh val/test có giống ảnh train bất thường không                                                       |
| `infer.py`              | Nạp mô hình, dự đoán, vẽ kết quả. Class `Classifier` được dùng lại bởi app                                      |
| `app.py`                | Giao diện Tkinter: ảnh, thư mục, webcam                                                                         |
| `threshold_analysis.py` | Phân tích ngưỡng "không chắc chắn" trên bộ ảnh webcam                                                           |
| `benchmark.py`          | Đo thời gian suy luận và kích thước mô hình                                                                     |
| `summarize.py`          | Tổng hợp kết quả thành bảng CSV và biểu đồ                                                                      |

---

## 10. Các quyết định thiết kế chính và lý do

| Quyết định                                          | Lý do                                                                                                                                                |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Làm phân loại, không làm phát hiện                  | Nhãn vị trí của dataset kém ý nghĩa với một số lớp (lỗi rỗ bề mặt thường chiếm gần cả ảnh), và bài toán phân loại phù hợp với thời gian và phần cứng |
| Gộp train + validation gốc rồi chia lại 70/15/15    | Dataset gốc không có tập test. Chia theo từng lớp để mỗi tập đủ cả 6 lớp                                                                             |
| Seed = MSV                                          | Chia dữ liệu và huấn luyện lặp lại được, kết quả giống nhau mỗi lần chạy                                                                             |
| Loại ảnh trùng y hệt trước khi chia                 | Tránh cùng một ảnh nằm ở cả train và test                                                                                                            |
| Chọn mô hình theo tập val, test chỉ dùng để báo cáo | Để số liệu test không bị lạc quan do tự chọn theo nó                                                                                                 |
| Augmentation chỉ áp dụng cho tập train              | Tập val/test phải là ảnh nguyên bản mới đánh giá đúng                                                                                                |
| Không đổi màu (hue) khi augmentation                | Ảnh gốc là ảnh xám. Chỉ đổi sáng và tương phản                                                                                                       |
| Có hai mức augmentation (cơ bản, mạnh)              | Để đo augmentation có thật sự giúp mô hình bền hơn với ảnh khác miền dữ liệu (webcam) không                                                          |
| Chọn ResNet18 pretrained làm mô hình chính          | Bền hơn nhiều trên ảnh xấu và webcam so với CNN tự xây, đổi lại nặng hơn và chậm hơn (vẫn đủ nhanh để chạy thời gian thực)                           |
| Có ngưỡng "không chắc chắn"                         | Khi độ tin cậy thấp, người dùng nên kiểm tra lại thay vì tin ngay kết quả                                                                            |
| Đánh giá riêng trên ảnh webcam                      | NEU là ảnh sạch từ thiết bị công nghiệp, đo trên đó chưa phản ánh việc dùng ngoài thực tế                                                            |

---

## 11. Hạn chế

- Chỉ **phân loại cả ảnh**, không định vị vùng lỗi, và mỗi ảnh chỉ được gán một nhãn dù thực tế một ảnh có thể chứa nhiều loại lỗi.
- Chỉ có 6 lớp lỗi, **không có lớp "không có lỗi" hay "không phải bề mặt thép"**. Nếu đưa vào ảnh bất kỳ (ví dụ khuôn mặt), mô hình vẫn buộc phải chọn một trong 6 lớp và có thể báo độ tin cậy rất cao. Ngưỡng không chặn được trường hợp này.
- Mô hình sai mà vẫn rất tự tin là có xảy ra, nên ngưỡng chỉ giảm chứ không loại bỏ hoàn toàn các lỗi.
- Bộ ảnh webcam nhỏ (120 khung từ 60 ảnh nguồn, một điều kiện chụp), ngưỡng 0,90 chọn trên chính bộ này, nên số liệu webcam chỉ có tính tham khảo.
- Dataset nhỏ (1800 ảnh), chụp từ một nguồn, nên chưa đại diện cho các dây chuyền khác.
- Đo độ bền trên ảnh bị làm xấu bằng thuật toán, không thay thế cho ảnh thật qua camera. Mô hình augmentation mạnh được huấn luyện với đúng loại biến đổi như mờ, nhiễu nên các con số đó có phần thuận lợi.
- Cặp lớp hay bị nhầm: rỗ bề mặt và lẫn tạp chất, trầy xước và lẫn tạp chất.

---

## 12. Xử lý lỗi thường gặp

| Hiện tượng                                                        | Nguyên nhân và cách xử lý                                                                                                                                                                                                       |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dòng đầu khi huấn luyện ghi `Thiet bi: CPU` dù máy có card NVIDIA | Chưa kích hoạt môi trường ảo (đầu dòng không có `(.venv)`), đang dùng Python khác. Chạy `.venv\Scripts\Activate.ps1` rồi chạy lại. Nếu đã kích hoạt mà vẫn vậy thì kiểm tra `nvidia-smi` và cài lại PyTorch bản CUDA ở bước 4.3 |
| `ModuleNotFoundError: No module named ...`                        | Chưa cài thư viện. Kích hoạt môi trường ảo rồi `pip install -r requirements.txt`                                                                                                                                                |
| `CUDA out of memory`                                              | Hết bộ nhớ card. Đóng bớt ứng dụng khác hoặc thêm `--batch 16`                                                                                                                                                                  |
| Lỗi khi tải dữ liệu (DataLoader, workers) trên Windows            | Thêm `--workers 0` vào lệnh `train.py`                                                                                                                                                                                          |
| Báo lỗi `running scripts is disabled` khi kích hoạt môi trường ảo | Chạy `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`                                                                                                                                                               |
| Không mở được webcam                                              | Đóng các ứng dụng đang dùng camera (Zoom, Zalo, Teams...), kiểm tra quyền camera của Windows                                                                                                                                    |
| `Khong tim thay anh trong: ...`                                   | Đường dẫn `--source` sai hoặc thư mục không có ảnh                                                                                                                                                                              |
| `evaluate.py --split webcam` báo lỗi                              | Chưa chụp bộ ảnh webcam (xem mục 7) hoặc thiếu thư mục của một lớp                                                                                                                                                              |
| Không tìm thấy file `.pt`                                         | Chưa huấn luyện (mục 6) hoặc chưa chép thư mục `models` từ bản nộp bài                                                                                                                                                          |

---

## 13. Tài liệu tham khảo

- Dataset: NEU Surface Defect Database, bản trên Kaggle: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
- K. Song, Y. Yan, _A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects_, Applied Surface Science, 2013.
- K. He, X. Zhang, S. Ren, J. Sun, _Deep Residual Learning for Image Recognition_, CVPR, 2016 (ResNet)
- PyTorch: https://pytorch.org, OpenCV: https://opencv.org
