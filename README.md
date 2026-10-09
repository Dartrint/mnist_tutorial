# MNIST Toolkit — nhận dạng chữ số viết tay

Dự án phân loại chữ số viết tay **MNIST** hoàn chỉnh, chạy được ngay, với hai backend:

| Backend | Công nghệ | Độ chính xác test | Ghi chú |
|---|---|---|---|
| `mnist.train` | **NumPy thuần** (MLP tự viết, không framework) | **97.47%** | Chỉ cần `numpy`, chạy trên mọi máy |
| `mnist.torch_cnn` | **PyTorch CNN** (tùy chọn) | **98.93%** | Cần cài `torch` (bản CPU) |

Toàn bộ pipeline được viết từ đầu: tải và giải mã file IDX, lan truyền xuôi/ngược,
các optimizer (SGD + momentum, Adam), hàm mất mát softmax cross-entropy, bộ đọc/ghi
ảnh PNG **không cần Pillow**, CLI huấn luyện – đánh giá – dự đoán, và một
**website vẽ tay để thao tác với model** (chỉ dùng thư viện chuẩn, không Flask/CDN).


## Kết quả tham chiếu (đã chạy thật, log trong `reports/`)

**MLP NumPy** — `--hidden 256,128 --optimizer sgd --lr 0.1 --momentum 0.9 --batch-size 512 --epochs 8`
(300 giây trên CPU, 235 146 tham số):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| train loss | 0.427 | 0.136 | 0.094 | 0.069 | 0.055 | 0.045 | 0.037 | 0.027 |
| val acc | 0.9470 | 0.9608 | 0.9668 | 0.9706 | **0.9720** | 0.9710 | 0.9720 | 0.9720 |

→ **test accuracy 0.9747** (loss 0.0810) trên 10 000 ảnh test, confusion matrix đầy đủ
trong `reports/mlp_eval.json`. Lớp dễ nhầm nhất là `4 ↔ 9` và `3 ↔ 5`, đúng như kỳ vọng
với một mô hình chỉ dùng lớp fully-connected.

Metrics thô: `reports/mlp_history.json`, `reports/mlp_eval.json`;
sprite sheet dự đoán: `reports/predictions.png`.

**CNN PyTorch** — `--epochs 3 --batch-size 128 --lr 1e-3` (420 giây trên CPU, ~0.5M tham số):

| Epoch | 1 | 2 | 3 |
|---|---|---|---|
| train loss | 0.770 | 0.110 | 0.070 |
| val acc | 0.9714 | 0.9824 | **0.9856** |

→ **test accuracy 0.9893** trên 10 000 ảnh test (chỉ 107 ảnh sai), confusion matrix
và precision/recall/F1 trong `reports/cnn_eval.json`. Nhờ tích chập giữ được cấu trúc
2D, số lỗi giảm từ 253 (MLP) xuống 107.

Metrics thô: `reports/cnn_history.json`, `reports/cnn_eval.json`;
sprite sheet dự đoán: `reports/cnn_predictions.png`.


## Điểm nổi bật

- **Không phụ thuộc nặng**: backend NumPy chỉ cần `numpy>=1.24`.
- **Tự viết mọi thứ**: `mnist/nn.py` chứa `Linear`, `ReLU`, `Dropout`, `softmax_cross_entropy`,
  `SGD`, `Adam` — kèm kiểm tra gradient bằng sai phân trung tâm trong test.
- **Tải dữ liệu tự động** từ nhiều mirror, kiểm tra MD5, cache lại trong `data/`.
- **CLI đầy đủ** cho huấn luyện, đánh giá (confusion matrix, precision/recall/F1) và dự đoán.
- **Dự đoán ảnh của bạn**: đọc PNG/PGM bằng codec tự viết, tự động đảo màu, xuất sprite sheet PNG.
- **Bộ test 98 case** chạy bằng `unittest` (không cần pytest), phủ cả gradient check và web API.

## Cài đặt

```bash
git clone https://github.com/Dartrint/mnist_tutorial.git
cd mnist_tutorial

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # chỉ cần numpy

# (tùy chọn) backend PyTorch bản CPU
pip install -r requirements-torch.txt
```

## Bắt đầu nhanh

```bash
make data        # tải MNIST (~12 MB) vào data/
make demo        # huấn luyện nhanh + đánh giá + vẽ sprite sheet
```

Có thể dùng CLI thống nhất `python -m mnist <lệnh>`:

```bash
python -m mnist data                                     # tải dữ liệu
python -m mnist train --epochs 8 --hidden 256,128
python -m mnist evaluate --model models/mlp.npz --show-confusion
python -m mnist predict  --model models/mlp.npz --sample 16 --ascii
python -m mnist cnn --epochs 3                           # cần torch
python -m mnist serve --port 8000                        # mở website
python -m mnist --help
```

Hoặc gọi trực tiếp từng module (tương đương):

```bash
python -m mnist.data                                     # tải dữ liệu
python -m mnist.train --epochs 10 --hidden 256,128 \
    --optimizer sgd --lr 0.1 --momentum 0.9 \
    --out models/mlp.npz --metrics-out reports/mlp_history.json

python -m mnist.evaluate --model models/mlp.npz --show-confusion
python -m mnist.predict  --model models/mlp.npz --sample 16 --save-grid reports/predictions.png
```

Muốn độ chính xác cao hơn (~99%):

```bash
pip install -r requirements-torch.txt
python -m mnist.torch_cnn --epochs 3 --out models/cnn.pt
python -m mnist.evaluate --model models/cnn.pt --show-confusion
```


## Website thao tác với model

```bash
make serve                      # hoặc: python -m mnist serve --port 8000
# mở http://127.0.0.1:8000
```

Server dùng `http.server` của thư viện chuẩn, giao diện HTML/CSS/JS thuần —
**không Flask, không CDN, không build step, không cần mạng**.

Tính năng:

- **Vẽ tay** trên canvas (hỗ trợ cả chuột và cảm ứng), tự dự đoán sau 400 ms ngừng vẽ.
- **Chọn model**: dropdown liệt kê mọi checkpoint trong `models/`, kèm loại (MLP/CNN)
  và độ chính xác test đọc từ `reports/`.
- **Xác suất 10 lớp** dạng thanh ngang, làm nổi bật lớp thắng, kèm confidence.
- **"Ảnh model nhìn thấy"**: ảnh 28×28 sau tiền xử lý (phóng to 10×) — thấy ngay vì sao
  model đoán đúng hay sai.
- **Ảnh test ngẫu nhiên**: lấy 8 ảnh từ test set, bấm vào để nạp vào canvas và dự đoán,
  có badge đỏ khi đoán sai so với nhãn thật.
- **Đánh giá model**: accuracy, per-class accuracy và confusion matrix 10×10 dạng heatmap.

### Tiền xử lý ảnh vẽ tay

Đây là phần quyết định để model dùng được với ảnh thật. Canvas được đưa qua
`preprocess_digit()`: tự phát hiện nền sáng/tối → đảo màu, cắt sát nét vẽ, scale cạnh lớn
nhất về 20 px (giữ tỉ lệ), dán vào khung 28×28 sao cho **khối tâm của nét vẽ trùng tâm** —
đúng quy trình tạo dữ liệu MNIST gốc.

Đo trên 300 ảnh test bị dời lệch + đảo màu (mô phỏng ảnh vẽ tay):

| Tiền xử lý | Accuracy |
|---|---|
| Chỉ resize | 10.0% |
| Cắt + scale 20×20 + căn khối tâm | **97.7%** |
| Ảnh MNIST gốc (giới hạn trên) | 98.7% |

### JSON API

| Endpoint | Mô tả |
|---|---|
| `GET /api/health` | trạng thái + danh sách model |
| `GET /api/models` | metadata checkpoint (loại, dung lượng, accuracy từ `reports/`) |
| `POST /api/predict` | `{model, image: dataURL, center}` → `{digit, confidence, probs, preview, elapsed_ms}` |
| `GET /api/samples?n=8&seed=0` | ảnh test ngẫu nhiên kèm nhãn thật |
| `POST /api/evaluate` | `{model, limit}` → accuracy, confusion matrix, per-class accuracy |

### Kiểm thử giao diện

`tests/test_webapp.py` khởi động **server thật** trên cổng ngẫu nhiên và gọi HTTP thật
(31 case: API, chặn path traversal, model không tồn tại, JSON hỏng, ảnh hỏng…).
Ngoài ra có script jsdom chạy đúng `app.js` trong DOM giả để kiểm tra luồng UI:

```bash
npm install jsdom                       # chỉ cần cho bước này
python -m mnist serve --port 8123 --quiet &
node scripts/frontend_smoke.mjs http://127.0.0.1:8123 my_digit.png 7
```

Kết quả đã chạy: **15/15 check PASS** — nạp danh sách model, vẽ → đoán đúng "7"
(confidence 99.98%), preview 28×28, biểu đồ 10 lớp, nút Xoá/Dự đoán, 8 ảnh test,
và panel đánh giá (4 số liệu + 10 thanh per-class + confusion matrix 121 ô).


## Cấu trúc dự án

```
.
├── mnist/
│   ├── __main__.py    # CLI thống nhất: python -m mnist <lệnh>
│   ├── data.py        # tải + giải mã file IDX, cache trong data/
│   ├── nn.py          # Linear, ReLU, Dropout, softmax-CE, SGD, Adam
│   ├── model.py       # MLP cấu hình được + lưu/đọc .npz
│   ├── train.py       # vòng lặp mini-batch, early stopping, CLI
│   ├── evaluate.py    # accuracy, confusion matrix, per-class report
│   ├── predict.py     # suy luận trên ảnh/test set, vẽ sprite sheet
│   ├── imageio.py     # đọc/ghi PNG, đọc PGM, resize (không cần Pillow)
│   ├── torch_cnn.py   # backend PyTorch CNN (tùy chọn)
│   ├── webapp.py      # web server (http.server) + JSON API
│   └── web/           # giao diện: index.html, app.js, style.css
├── examples/demo.py   # demo end-to-end dưới 1 phút
├── scripts/frontend_smoke.mjs  # test giao diện bằng jsdom (tùy chọn)
├── tests/             # 98 unittest: gradient check, data, imageio, model, web API
├── reports/           # metrics JSON + ảnh sinh ra khi chạy
├── Makefile           # make data | train | evaluate | predict | serve | test | demo
└── pyproject.toml     # metadata + cấu hình pytest/ruff
```

## Tham chiếu CLI

`python -m mnist --help` liệt kê tất cả lệnh: `data`, `train`, `cnn`, `evaluate`, `predict`.

### `python -m mnist.train` — huấn luyện MLP

| Tùy chọn | Mặc định | Mô tả |
|---|---|---|
| `--epochs` | `10` | Số epoch |
| `--batch-size` | `128` | Kích thước mini-batch |
| `--hidden` | `256` | Số neuron lớp ẩn, phân cách bằng dấu phẩy (`512,256`) |
| `--dropout` | `0.0` | Xác suất dropout sau mỗi lớp ẩn |
| `--lr` | `0.1` | Learning rate |
| `--optimizer` | `sgd` | `sgd` (có momentum) hoặc `adam` |
| `--momentum` | `0.9` | Momentum cho SGD |
| `--weight-decay` | `0.0` | L2 regularization |
| `--init` | `he` | Khởi tạo trọng số: `he` hoặc `xavier` |
| `--val-size` | `5000` | Số mẫu tách từ train làm validation |
| `--patience` | – | Early stopping sau N epoch không cải thiện |
| `--limit` | – | Chỉ dùng N ảnh đầu (chạy thử nhanh) |
| `--out` | `models/mlp.npz` | Nơi lưu model (`''` để không lưu) |
| `--metrics-out` | – | Ghi lịch sử huấn luyện ra JSON |

### `python -m mnist.evaluate` — đánh giá

`--model` (mặc định `models/mlp.npz`), `--split train|test`, `--limit`,
`--show-confusion`, `--json out.json`.
Hỗ trợ cả checkpoint `.npz` (MLP) lẫn `.pt` (CNN).

### `python -m mnist.predict` — dự đoán

`--model`, `--sample N` (lấy ngẫu nhiên N ảnh test), `--image file.png` (lặp lại được),
`--ascii` (in ra ASCII art), `--invert auto|yes|no`, `--save-grid out.png`.

### `python -m mnist.torch_cnn` — huấn luyện CNN

`--epochs 3 --batch-size 128 --lr 1e-3 --dropout 0.25 --threads N`, cùng `--out`/`--metrics-out`.

## Kiến trúc mô hình

**MLP (NumPy)** — `mnist/model.py`:

```
784 → Linear → ReLU → [Dropout] → Linear → ReLU → [Dropout] → ... → Linear(10) → softmax
```

Mặc định `--hidden 256,128` ⇒ 235 146 tham số. Huấn luyện bằng SGD + momentum
(lr 0.1) với He initialization; checkpoint lưu dạng `.npz` gồm cả cấu hình kiến trúc
nên `MLP.load()` dựng lại đúng model mà không cần truyền tham số.

**CNN (PyTorch)** — `mnist/torch_cnn.py`:

```
Conv(1→32,3×3) → ReLU → Conv(32→32,3×3) → ReLU → MaxPool2 → Dropout(0.25)
Conv(32→64,3×3) → ReLU → Conv(64→64,3×3) → ReLU → MaxPool2 → Dropout(0.25)
Flatten → Linear(3136→128) → ReLU → Dropout(0.5) → Linear(128→10)
```

Optimizer AdamW + OneCycleLR, chọn checkpoint theo validation accuracy.

## Dự đoán ảnh chữ số của bạn

```bash
python -m mnist.predict --model models/mlp.npz --image my_digit.png --ascii
```

Yêu cầu ảnh: PNG/PGM (thang xám hoặc RGB), nền sáng chữ đen **hoặc** nền đen chữ trắng —
chương trình tự phát hiện và đảo màu (`--invert auto`), sau đó resize về 28×28 và
chuẩn hóa về `[0, 1]` đúng như dữ liệu MNIST. Kết quả in ra kèm độ tin cậy (confidence)
và xác suất từng lớp.

## Kiểm thử

```bash
make test        # python -m unittest discover -s tests -t . -v
```

`make test` chạy 98 case (không cần dữ liệu MNIST thật — các test dùng dữ liệu tổng hợp,
riêng test dữ liệu thật sẽ tự skip nếu `data/` trống). Bao gồm:

- **Gradient check** bằng sai phân trung tâm (float64) cho `Linear`, `ReLU`, `Dropout`.
- Tính chất softmax (tổng bằng 1, bất biến với phép dịch) và loss so với công thức tay.
- Parser IDX với dữ liệu tổng hợp; kiểm tra shape/chuẩn hóa của `load_mnist`.
- Round-trip PNG (xám + RGB), đọc PGM, resize, `prepare_digit`.
- Optimizer hội tụ trên bài toán bình phương tối thiểu.
- Lưu/đọc model `.npz` và `.pt`.
- Dispatcher CLI (`python -m mnist`), parse tham số và các helper trong `utils`.
- **Web API**: mọi endpoint qua HTTP thật, gồm cả các trường hợp lỗi (400/404/405, path traversal).
- **Tiền xử lý ảnh**: đọc/ghi PNG, căn khối tâm, tự phát hiện đảo màu.

## Ghi chú kỹ thuật

- **Không dùng torchvision**: CNN đọc dữ liệu qua chính `mnist.data`, nên chỉ cần `torch`.
- **Không dùng Pillow**: `mnist/imageio.py` tự cài đặt codec PNG (zlib + struct, hỗ trợ
  filter 0–4, ảnh xám/RGB/RGBA 8-bit) và đọc PGM.
- **Mirror dữ liệu**: thử lần lượt `ossci-datasets.s3.amazonaws.com` và
  `storage.googleapis.com/cvdf-datasets`; xác thực MD5 trước khi dùng.
- **Tái lập kết quả**: mọi tham số ngẫu nhiên đều đi qua `--seed`.

## Giấy phép

MIT.

