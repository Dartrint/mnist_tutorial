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
- **Bộ test 118 case** chạy bằng `unittest` (không cần pytest), phủ cả gradient check, web API và kiểm tra tương thích Windows.

## Yêu cầu hệ thống

| Mục | Yêu cầu |
|---|---|
| Hệ điều hành | Windows 10/11, macOS 12+, hoặc Linux |
| Python | **3.10 trở lên** (bộ test trong repo chạy trên 3.11) |
| Dung lượng | ~15 MB (code + dữ liệu MNIST 12 MB); thêm ~250 MB nếu cài PyTorch CPU |
| Khác | Không cần quyền admin, không cần GPU, sau khi tải dữ liệu thì không cần mạng |
| Tùy chọn | `make` (chỉ POSIX) hoặc Node.js ≥ 18 (chỉ để chạy smoke test giao diện) |

Mọi thứ sinh ra (`data/`, `models/`, `reports/`) nằm trong thư mục dự án — không ghi
gì ra ngoài hệ thống.

### Lệnh Python theo hệ điều hành

| Hệ điều hành | Dùng | Ghi chú |
|---|---|---|
| Windows | **`py -3`** (khuyến nghị) hoặc `python` | `python3` **không tồn tại** trên Windows |
| macOS / Linux | `python3` | một số bản cũng có `python` |

Các ví dụ bên dưới ghi `python` cho gọn; trên Windows hãy thay bằng `py -3` nếu gõ
`python` bị mở Microsoft Store hoặc báo *not recognized*.

## Cài đặt

### Windows — PowerShell (khuyến nghị)

```powershell
git clone https://github.com/Dartrint/mnist_tutorial.git
cd $HOME\mnist_tutorial

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt          # chỉ cần numpy
```

Nếu PowerShell báo *running scripts is disabled on this system*:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Windows — cmd.exe

```bat
git clone https://github.com/Dartrint/mnist_tutorial.git
cd %USERPROFILE%\mnist_tutorial

py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
git clone https://github.com/Dartrint/mnist_tutorial.git
cd mnist_tutorial

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### (tùy chọn) PyTorch bản CPU cho CNN

```powershell
pip install -r requirements-torch.txt
```

Nếu mạng công ty chặn index của PyTorch:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Không cài cũng không sao — MLP, web UI và toàn bộ CLI chỉ cần `numpy`.

## Chạy nhanh

Ba cách **tương đương nhau**, chọn cách hợp với máy bạn:

| Việc | Mọi hệ điều hành (khuyến nghị) | Chỉ khi có `make` (macOS/Linux/Git Bash) |
|---|---|---|
| Tải dữ liệu | `python scripts/tasks.py data` | `make data` |
| Huấn luyện MLP | `python scripts/tasks.py train` | `make train` |
| Huấn luyện CNN | `python scripts/tasks.py cnn` | `make cnn` |
| Đánh giá model | `python scripts/tasks.py evaluate` | `make evaluate` |
| Dự đoán + sprite sheet | `python scripts/tasks.py predict` | `make predict` |
| Mở website | `python scripts/tasks.py serve` | `make serve` |
| Chạy test | `python scripts/tasks.py test` | `make test` |
| Demo dưới 1 phút | `python scripts/tasks.py demo` | `make demo` |

`scripts/tasks.py` chỉ dùng thư viện chuẩn và chạy **y hệt trên PowerShell, cmd, macOS
và Linux**; tham số thêm được truyền thẳng xuống CLI:

```powershell
python scripts/tasks.py                 # liệt kê tất cả task
python scripts/tasks.py train --epochs 8 --hidden 256,128
python scripts/tasks.py serve --port 8080
```

Hoặc gọi thẳng CLI (không cần `make` lẫn `tasks.py`):

```powershell
python -m mnist data
python -m mnist train --epochs 8 --hidden 256,128 --batch-size 512
python -m mnist evaluate --model models/mlp.npz --show-confusion
python -m mnist predict --model models/mlp.npz --sample 16 --ascii
python -m mnist serve --port 8000
python -m mnist --help
```

> ⚠️ **Phải chạy lệnh từ thư mục gốc dự án** (nơi có thư mục `mnist\`). Chạy ở thư mục
> khác sẽ báo `ModuleNotFoundError: No module named 'mnist'`.

> **Nối dòng:** dấu `\` chỉ đúng trong bash. PowerShell dùng backtick `` ` ``, cmd dùng
> `^`. An toàn nhất là viết lệnh trên một dòng như các ví dụ trên.

Kết quả nằm trong `models\` và `reports\` (Windows) hoặc `models/`, `reports/` — dùng
dấu `/` trong tham số cũng chạy tốt trên Windows: `--out models/mlp.npz`.

## Website thao tác với model

```powershell
python scripts/tasks.py serve            # hoặc: python -m mnist serve --port 8000
# rồi mở http://127.0.0.1:8000 trên trình duyệt
```

Windows: `Ctrl+C` trong cửa sổ terminal để dừng server. Mặc định server chỉ bind
`127.0.0.1` nên **không** bị Windows Defender Firewall hỏi. Nếu muốn máy khác trong LAN
truy cập, chạy `--host 0.0.0.0` và cho phép cổng đó ở firewall (xem mục Xử lý sự cố).

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

Đo trên 150 ảnh test đặt lệch trong khung 280×280 nền trắng (mô phỏng ảnh vẽ tay):

| Tiền xử lý | Accuracy |
|---|---|
| Resize thẳng về 28×28 | 67.3% |
| Cắt + scale 20×20 + căn khối tâm | **98.0%** |
| Ảnh MNIST gốc (giới hạn trên) | 98.7% |

Cả CLI (`python -m mnist predict`) và web UI đều dùng pipeline này;
cờ `--no-center` để tắt căn khối tâm khi ảnh đã đúng chuẩn MNIST.

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

```powershell
# Cửa sổ terminal 1 — chạy server, để nguyên cửa sổ này:
python -m mnist serve --port 8123 --quiet

# Cửa sổ terminal 2 — chạy kiểm tra:
npm install jsdom
node scripts/frontend_smoke.mjs http://127.0.0.1:8123 my_digit.png 7
```

`my_digit.png` là ảnh bạn tự vẽ (nền đen nét trắng hoặc ngược lại, kích thước bất kỳ)
và `7` là nhãn đúng để đối chiếu. Cách này chạy giống nhau trên Windows, macOS và Linux.

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
├── scripts/
│   ├── tasks.py       # task runner đa nền tảng (thay cho make trên Windows)
│   └── frontend_smoke.mjs      # test giao diện bằng jsdom (tùy chọn, cần Node.js)
├── tests/             # 118 unittest: gradient check, data, imageio, model, web API, Windows
├── reports/           # metrics JSON + ảnh sinh ra khi chạy
├── Makefile           # tiện ích cho macOS/Linux/Git Bash (không bắt buộc)
├── ci/                # template GitHub Actions (copy vào .github/workflows/ để bật)
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

```powershell
python -m mnist.predict --model models/mlp.npz --image my_digit.png --ascii
# Windows: đường dẫn có dấu cách thì nhớ quote -> --image "C:\anh\chu so.png"
```

Yêu cầu ảnh: PNG/PGM (thang xám hoặc RGB), nền sáng chữ đen **hoặc** nền đen chữ trắng —
chương trình tự phát hiện và đảo màu (`--invert auto`), rồi **cắt sát nét, scale về 20×20
và căn khối tâm vào khung 28×28** (đúng chuẩn MNIST) trước khi chuẩn hóa về `[0, 1]`.
Kết quả in ra kèm độ tin cậy (confidence) và xác suất từng lớp.

Thêm `--ascii` để xem ảnh dưới dạng ký tự ngay trong terminal, `--no-center` nếu ảnh của
bạn đã đúng chuẩn MNIST, `--invert yes|no` để ép chiều màu.

## Kiểm thử

```powershell
python scripts/tasks.py test            # tương đương lệnh dưới
python -m unittest discover -s tests -t . -v
```

Bộ test chạy 118 case (không cần dữ liệu MNIST thật — các test dùng dữ liệu tổng hợp,
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
- **Tương thích Windows**: CLI không crash trên code page cũ (cp437/cp1258/ascii),
  không có path `/tmp` cứng, không import module chỉ có trên POSIX.
- **Task runner**: mọi task trong `scripts/tasks.py` đều được kiểm tra (map lệnh, `clean` an toàn).

## Xử lý sự cố (ưu tiên Windows)

| Triệu chứng | Nguyên nhân | Cách sửa |
|---|---|---|
| `'python3' is not recognized` | Windows không có `python3` | dùng `py -3` hoặc `python` |
| `'python' is not recognized`, hoặc gõ `python` lại mở Microsoft Store | chưa cài Python, hoặc alias của Store đang bật | cài từ python.org (tick *Add python.exe to PATH*), dùng `py -3`; hoặc tắt *App execution aliases* → `python.exe` |
| `'source' is not recognized` (PowerShell/cmd) | `source` chỉ có trong bash | PowerShell: `.\.venv\Scripts\Activate.ps1` — cmd: `.venv\Scripts\activate.bat` |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | ExecutionPolicy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` rồi activate lại |
| `'make' is not recognized` | `make` không có sẵn trên Windows | dùng `python scripts/tasks.py <task>` (khuyến nghị), hoặc `winget install ezwinports.make`, hoặc mở Git Bash |
| `ModuleNotFoundError: No module named 'numpy'` | chưa activate venv, hoặc chưa `pip install` | activate venv rồi `pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'mnist'` | đang chạy lệnh ngoài thư mục gốc dự án | `cd` vào thư mục chứa `mnist\` rồi chạy lại |
| `pip` chậm/lỗi SSL khi tải dữ liệu (mạng công ty) | proxy chặn | thêm `--trusted-host pypi.org --trusted-host files.pythonhosted.org`; script tải MNIST đã tự thử lại không kiểm tra chứng chỉ |
| `[WinError 10048]` hoặc `Address already in use` | cổng đang bị chiếm | đổi cổng `--port 8080`, hoặc `netstat -ano \| findstr :8000` rồi `taskkill /PID <pid> /F` |
| Máy khác không vào được web UI | chỉ bind loopback / firewall | chạy `--host 0.0.0.0` và thêm Inbound rule TCP cho cổng đó trong Windows Defender Firewall |
| Tiếng Việt hiện thành `?` trong cmd/PowerShell | code page cũ (437/1258) | `set PYTHONUTF8=1` (cmd), `$env:PYTHONUTF8="1"` (PowerShell), hoặc dùng Windows Terminal |
| `UnicodeEncodeError` khi in ra console | code page không mã hoá được ký tự | chương trình đã tự thay ký tự nên không còn crash; đặt thêm `PYTHONUTF8=1` để hiện đúng tiếng Việt |
| Cài `torch` lâu, tốn dung lượng | wheel mặc định kéo theo CUDA | chỉ cài khi cần CNN, và cài bản CPU như hướng dẫn ở trên |

Kiểm tra nhanh môi trường:

```powershell
py -3 --version
py -3 -c "import sys, numpy; print(sys.executable, numpy.__version__)"
```

## Ghi chú kỹ thuật


- **Không dùng torchvision**: CNN đọc dữ liệu qua chính `mnist.data`, nên chỉ cần `torch`.
- **Không dùng Pillow**: `mnist/imageio.py` tự cài đặt codec PNG (zlib + struct, hỗ trợ
  filter 0–4, ảnh xám/RGB/RGBA 8-bit) và đọc PGM.
- **Mirror dữ liệu**: thử lần lượt `ossci-datasets.s3.amazonaws.com` và
  `storage.googleapis.com/cvdf-datasets`; xác thực MD5 trước khi dùng.
- **Tái lập kết quả**: mọi tham số ngẫu nhiên đều đi qua `--seed`.

## Giấy phép

MIT.

