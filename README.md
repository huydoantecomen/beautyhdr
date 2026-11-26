# Beauty HDR 0.14.0: phục hồi, làm nét, phóng to ảnh trên PC

Phần mềm desktop chạy hoàn toàn trên máy: có GPU thì dùng GPU, không có thì dùng CPU. Ảnh không gửi đi đâu, ảnh gốc không bao giờ bị ghi đè.

## Tính năng (theo đặc tả, phiên bản PC)

| Phần | Có trong 0.8 |
| --- | --- |
| Nhập ảnh | JPG, PNG, WebP, BMP, TIFF; kéo thả; xoay đúng theo EXIF; báo lỗi rõ khi ảnh hỏng, sai định dạng, quá 80 MB |
| Phân tích ảnh | độ mờ, mức nhiễu, sáng/tối, tương phản, độ bão hoà, số mặt và mặt nào xử lý được; gợi ý chế độ + preset |
| Chế độ | Cải thiện nhanh, Chân dung AI, Ảnh cũ, Phóng to, Tự chỉnh |
| Mức nét | thanh 1–6 từ nhẹ đến mạnh: 1 không AI (nhanh nhất), 2 AI nhẹ 5 MB, 3–5 thêm model mặt và AI mạnh dần, 5 là Rõ nét tối đa, 6 Cao nhất là mức 5 với model mới nhất, nặng nhất (RestoreFormer++, DAT-2). Tự động chọn mức 1–5 theo độ mờ đo được: ảnh mờ ít dùng model nhẹ, mờ nhiều dùng model nặng. Chọn tay được Model mặt (GFPGAN, RestoreFormer++, RestoreFormer) và Model nền (compact, RRDB, RealPLKSR, DAT-2) |
| Sửa phơi sáng | tự sửa ảnh quá tối, quá sáng, ngược sáng (thanh Tự sửa sáng). Giữ màu, không làm cháy vùng sáng; quá sáng thì hạ sáng theo vùng; ngược sáng thì nâng sáng người, giữ trời. Không đụng ảnh vốn tối hoặc sáng như ảnh đêm, logo nền trắng, trang chữ |
| Tông tươi sáng | bước cuối (thanh Tươi sáng, mặc định 0,6): nâng tông giữa, làm trong vùng tối, tăng độ rực nhẹ, da trong vùng mặt sáng hồng hơn. Màu rực sáng đi qua vai mềm nên không cháy thêm. Logo, hình đồ hoạ, ảnh đêm, ảnh vũ trụ giữ gần như nguyên |
| Ảnh nhỏ | Phóng to "Tự động" (mặc định): cạnh dài dưới 1000 px phóng 4×, còn lại 2×. Ảnh bị phóng to nên mờ: đo mức mờ, thu nhỏ lại rồi để AI vẽ lại chi tiết |
| Cảnh báo | ảnh mờ rất nặng, ảnh rất nhỏ (dưới 400 px): báo trước AI chỉ làm rõ được một phần, nét mặt có thể không giống thật hoàn toàn |
| Preset | Tự động, Tự nhiên, Rõ nét, Chân dung, Rõ nét tối đa (kiểu Remini), Ảnh cũ, Ảnh tối, Tuỳ chỉnh |
| Sửa màu theo hãng | đọc EXIF Make/Model để chọn bộ lọc: iPhone, Samsung, Xiaomi, Oppo/realme/OnePlus, Vivo, Huawei/Honor, Pixel, máy ảnh, Khác, hoặc Tắt (chọn tay được). Cân bằng trắng tự động gỡ ám vàng/đỏ/cam (vùng mặt không tính vào ước lượng vì da vốn ấm), sửa da đỏ/vàng trong vùng mặt, giảm bão hoà và độ nét cho máy vốn rực màu, nét mạnh. Ảnh qua Zalo/Facebook mất EXIF thì dùng bộ Khác |
| Tăng màu, độ bóng da | giãn mức sáng, rõ khối, độ rực (da tăng ít), da mịn nhưng giữ vân da, ánh bóng nhẹ |
| Điều chỉnh | Phóng to (Tự động, 1×, 2×, 4×), Độ nét, Khử nhiễu (tự động hoặc 0–10), Tương phản (CLAHE), Độ sáng, Tự sửa sáng, Tươi sáng, Retinex cho ảnh tối, Độ trung thực AI, Cường độ chung, Đặt lại |
| Xem trước | thanh so sánh trước/sau cùng tỉ lệ, zoom bằng con lăn, kéo ảnh, 100%/200%/vừa khung |
| Thao tác | Huỷ giữa chừng (Esc), Hoàn tác (Ctrl+Z), Khôi phục ảnh gốc |
| Xuất ảnh | JPG (mặc định chất lượng 94), PNG, WebP; tên file tự tránh trùng |
| Lịch sử | thumbnail, ngày giờ, chế độ, kích thước; mở lại, mở thư mục, xoá bản ghi, xoá cả file ra |
| Model | manifest có phiên bản, dung lượng, SHA-256, giấy phép; tải khi cần, kiểm checksum, thử lại 3 lần; màn Quản lý model |
| Dự phòng | thiếu model / mất mạng / lỗi AI → tự chuyển sang Cải thiện nhanh và ghi lý do; GPU thiếu bộ nhớ → tự chia ô nhỏ hơn; model nền mới (DAT-2, RealPLKSR) lỗi trên thiết bị → dùng Real-ESRGAN compact |
| Cấu hình máy | đọc CPU, RAM, GPU, bộ nhớ GPU (cả GPU NVIDIA khi bản cài chỉ chạy CPU, qua nvidia-smi); ước tính thời gian của model đang chọn cho ảnh đang mở; cảnh báo khi model nặng chạy trên CPU, GPU ít bộ nhớ, RAM ít, hoặc có GPU NVIDIA mà bản cài chỉ chạy CPU. Chỉ cảnh báo, không chặn. Quản lý model có cột ước tính thời gian từng model trên máy đang dùng |

## Cài đặt trên Windows

**Một file cài đặt (dễ nhất):** bấm đúp `dong_goi_exe.bat`. Script tự cài Python 3.12 và Inno Setup nếu máy chưa có, đóng gói, tự kiểm rồi để lại **một** file `NetAnh-<phiên bản>-setup.exe` ngay trong thư mục này (mọi file tạm nằm trong `_build\`). Bấm đúp file đó để cài, mở Beauty HDR từ menu Start. Lần đầu khoảng 20–40 phút, cần mạng. Bản gửi người dùng (`tools/dong_goi_nguoi_dung.py`) chỉ để ngoài cùng `TAO_FILE_CAI_DAT.bat`, `HUONG_DAN.txt` và thư mục `ma-nguon`.

**Chạy thẳng từ mã nguồn** (để sửa, thử thuật toán):

1. Cài **Python 3.11 hoặc 3.12** từ python.org, tick **Add python.exe to PATH**.
2. Máy có GPU NVIDIA: cập nhật driver NVIDIA mới nhất.
3. Bấm đúp `cai_dat.bat` (5–15 phút, khoảng 3–5 GB).
4. Bấm đúp `chay.bat` để mở phần mềm.

Chế độ Cải thiện nhanh chạy ngay không cần tải gì. Các chế độ AI tải model ở lần đầu (khoảng 610 MB, từ GitHub) vào `%USERPROFILE%\.net_anh\weights`; hoặc vào **Quản lý model → Tải tất cả** trước.

Nếu cài PyTorch CUDA bị lỗi: vào https://pytorch.org/get-started/locally/, chọn Windows + Pip + bản CUDA, chạy lệnh đó sau khi `.venv\Scripts\activate`.
GPU AMD / Intel trên Windows (thử nghiệm): `.venv\Scripts\pip install torch-directml`.

## Linux / macOS

```bash
./cai_dat.sh
. .venv/bin/activate
python -m net_anh
```

## Dòng lệnh

```bash
python -m net_anh.cli anh.jpg                               # tự phân tích và chọn chế độ
python -m net_anh.cli anh.jpg --mode portrait --fidelity 0.6
python -m net_anh.cli thu_muc -o ket_qua --mode upscale --upscale 4
python -m net_anh.cli anh.jpg --preset low_light --format webp
python -m net_anh.cli anh.jpg --phone oppo --wb 0.8 --skin-tone 1   # chọn tay bộ lọc hãng và mức sửa màu
python -m net_anh.cli anh.jpg --level 2                     # mức nét 1–6 (2 = AI nhẹ, nhanh)
python -m net_anh.cli anh.jpg --level 6                     # Cao nhất: RestoreFormer++ + DAT-2 (nên có GPU)
python -m net_anh.cli anh.jpg --face-model restoreformer_pp --bg-model plksr   # chọn tay model
python -m net_anh.cli anh.jpg --exposure 0                  # tắt tự sửa sáng
```

## Kiểm thử và đo chất lượng

```bash
pip install pytest
python -m pytest tests            # 58 test: đọc/ghi (cả đường dẫn có dấu tiếng Việt), EXIF, không ghi đè, cổ điển, router, mức nét 1–6, sửa sáng, màu, lịch sử, checksum, huỷ, dự phòng, model mới, ước tính thời gian và cảnh báo cấu hình máy
python tools/evaluate.py thu_muc_anh_net --csv ket_qua.csv
python tools/evaluate_general.py --csv ket_qua_chung.csv   # ảnh không mặt, ảnh mẫu có sẵn
python tools/export_onnx.py --dn 0.5 -o realesr_compact_dn50.onnx   # model cho Android
python tools/wb_bench.py              # đo cân bằng trắng trên ảnh phủ ám giả lập
python tools/phone_cast_demo.py       # lưới ảnh trước/sau cho ám màu kiểu điện thoại
python tools/stress_test.py           # 30 ca qua dòng lệnh thật: lỗi, thời gian, bộ nhớ, lệch màu
python tools/level_bench.py           # chất lượng và thời gian từng model theo mức mờ
python tools/exposure_bench.py        # ảnh quá tối, quá sáng, ngược sáng, rất mờ (giả lập) và ảnh tối/sáng tự nhiên
python tools/make_exposure_fixtures.py   # ảnh mẫu để kiểm bản Android sửa sáng giống bản PC
python tools/tone_bench.py --grid luoi_tong.jpg   # độ sáng, độ tươi màu, mức đen, điểm cháy ở các mức Tươi sáng
python tools/make_tone_fixtures.py   # ảnh mẫu để kiểm bản Android tông tươi sáng giống bản PC
python tools/small_bench.py          # ảnh nhỏ (thu nhỏ 2–4 lần) và ảnh bị phóng to nên mờ: phóng 2× so với 4×
python tools/model_bench.py --mat --musiq musiq.pth --csv mat.csv   # model mặt: IDS, SSIM, GMSD, MUSIQ, thời gian
python tools/model_bench.py --nen --model dat2=4xRealWebPhoto_v4_dat2.safetensors --csv nen.csv   # model cả ảnh
python tools/speed_bench.py          # tốc độ từng model trên máy đang chạy (số trong net_anh/hardware.py)
```

`tools/evaluate.py` lấy ảnh gốc nét, làm hỏng giả lập (blur, thu nhỏ 2×, nhiễu, JPEG) ở 2 mức, chạy từng chế độ rồi so với ảnh gốc: **IDS** (ArcFace cosine, giữ đúng người), **PSNR** trên mặt và toàn ảnh, **độ nét so với gốc**, thời gian.

## Đóng gói .exe

Xem **HUONG_DAN_BUILD.md**. Tóm tắt: `dong_goi_exe.bat` (hoặc `packaging\build_windows.ps1 -Flavor cpu|cuda`) cho ra một file `NetAnh-<phiên bản>-setup.exe` cạnh file .bat (bản GPU NVIDIA trên 2 GB: thư mục `NetAnh-<phiên bản>-GPU-NVIDIA` gồm setup.exe và các file .bin), kèm sẵn model và tự kiểm trước khi xuất. File trung gian ở `_build\` (`_build\dist\NetAnh\NetAnh.exe` chạy được không cần cài). Không có máy Windows thì dùng GitHub Actions (`.github/workflows/build-windows.yml`).

## Model và giấy phép

| Model | Dùng cho | Dung lượng | Giấy phép |
| --- | --- | --- | --- |
| GFPGAN v1.4 | phục hồi khuôn mặt | 349 MB | Apache-2.0 |
| RestoreFormer (bản phát hành GFPGAN 1.3.4) | model mặt chọn tay | 291 MB | Apache-2.0 |
| RestoreFormer++ (wzhouxiff/RestoreFormerPlusPlus 1.0.0) | model mặt chọn tay, mức 6 | 294 MB | Apache-2.0 |
| Real-ESRGAN compact (realesr-general-x4v3 + bản wdn) | làm nét, phóng to toàn ảnh (mặc định từ 0.3) | 4,9 MB mỗi file | BSD-3-Clause |
| Real-ESRGAN x2plus / x4plus | tuỳ chọn `sr_model="rrdb"` | 67 MB mỗi file | BSD-3-Clause |
| 4xRealWebPhoto_v4_dat2 (DAT-2), tác giả Philip Hofmann | model nền chọn tay, mức 6 | 140 MB | CC-BY-4.0 |
| 4xNomosWebPhoto_RealPLKSR, tác giả Philip Hofmann | model nền chọn tay | 30 MB | CC-BY-4.0 |
| RetinaFace ResNet50 (facexlib) | tìm mặt + 5 landmark | 109 MB | MIT |
| ParseNet (facexlib) | tách vùng mặt khi ghép | 85 MB | MIT |
| Đoán tuổi ResNet-10 (Cydral Technology, dlib-models; ONNX từ gói age-and-gender của mowshon, MIT) | preset theo đối tượng | 10,8 MB | CC0 1.0 |
| Phân loại giới tính (Cydral Technology, dlib-models; ONNX như trên) | preset theo đối tượng | 0,5 MB | CC0 1.0 |
| LaMa big-lama (Suvorov và cộng sự, bản TorchScript của Sanster) | vá vết xước, bụi (khôi phục ảnh cũ) | 206 MB | Apache-2.0 |
| U-Net dò xước (Microsoft, Bringing Old Photos Back to Life) | dò xước, nếp gãy (khôi phục ảnh cũ) | 75 MB (fp16) | MIT |
| ArcFace IR-SE50 (facexlib) | chỉ dùng trong `tools/evaluate.py` | 175 MB | MIT |

Ảnh UTKFace (nghiên cứu phi thương mại) và model InsightFace chỉ dùng để đo trong `tools/`, không đóng gói. CodeFormer, SUPIR, HYPIR **không** được dùng vì giấy phép không cho dùng thương mại. Hai model của Philip Hofmann (github.com/Phhofm/models) theo CC-BY-4.0: dùng, bán được, phải ghi tên tác giả (cột Giấy phép trong Quản lý model và bảng này). Chúng được nạp bằng thư viện spandrel (MIT). Model mặt và model nền mới chỉ tải khi được chọn. SHA-256 từng file nằm trong `net_anh/weights.py`.

## Cấu trúc mã

```
net_anh/
  pipeline.py   validate, đọc/ghi, Enhancer: các chế độ, Real-ESRGAN chia ô, GFPGAN, ghép, huỷ, dự phòng
  analysis.py   phân tích ảnh, luật mặt hợp lệ, router preset Tự động
  levels.py     thang mức nét 1–6, mức Tự động theo độ mờ, model mặt và model nền chọn tay
  hardware.py   cấu hình máy (CPU, RAM, GPU, bộ nhớ GPU), ước tính thời gian, cảnh báo model nặng
  exposure.py   sửa ảnh quá tối, quá sáng, ngược sáng
  tone.py       tông tươi sáng, trong trẻo (bước cuối)
  wb.py, phone.py, color.py   cân bằng trắng, tông da, bộ lọc theo hãng, tăng màu, độ bóng da
  settings.py   chế độ, preset, thông số chỉnh tay
  classic.py    khử nhiễu theo mức nhiễu, Retinex, CLAHE, unsharp, frequency separation
  history.py    lịch sử SQLite + thumbnail
  weights.py    manifest model, tải, SHA-256, thử lại
  device.py     CUDA / XPU / DirectML / Metal / CPU, bỏ card mà bản PyTorch không chạy được
  gpu_profile.py dòng card, kiểu số có chạy thử, cắt ô theo VRAM, bảng khuyến nghị
  batch.py      làm thư mục: đọc trước, ghi luồng riêng, 2 ảnh song song trên GPU lớn
  gpu_bench.py  đo tốc độ card (--do-gpu)
  identity.py   ArcFace để đo IDS
  gui.py, cli.py
  server.py     máy chủ cho app điện thoại: HTTP API v1, mã ghép nối, hàng đợi một GPU, tìm máy qua UDP
  server_gui.py cửa sổ máy chủ (NetAnh.exe --may-chu)
  archs/        kiến trúc mạng GFPGAN, RestoreFormer/RestoreFormer++, Real-ESRGAN (trích mã gốc, Apache 2.0);
                DAT-2 và RealPLKSR nạp bằng thư viện spandrel
tests/          pytest
tools/          đo chất lượng, ảnh mẫu cho Android, xuất ONNX
```

## Thay đổi ở 0.3

Mạng làm nét nền đổi sang Real-ESRGAN compact, model mà Upscayl dùng. Trên 12 ảnh không mặt nó cho SSIM cao hơn và độ nét sát ảnh gốc hơn RRDB (0,98 so với 1,99 lần), giữ danh tính mặt ngang nhau, nhanh gấp khoảng 4 lần. Mức khử nhiễu là tỉ lệ trộn trọng số với bản wdn: thanh Khử nhiễu 0–10 tương ứng 0–100 %, tự động là 50 %, Ảnh cũ là 100 %. Ảnh tối không còn bật CLAHE cùng Retinex.

## Thay đổi ở 0.5

Sửa ám vàng, đỏ của ảnh điện thoại và bộ lọc theo hãng (đọc EXIF). Cân bằng trắng ước lượng nguồn sáng ở không gian tuyến tính, bỏ vùng mặt khỏi ước lượng và chỉ gỡ 60–70 % ám, vì gỡ hết làm ảnh lạnh và mất không khí. Thông số từng hãng là điểm xuất phát, chưa chỉnh trên ảnh thật của từng hãng.

## Máy chủ cho điện thoại

App Android bật **Chất lượng cao nhất** thì gửi ảnh tới PC này để xử lý bằng model nặng nhất trên GPU, máy chủ không
có thì điện thoại tự xử lý trên máy. Mở **Beauty HDR - Máy chủ cho điện thoại** (menu Start), `may_chu.bat` hoặc
`python -m net_anh.server`. Cài đặt, ghép nối, dùng ở ngoài qua Tailscale, API: [docs/MAY_CHU.md](docs/MAY_CHU.md).

## Thay đổi ở 0.16.0 (cổng kết nối API xử lý ảnh ngoài)

- Máy chủ cho điện thoại nhận thêm nguồn xử lý ngoài: fal.ai Clarity, Stability AI Upscale (Conservative, Fast),
  Replicate (model tự chọn). Khoá API chỉ đặt trên máy chủ (biến môi trường hoặc `~/.net_anh/providers.json`), app chọn
  bằng `engine=<id>` và phải gửi `consent=1` sau khi hỏi người dùng.
- Kết quả từ API ngoài được kiểm giữ khuôn mặt bằng ArcFace; mặt bị đổi (IDS < 0,70) hoặc API lỗi, hết credit thì máy
  chủ tự xử lý bằng pipeline của máy này và ghi lý do vào báo cáo. Ảnh gửi đi bỏ sạch EXIF.
- Thử nhanh không cần máy chủ: `python -m net_anh.providers anh.jpg --engine fal-clarity`. Chi tiết:
  [docs/API_NGOAI.md](docs/API_NGOAI.md).

## Thay đổi ở 0.15.1 (sửa từ máy thật)

- "Chọn GPU mà render lại dùng CPU": model AI vẫn chạy trên GPU (ô Nền ghi "(fp32)" chỉ khi chạy GPU), nhưng các bước
  màu, tông, khử bóng da, ghép mặt là xử lý ảnh trên CPU và chiếm phần lớn thời gian; đồng hồ GPU của Windows / Game Bar
  chỉ đếm phần 3D, không đếm CUDA (xem Task Manager → GPU → "Cuda" hoặc "Compute"). Tóm tắt kết quả giờ ghi tách thời
  gian model AI trên GPU và phần xử lý trên CPU. Khử bóng da nhanh hơn khoảng 3 giây mỗi ảnh (làm mờ σ lớn trên ảnh thu
  nhỏ, tính bản đồ bóng một lần; lệch trung bình 0,03 mức xám). Chọn GPU mà ảnh thật sự chạy CPU (GPU lỗi giữa chừng):
  thanh trạng thái báo ⚠ kèm lý do.
- Khu tiến độ mới ở đáy vùng ảnh (thay thanh 4 px trong thanh trạng thái bị cắt chữ): tên bước đang làm, thanh % cao
  theo cỡ chữ, thời gian đã chạy và còn lại ước tính, nút Huỷ; cỡ theo cỡ chữ hệ thống nên không bị cắt ở DPI cao.

## Thay đổi ở 0.15.0 (tối ưu theo dòng card đồ hoạ)

- Tự nhận dòng card và bộ nhớ (`net_anh/gpu_profile.py`): NVIDIA theo thế hệ chip đọc từ driver (GTX 900, GTX 10, GTX 16,
  RTX 20/30/40/50), Intel Arc / Iris Xe, AMD (ROCm, DirectML), Apple. Ô Thiết bị ghi dòng card, kiểu số, cỡ ô, cách làm thư
  mục và mức hỗ trợ; nút **Khuyến nghị theo card…** (và menu Trợ giúp) mở bảng dưới, tô đậm dòng của máy này.
- Kiểu số có kiểm tra tự động: mỗi model nền khi nạp lên GPU chạy một ảnh mẫu (có mảng sáng phẳng, chỗ làm fp16 hay tràn
  số) ở fp32 và ở kiểu nhanh hơn; lệch quá 3 mức xám, ra NaN hay ra đen thì dùng fp32 cho model đó. Khi chạy thật, ô nào
  hỏng vẫn tự chạy lại fp32 như 0.8.1. GTX 16 và GTX 10 luôn fp32 (không thử). RTX 30 trở lên: fp16 cho Real-ESRGAN,
  RealPLKSR; DAT-2 thử bf16 (fp16 của DAT-2 ra NaN, đo được trên CPU), bật TF32. Ép kiểu để đo: `NETANH_PRECISION=fp32`.
- Cắt ô theo bộ nhớ GPU còn trống (Settings.tile mặc định −1 = tự động, CLI `--tile -1`): card 4–6 GB cắt ô nhỏ cho
  model nặng, card 12 GB trở lên gần như không phải cắt với model nhẹ, ô DAT-2 lớn gấp 2–3 lần trước (ít đường ghép,
  nhanh hơn). Thiếu bộ nhớ thật vẫn tự chia đôi ô. CPU giữ ô 400 px.
- Làm cả thư mục nhanh hơn (`net_anh/batch.py`, dùng cho cả giao diện và dòng lệnh): model nạp một lần, ảnh kế tiếp được
  đọc trước trong lúc ảnh hiện tại đang chạy, ghi file ở luồng riêng; GPU NVIDIA từ 11 GB VRAM, máy 16 GB RAM, 8 luồng
  thì chạy 2 ảnh song song (mỗi luồng một bộ model, chia đôi bộ nhớ khi tính ô). Ép số luồng: CLI `--song-song N`, hoặc
  biến `NETANH_BATCH_WORKERS`.
- GTX 10 / GTX 900: PyTorch CUDA 12.8 trở lên không còn mã máy cho Pascal, Maxwell (đọc được trong bản PyTorch mới:
  sm_75 trở lên). Thêm gói **cuda_old** (PyTorch CUDA 12.6) ở BeautyHDR.exe và `build_windows.ps1 -Flavor cuda_old`;
  BeautyHDR.exe tự chọn theo tên card và hỏi chuyển nếu máy GTX 10 đã lỡ cài bản GPU thường. App không còn liệt kê card
  mà bản PyTorch đang cài không chạy được (trước đây chạy là lỗi "no kernel image"), mà ghi lý do.
- Tăng tốc theo card × model: model tích chập (compact, RealPLKSR, x4plus) ở fp16/bf16 trên card có Tensor Core (RTX 20
  trở lên) xếp dữ liệu kênh cuối (channels_last, ảnh ra không đổi; ép `NETANH_CHANNELS_LAST=0|1`). Model mặt GFPGAN,
  RestoreFormer chạy thử bf16 trên RTX 30 trở lên (fp16 bị loại: GFPGAN fp16 lệch 91 mức xám, đo trên CPU), ra NaN thì
  tự về fp32. Làm thư mục từ 3 ảnh trên NVIDIA: bật cuDNN tự chọn thuật toán (`NETANH_CUDNN_BENCHMARK=0` để tắt). Bảng
  **Khuyến nghị theo card…** có thêm bảng từng model (kiểu số, kênh cuối, cỡ ô, thời gian ước lượng) cho card của máy.
- Dữ liệu card × model tách khỏi code, để sau này thêm model nặng hơn: `net_anh/du_lieu/model_profiles.json` khai mỗi
  model một dòng (vai trò, kiến trúc, chạy được fp16/bf16 không, bộ nhớ mỗi điểm ảnh, tốc độ CPU, VRAM tối thiểu, cần bf16
  thật không, ngưỡng chậm). Bảng khuyến nghị, cắt ô, ước lượng tốc độ và kiểm phần cứng (app ghi chú khi model chậm hoặc
  thiếu VRAM trên card này) đều dựng từ file đó. File đo (`--do-gpu`) có thêm `schema` và danh sách bản ghi phẳng (một số
  một dòng: dòng card, card, model, kiểu số, loại số đo, giá trị, cỡ ô, kênh cuối, bản app/PyTorch/driver);
  `python tools/nhap_so_do.py <file.json>` gộp vào `net_anh/du_lieu/so_do_gpu.json`, và app dùng trung vị số đo thật
  thay cho ước lượng (bảng ghi "(đo)"). `NETANH_DATA_DIR` trỏ tới thư mục khác để thử bảng mới không phải build lại.
- Intel Arc / Iris Xe (gói xpu từ 0.14.1) giữ fp32, cắt ô theo bộ nhớ; AMD chưa có gói cài sẵn (ROCm hoặc DirectML tự cài
  từ mã nguồn), thử nghiệm.
- Đo trên card thật: **Trợ giúp → Đo tốc độ card…** hoặc `NetAnh.exe --do-gpu` (`python app.py --do-gpu`, thêm `--nhanh`):
  đo từng model nền ở fp32/fp16/bf16 (lệch so với fp32, thời gian ô 256 và 512 px, byte bộ nhớ mỗi điểm ảnh), GFPGAN, cả
  quy trình 1 và 2 luồng, ghi file `BeautyHDR-do-gpu-….txt` ra Desktop. Các số tốc độ GPU trong bảng là ước lượng, chưa
  đo trên card thật.

Bảng khuyến nghị (cột Cắt ô: cạnh ô lớn nhất theo VRAM, compact fp16 / DAT-2 bf16 hoặc fp32, khi card còn trống 90 % bộ nhớ):

| Dòng card | Ví dụ | Kiểu số (thử theo thứ tự) | Cắt ô (compact / DAT-2) | Nên dùng | Hàng loạt | Mức hỗ trợ |
|---|---|---|---|---|---|---|
| GeForce RTX 50 (Blackwell) | RTX 5060, 5070, 5080, 5090 | fp16 → bf16 → fp32 | 8 GB: 1536 / 576 px · 12 GB: 1536 / 736 px · 16 GB: 1536 / 864 px · 32 GB: 1536 / 1248 px | mức 1–6 | 2 ảnh song song khi ≥ 11 GB VRAM, 16 GB RAM, 8 luồng | đầy đủ |
| GeForce RTX 40 (Ada) | RTX 4060, 4070, 4080, 4090 | fp16 → bf16 → fp32 | 8 GB: 1536 / 576 px · 12 GB: 1536 / 736 px · 16 GB: 1536 / 864 px · 24 GB: 1536 / 1088 px | mức 1–6 | 2 ảnh song song khi ≥ 11 GB VRAM, 16 GB RAM, 8 luồng | đầy đủ |
| GeForce RTX 30 (Ampere) | RTX 3050, 3060, 3070, 3080, 3090 | fp16 → bf16 → fp32 | 4 GB: 1344 / 352 px · 8 GB: 1536 / 576 px · 12 GB: 1536 / 736 px · 24 GB: 1536 / 1088 px | mức 1–6 từ 6 GB; 4 GB nên mức 1–5 | 2 ảnh song song khi ≥ 11 GB VRAM, 16 GB RAM, 8 luồng | đầy đủ |
| GeForce RTX 20 (Turing có Tensor Core) | RTX 2060, 2070, 2080, Titan RTX | fp16 → fp32 | 6 GB: 1536 / 320 px · 8 GB: 1536 / 384 px · 11 GB: 1536 / 480 px | mức 1–6 | 2 ảnh song song khi ≥ 11 GB VRAM, 16 GB RAM, 8 luồng | đầy đủ |
| GeForce GTX 16 (Turing không Tensor Core) | GTX 1630, 1650, 1660, 1660 Ti, 1660 Super, MX450 | fp32 | 4 GB: 960 / 224 px · 6 GB: 1280 / 320 px | mức 1–6 từ 6 GB; 4 GB nên mức 1–5 | 1 ảnh, đọc/ghi file song song | đầy đủ |
| GeForce GTX 10 (Pascal) | GTX 1050, 1050 Ti, 1060, 1070, 1080, MX150–MX350 | fp32 | 2 GB: 448 / 128 px · 4 GB: 960 / 224 px · 6 GB: 1280 / 320 px · 8 GB: 1536 / 384 px · 11 GB: 1536 / 480 px | mức 1–6 từ 6 GB; 2 GB nên mức 1–5 | 2 ảnh song song khi ≥ 11 GB VRAM, 16 GB RAM, 8 luồng | đầy đủ |
| GeForce GTX 900 / 750 (Maxwell) | GTX 750 Ti, 950, 960, 970, 980 | fp32 | 2 GB: 448 / 128 px · 4 GB: 960 / 224 px | mức 1–6 từ 6 GB; 2 GB nên mức 1–5 | 1 ảnh, đọc/ghi file song song | thử nghiệm |
| Intel Arc (XPU) | Arc A380–A770, B570, B580, Arc trong Core Ultra | fp32 | 6 GB: 1280 / 320 px · 8 GB: 1536 / 384 px · 12 GB: 1536 / 512 px · 16 GB: 1536 / 608 px | mức 1–6 | 1 ảnh, đọc/ghi file song song | thử nghiệm |
| Intel Iris Xe / UHD (XPU) | Iris Xe, UHD 7xx | fp32 | theo bộ nhớ đọc được, không đọc được thì 400 px | mức 1–5; mức 6 chậm | 1 ảnh, đọc/ghi file song song | thử nghiệm |
| AMD Radeon qua ROCm | RX 7600–7900, RX 9060–9070 (ROCm trên Linux, bản xem trước trên Windows) | fp16 → fp32 | 8 GB: 1536 / 384 px · 16 GB: 1536 / 608 px · 24 GB: 1536 / 768 px | mức 1–6 | 2 ảnh song song khi ≥ 11 GB VRAM, 16 GB RAM, 8 luồng | thử nghiệm |
| AMD / Intel qua DirectML | Radeon RX 5000–9000, Vega, card tích hợp | fp32 | theo bộ nhớ đọc được, không đọc được thì 400 px | mức 1–5; mức 6 chậm | 1 ảnh, đọc/ghi file song song | thử nghiệm |
| Apple Silicon (Metal) | M1–M4 | fp32 | theo bộ nhớ đọc được, không đọc được thì 400 px | mức 1–5; mức 6 chậm | 1 ảnh, đọc/ghi file song song | thử nghiệm |
| CPU | Mọi máy | fp32 | 400 px | mức 1–5 (model nhẹ); mức 6 rất chậm | 1 ảnh, đọc/ghi file song song | đầy đủ |

## Thay đổi ở 0.14.2 (màu cảnh ảnh cũ, Tô màu AI, preset ảnh thẻ; gộp sửa loang 0.14.1)

- Gộp bản sửa loang màu tông mẫu (look.py: nền chỉ làm phẳng khi thật sự trơn, "da" theo màu chỉ quanh mặt, gỡ ám chỉ vùng trắng thật)
  vào nhánh có BeautyHDR.exe (0.14.1 nhận diện card NVIDIA/AMD/Intel). Trước đây hai thay đổi này nằm ở hai nơi.
- Khôi phục ảnh cũ, ô "Nâng màu sặc sỡ cho cảnh" (Chuyên sâu; Settings.scene_vivid, CLI --scene-vivid 0..1): ảnh cũ chụp cảnh
  (nền không đồng màu) được gỡ ám vùng sáng, kéo sáng, ấm vùng tối/giữa và tăng độ rực; bỏ qua ảnh đen trắng và ảnh thẻ nền đồng màu.
- Ô "Tô màu AI cho cảnh" (Settings.scene_ai, CLI --scene-ai, cần ô trên): DDColor (Apache-2.0, net_anh/colorize.py, archs/ddcolor_arch)
  đoán lại màu theo ngữ nghĩa (áo trắng, quần jeans, cây xanh), ghép a, b ở cuối pipeline. Model 912 MB tải riêng ở lần đầu bật,
  không nằm trong bộ model mặc định. Ảnh còn đủ màu (sắc độ Lab trung bình trên 10, như ảnh điện thoại) được bỏ qua vì tô lại ra loè loẹt;
  ảnh phai dưới 7,5 tô đủ (restore.ai_color_weight). Ngưỡng mới chỉnh trên 1 ảnh mẫu; chưa có ở chế độ Đơn giản và Android.
- Thư mục presets/: 4 bộ preset ảnh thẻ (nam/nữ, trẻ/lớn tuổi; bộ lớn tuổi không trẻ hoá) + HUONG_DAN.txt cách nạp
  (Chuyên sâu > Quản lý… > Nạp từ file…). tools/id_preset_bench.py đo tuổi trước/sau.

## Thay đổi ở 0.13.1

- Sửa lỗi sập `UnicodeEncodeError: 'charmap' codec can't encode character` khi chạy `python app.py --may-chu` (và mọi lệnh in chữ Việt)
  trên console Windows dùng cp1252: net_anh/__init__.py ép stdout/stderr sang UTF-8 (thay ký tự không hiện được bằng ?) ngay khi nạp gói,
  phủ app.py, python -m net_anh.*, máy chủ, dòng lệnh; chay.bat, may_chu.bat, cai_dat.bat, dong_goi_exe.bat đặt thêm chcp 65001 và PYTHONUTF8=1.
  Test tests/test_core.py::test_vietnamese_output_survives_cp1252_console chạy với PYTHONIOENCODING=cp1252.

## Thay đổi ở 0.13.0: một file exe duy nhất

- `BeautyHDR.exe` (launcher/beautyhdr_launcher.py, chỉ thư viện chuẩn Python) là file duy nhất người dùng tải. Mở lên: kiểm
  `update.json` ở Releases mới nhất; chưa cài thì tải gói app (CPU hoặc GPU NVIDIA, hỏi nếu máy có card), tải các model cần
  thiết (NetAnh.exe --tai-model --progress-file; model vá xước tách từ zip ngay trong app), rồi chạy app. Có bản mới thì
  tự cập nhật, model giữ nguyên. Tải dở mở lại thì làm tiếp (HTTP Range), kiểm SHA-256, kiểm ổ trống trước khi tải, báo lỗi
  rõ nếu hết chỗ hoặc mất mạng. Dữ liệu ở %LOCALAPPDATA%\BeautyHDR (app, weights, downloads, state.json).
- Bản GPU chạy được cả bằng CPU; chọn Tự động / GPU / CPU trong app (ô Thiết bị).
- .github/workflows/release-exe.yml: build gói app cpu + cuda, cắt phần dưới 1,9 GB, build BeautyHDR.exe, tạo update.json và
  đăng lên Releases. Kho phải công khai (UPDATE_REPO trong launcher hoặc biến BEAUTYHDR_REPO).
- Cách cài cũ (TAO_FILE_CAI_DAT.bat, setup.exe, -CapNhat) vẫn còn trong mã nhưng không còn là cách phát hành.
## Thay đổi ở 0.14.0 (tông mẫu Trong sáng, Da đẹp ảnh thẻ)

- Đo 4 ảnh thẻ mẫu của huy bằng tools/look_stats.py (ParseNet tách da, môi, tóc, nền; Lab): da L ~74 a ~10 b ~13 (góc màu ~50°),
  môi a ~33 b ~15, tóc đen trung tính (sắc độ 2–5), nền phẳng, dải sáng tối p5–p95 rộng (85–96), vân da ~2. Ảnh gốc của huy lệch:
  da đỏ (a 15–19) hoặc vàng (b 28–30), tóc nâu đỏ, nền loang, ảnh mờ sương (p5–p95 52–67).
- net_anh/look.py (không model): da lệch màu hằng số về đích (giữ vân), nâng sáng kiểu screen, đều màu; môi về hồng đỏ; tóc về đen;
  vùng trắng về trắng hơi lạnh; kéo giãn dải sáng tối; làm phẳng nền đồng màu (không đụng viền tóc, người). Da đẹp ảnh thẻ thêm
  làm sạch mụn, đốm sạm, vết đỏ ở tầng giữa (chỗ lệch lớn như lỗ mũi, mí mắt giữ nguyên) và làm đều vùng sạm rộng.
- Settings.look (dict) và look_amount; mặc định tắt, không preset cũ nào đổi. Hai bộ preset có sẵn "Trong sáng (ảnh thẻ)" và
  "Da đẹp ảnh thẻ" mang tông mẫu; file .netanh.json định dạng 3 có thêm khoá "look" (đọc được file 1, 2).
- GUI: Đơn giản nhóm "Tuỳ chọn thêm" ô "Tông màu"; Chuyên sâu nhóm "Ánh sáng và tông" ô "Tông mẫu" + thanh "Độ mạnh tông mẫu".
  Giữ khi đổi loại ảnh, mức nét, mở ảnh. CLI: --look trong_sang|da_dep_the|off, --look-amount, hoặc --preset-set trong-sang / da-dep-the.
- Đo trên 2 ảnh gốc của huy (CPU, mức Tự động): da đỏ a 19 → 12–13, vàng b 28 → 16, tóc nâu đỏ → sắc độ 4–8, nền loang → phẳng.
  Test: 121 qua. Chưa thử trên Windows thật, chưa có trên Android.

## Thay đổi ở 0.13.0 (khử bóng loáng da)

- Tuỳ chọn mới **Khử bóng da** (không bắt buộc, mặc định tắt): ô "Khử bóng loáng da" ở Đơn giản (nhóm Tuỳ chọn thêm) và
  ô + thanh cường độ ở Chuyên sâu (nhóm Khuôn mặt). Hai ô là một thông số `Settings.deshine` (0 = tắt, tích lần đầu 60 %);
  đổi loại ảnh, mức nét, mở ảnh mới không làm mất; lịch sử và hoàn tác nhớ đúng.
- Thuật toán `net_anh/deshine.py` (không model, CPU): vùng da từ ParseNet (da mặt, mũi; không đụng mắt, mày, môi, tóc),
  bản đồ bóng = vùng sáng hơn da xung quanh **và** nhạt màu hơn (ánh gương trắng), kéo tầng giữa về gần da nền (giữ 25 %
  phần nhô để còn khối), trả màu da, xoá đốm bóng nhỏ ở chóp mũi, nhân trung; vân da gần như giữ nguyên; chỗ cháy sáng
  được thêm lại chút vân. Chạy ở bước cuối (sau Tươi sáng, trước Đổi nền), ngay trên điểm ảnh của ảnh ra.
- Đo trên ảnh chân dung da dầu của huy (Tự động, phóng 2×, CPU): vùng bóng 12,3 % → 7,9 % da mặt, sáng p99 của da
  89,1 → 87,7, độ vân da (tầng σ 2 px) 0,29 → 0,29. Thêm ~1–2 s mỗi ảnh.
- Preset theo đối tượng có thêm nút **Khử bóng** (−1..1, cộng vào thanh chung, chỉ khi ô được tích): bộ mặc định Nam +0,1,
  Trẻ em −0,2. File `.netanh.json` lên định dạng 2; file định dạng 1 vẫn nạp được. Android 0.8/0.9 nạp file mới vẫn được
  (báo bỏ qua "matte").
- Dòng lệnh `--deshine 0..1`; máy chủ nhận `deshine=0..1` (`info.features` có "deshine").
- Chưa chạy thử trên Windows thật.

## Thay đổi ở 0.12.2

- Khi đang Làm nét, khoá mọi chỉnh sửa (thông số, preset, lịch sử, mở/đóng ảnh, thiết bị); nút Làm nét thành nút Huỷ.
- Thanh tiến trình có % và bước đang chạy (Làm nét nền 40%, Mặt 2/5…), thời gian đã chạy / ước tính; cảnh báo khi chạy lâu
  hơn ước tính hoặc một bước đứng yên; lỗi báo ngay bước nào và lý do.
- Ô Thiết bị: Tự động / GPU / CPU, báo đang dùng gì; GPU xám kèm lý do (bản cài chỉ có CPU thì chỉ cách chuyển).
- Nút Đơn giản / Chuyên sâu đặt ở đầu cột phải (không còn ở góc thanh menu bị lớp phủ FPS che).
- build_windows.ps1: -CapNhat (chép đè lên bản đã cài, không tải lại model/torch), bản đầy đủ hỏi tạo bản GPU khi máy có card NVIDIA.

## Thay đổi ở 0.12.1

- Một thanh mức nét chung cho hai chế độ: Tự động, 1 Nhẹ, 2 Vừa, 3 Rõ, 4 Rất rõ, 5 Tối đa, 6 Cao nhất. Chọn ở chế độ
  này thì chế độ kia sáng cùng nút (0.12.0 có 4 nút tên riêng ở Đơn giản, lệch với mức 1–6 ở Chuyên sâu). Mô tả từng
  mức xem ở chú thích khi di chuột.
- Bỏ các dòng chú thích thừa (mô tả mức, mô tả chế độ, ghi chú nhóm Khôi phục, Preset theo đối tượng, gợi ý dài ở
  Phân tích ảnh; gợi ý vẫn ghi vào Nhật ký). Ước tính thời gian ở Chuyên sâu chỉ hiện khi có cảnh báo.

## Thay đổi ở 0.12.0 (giao diện mới hai chế độ, theo demo huy duyệt)

- Giao diện kiểu app chỉnh ảnh desktop (Lightroom, Capture One): nền xám trung tính quanh ảnh, góc gần vuông, không
  bóng đổ hay gradient, biểu tượng nét mảnh một màu, một màu nhấn xanh thép cho nút Làm nét và thanh tiến độ.
- Thanh menu Tệp, Sửa, Ảnh, Xem, Model, Trợ giúp; thanh công cụ Mở ảnh, Cả thư mục, Hoàn tác, Đóng ảnh, Xuất ảnh.
  Thông báo và tiến độ (kèm Huỷ) ở thanh trạng thái; nhật ký ở Xem → Nhật ký.
- Hai chế độ, đổi bằng chữ Đơn giản | Chuyên sâu ở góc phải thanh menu (Ctrl+1, Ctrl+2), app nhớ lần sau.
  Hai chế độ dùng chung một bộ thông số: đổi qua lại không mất gì.
  - Đơn giản: Phân tích ảnh → Loại ảnh (Tự động, Chân dung, Ảnh cũ, Ảnh thẻ đổi nền, Phong cảnh đồ vật, Ảnh tối) →
    Mức nét → Làm nét.
    Ảnh thẻ hiện ô màu nền, Ảnh cũ hiện nút Tô vùng hỏng nặng. Loại ảnh: Chân dung = preset Rõ nét tối đa,
    Ảnh cũ = Khôi phục ảnh cũ, Ảnh tối = Ảnh tối (Retinex), Phong cảnh = chế độ Phóng to. Mở ảnh mới vẫn giữ lựa chọn.
  - Chuyên sâu: cột trái Preset (kiểu làm nét, bộ preset theo đối tượng: Nạp, Xuất, Quản lý) và Lịch sử; cột phải
    histogram, ô tìm thông số (không cần gõ dấu), 8 nhóm thu gọn được có Đặt lại từng nhóm; Khôi phục ảnh cũ và
    Đổi nền bật/tắt bằng ô tích trên tiêu đề nhóm.
- Thanh dưới ảnh: Trước | Chia đôi | Sau, Vừa | 100% | 200%.
- Xem → Cỡ chữ giao diện (9–13 pt). Lần đầu app tự chọn chữ lớn hơn trên màn rộng (2560 px để tỉ lệ 100 %).
- Cửa sổ nhỏ nhất 980×620; các ô chọn co theo cột, không còn bị cắt mép phải.
- Thuật toán, model, CLI, máy chủ cho điện thoại không đổi so với 0.11.0.

## Thay đổi ở 0.11.0 (đợt 2 chất lượng, theo kết quả chấm mù của huy)

Số đo và ảnh so sánh: tools/exp_*.py (mặt, vá xước, da và viền, màu) và trang chấm mù A/B (31 cặp, mù tên bản).

- **Vá xước giữ ngũ quan** (`restore.feature_guard`): bộ dò đốm 0.10 bắt nhầm 22–63 % vùng mắt, lông mày và 16–55 % môi trên mặt nhỏ, LaMa vẽ lại lòng đen và ánh mắt thành da trơn. Nay vùng mắt, lông mày, mũi, miệng, kính (theo 5 điểm mốc + ParseNet) không bị coi là bụi, trên đầu chỉ nhận đốm trắng rõ; vết xước đi qua mặt vẫn vá nhưng chỉ đúng nét xước. Bắt nhầm về ~0 %, vết xước giả lập thu hồi 56 % (0.10: 61 %). Chấm mù: hơn 1, ngang 2.
- **Phục hồi màu phai giữ màu da** (`Settings.fade_style = "giu_da"`, mặc định; `--fade-style 0.10` cho bản cũ): chỉ gỡ ám khi có dấu hiệu phai hoặc giấy ố (`restore.fade_cast`), ánh sáng ấm có chủ ý chỉ được kéo tương phản (ΔE so với ảnh vào 9,3 → 0,8), độ rực thêm theo màu da và kiểu vibrance (môi ảnh ám vàng +30 → +7 độ rực), da lệch góc màu tự nhiên được xoay về. Chấm mù: hơn 4, ngang 5, kém 1 (ảnh nhóm nắng chiều).
- **Độ bóng da của mức 5 và "Rõ nét tối đa" về 0** (0,5 trước đó): chấm mù ưa bản tắt hơn; đo cho thấy quầng sáng quanh viền mặt giảm 0,35–0,56 mức sáng. Mức 4 giữ 0,2.
- **Mức AI cho mặt giữ nguyên**: chấm mù ưa 70 % AI (mức 5) hơn 50 % và 30 % ở mặt lớn, nên Tự động vẫn dùng Rõ nét tối đa cho chân dung.
- **Ảnh nhóm, toàn thân** (từ thread "Cải thiện ảnh toàn thân và phong cảnh"): số mặt tối đa mặc định 5 → 24 (`Settings.max_faces`, `--max-faces`). Nguyên nhân thật: ảnh đội bóng 21 mặt chỉ 5 mặt qua model mặt, 16 mặt còn lại bị mạng nền vẽ thành mặt nạ có vạch ngang. Mặt hai mắt cách < 10 px (`pipeline.AI_MIN_EYE`, đổi được bằng biến môi trường `NETANH_AI_MIN_EYE`) không chạy model mặt mà trộn lại nét gốc theo `face_ai`.
- **Mặt rất nhỏ** (hai mắt cách < 32 px, ảnh nhóm): model mặt bịa ra tóc vàng, mắt xanh. Giới hạn trọng số AI bị huy chấm thua nên bỏ; thêm tuỳ chọn `Settings.small_face_sr` / `--small-face-sr` (phóng AI vùng mặt trước khi căn: ít bịa hơn, ít nét hơn), tắt mặc định, chờ chấm vòng 2.

## Thay đổi ở 0.10.3 (tên và icon mới)

- App đổi tên thành **Beauty HDR** (cửa sổ, menu Start, bộ cài, biểu tượng ngoài màn hình) và có icon mới hình chân dung nửa vỡ hạt nửa rõ nét (`packaging/net_anh.ico`, `net_anh/assets/icon.png`; vẽ lại bằng `/mnt/project-files/ten-icon/make_assets.py`).
- Giữ nguyên tên file `NetAnh.exe`, thư mục cài `NetAnh` và mã bộ cài (AppId), nên cài đè lên bản cũ được và không phải tải lại model. Định dạng file preset `.netanh.json` và giao thức máy chủ với điện thoại không đổi.
- Sửa 4 lỗi từ review mã:
  - Ảnh màu không còn bị đổi thành đen trắng khi bật phục hồi màu phai: `restore.is_mono()` xét vùng giữa và lưới 4×4 ô phủ cả ảnh (bỏ 4 % sát mép), chỉ cần một ô có màu rõ là giữ màu (vd vùng giữa xám, áo đỏ sát mép).
  - Bộ đếm mặt báo đúng: mặt bị lỗi model (dùng lại mặt gốc) không còn tính là "đã làm nét". Báo cáo ghi thêm số mặt lỗi model giữ nguyên; mặt lỗi cũng không bị khớp màu hay trộn nữa. Danh sách ghép vẫn đủ các mặt.
  - Ảnh TIFF số thực (mode F) chuẩn hoá 0..1 không còn mở ra gần như đen; giá trị NaN/vô cực thành đen.
  - Ảnh dải rất dài và hẹp không còn lỗi khi thu nhỏ (process(), máy chủ trả ảnh cho điện thoại, dò đốm bụi): dùng `resize_long()` / chặn mỗi cạnh tối thiểu 1 px.
  - Xử lý cả thư mục (giao diện và dòng lệnh) có tổng kết tách rõ: ảnh làm bằng AI đủ, AI một phần (có mặt lỗi model giữ nguyên), CHỈ XỬ LÝ NHANH vì AI lỗi (kèm tên file; giao diện hiện cảnh báo), chế độ Nhanh theo lựa chọn, và ảnh lỗi. Báo cáo từng ảnh có cờ `ai_fallback` (cả trong phản hồi máy chủ cho điện thoại).

## Thay đổi ở 0.10.2 (máy chủ cho Android 0.8)

- Máy chủ nhận thêm tham số từ điện thoại: khôi phục ảnh cũ (`restore=1`), đổi nền (`bg`), đối tượng (`subject`) và kiểu mặt
  của bộ preset đang dùng trên điện thoại (`styles`). `GET /api/v1/info` có thêm `features` để app biết máy tính hiểu
  các tham số này; máy chủ bản cũ thì app báo cần cập nhật. Chi tiết: docs/MAY_CHU.md.
- tools/make_android_fixtures.py: ảnh mẫu kiểm tra bản Android (kiểu mặt, tuổi/giới tính, đổi nền, khôi phục, cân bằng
  trắng lấy da làm mốc).

## Thay đổi ở 0.10.1 (vùng hỏng nặng)

- Nút **Vẽ vùng hỏng nặng…** (CLI `--mask mat_na.png`, trắng = hỏng): tô lên lá vàng, vết ố, chỗ rách che mất chi tiết.
  App vá chỗ đó bằng LaMa, rồi mặt trúng vùng hỏng được dựng lại bằng RestoreFormer++ (nếu đã tải) trong đúng vùng tô.
- Mặt nhỏ thì bớt dựng lại: đo trên ảnh lớp học, mặt trẻ em ~40 px bị phủ lá vàng mà để model mặt tự dựng thì ra người
  lớn đeo kính. Mặt dưới 30 px chỉ giữ phần đã vá, 60–90 px dựng một phần, tối đa 80 %.
- Không tự dò được mảng hỏng lớn: lá vàng không phân biệt được với họa tiết thật (áo hoa, áo chấm bi) bằng màu hay độ
  nhám (đã thử), nên để người dùng tô. Preset Khôi phục ảnh cũ làm tối đa 24 mặt (trước là 5).
- Settings mới: `damage_mask` (đường dẫn PNG). Model: không thêm file nào (dùng lại LaMa, RestoreFormer++).

## Thay đổi ở 0.10.0 (khôi phục ảnh cũ)

Cho ảnh in cũ chụp lại bằng điện thoại: xước, bụi, vết ố vàng, nếp nhăn màng bọc, ảnh phai màu, ảnh nằm ngang.
Chọn preset **Khôi phục ảnh cũ** hoặc tích “Khôi phục ảnh cũ” (CLI `--preset restore`). Các bước (`net_anh/restore.py`):

1. Xoay thẳng ảnh nằm ngang/ngược: dò mặt ở 4 hướng, chỉ xoay khi một hướng có nhiều mặt hơn hẳn.
2. Dò vết hỏng: U-Net dò xước/nếp gãy của Microsoft “Bringing Old Photos Back to Life” (MIT), cộng bộ dò đốm bụi và
   đốm vàng bằng top-hat OpenCV (chỉ bắt hạt nhỏ tròn, không bắt cạnh áo hay viền màu).
3. Vá bằng LaMa big-lama (Apache-2.0), chia ô 640 px quanh vùng hỏng; thiếu model thì dùng `cv2.inpaint`. Luôn fp32.
4. Phục hồi màu phai: kéo giãn từng kênh theo phân vị vùng giữa ảnh (viền giấy không tính), cân bằng trắng mềm, tăng độ
   rực. Ảnh đen trắng chỉ chuẩn mức sáng và gỡ ám màu giấy; **chưa tô màu** (model tô màu nằm trên máy chủ bị chặn).
5. Làm nét như cũ (mặt GFPGAN giữ 60 % AI, nền Real-ESRGAN compact, phóng tự động) rồi tông tươi sáng.

Giới hạn: vết bẩn lớn trên mặt (mảng vàng to hơn hạt bụi) vẫn còn; chưa tự cắt viền giấy. Model mới tải ở lần đầu:
LaMa 206 MB, dò xước 75 MB (lấy đúng một phần của file zip 2 GB của Microsoft bằng HTTP Range, ~450 MB tải về).
Thêm vào `Settings`: `repair`, `fade`, `auto_rotate` (mặc định tắt, ảnh cũ chạy y như trước).

## Thay đổi ở 0.9.2

- Đổi màu nền: viền tóc hết ám màu nền cũ; khe nền kín trong lọn tóc cũng được đổi; tàn dư cùng sắc nền sát mép bị bỏ.

## Thay đổi ở 0.9.1 (đổi màu nền đồng màu)

- Tích "Đổi màu nền (nền đồng màu)" rồi chọn Trắng, Xanh, Xanh nhạt, Đỏ hoặc “Màu…” bất kỳ (CLI --bg-color #RRGGBB hoặc
  trang, xanh, xanh-nhat, do, xam). Dùng cho ảnh thẻ, phông trơn; chạy ở cuối cùng, trên ảnh đã làm nét (net_anh/backdrop.py,
  không cần model AI, ~1 giây mỗi ảnh).
- Cách làm: dò nền từ viền trên và hai bên (≥ 60 % mẫu cùng màu thì coi là đồng màu), tách theo khoảng cách màu chỉ trong
  vùng nối liền với viền (áo cùng màu nền ở trong người không bị đổi), mặt nạ mềm cho sợi tóc, gỡ ám nền cũ ở viền để không
  còn quầng xanh, bỏ vết bẩn rời trên phông, rồi ghép lên màu mới. Nền có vật thể hay nhiều màu thì không đổi và báo lý do.
- Giới hạn: khe nền kín giữa tay và thân giữ nguyên màu cũ; áo cùng màu nền và dính nền sẽ bị đổi theo nền.

## Thay đổi ở 0.9 (preset theo đối tượng)

- Mỗi khuôn mặt được đoán tuổi, giới tính (net_anh/subject.py; model Cydral CC0, 11 MB, chạy bằng OpenCV DNN) rồi
  chỉnh theo kiểu của nhóm: Trẻ em (0–12 tuổi), Nam, Nữ, Người lớn tuổi (từ 60), Người lớn (trung tính khi không chắc).
  Ảnh gia đình: bố, mẹ, con mỗi người một kiểu. Ô "Đối tượng" ở cột trái: Tự động, ép một nhóm, hoặc Tắt (CLI --subject).
- Một **bộ preset** có 5 dòng (mỗi nhóm một kiểu) × 6 nút: Giữ nét thật, Ngũ quan, Mịn da, Bóng da, Sáng da, Hồng hào
  (net_anh/facestyle.py). Kiểu cộng thêm vào mức nét, không thay mức nét; chỉ chỉnh da, mũi, tai trên mặt đã căn, không
  đụng mắt, môi, tóc. Chạy cả ở mức 1–2 (không model mặt).
- Quản lý… : xem bộ có sẵn, Tạo bản sao rồi sửa, đổi tên, xoá, **Xuất ra file** `.netanh.json` và **Nạp từ file** để chia sẻ
  giữa người dùng (JSON thuần số và chữ, tối đa 512 KB; giá trị ngoài khoảng bị kẹp lại và báo, khoá lạ bị bỏ). CLI:
  --preset-file, --preset-set. Cùng định dạng cho bản Android.
- Độ chính xác (UTKFace 800 mặt, đo nội bộ, không phát hành): người châu Á đúng nhóm 87 % (ảnh gốc), 82 % (mờ);
  trẻ em đúng 97–100 %, nữ 90 %, nam 81–85 %, người lớn tuổi 57–79 % (hay bị đoán là người lớn thường). Không chắc thì
  dùng preset trung tính, chọn tay được.
- Model tuổi, giới tính (age-v1.onnx, gender-v1.onnx) tải lần đầu như các model khác, đóng kèm trong bộ cài.

## Thay đổi ở 0.8.1 (sửa lỗi theo phản hồi trên máy thật)

- Nền ra mảng đen trên GTX 1650/1660 (fp16 tràn số): dòng GTX 16xx luôn chạy fp32; với mọi GPU, ô nào ra NaN hoặc đen
  thì tự chạy lại fp32 (ghi chú trong Nhật ký). DAT-2 chỉ dùng bf16 trên GPU từ RTX 30 (trước đây Turing báo hỗ trợ bf16
  giả lập, DAT-2 lỗi rồi lặng lẽ quay về compact nên đổi model không thấy khác).
- Ghi rõ model đã thật sự chạy: "Mặt: … · Nền: … (fp16/bf16/fp32)" trên ảnh Sau, thanh trạng thái, Nhật ký, Lịch sử;
  model chọn bị thay vì lỗi thì ghi "⚠ … dùng Real-ESRGAN compact thay".
- Ghép ô: hai ô kề nhau chồng nhau và được trộn dần, hết đường nối dọc (DAT-2 cho vân khác nhau giữa các ô).
- Cân bằng trắng lấy da làm mốc: chỉ được kéo màu da về phía da tự nhiên, không làm da hồng thành vàng nhợt khi nền là
  mảng màu rực (phông xanh, áo cam). Sửa tông da chỉ kéo da đỏ hẳn (góc màu < 36°), giữ da hồng hào.
- Rõ khối (Tươi sáng, Tăng màu) dùng lọc dẫn hướng giữ cạnh: hết quầng sáng quanh tóc, vai trên nền sáng.
- Mặt: giữ dáng, màu môi gốc (ô "Giữ dáng, màu môi gốc", mặc định bật; CLI --ai-lips để tắt); viền má, quai hàm lấy
  85 % từ model mặt nên không còn nhòe theo ảnh gốc; mặt nạ ghép mặt gọn hơn (nới 10 px, mềm σ 6 thay vì σ 11 × 2).
- Sửa nhãn tách vùng mặt (ParseNet): vùng "mắt, môi" trước đây gồm cả tóc và mũi, thiếu một bên lông mày và miệng.
- Giao diện: cột trái co giãn theo cửa sổ (không còn bị cắt mép), kéo vạch chia để nới rộng; thanh trên cùng có Mở ảnh,
  Xuất ảnh, cỡ xuất (Giữ nguyên, 1K 1920 px, 2K 2560 px, 4K 3840 px theo cạnh dài; CLI --size), Đóng ảnh (Ctrl+W).
- Gộp chế độ máy chủ cho điện thoại (xem mục Máy chủ cho điện thoại, docs/MAY_CHU.md).

## Thay đổi ở 0.8

huy muốn có model mới nhất, nặng cũng được, chọn tay được, app kiểm cấu hình máy và cảnh báo. Đo bằng `tools/model_bench.py` trên ảnh làm hỏng giả lập (có ảnh gốc để so), CPU 4 luồng.

Model mặt: 21 mặt căn 512 px × 3 mức hỏng kiểu GFPGAN (mờ, thu nhỏ 2/4/8 lần, nhiễu, JPEG), đầu ra thuần của model (100 % AI). IDS là độ giống người thật (ArcFace), MUSIQ là điểm cảm nhận không cần ảnh gốc (0–100).

| Model | IDS mờ nhẹ | IDS mờ vừa | IDS mờ nặng | SSIM | MUSIQ | Giây mỗi mặt (CPU) |
| --- | --- | --- | --- | --- | --- | --- |
| GFPGAN 1.4 (mặc định) | 0,957 | 0,916 | 0,643 | 0,818 | 74,6 | 1,2 |
| RestoreFormer | 0,910 | 0,881 | 0,688 | 0,744 | 73,6 | 2,6 |
| RestoreFormer++ | **0,965** | **0,932** | **0,714** | 0,777 | 72,8 | 2,6 |

RestoreFormer++ giữ đúng người hơn GFPGAN ở 15, 18 và 21 trên 21 mặt (mờ nhẹ, vừa, nặng): mờ nặng thì GFPGAN hay vẽ ra người khác, trẻ hơn. GFPGAN mịn và "đẹp" hơn một chút (SSIM, MUSIQ cao hơn), nhanh gấp đôi, nên vẫn là mặc định.

Model cả ảnh: 9 ảnh (5 có người, 4 không) × 3 kiểu hỏng (thu nhỏ 4 lần; thu 3 lần rồi phóng lại nên mờ; mờ, nhiễu, JPEG 60), chạy như app (đo độ mờ, thu nhỏ, phóng 4×, đưa về cỡ gốc).

| Model | SSIM | GMSD (thấp = tốt) | MUSIQ | Độ nét so với gốc | IDS mặt | Giây mỗi MP ảnh vào mạng (CPU) |
| --- | --- | --- | --- | --- | --- | --- |
| Real-ESRGAN compact (mặc định) | 0,848 | 0,083 | 64,6 | 0,54 | 0,783 | 7,8 |
| Real-ESRGAN x4plus (RRDB) | 0,847 | 0,081 | 70,1 | 0,69 | 0,781 | 127 |
| RealPLKSR | 0,825 | 0,081 | **72,2** | 1,19 | 0,781 | 67 |
| DAT-2 | **0,850** | **0,070** | 67,9 | 0,63 | **0,807** | 446 |

Tốc độ đo riêng bằng `tools/speed_bench.py` (CPU 4 luồng, ô 256 px, máy không chạy việc khác). Ảnh mờ được thu nhỏ trước khi vào mạng nên chạy nhanh hơn: mờ 2 lần thì chỉ còn một phần tư số điểm ảnh. App lấy các số này, nhân theo số luồng CPU hoặc chia cho tốc độ GPU (ước lượng, chưa đo trên GPU thật), để báo thời gian trước khi chạy.

- DAT-2 sát ảnh gốc nhất (GMSD tốt hơn compact ở 26 trên 27 ca, mặt giống người hơn), lông, tóc, vải ra tự nhiên; đổi lại rất chậm (gấp khoảng 57 lần compact): ảnh nét 2 MP trên CPU 4 luồng mất khoảng 15 phút (CPU nhiều nhân hơn thì nhanh hơn tương ứng), nên có GPU NVIDIA.
- RealPLKSR nét gắt nhất (MUSIQ cao nhất) nhưng làm nét quá ảnh gốc 1,2 lần, SSIM thấp nhất; nhanh hơn DAT-2 khoảng 7 lần, hợp máy không có GPU muốn ảnh "nét căng".
- Mức 6 "Cao nhất" = mức 5 (Rõ nét tối đa) với RestoreFormer++ cho mặt và DAT-2 cho cả ảnh. Tự động vẫn chỉ chọn mức 1–5.
- Kiểm cấu hình máy (`net_anh/hardware.py`): CPU, RAM, GPU, bộ nhớ GPU; ước tính thời gian cho ảnh đang mở và cảnh báo (không chặn) khi chạy model nặng trên CPU, GPU dưới 4 GB với DAT-2, RAM ít, hoặc máy có GPU NVIDIA mà bản cài chỉ chạy CPU. Quản lý model có cột ước tính thời gian trên máy đang dùng và nút tải bộ chất lượng cao nhất (~430 MB).
- DAT-2 trên GPU NVIDIA chạy bf16 nếu GPU hỗ trợ (RTX 30 trở lên), không thì fp32, chia ô 256 px (đỉnh bộ nhớ khoảng 1,1 GB mỗi ô ở fp32). Model mới lỗi trên thiết bị thì app dùng Real-ESRGAN compact và ghi lý do.
- Không thêm: CodeFormer, SUPIR, HYPIR (giấy phép cấm dùng thương mại); DiffBIR, PMRF, OSEDiff (model khuếch tán, nặng 5–12 GB, chỉ tải được từ huggingface.co, máy đo bị chặn nên chưa đo); OSDFace chưa công bố model; 4xBHI_dat2_real (bản DAT-2 khác của cùng tác giả: sát gốc hơn một chút nhưng mềm, MUSIQ 61 so với 68, độ nét 0,45 so với 0,70, chưa rõ giấy phép).
- Ước tính thời gian tính cả các bước ngoài model (tìm mặt, ghép mặt, chỉnh màu, tông): đo từng bước ở mức 5 và 6 trên ảnh 288×360 đến 1281×1600, ước tính lệch thời gian thật dưới 12 % (CPU).
- Bộ cài cho phép ghi thư mục weights cạnh NetAnh.exe, để tải thêm model cả khi chọn cài cho mọi người dùng.
- Sửa `build_windows.ps1`: chữ in ra của winget lẫn vào đường dẫn python.exe, ISCC.exe (lỗi gặp khi chạy thật trên Windows 11).

## Thay đổi ở 0.7

- Tông tươi sáng, trong trẻo (`net_anh/tone.py`, thanh Tươi sáng, mặc định 0,6) vì ảnh sau sửa trông u ám. Đo trên 15 ảnh (`tools/tone_bench.py`, ảnh gốc và 3 kiểu hỏng): độ sáng L +3,0, L mặt +6,0, độ tươi màu +13 %, mức đen sâu hơn (4,6 → 3,9), điểm cháy giữ 2,0 %. Ở mức 1: L mặt +9,9, độ tươi màu +21 %.
- Màu rực sáng khi nâng tông đi qua vai mềm thay vì cắt ở 255: ảnh tông ấm ở mức 1 trước đây cháy 23 % điểm ảnh (gốc 3,9 %), nay 4,0 %.
- Nhận hình đồ hoạ: giãn dải giá trị trước khi đếm màu, ảnh chụp tối không còn bị coi là logo. Bước tông xét đồ hoạ trên ảnh vào (ảnh ra của AI có thêm màu trung gian).
- Phóng to Tự động theo cỡ ảnh: cạnh dài dưới 1000 px phóng 4×, còn lại 2× (chọn tay 1×, 2×, 4× như cũ). Đo `tools/small_bench.py` trên 9 ảnh thu nhỏ 4 lần (75–300 px): xem cùng cỡ, ảnh 4× nét gấp khoảng 2,5 lần ảnh 2× (ảnh có mặt 0,71 so với 0,30 lần độ nét ảnh gốc, ảnh không mặt 0,39 so với 0,15), SSIM và độ giống mặt ngang nhau.
- Ảnh bị phóng to nên mờ (phóng ×2 đến ×4): AI khử mờ đưa độ nét từ 0,05–0,21 lần ảnh gốc (phóng thường) lên 0,6–1,0 ở ảnh có mặt. Mức mờ đo được thấp hơn hệ số phóng thật một chút và giữ như vậy: thử dùng đúng hệ số (×3, ×4) làm SSIM giảm 0,01–0,04.
- Mặt rất nhỏ (36–59 px): mức 3 giữ đúng người nhất (IDS 0,88–0,97); mức 4–5 nét hơn nhưng IDS giảm 0,01–0,05, nên Tự động vẫn chọn mức 3.
- Cảnh báo ảnh rất nhỏ (cạnh dài dưới 400 px). Gồm cả các bản sửa lỗi của 0.6.1.
- Một file cài đặt: `dong_goi_exe.bat` tự cài Python, Inno Setup nếu thiếu và để lại đúng một file `NetAnh-<phiên bản>-setup.exe` cạnh nó; file tạm gom vào `_build\`, model tải một lần rồi dùng lại. Bản gửi người dùng (`tools/dong_goi_nguoi_dung.py`) ngoài cùng chỉ có `TAO_FILE_CAI_DAT.bat`, `HUONG_DAN.txt` và thư mục `ma-nguon`. Phần Python đã chạy thử trong đường dẫn có dấu tiếng Việt và dấu cách.

## Thay đổi ở 0.6.1 (audit, chỉ sửa lỗi)

- Bản exe: bỏ `torch._dynamo`, `torch._inductor`, `sympy` khỏi danh sách loại trừ của PyInstaller. torchvision (facexlib dùng để dò mặt) cần chúng; thiếu thì tự kiểm khi build báo KHÔNG ĐẠT và exe không chạy được chế độ AI.
- Bộ cài CPU ra một file `setup.exe` duy nhất (trước đây luôn tách thêm file `.bin`, người chỉ tải `setup.exe` sẽ cài lỗi). Bản CUDA vẫn tách vì vượt 2 GB.
- Phóng to AI ghép từng ô thẳng vào ảnh 8 bit, không giữ tensor số thực cả ảnh ra: ảnh điện thoại 12 MP ở chế độ AI trước đây cần hơn 8 GB RAM.
- Ảnh scan xám 16 bit không còn ra trắng xoá; ảnh quá nhiều điểm ảnh báo lỗi rõ ràng thay vì làm app im lặng.
- Model: kiểm SHA-256 trên file tạm rồi mới đổi tên, dừng khi tải thiếu hoặc thừa byte.
- Xử lý thư mục: một ảnh lỗi (hết RAM, ổ đầy…) không dừng cả thư mục. Thiếu RAM ở chế độ AI thì rơi về Cải thiện nhanh.
- Nút "Tải tất cả model" chỉ tải model app cần và model đang chọn (khoảng 550 MB thay vì 1,2 GB).

## Thay đổi ở 0.6

- Thang mức nét 1–5 trên giao diện, Tự động chọn mức theo độ mờ. Đo bằng `tools/level_bench.py`: ảnh mờ nhẹ thì mức 2 (khoảng 1 giây) giữ mặt gần bằng model mặt (khoảng 7 giây); từ mờ 2× trở lên model mặt mới đáng. RRDB không tốt hơn compact ở mức mờ nào mà chậm gấp 5–15 lần, nên chỉ để chọn tay.
- Sửa phơi sáng mới (`net_anh/exposure.py`) thay Retinex trong Tự động. Retinex làm ảnh tối sáng quá tay và giảm độ giống mặt; sửa mới đưa độ sáng về gần ảnh gốc (xem số đo trong tài liệu dự án).
- Ảnh mờ rất nặng: khử mờ mạnh hơn mức hiện tại không làm ảnh tốt hơn khi đo, nên giữ mức trần và thêm cảnh báo.
