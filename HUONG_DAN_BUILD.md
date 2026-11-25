# Hướng dẫn build Beauty HDR cho Windows

Có hai cách build. Cả hai cho ra cùng một thứ: **một file** `NetAnh-<phiên bản>-setup.exe` (bản GPU NVIDIA trên 2 GB thì là thư mục `NetAnh-<phiên bản>-GPU-NVIDIA` gồm setup.exe và các file .bin). Bấm đúp để cài cho riêng người dùng, không cần quyền quản trị; mở Beauty HDR từ menu Start.

File trung gian nằm hết trong `_build\`: môi trường build `_build\venv-<cpu|cuda>`, model đã tải `_build\weights` (lần sau dùng lại), bản chạy không cần cài `_build\dist\NetAnh\NetAnh.exe`, log tự kiểm `_build\tu_kiem_<cpu|cuda>.log`. Xoá `_build` khi không cần nữa.

## Hai bản

| Bản | Dành cho | Dung lượng bộ cài (ước tính) |
|---|---|---|
| `cpu` | Mọi máy Windows 10/11 64-bit. Chân dung AI mất khoảng 5–15 giây mỗi ảnh. | khoảng 0,9–1,2 GB (có model) |
| `cuda` | Máy có GPU NVIDIA và driver mới (từ 2024 trở đi). Nhanh hơn khoảng 10–30 lần. | khoảng 3–3,5 GB, tách thành `setup.exe` và các file `.bin` |

Bản `cuda` vẫn chạy được trên máy không có GPU (tự chuyển sang CPU), chỉ là nặng hơn.

## Cách 1: build trên máy Windows của bạn

Bấm đúp `dong_goi_exe.bat` (bản gửi người dùng: `TAO_FILE_CAI_DAT.bat`). Xong thì Explorer tự mở và chọn sẵn file cài đặt. Script tự lo phần còn thiếu:

- **Python**: dùng Python 3.10–3.13 có sẵn (`py -3.12`, `py -3.11`, `py -3.13`, `python`); không có thì tự cài Python 3.12 cho riêng tài khoản (winget, không có winget thì tải từ python.org).
- **Inno Setup 6** (làm file cài đặt): không có thì tự cài (winget, không được thì tải từ jrsoftware.org và cài cho riêng tài khoản).

Đường dẫn có dấu tiếng Việt và dấu cách (ví dụ `D:\nét ảnh\...`) chạy được. Muốn chọn bản hoặc nơi để file, mở PowerShell trong thư mục mã nguồn rồi chạy:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1                 # bản CPU, kèm model, có bộ cài
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -Flavor cuda    # bản GPU NVIDIA
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -NoModels       # không kèm model (app tự tải khi cần)
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -OutDir D:\ra    # để file cài đặt vào thư mục khác
```

Script làm lần lượt các bước sau:

1. Tìm hoặc cài Python, tạo môi trường build riêng `_build\venv-cpu` hoặc `_build\venv-cuda`, không đụng tới môi trường đang dùng.
2. Cài PyTorch đúng bản và thư viện.
3. Đóng gói bằng PyInstaller theo `packaging\net_anh.spec`.
4. Tải 5 model, khoảng 620 MB, vào `_build\weights` và kiểm SHA-256 từng file (đã có thì không tải lại).
5. Chạy `NetAnh.exe --tu-kiem --can-ai` để thử chế độ Nhanh và Chân dung AI ngay trên bản đã đóng gói. Nếu không đạt thì dừng và báo lỗi.
6. Tìm hoặc cài Inno Setup, tạo file cài đặt rồi chép ra cạnh file .bat.

Lần đầu mất khoảng 20–40 phút, chủ yếu là thời gian tải PyTorch, model và nén bộ cài. Lỗi giữa chừng thì bấm lại file .bat: phần đã tải không phải tải lại.

## Cách 2: để GitHub build (không cần máy Windows)

1. Tạo một repo GitHub (để riêng tư cũng được) và đẩy toàn bộ thư mục `net-anh-pc` lên. Workflow nằm sẵn ở `.github/workflows/build-windows.yml`.
2. Vào tab **Actions**, chọn **Build Windows**, rồi bấm **Run workflow**. Chọn bản `cpu`, `cuda` hoặc `both`, và chọn có kèm model hay không.
3. Đợi khoảng 20–40 phút, sau đó tải bộ cài ở mục **Artifacts** của lần chạy (bản cpu là một file `NetAnh-<phiên bản>-setup.exe`). Lần chạy đó có kèm file `tu_kiem_*.log` cho biết kết quả tự kiểm.
4. Muốn ra bản phát hành thì đẩy tag, ví dụ `git tag v0.3.0 && git push --tags`. Cả hai bản sẽ tự build và được đăng lên **Releases**.

## Kiểm tra trên máy người dùng

- `NetAnh.exe --tu-kiem` ghi ra `tu_kiem.log` cạnh file exe. File này cho biết GPU hay CPU đang được dùng, model nào đã có, và kết quả chạy thử.
- Khi cần xem lỗi trực tiếp, build bản có cửa sổ dòng lệnh: đặt `set NET_ANH_CONSOLE=1` rồi build lại.
- `NetAnh.exe --cli anh.jpg --mode portrait` chạy chế độ dòng lệnh, giống `python -m net_anh.cli`.

## Model nằm ở đâu

Thứ tự app tìm model:

1. Biến môi trường `NET_ANH_WEIGHTS`, nếu có đặt.
2. Thư mục `weights` cạnh `NetAnh.exe`. Bộ cài chép model vào đây.
3. Nếu thư mục đó không ghi được (ví dụ khi đặt trong Program Files) và chưa có đủ model, app dùng `%LOCALAPPDATA%\NetAnh\weights`.

Bộ cài mặc định cài cho riêng người dùng vào `%LOCALAPPDATA%\Programs\NetAnh`, nên không cần quyền quản trị. ArcFace (dùng để đo) và RRDB (mạng nền cũ) không được đóng gói kèm; app chỉ tải khi thật sự cần.

## Đã thử

Gói PyInstaller đã được build và tự kiểm đạt trên Linux (máy cloud): chế độ Nhanh mất 0,05 giây, Chân dung AI 1,3 giây trên CPU với ảnh thử. Ở 0.7, phần Python (tải và kiểm model, tự kiểm có AI, đọc và ghi ảnh) đã chạy thử trong thư mục có dấu tiếng Việt và dấu cách. Bản 0.7.0 đã chạy thật trên Windows 11 (thư mục có dấu cách và dấu ngoặc): bước 1–5 đạt, kể cả tự kiểm bản đóng gói trên CPU. Lỗi ở bước tự cài Python và Inno Setup bằng winget (chữ winget in ra lẫn vào đường dẫn python.exe, ISCC.exe) đã sửa trong `build_windows.ps1`: output của winget đưa thẳng ra màn hình, cài xong thì tìm lại file exe theo đường dẫn chuẩn.
