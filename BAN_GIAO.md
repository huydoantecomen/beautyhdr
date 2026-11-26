# BÀN GIAO DỰ ÁN BEAUTY HDR (viết 2026-11-26, cho phiên Claude tiếp quản)

Đọc hết file này trước khi đụng mã. Mọi số liệu ở đây lấy từ project memory, các thread và kho GitHub tại thời điểm viết;
nếu thấy khác, tin mã và git log hơn file này. File không chứa khoá API, mật khẩu hay token, và cũng không được thêm vào.

---

## 1. App là gì, cho ai

- **Beauty HDR** (tên cũ "Nét Ảnh"): làm nét, phục hồi, phóng to ảnh chân dung và ảnh cũ. Trọng tâm: ảnh thẻ, chân dung
  chụp điện thoại của người Việt, ảnh in cũ chụp lại. Chạy trên máy (offline), ảnh không gửi đi đâu trừ khi người dùng
  chọn máy chủ.
- Chủ dự án: **huy** (GitHub `huydoantecomen`). Viết tiếng Việt, không rành kỹ thuật sâu: **mọi trả lời, hướng dẫn,
  tên nút đều bằng tiếng Việt**, hướng dẫn phải nói cụ thể thư mục nào, bấm file nào.
- Hai sản phẩm:
  - **PC** (Python + PySide6 + PyTorch): nơi thử thuật toán và đo, cũng là bản người dùng chính. Kho này.
  - **Android** (Kotlin + Compose + OpenCV + ONNX Runtime): kho **riêng tư** `huydoantecomen/Creator-ID-360`
    (phiên hiện tại không có quyền đọc kho đó; xin huy thêm vào nếu cần).
- Hướng kết hợp (huy chọn 2026-11-25): Android làm nhẹ trên máy, "Chất lượng cao nhất" gửi lên máy chủ. Máy chủ hiện là
  PC của huy (`NetAnh.exe --may-chu`), sau có thể là GPU thuê. **Thuê VPS/GPU chỉ làm khi huy bảo.**

## 2. Kiến trúc và thư mục (PC)

```
net_anh/                 gói chính (bản thường)
  pipeline.py            Enhancer.process(): đọc/kiểm ảnh → phân tích → mô hình nền (Real-ESRGAN… chia ô) → mặt
                         (GFPGAN/RestoreFormer++, ghép ParseNet) → màu/tông → khử bóng → tông mẫu (look) → đổi nền;
                         Report ghi model thật đã chạy, fallback, thời gian GPU/CPU (model_s)
  analysis.py            đo mờ/nhiễu/sáng, đếm mặt, router preset Tự động
  levels.py              thang mức nét Tự động + 1–6 (1 không AI … 5 Rõ nét tối đa, 6 Cao nhất = RF++ + DAT-2)
  settings.py            Settings (mọi thông số), preset (auto, max, portrait, restore, low_light…)
  wb.py phone.py color.py tone.py exposure.py classic.py   màu, bộ lọc theo hãng, tông tươi sáng, phơi sáng, cổ điển
  facestyle.py           preset theo đối tượng (tuổi/giới tính), PresetSet, file .netanh.json (VERSION 3), thư viện
                         ~/.net_anh/presets; bộ có sẵn: default, trong-sang, da-dep-the + 4 bộ ảnh thẻ (từ 0.15.2)
  subject.py             đoán tuổi/giới tính (ONNX Cydral, CC0) bằng cv2.dnn
  look.py                tông mẫu "Trong sáng", "Da đẹp ảnh thẻ" (không model), có chốt chặn ảnh không phải ảnh thẻ
  deshine.py             khử bóng loáng da (ô tích, mặc định tắt)
  backdrop.py            đổi màu nền đồng màu (không model); làm lại mặt nạ ở 0.15.2
  restore.py colorize.py khôi phục ảnh cũ: dò xước U-Net + LaMa, màu phai, scene_vivid, tô màu DDColor (scene_ai)
  gpu_profile.py device.py hardware.py gpu_bench.py batch.py   nhận card, chọn fp32/fp16/bf16 có chạy thử, cắt ô theo
                         VRAM, đo tốc độ (--do-gpu), làm thư mục song song
  du_lieu/               model_profiles.json (khai báo model ↔ phần cứng), so_do_gpu.json (kho số đo thật),
                         anh-the-4-nhom.netanh.json (preset ảnh thẻ có sẵn)
  weights.py drive.py drive_models.json   manifest model (SHA-256, giấy phép), tải từ Google Drive, dự phòng GitHub gốc
  server.py server_gui.py máy chủ HTTP API v1 cho Android (cổng 8765, tìm máy UDP 8766, mã ghép nối 6 số)
  providers.py           (chỉ có ở PR #4) cổng API ngoài fal.ai / Stability / Replicate, gọi từ máy chủ
  gui.py                 giao diện Qt hai chế độ Đơn giản / Chuyên sâu (một file lớn)
  cli.py                 dòng lệnh, cùng thông số với GUI
  archs/                 kiến trúc mạng vendor (GFPGAN, RestoreFormer, Real-ESRGAN, DDColor…) kèm LICENSE
net_anh_dev/             (chỉ có ở PR #3) bản nhà phát triển: xuất prompt ChatGPT/Gemini/Grok. CẤM lọt vào bản thường
launcher/                BeautyHDR.exe một file tự tải app + model (chỉ làm khi huy yêu cầu, xem mục 4)
packaging/               net_anh.spec (PyInstaller), NetAnh.iss (Inno Setup), build_windows.ps1, cap_nhat.ps1,
                         nguoi_dung/ (.bat + HUONG_DAN*.txt đưa vào gói người dùng), cloud/Dockerfile (chưa chạy thử)
presets/                 file preset ảnh thẻ để nạp tay/chia sẻ
tools/                   đo chất lượng, bench, dong_goi_nguoi_dung.py (danh sách file gói người dùng), make_update_json…
tests/                   pytest
.github/workflows/       build-windows.yml (tay), release-exe.yml (tag v*), kiem-cap-nhat.yml (kiểm CAP_NHAT trên Windows)
docs/                    MAY_CHU.md (máy chủ), API_NGOAI.md (PR #4)
```

Model dùng (giấy phép cho phép thương mại): GFPGAN 1.4, RestoreFormer/RestoreFormer++, Real-ESRGAN compact (mặc định
nền) / x2plus / x4plus, DAT-2 và RealPLKSR của Philip Hofmann (CC-BY-4.0, **phải ghi tên tác giả**), RetinaFace,
ParseNet (facexlib), Cydral tuổi/giới tính (CC0), LaMa, U-Net Microsoft dò xước, DDColor (912 MB, chỉ tải khi bật).
**Không dùng** CodeFormer, SUPIR, HYPIR (cấm thương mại), InsightFace genderage/MiVOLO/FairFace (chỉ để đo nội bộ).

Nhãn ParseNet facexlib đúng: 0 nền, 1 da, 2 mũi, 3 kính, 4–5 mắt, 6–7 mày, 8–9 tai, 10 miệng, 11–12 môi, 13 tóc,
14 mũ, 15 khuyên tai, 16 vòng cổ, 17 cổ, 18 áo (≤ 0.8.0 từng dùng nhầm thứ tự CelebAMask).

### Android (kho Creator-ID-360)
- Module `core` (Kotlin JVM + OpenCV): port thuật toán PC, kiểm khớp từng điểm ảnh bằng ảnh mẫu do `tools/make_*_fixtures.py`
  của PC tạo. 85 test lõi: `./gradlew -PcoreOnly :core:test` (chạy được trên cloud).
- Module `app` (Compose): **chỉ build được qua GitHub Actions** của kho đó (cloud chặn dl.google.com). Đẩy main là tự build,
  đăng Releases v<versionName>; APK > 100 MB nên tải bằng Releases API
  (`gh api -H "Accept: application/octet-stream" repos/huydoantecomen/Creator-ID-360/releases/assets/<id>`).
- Không có ParseNet trên điện thoại: mặt nạ da/môi/tóc dựng từ 5 điểm mốc RetinaFace + màu.
- Khoá ký cố định từ 0.9.0 (cài đè giữ dữ liệu; 0.8.1 trở về trước phải gỡ một lần).

## 3. Trạng thái các bản và PR (2026-11-26 ~16h UTC)

| Bản | Ở đâu | Trạng thái |
|---|---|---|
| PC 0.14.1 | `main` + Release v0.14.1 (exe) | bản phát hành exe mới nhất |
| PC 0.14.2 | PR #1, nhánh `claude/project-thread-ampo6r` → `main` | chờ huy merge |
| PC 0.15.0 → 0.15.2 | PR #2, nhánh `claude/project-thread-r8nke4` (base `ampo6r`) | 0.15.2 commit 1093c0b; gói zip đã giao; chờ huy thử + merge |
| Bản dev 0.15.0 | PR #3, nhánh `claude/project-thread-fzn3gn` (base `r8nke4` ở 062e49d) | chờ merge, chưa thử Windows |
| PC 0.16.0 (API ngoài) | PR #4, nhánh `claude/project-thread-jx4n06` (base `r8nke4` ở 062e49d) | chờ merge, chưa thử khoá thật |
| PC 0.16.0 (Nguồn xử lý + màu cảnh ở Đơn giản) | PR #6, nhánh `claude/project-thread-9g3kiv` (base `r8nke4` ở 0.15.1, đã gộp PR #4) | gói 0.16.0 đã giao; chờ merge, chưa thử Windows/khoá thật |
| Tài liệu bàn giao | PR #5, nhánh `claude/project-thread-81lvh2` → `main` | chỉ tài liệu |
| Android 0.10.1 | main Creator-ID-360 (versionCode 15) | đã giao Full + Cập nhật, chưa thử máy thật |
| Preset ảnh thẻ (file) | nhánh `claude/project-thread-tso5vb`, chưa PR | đã gộp vào PR #2 ở 0.15.2; nhánh này coi như cũ |

Tóm tắt từng PR:
- **#1 (0.14.2)**: sửa loang màu look.py, scene_vivid, Tô màu AI DDColor, `restore.ai_color_weight`, thư mục presets/.
- **#2 (0.15.x)**: gpu_profile, cắt ô theo VRAM, batch song song, gói `cuda_old` cho GTX 10/900, dữ liệu card×model tách
  file; 0.15.1 khu tiến độ + tách thời gian GPU/CPU + cảnh báo chạy CPU + gói Cập nhật kiểu "chạy thẳng"; 0.15.2 làm lại
  mặt nạ Đổi nền + 4 preset ảnh thẻ có sẵn trong app.
- **#3 (dev)**: gói `net_anh_dev/`, `tests/test_dev_exclusion.py`, `SKIP_DIRS` có `net_anh_dev`, spec excludes.
- **#4 (0.16.0)**: `net_anh/providers.py`, server `engine=<id>&consent=1&guard=0|1`, kiểm IDS ArcFace (< 0,70 thì xử lý
  bằng pipeline máy này), `docs/API_NGOAI.md`. Khoá chỉ đọc từ biến môi trường / `~/.net_anh/providers.json` trên máy chủ.

- **#6 (0.16.0)**: `net_anh/sources.py`, `gui_sources.py`: ô Nguồn xử lý Trên máy này / Máy chủ GPU (ghép nối mã 6 số,
  gửi `settings=<JSON>`) / API ngoài (khoá người dùng nhập, hỏi đồng ý mỗi lần, kiểm giữ mặt); Đơn giản > Ảnh cũ có Nâng
  màu cảnh + Tô màu AI. **Đã chứa PR #4** nhưng **thiếu 0.15.2** (Đổi nền mới, 4 preset ảnh thẻ).

**Thứ tự merge (bắt buộc):** #1 → main; đổi base #2 sang main rồi merge; đổi base #3 và #6 sang main (#4 đã nằm trong #6, đóng #4
hoặc merge #4 trước #6), **merge main vào từng nhánh** (#3 thiếu 0.15.1/0.15.2, #6 thiếu 0.15.2), giải xung đột (`net_anh/__init__.py` version, `tools/dong_goi_nguoi_dung.py`,
`gui.py`, README), chạy lại test rồi merge. Phiên bản sau cùng: 0.16.x. **Không tự merge**: chỉ merge khi huy bảo rõ.

Thread khác đang chạy lúc viết (có thể đã đẩy thêm commit, xem `git log` các nhánh):
- "Đưa tính năng còn thiếu vào sẵn": đã ra PR #6 (0.16.0). Còn thiếu: Android chưa có ô nguồn/đồng ý; làm thư mục qua
  máy chủ; small_face_sr, fade_style, max_faces chỉ có ở CLI.
- "Audit chức năng ảnh hưởng kết quả": chứng minh từng thông số Chuyên sâu 0.15.1 thật sự đổi kết quả render (~200 lượt),
  sửa lỗi tìm được, mở PR.
- "Tối ưu app cho mọi dòng card": vừa giao 0.15.2, chờ huy thử.

## 4. Chạy, test, build, đóng gói, phát hành

### Chạy từ mã nguồn
- Windows (huy dùng cách này): `cai_dat.bat` (tạo `.venv`, cài torch đúng GPU) rồi `chay.bat`. Máy chủ: `may_chu.bat`.
- Linux: `./cai_dat.sh; . .venv/bin/activate; python -m net_anh` (GUI) hoặc `python -m net_anh.cli anh.jpg --level 5`.
- Model mặc định ở `%USERPROFILE%\.net_anh\weights` (hoặc biến `NET_ANH_WEIGHTS`).

### Test (cloud Linux, không GPU)
```bash
pip install torch torchvision                # từ PyPI; index download.pytorch.org bị chặn
pip install -r requirements.txt pytest scikit-image opencv-python-headless
sudo apt-get install -y libegl1 libgl1 libxkbcommon0   # cho test GUI (QT_QPA_PLATFORM=offscreen)
python -m pytest tests -q
python -m pytest net_anh_dev/tests -q        # chỉ trên nhánh PR #3
```
Số test đã báo: PR #1 145 qua; PR #2 184 (bộ đầy đủ); PR #3 254 qua + 6 bỏ qua; PR #4 253 qua + 13 bỏ qua. GPU được giả
lập bằng thiết bị "cuda" chạy trên CPU; **chưa có lần đo nào trên GPU thật**. Khi sửa, chỉ chạy lại test hỏng.

### Build exe Windows
- Cloud **không** build được exe (Wine bị chặn, pwsh không tải được). Chỉ build qua GitHub Actions (`build-windows.yml`,
  chạy tay; flavor cpu / cuda / cuda_old / both) hoặc trên máy huy (`TAO_FILE_CAI_DAT.bat` → `packaging/build_windows.ps1`).
- Trên Linux có thể build PyInstaller để tự kiểm: `NetAnh --tu-kiem --can-ai` phải ĐẠT trước khi giao.
- Mỗi lần nâng torch/torchvision phải build thử (đã từng vỡ: torchvision ≥ 0.29 đổi tên `_C_stable`, spec đã gom).

### Đóng gói: QUY ƯỚC 4 GÓI (huy chốt 2026-11-25, đang áp dụng)
Mỗi lần cập nhật chỉ đưa 4 gói vào `/mnt/project-files/` (thư mục dự án):

| Gói | Tên file | Bên trong |
|---|---|---|
| PC Full | `BeautyHDR-<v>-TronBo.zip` | `BeautyHDR-<v>/TAO_FILE_CAI_DAT.bat`, `HUONG_DAN.txt`, `ma-nguon/` (không kèm model, tự tải; không kèm APK) |
| PC Cập nhật | `BeautyHDR-<v>-CapNhat.zip` | `CAP_NHAT.bat`, `CAP_NHAT_CHAY_THANG.bat`, `CHUYEN_SANG_GPU.bat`, `HUONG_DAN_CAP_NHAT.txt`, `ma-nguon/` (bỏ tests/tools) |
| Android Full | `BeautyHDR-Android-<v>-Full.zip` | 2 APK (arm64-v8a, armeabi-v7a), `DOC_TRUOC.txt`, `android-ma-nguon/` |
| Android Cập nhật | `BeautyHDR-Android-<v>-CapNhat.zip` | APK arm64 + `DOC_TRUOC.txt` (nói rõ cài đè từ bản nào) |

- Danh sách file gói PC lấy từ `tools/dong_goi_nguoi_dung.files()` (bỏ `.git`; trên PR #3 trở đi `SKIP_DIRS` có `net_anh_dev`).
  Script dựng TronBo/CapNhat của các thread trước nằm ở scratchpad, đã mất: mở zip 0.15.2 trong thư mục dự án làm mẫu
  cấu trúc rồi dựng lại.
- Gói Cập nhật giải nén ở đâu cũng được; `CAP_NHAT.bat` gọi `ma-nguon/packaging/cap_nhat.ps1`: tự tìm bản cũ, hỏi
  Yes/No/Cancel, hộp chọn thư mục, ghi đè mã (giữ `_build`, `.venv`, model, lịch sử, preset). Kiểu "chạy thẳng" (có `.venv`,
  chưa cài setup.exe, **đúng máy huy**) chỉ ghi đè mã, pip -r requirements, kiểm import, mở `chay.bat`.
- Bản dev: `python net_anh_dev/dong_goi.py <thư mục>` → `BeautyHDR-<v>-dev-TronBo.zip` (gói thứ 5, chỉ khi có thay đổi dev).
- Không tạo lại các gói kiểu cũ (`-Windows.zip`, trọn bộ kèm APK).

### Model
- 16 file model trên Google Drive của huy (thư mục chia sẻ "bất kỳ ai có liên kết"; id thư mục nằm trong
  `net_anh/drive_models.json`) và bản sao trên Telegram của huy. App tải từ Drive, kiểm SHA-256, dự phòng nguồn GitHub
  gốc. Cloud không vào được drive.google.com nên chưa kiểm phần liệt kê Drive với Google thật.

### Phát hành exe (BeautyHDR.exe một file tự cập nhật)
- Exe đọc `update.json` ở Release mới nhất của kho **công khai** này. `release-exe.yml` chạy khi đẩy tag `v<version>`
  (trùng `net_anh/__init__.py`) hoặc chạy tay. Đẩy mã lên main **không** đủ để exe nhận bản mới.
- Thực tế: phiên Claude đẩy tag hay lỗi, workflow_dispatch hay trả 500/502 (thử lại). Release đã có: v0.14.0, v0.14.1.
- **Chỉ phát hành exe khi huy yêu cầu**; đừng tự đăng hay nhắc huy về exe.

## 5. Các quyết định của huy (không tự đổi)

- **Ngôn ngữ**: tiếng Việt mọi nơi. Huy không thích chú thích dài trên giao diện, giải thích để trong tooltip.
- **GTX 16xx luôn fp32** (fp16 ra nền đen trên GTX 1660 Ti của huy; Turing không Tensor Core). GTX 10/900 cũng fp32.
- **Hai chế độ giao diện** Đơn giản / Chuyên sâu kiểu Lightroom (xám trung tính, bo 2 px, ô tích thay công tắc; bản "sặc
  mùi AI" bị chê). Một thanh mức nét chung "Tự động, 1 Nhẹ … 6 Cao nhất" ở cả hai chế độ. Một bộ thông số chung.
- **Tự động gặp chân dung dùng "Rõ nét tối đa"** (kiểu Remini); huy chấm mù ưa 70 % AI cho mặt. Tông tươi sáng mặc định 0,6.
- **Tông mẫu** Trong sáng / Da đẹp ảnh thẻ; huy thích da hồng hào. Đổi màu theo vùng phải có chốt chặn khi ảnh không
  phải ảnh thẻ (bài học lỗi loang 0.14.0).
- **Preset theo đối tượng** (trẻ em/người lớn/người già, nam/nữ) xuất/nạp `.netanh.json` để chia sẻ; đổi định dạng thì tăng
  VERSION và vẫn đọc bản cũ. **Người già không làm trẻ hoá**: giữ nếp nhăn, đồi mồi; kiểm bằng đo tuổi trước/sau.
- **Khử bóng loáng da** là ô tích tùy chọn, mặc định tắt.
- **Chỉ 4 gói phát hành** (mục 4). Exe một file chỉ khi huy yêu cầu.
- **Bản dev tách riêng**: xuất prompt chỉ có trong `net_anh_dev`; bản thường (PC Full, Cập nhật, exe, Android) không được có
  mã, menu hay chữ "ChatGPT/Gemini/Grok/xuất prompt". `tests/test_dev_exclusion.py` chặn. Kho công khai nên mã dev vẫn
  đọc được trên GitHub.
- **API ngoài** chỉ dùng thử/dự phòng, luôn gọi qua máy chủ của mình, cần người dùng đồng ý; khoá nằm trên máy chủ, không
  bao giờ trong APK hay gói PC.
- Thông số GPU đo từ máy thật sẽ dùng để phát triển tiếp khi thêm model nặng: giữ dữ liệu tách khỏi mã
  (`du_lieu/model_profiles.json`, `so_do_gpu.json`, `tools/nhap_so_do.py`).

## 6. Lỗi đã biết và việc đang dở

- **Đổi nền lẹm, răng cưa viền tóc/vai** (huy báo trên 0.15.1): đã làm lại ở 0.15.2, mới kiểm trên ảnh của huy và ảnh
  dựng; **chờ huy thử trên Windows**. Ảnh trước/sau: `/mnt/project-files/doi-nen/truoc-sau-0.15.2.png`.
- **4 preset ảnh thẻ có sẵn**: xong ở 0.15.2 (Nam trẻ, Nữ trẻ, Nam lớn tuổi, Nữ lớn tuổi; bản nạp tay trùng thì dọn, đã sửa
  thì giữ thêm "(của bạn)"). Còn thiếu ảnh thẻ gốc nam trẻ, nam 60+, nữ 60+ của huy để tinh chỉnh.
- **Đưa tính năng vào chế độ Đơn giản**: xong ở PR #6 (0.16.0), chưa thử Windows.
- **Audit thông số Chuyên sâu**: thread khác đang làm.
- **File đo GPU của huy**: chờ huy chạy Trợ giúp → Đo tốc độ card (`NetAnh.exe --do-gpu`), gửi `BeautyHDR-do-gpu-*.txt`
  để thay số ước lượng (SPEEDUP, ACT_BYTES_PER_PX) bằng số thật (`tools/nhap_so_do.py`).
- Chưa thử trên phần cứng thật: GPU (mọi dòng), AMD/Intel, khoá API thật, APK Android 0.10.x trên máy huy (Honor
  VKI-AN00, cỡ chữ lớn).
- Ảnh phong cảnh "không đạt": chờ huy gửi ảnh lỗi cụ thể. Compact vẽ gạch như sơn dầu; DAT-2 tự nhiên hơn nhưng chậm 8×.
- look.py: ánh sáng xiên mạnh làm sáng lệch nửa mặt; ảnh đen trắng bị nhuộm màu da (chưa chặn).
- DDColor còn sai vài màu (áo vàng ra kem, quần be ra trắng, da nhợt hồng).
- Android còn nợ: chưa có ô khử bóng; máy chủ PC chưa nhận look/scene_vivid từ Android; token máy chủ chưa mã hoá Keystore.
- Bộ lọc theo hãng điện thoại chưa hiệu chỉnh trên ảnh thật (cần 3–5 ảnh gốc mỗi hãng).
- Lộ trình 3 giai đoạn theo app tương tự (doc https://claude.ai/artifact/5BWRcRLZhudchsd2GNidqB): chờ huy chốt thứ tự,
  cách thu tiền, Google Play, ngân sách.

## 7. Môi trường

- **Máy huy**: Windows 11, GPU **GTX 1660 Ti 6 GB**, Python 3.14 (`.venv` trong `ma-nguon`), chạy thẳng bằng `chay.bat`,
  thư mục kiểu `D:\hight net\BeautyHDR-0.15.0-dev-TronBo\BeautyHDR-0.15.0-dev\ma-nguon` (đường dẫn có dấu cách, có thể có
  dấu tiếng Việt), model ở `C:\Users\<user>\.net_anh\weights`. Điện thoại Honor VKI-AN00.
- **Cloud Claude**: Linux, CPU 4 luồng, **không GPU**, không Wine, không pwsh. Bị chặn: dl.google.com (Android SDK),
  download.pytorch.org (dùng torch PyPI), drive.google.com, wikimedia; blob artifact của Actions bị proxy chặn.
  huggingface.co đã được mở (dùng cho DDColor). Ảnh mẫu tải được qua raw.githubusercontent / git clone.
- Thư mục dự án chung `/mnt/project-files/` (nhiều thread cùng ghi): đọc lại ngay trước khi ghi, không `rm -rf`, không ghi
  đè gói của thread khác, chọn số bản sau khi xem file đã có.

## 8. Bẫy đã gặp (đừng lặp lại)

1. **Hai bản "0.14.1"**: zip `net-anh-pc-0.14.1.zip` (sửa loang look.py) khác `main` 0.14.1 (GPU, launcher, Drive). Từ
   0.14.2 lấy **kho GitHub** làm gốc; không đóng gói từ các zip `net-anh-pc-*.zip` cũ.
2. PR #3, #4 dựng trên 0.15.0 (062e49d), **thiếu 0.15.1/0.15.2**; PR #6 dựng trên 0.15.1, **thiếu 0.15.2**; phải merge nhánh mới vào trước khi merge/đóng gói.
3. PowerShell: gọi script bằng mảng tham số truyền theo vị trí → `-CapNhat` rơi vào `-Flavor`. Luôn splat bảng băm
   `@{CapNhat=$true}`. Trong hàm PS mọi output không bắt đều thành giá trị trả về (dùng `| Out-Host`). Script inline của
   workflow không có BOM → so mẫu ASCII.
4. PyInstaller **bỏ lặng** file lỗi cú pháp; `net_anh.spec` có `_check_own_modules()` dừng build. Không thêm lại
   `torch._dynamo/_inductor/sympy` vào EXCLUDES (làm hỏng torchvision/facexlib).
5. `torch.cuda.is_bf16_supported()` trả True giả trên Turing: chỉ dùng bf16 khi compute capability ≥ 8. DAT-2 fp16 ra NaN.
   GFPGAN fp16 lệch nặng, lỗi với channels_last.
6. PyTorch PyPI mới (cu128+) chỉ có sm_75+: GTX 10/900 cần gói `cuda_old` (cu126).
7. "Chọn GPU mà render chạy CPU": model chạy GPU đúng, nhưng màu/tông/khử bóng/ghép mặt chạy CPU và chiếm phần lớn thời
   gian; overlay GPU của Windows không đếm CUDA. Đừng chẩn đoán nhầm.
8. Console Windows cp1252 làm sập khi in tiếng Việt: `net_anh/__init__._utf8_streams`, `.bat` có `chcp 65001` + `PYTHONUTF8`.
9. ArcFace IDS **không** bắt được lỗi "mặt nạ"/vạch ngang trên ảnh nhóm: phải xem ảnh. PSNR ưu ái ảnh mờ: đo thêm SSIM/GMSD.
10. Tính năng đổi màu theo vùng phải thử trên ảnh ngoài trời nền phức tạp trước khi giao. DDColor phải áp ab **ở cuối**
    pipeline; ảnh còn màu (sắc độ Lab TB > 10) bỏ qua tô màu AI.
11. Nhiều thread từng ghi đè gói của nhau (sự cố 0.10.1/0.10.2). Mỗi thread dùng nhánh riêng; không đẩy lên nhánh của PR khác.
12. Đừng hứa "đã thử trên Windows/GPU" khi chỉ chạy trên cloud; luôn ghi rõ phần chưa thử.

## 9. Cách làm việc với huy

- Hỏi ít, làm theo cách hiểu hợp lý nhất rồi nói rõ đã hiểu thế nào. Báo kết quả ngắn, bắt đầu bằng kết quả.
- Mỗi lần giao bản mới: đưa đúng 4 gói, hướng dẫn cụ thể (giải nén ở đâu, bấm file nào, chọn thư mục nào), ghi phần chưa thử.
- Đổi mặc định chất lượng chỉ khi huy chấm (trang A/B chấm mù) hoặc số đo tốt hơn rõ.
- Không merge PR, không phát hành, không thuê GPU, không gửi gì ra ngoài khi huy chưa bảo.
- Tài liệu tham khảo: doc kiến trúc/đo https://claude.ai/artifact/SpVNhxRkDSyLWP7ks4i3Fr, kịch bản card × model
  https://claude.ai/artifact/ERXXmr2zJz4MG3fLznDSo1, demo giao diện PC https://claude.ai/artifact/A4TnL3kGdk1MD4zLdZfWSX,
  demo mobile https://claude.ai/artifact/2kiyjY2B5wY6wrZ72udNr4.
