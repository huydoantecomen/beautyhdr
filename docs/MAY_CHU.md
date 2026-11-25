# Máy chủ cho điện thoại

App Nét Ảnh trên Android xử lý nhẹ ngay trên máy (Real-ESRGAN compact). Bật **Chất lượng cao nhất** thì ảnh được gửi
tới máy tính chạy Nét Ảnh (GPU, model nặng nhất: RestoreFormer++ cho mặt, DAT-2 cho cả ảnh), xử lý xong trả về điện
thoại. Máy tính không bật, mất wifi, hay máy chủ lỗi thì điện thoại **tự xử lý trên máy** và báo lý do.

## Cài lần đầu (PC Windows)

1. Mở **Nét Ảnh - Máy chủ cho điện thoại** trong menu Start (bản cài), hoặc `may_chu.bat` (bản mã nguồn), hoặc
   `NetAnh.exe --may-chu`.
2. Windows hỏi quyền tường lửa: tick **Mạng riêng (Private)**, bấm Cho phép. Bộ cài chạy với quyền quản trị thì đã tự
   mở cổng 8765 (TCP) và 8766 (UDP) cho mạng riêng.
   Nếu wifi nhà đang để kiểu **Công cộng (Public)**: Cài đặt › Mạng và Internet › Wi-Fi › tên wifi › chọn **Riêng tư**.
3. Bấm **Tải sẵn model Cao nhất (~450 MB)** một lần để lần gửi ảnh đầu không phải chờ tải.
4. Cửa sổ hiện **địa chỉ** (vd. `192.168.1.10:8765`) và **mã ghép nối 6 số**.

## Ghép nối điện thoại

1. App Nét Ảnh › chọn ảnh › thẻ **Chất lượng cao nhất** › **Cài đặt máy chủ**.
2. **Tìm máy tính trong wifi** (hoặc gõ địa chỉ trên cửa sổ máy chủ) › nhập mã 6 số › **Ghép nối**.
3. Mỗi mã chỉ dùng được một lần (ghép xong mã tự đổi). Sai 5 lần thì khoá 60 giây.
   **Huỷ mọi điện thoại đã ghép** trên cửa sổ máy chủ để thu hồi quyền.

## Dùng khi ở ngoài (không cùng wifi)

- **Tailscale** (khuyên dùng, miễn phí cho cá nhân): cài Tailscale trên PC và điện thoại, đăng nhập cùng tài khoản.
  Cửa sổ máy chủ hiện thêm địa chỉ `100.x.x.x:8765`: nhập địa chỉ đó vào app. Kết nối được mã hoá sẵn, không cần mở cổng
  modem.
- **Cloudflare Tunnel**: `cloudflared tunnel --url http://localhost:8765` cho địa chỉ `https://….trycloudflare.com`,
  nhập nguyên địa chỉ đó vào app. Địa chỉ đổi mỗi lần chạy lệnh (trừ khi tạo tunnel có tên).
- Không nên mở cổng 8765 trên modem ra Internet: trong mạng nhà máy chủ dùng HTTP thường (không mã hoá).

PC phải bật và để cửa sổ máy chủ mở. Tắt chế độ ngủ khi cắm điện nếu muốn dùng lúc ở ngoài.

## Tốc độ

Đo trên máy cloud không GPU (CPU 4 luồng, chỉ để biết thứ tự), mức 6: ảnh 336×434 một mặt ra 1344×1736 mất 40 s;
ảnh 395×400 có 5 mặt ra 1580×1600 mất 110 s; ảnh 480×241 không mặt (chỉ DAT-2) 63 s. GTX 1660 Ti nhanh hơn nhiều lần nhưng chưa đo thật: huy chạy thử rồi báo số giúp.
Ảnh điện thoại 12 MP phóng 2× rồi thu về cạnh dài 4096 px (giới hạn RAM điện thoại).

## Dòng lệnh

    python -m net_anh.server                  # cổng 8765, in địa chỉ và mã ghép nối
    python -m net_anh.server --prepare        # tải sẵn model mức 6 rồi chạy
    python -m net_anh.server --new-code       # đổi mã ghép nối
    python -m net_anh.server --unpair-all     # huỷ mọi điện thoại đã ghép
    NetAnh.exe --may-chu --dong-lenh          # như trên, từ bản exe (không mở cửa sổ)

Dữ liệu ghép nối: `~/.net_anh/server.json` (mã hiện tại, token đã băm SHA-256, tên điện thoại). Ảnh chỉ nằm trong RAM,
file tạm xoá ngay sau khi đọc, kết quả xoá sau 15 phút.

## Chuyển sang GPU thuê sau này (RunPod, Modal…)

App điện thoại không phải sửa: chỉ đổi địa chỉ máy chủ sang `https://…` của GPU thuê. Trên máy thuê chạy đúng lệnh
`python -m net_anh.server --no-pair --no-discovery --prepare` với biến môi trường `NET_ANH_TOKENS=<token dài ngẫu
nhiên>` (tắt ghép nối bằng mã, chỉ nhận token đặt sẵn), sau HTTPS của nhà cung cấp. API tách gửi việc / hỏi trạng
thái / tải ảnh nên không vướng giới hạn thời gian mỗi yêu cầu (Cloudflare 100 s). `packaging/cloud/Dockerfile` là
bản mẫu, **chưa chạy thử**. Lúc đó cần thêm: ô nhập token trên app (hiện chỉ ghép bằng mã), giới hạn số ảnh mỗi ngày.

# API v1

Mọi lỗi trả JSON `{"error": "thông báo tiếng Việt", "code": "..."}`. Xác thực: `Authorization: Bearer <token>`.

| Yêu cầu | Mô tả |
|---|---|
| `GET /api/v1/info` | Không cần token. `{"app":"netanh","api":1,"version","name","device","gpu","busy","queue","port","max_upload_mb","levels","pairing"}` |
| `POST /api/v1/pair` | `{"code":"123456","device":"Pixel 8"}` → `{"token","name"}`. 403 `wrong_code`, 429 `locked`, 403 `pairing_disabled` |
| `GET /api/v1/ping` | Kiểm tra token: 200 kèm info, hoặc 401 `unauthorized` |
| `POST /api/v1/jobs?level=6&phone=auto&upscale=0&max_side=4096&quality=94` | Thân là file ảnh (JPG, PNG, WebP, BMP, TIFF; nhận theo nội dung, không theo Content-Type; tối đa 40 MB). → 202 `{"id","state":"queued","position"}`. 415 `bad_image`, 413 `too_large`, 503 `queue_full` (8 việc) |
| tham số thêm (0.10.2, `info.features`) | `restore=1` khôi phục ảnh cũ (xoay, vá xước/bụi bằng LaMa, màu phai); `bg=RRGGBB` đổi nền đồng màu; `subject=auto\|child\|man\|woman\|elder\|adult\|off`; `styles=<JSON>` kiểu mặt của bộ preset trên điện thoại (kiểm và kẹp như file .netanh.json, tối đa 8 KB). Sai → 400 `bad_restore`, `bad_bg`, `bad_subject`, `bad_styles` |
| `GET /api/v1/jobs/<id>` | `{"id","state","progress","position","elapsed","error"?,"report"?}`; state: `queued`, `running`, `done`, `error`, `cancelled` |
| `GET /api/v1/jobs/<id>/result` | `image/jpeg`; 409 `not_ready` nếu chưa xong |
| `DELETE /api/v1/jobs/<id>` | Huỷ (việc đang chạy dừng ở lần kiểm huỷ kế tiếp) |

Tham số: `level` = `auto` hoặc 1–6 (như thang mức nét của PC; 6 = RestoreFormer++ + DAT-2); `phone` = khoá hãng
(`apple`, `samsung`, `xiaomi`, `oppo`, `vivo`, `huawei`, `google`, `camera`, `generic`, `off`) hoặc `auto` (đọc EXIF);
`upscale` = 0 (tự động), 1, 2, 4; `max_side` = cạnh dài tối đa ảnh trả về (0 = không giới hạn).

`report` khi xong: `mode`, `device`, `seconds`, `faces_found`, `faces_restored`, `notes`, `width`, `height`,
`input_width`, `input_height`, `level`, `face_model`, `bg_model`, `phone`, `phone_desc`, `reasons`, `analysis`,
`server`, `summary`.

Tìm máy tự động: điện thoại gửi gói UDP `NETANH_DISCOVER_V1` tới cổng 8766 (broadcast), máy chủ trả JSON
`{"app":"netanh","api":1,"version","name","device","gpu","port"}`.
