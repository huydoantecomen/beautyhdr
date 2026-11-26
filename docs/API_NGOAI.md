# Cổng kết nối API xử lý ảnh ngoài (từ 0.16.0)

Máy chủ Beauty HDR có thể gửi ảnh tới dịch vụ ngoài (fal.ai, Stability AI, Replicate) thay cho pipeline của máy này.
App điện thoại / PC vẫn chỉ nói chuyện với máy chủ Beauty HDR qua API v1, nên thêm hay bỏ một dịch vụ không phải phát
hành lại app. Mã: `net_anh/providers.py`, phần hàng đợi trong `net_anh/server.py`.

```
Điện thoại ──API v1──▶ Máy chủ Beauty HDR ──┬─▶ pipeline máy này (local, mặc định)
                       (giữ khoá API)        └─▶ API ngoài ──▶ kiểm giữ mặt (ArcFace) ──▶ trả ảnh
                                                     │ lỗi / đổi mặt
                                                     └──────────▶ pipeline máy này + ghi chú lý do
```

## Bật một dịch vụ

Đặt khoá trên **máy chủ** (không bao giờ đặt trong app: ai giải nén APK cũng lấy được):

| Dịch vụ | Biến môi trường | Nguồn xử lý (id) | Giá công khai (11/2026) |
|---|---|---|---|
| fal.ai Clarity Upscaler (khuếch tán) | `FAL_KEY` | `fal-clarity` | 0,03 USD mỗi megapixel |
| Stability AI Upscale | `STABILITY_API_KEY` | `stability-conservative`, `stability-fast` | 40 / 2 credit mỗi ảnh |
| Replicate (model tự chọn) | `REPLICATE_API_TOKEN` + `NETANH_REPLICATE_MODEL=owner/name[:version]` | `replicate` | theo giây GPU |

Hoặc ghi vào `~/.net_anh/providers.json` (Windows: `%USERPROFILE%\.net_anh\providers.json`), mọi trường tuỳ chọn:

```json
{
  "fal": {"key": "...", "creativity": 0.25, "resemblance": 0.75},
  "stability": {"key": "...", "creativity": 0.25, "prompt": "high quality portrait photo"},
  "replicate": {"key": "...", "model": "owner/name", "image_field": "image", "input": {"scale": 2},
                "label": "Tên hiện trên app", "price": "~0,01 USD/ảnh", "max_pixels": 4000000},
  "disabled": ["stability-fast"],
  "guard_ids": 0.7
}
```

Khởi động lại máy chủ; dòng `API ngoài sẵn sàng: …` hiện ra. Thử một ảnh không cần máy chủ:

```
python -m net_anh.providers                                   liệt kê nguồn đã cấu hình
python -m net_anh.providers anh.jpg --engine fal-clarity -o ra.png
```

## API v1: tham số mới của `POST /api/v1/jobs`

| Tham số | Ý nghĩa |
|---|---|
| `engine=<id>` | Nguồn xử lý. Không gửi = `local`. Danh sách ở `GET /api/v1/info` → `engines` (id, label, provider, external, price, note; **không** có khoá) |
| `consent=1` | Bắt buộc khi `engine` là dịch vụ ngoài: app phải hỏi người dùng trước (tên dịch vụ + giá). Thiếu → 400 `consent_required` |
| `guard=0\|1` | Kiểm giữ khuôn mặt (mặc định 1). Mặt nào có IDS dưới `guard_ids` (0,70) → dùng kết quả pipeline máy này |

Lỗi mới: 400 `bad_engine` (máy chủ không có nguồn đó), 400 `consent_required`, 400 `bad_guard`.

`report` khi xong thêm: `engine` (nguồn thật đã tạo ảnh), `engine_label`, `ids` (IDS từng mặt), và khi phải quay về máy
này: `engine_requested` + dòng đầu `notes` nói lý do (API lỗi, hết credit, đổi khuôn mặt). `mode` = `external` khi ảnh
lấy từ API ngoài.

Việc gửi ra ngoài chạy phần mạng ở luồng riêng (GPU máy chủ không ngồi chờ), rồi quay lại hàng đợi chính để kiểm mặt.
Tối đa 8 việc cả hai loại; huỷ (`DELETE`) cũng huỷ bên fal.ai / Replicate.

## Riêng tư

- Ảnh gửi đi được giải mã rồi nén lại từ điểm ảnh: mất toàn bộ EXIF (GPS, tên máy, thời gian).
- Thu nhỏ theo giới hạn từng API (Stability Conservative ≤ 9,4 MP, Fast ≤ 1 MP; fal ảnh ra ≤ 16 MP).
- Chỉ gửi khi có `consent=1`. Ảnh gốc và kết quả trên máy chủ vẫn xoá sau 15 phút như trước.
- Điều khoản lưu ảnh của từng dịch vụ khác nhau (Replicate ghi xoá đầu vào/ra sau 1 giờ; Stability giữ kết quả 24 giờ ở
  chế độ bất đồng bộ). Đọc điều khoản trước khi bật cho người dùng thật.

## Chưa làm

- App PC có ô Nguồn xử lý và hộp hỏi đồng ý từ 0.16.0 (gọi API thẳng từ PC bằng khoá người dùng nhập, hoặc qua
  máy chủ GPU). App Android chưa có.
- Chưa thử với khoá thật: bộ nối viết theo tài liệu công khai của từng dịch vụ (11/2026), test dùng mạng giả lập.
  fal.ai chưa ghi rõ có nhận ảnh dạng data URI hay không; nếu bị từ chối cần thêm bước tải ảnh lên kho của fal.
- Chưa tính credit/giới hạn theo người dùng.
