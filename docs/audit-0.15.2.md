# Audit thông số Chuyên sâu (PC 0.15.1 → 0.15.2), 2026-11-26

Đo trên CPU 4 nhân, fp32, không có GPU. Nền so sánh: mức 3 "Rõ" (Chân dung AI, Trung thực 70 %, Độ nét 50 %, phóng 2×).
Ảnh thử 640 px: chân dung nữ, ảnh thẻ nữ trung niên nền xanh, ảnh thẻ nam nền trắng (làm mờ 2× + nhiễu nhẹ), ảnh tối −2 EV,
ảnh ám vàng, ảnh nhiễu σ 14, ảnh nhạt, da ngả đỏ, da ngả vàng, ảnh cũ đen trắng, ảnh cũ chụp nhóm ngoài trời.
229 lượt render. Số liệu: /mnt/project-files/audit-0.15/ket-qua-do.json; ảnh so sánh: audit-0.15/anh-so-sanh/.
Công cụ: tools/audit_thong_so/ (audit.py chạy đo, guicheck.py kiểm nối giao diện, sheets.py dựng ảnh so sánh).
Độ lệch nền: chạy lại cùng thông số lệch ~0,04 mức xám (cả ảnh) / 0,11 (vùng mặt), nên dưới mức đó coi như không đổi.

## Lỗi đã sửa (nhánh claude/project-thread-yq2zwe, chưa PR, chưa merge)
1. Tương phản không tác dụng ở mức 2–6 (Chân dung, Phóng to): CLAHE chỉ chạy ở mức 1 và Ảnh cũ. Tự động còn bật 67 % cho
   ảnh nhạt rồi ghi "bật CLAHE 2.0" mà không có hiệu lực. Sau sửa: mức 3, độ lệch chuẩn L 11,5 → 14,6 / 16,8 / 18,2 (0,33/0,67/1).
   Lưu ý: ảnh nhạt chạy Tự động ở mức 2–5 sẽ ra khác bản cũ (tương phản cao hơn), đúng như thiết kế ghi.
2. Khử nhiễu chỉnh tay không tác dụng khi Model nền là RRDB, RealPLKSR, DAT-2 (chỉ compact có nút khử nhiễu). Sau sửa:
   PLKSR nhiễu 3,17 → 0,56 (mức 4) / 0,40 (mức 10). Khử nhiễu tự động với các model này vẫn không khử (giữ như cũ).
3. Mức 2 của ảnh cũ đặt sr_denoise = 1 (khử hạt mạnh) nhưng giao diện bỏ mất vì không có ô; nay giữ các thông số ẩn.
Có test: test_contrast_works_in_ai_modes, test_gui_keeps_hidden_level_fields (hỏng trên 0.15.1, qua trên bản sửa).

## Bảng kết quả
| Thông số | Kết quả | Số đo (mặc định → giá trị khác) | Ghi chú |
|---|---|---|---|
| Đối tượng | Có tác dụng, đúng | lệch mặt 0,9–2,3 khi ép nhóm khác / tắt; Tự động trùng đúng nhóm (nữ→Nữ, nam→Nam) | |
| Trung thực AI | Có, đúng | nét mặt 42 (0) → 32 (0,35) → 26 (0,7) → 24 (1) | |
| Trung thực mắt, môi | Có, đúng; phần môi bị Giữ môi ghi đè | lệch mắt/môi tăng khi AI nhiều; tắt Giữ môi: lệch môi 3,3 thay vì 0,75 | chỉ khi bỏ tick "Mắt, môi như cả mặt" |
| Giữ dáng, màu môi gốc | Có | lệch môi 4,4 (Trung thực 0: 6,8) | |
| Sửa màu da | Có điều kiện | da đỏ: góc màu 26° → 30° → 34°; da vàng: 65° → 61° → 60° | chỉ chạy khi góc màu da ngoài 36–60° hoặc quá đậm; ảnh thường không đổi |
| Độ bóng da | Có, đúng | da L 73 → 75 → 77 (0/0,5/1) | |
| Khử bóng da | Có, đúng | điểm bóng 0,73 % → 0,02 / 0,02 / 0,01 % | chỉ khi da có vệt bóng; preset Nam (matte 0,1) chỉ cộng thêm khi đã tick |
| Tự sửa sáng | Có, đúng | ảnh tối L 38 (0) → 45 (0,5) → 50 (1); ảnh thường không đổi | bị Retinex ghi đè (thanh bị khoá khi bật Retinex) |
| Độ sáng | Có, đúng (mức 1 và 3) | L 54 / 61 / 65 / 69 (−0,5/0/0,25/0,5) | |
| Tương phản | LỖI ở mức 2–6, đã sửa | xem trên; mức 1: 11,6 → 16,9; Ảnh cũ: 15,1 → 17,0 | |
| Tươi sáng | Có, đúng | da nam L 47 → 51 → 54 → 58 (0/0,3/0,6/1) | ảnh thẻ nền xanh: da sáng lên, nền đậm hơn nên L cả ảnh giảm nhẹ |
| Ảnh tối: Retinex | Có | L 50 → 68 (mức 3), 50 → 63 (mức 1) | |
| Tông mẫu | Có | da nam L 54 → 66; da nữ đỏ a 16 → 12 | |
| Độ mạnh tông mẫu | Có, tuyến tính | lệch 0 / 2,1 / 4,0 (0/0,5/1) | chỉ khi đã chọn tông mẫu |
| Máy chụp | Có | ảnh ám vàng: độ rực 11,3 (Khác) / 9,8 (iPhone) / 9,2 (Oppo) / 20,8 (Tắt) | |
| Cân bằng trắng | Có điều kiện, đúng | ảnh ám: độ rực 20,6 → 16,4 → 11,3 → 8,1 (0/0,3/0,6/1) | ảnh không ám không đổi (đúng thiết kế) |
| Tăng màu | Có, đúng | độ rực ảnh thẻ 29,4 → 31,5; độ nét 22 → 51 | |
| Độ nét | Có nhưng YẾU ở mức 3 | độ nét cả ảnh 20,6 → 21,8 → 23,1 (0/0,5/1); lệch < 0,1 | không làm nét vùng mặt (mặt do model mặt); mức 1: 9,2 → 12,8 |
| Khử nhiễu | Có với compact và mức 1; LỖI với model nền nặng, đã sửa | compact: nhiễu 1,99 (0) → 1,18 (4) → 0,40 (10) | |
| Khử nhiễu tự động | Có | | khi bật thì thanh khoá |
| Khôi phục ảnh cũ | Có | ảnh cũ: lệch 28, độ nét 17 → 32, gỡ ám giấy | |
| Màu cảnh (ảnh cũ) | Có điều kiện | ảnh cảnh: độ rực 24 → 30 | bỏ qua ảnh đen trắng và ảnh nền đồng màu |
| Tô màu AI (DDColor) | Không đổi trên ảnh thử | lệch 0,03 | luật có sẵn: ảnh còn đủ màu (sắc độ > 10) thì bỏ qua; CHƯA KIỂM trên ảnh phai gần hết màu |
| Đổi nền | Có, đúng | nền xanh → trắng / đỏ, mặt lệch ≤ 0,4 | |
| Cường độ chung | Có, tuyến tính | lệch 3,2 (0) / 1,65 (0,5) | |
| Chế độ | Có | Tự chỉnh cho kết quả GIỐNG HỆT Cải thiện nhanh | không phải lỗi nhưng dễ hiểu nhầm |
| Phóng to | Có | Tự động = 4× cho ảnh < 1000 px | |
| Model mặt | Có | RestoreFormer nét hơn GFPGAN | |
| Model nền | Có | PLKSR nét nhất, DAT-2 mịn nhất | |
| Mức nét 1–6 | Có, tăng dần | nét mặt 9 / 11 / 12 / 22 / 34 / 39 (mức 1→6); mức 6 chạy 67 s CPU | |
| Thiết bị | CHƯA KIỂM | | không có GPU; không đổi thông số render, chỉ chọn máy chạy |

## Kiểm nối giao diện (offscreen)
27/29 điều khiển đổi đúng trường Settings. 2 thanh không đổi khi bị khoá (Trung thực mắt môi khi tick "như cả mặt",
Khử nhiễu khi tick tự động), đúng thiết kế. Thanh Tương phản vẫn bật ở mọi chế độ (nay đã có tác dụng).

## Còn lại / chưa kiểm
- GPU (fp16/bf16, ô Thiết bị): chưa đo được trên cloud.
- Tô màu AI trên ảnh phai thật sự gần hết màu.
- Độ nét ở mức 2–6 yếu (độ nét cả ảnh chỉ tăng ~12 % khi kéo 0 → 1): chưa sửa vì là thay đổi cảm nhận, cần huy quyết.
- Chế độ "Tự chỉnh" trùng "Cải thiện nhanh": chưa đổi.
