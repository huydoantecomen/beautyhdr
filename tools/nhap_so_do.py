"""Gộp file đo card (BeautyHDR-do-gpu-*.json, tạo bởi Trợ giúp → Đo tốc độ card / NetAnh.exe --do-gpu) vào kho số đo
net_anh/du_lieu/so_do_gpu.json. App dùng trung vị các số đo cùng dòng card × model × kiểu số thay cho ước lượng (bảng
Khuyến nghị theo card, cắt ô theo bộ nhớ, kiểm phần cứng khi thêm model nặng hơn).

    python tools/nhap_so_do.py BeautyHDR-do-gpu-GTX-1660-Ti-20261127-0900.json [file khác...]
    python tools/nhap_so_do.py --xem          # tóm tắt kho số đo theo dòng card × model

Bản ghi trùng (cùng card, model, kiểu số, loại số đo, cỡ ô, kênh cuối, ngày giờ) chỉ giữ một.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STORE = ROOT / "net_anh" / "du_lieu" / "so_do_gpu.json"
KEY = ("card", "model", "dtype", "metric", "tile", "channels_last", "workers", "date")


def records_of(rep: dict) -> list[dict]:
    if rep.get("records"):
        return list(rep["records"])
    from net_anh.gpu_bench import records  # file đo cũ chưa có bản ghi: dựng lại từ số thô
    return records(rep)


def merge(store: dict, new: list[dict]) -> int:
    seen = {tuple(r.get(k) for k in KEY) for r in store.setdefault("records", [])}
    n = 0
    for r in new:
        k = tuple(r.get(k) for k in KEY)
        if k not in seen and r.get("family") and r.get("model") and r.get("metric"):
            store["records"].append(r)
            seen.add(k)
            n += 1
    return n


def summary(store: dict) -> str:
    from net_anh import gpu_profile as g

    groups: dict = {}
    for r in store.get("records", []):
        groups.setdefault((r["family"], r["model"], r["dtype"], r["metric"]), []).append(r)
    lines = []
    for (fam, model, dt, metric), rs in sorted(groups.items()):
        v = g.measured(fam, model, dt, metric, rs)
        cards = sorted({r.get("card", "?") for r in rs})
        lines.append(f"{fam:10} {model:20} {dt:5} {metric:22} {v if v is not None else '-':>10}  "
                     f"({len(rs)} số, {', '.join(cards)[:60]})")
    return "\n".join(lines) or "(chưa có số đo)"


def main(argv: list[str]) -> int:
    store = json.loads(STORE.read_text(encoding="utf-8")) if STORE.is_file() else {"schema": 1, "records": []}
    if not argv or argv == ["--xem"]:
        print(summary(store))
        return 0
    total = 0
    for a in argv:
        rep = json.loads(Path(a).read_text(encoding="utf-8"))
        n = merge(store, records_of(rep))
        total += n
        print(f"{a}: thêm {n} số đo ({rep.get('device', '?')})")
    STORE.write_text(json.dumps(store, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Đã ghi {STORE} ({len(store['records'])} số đo, thêm {total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
