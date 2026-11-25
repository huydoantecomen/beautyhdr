"""Nén thư mục app đã đóng gói (dist/NetAnh) thành zip rồi cắt thành các phần dưới 1,9 GB (GitHub giới hạn 2 GB mỗi file),
ghi <flavor>.json (tên, dung lượng, SHA-256 từng phần) cho tools/make_update_json.py.

  python tools/make_release_parts.py _build/dist/NetAnh out cpu
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

PART = int(1.9 * (1 << 30))


def main(src: str, out: str, flavor: str) -> None:
    src_dir, out_dir = Path(src), Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    zpath = out_dir / f"BeautyHDR-app-{flavor}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(src_dir.rglob("*")):
            if f.is_file() and "weights" not in f.relative_to(src_dir).parts[:1]:
                z.write(f, Path("NetAnh") / f.relative_to(src_dir))
    parts = []
    with open(zpath, "rb") as f:
        i = 0
        while True:
            chunk = f.read(PART)
            if not chunk:
                break
            i += 1
            p = out_dir / f"BeautyHDR-app-{flavor}.zip.{i:03d}"
            p.write_bytes(chunk)
            parts.append({"name": p.name, "size": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
    zpath.unlink()
    (out_dir / f"{flavor}.json").write_text(json.dumps({"flavor": flavor, "parts": parts}, indent=1), encoding="utf-8")
    print(flavor, [(p["name"], p["size"]) for p in parts])


if __name__ == "__main__":
    main(*sys.argv[1:4])
