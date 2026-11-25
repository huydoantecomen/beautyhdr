"""Gộp các <flavor>.json (từ make_release_parts.py) thành update.json cho BeautyHDR.exe.

  python tools/make_update_json.py <phiên bản> <repo> <thư mục chứa *.json> > update.json
URL tải: https://github.com/<repo>/releases/download/v<phiên bản>/<tên phần>
"""
import json
import sys
from pathlib import Path


def main(version: str, repo: str, folder: str) -> None:
    flavors = {}
    for f in sorted(Path(folder).glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for p in d["parts"]:
            p["url"] = f"https://github.com/{repo}/releases/download/v{version}/{p['name']}"
        flavors[d["flavor"]] = {"parts": d["parts"]}
    if "cpu" not in flavors:
        raise SystemExit("thiếu bản cpu")
    print(json.dumps({"version": version, "flavors": flavors}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
