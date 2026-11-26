"""In prompt cho hệ render ngoài từ dòng lệnh (bản nhà phát triển), không cần ảnh hay model:

    python -m net_anh_dev prompt --preset max --target grok --lang vi
    python -m net_anh_dev prompt --level 4 --preset-set trong-sang --bg-color xanh --size 800x600 --json
"""
from __future__ import annotations

import argparse
import sys

from net_anh import __version__, backdrop, facestyle
from net_anh.levels import LEVELS, level_settings
from net_anh.settings import PRESETS, Settings

from . import prompt as P


def parse(argv):
    p = argparse.ArgumentParser(prog="python -m net_anh_dev prompt", description="Xuất prompt theo thông số Beauty HDR")
    p.add_argument("--preset", choices=list(PRESETS), default="auto")
    p.add_argument("--level", type=int, choices=list(LEVELS))
    p.add_argument("--preset-set", default="default", help="mã bộ preset có sẵn: " + ", ".join(sorted(facestyle.BUILTIN_IDS)))
    p.add_argument("--subject", choices=["auto", *facestyle.SUBJECT_KEYS, "off"], default="auto")
    p.add_argument("--bg-color", help="#RRGGBB hoặc trang, xanh, xanh-nhat, do, xam")
    p.add_argument("--deshine", type=float)
    p.add_argument("--size", help="cỡ ảnh gốc, vd 800x600")
    p.add_argument("--target", choices=list(P.TARGETS), default="chatgpt")
    p.add_argument("--lang", choices=list(P.LANGS), default="en")
    p.add_argument("--json", action="store_true", help="in mọi hệ × ngôn ngữ dạng JSON")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse(sys.argv[1:] if argv is None else argv)
    s = Settings().with_(**PRESETS[a.preset][1])
    if a.level:
        s = level_settings(a.level, s)
    ps = next((x for x in facestyle.BUILTIN_SETS if x.id == a.preset_set), None)
    if ps is None:
        print(f"Không có bộ preset “{a.preset_set}”", file=sys.stderr)
        return 2
    s = s.with_(face_styles=facestyle.styles_to_settings(ps), look=dict(ps.look) if ps.look else None,
                subject=a.subject)
    if a.bg_color:
        try:
            s = s.with_(bg_color=backdrop.parse_color(a.bg_color))
        except ValueError as e:
            print(f"--bg-color: {e}", file=sys.stderr)
            return 2
    if a.deshine is not None:
        s = s.with_(deshine=a.deshine)
    size = None
    if a.size:
        try:
            w, h = (int(x) for x in a.size.lower().split("x"))
            size = (w, h)
        except ValueError:
            print("--size: dạng RỘNGxCAO, vd 800x600", file=sys.stderr)
            return 2
    ctx = P.Context(level=a.level, size=size, preset_name=ps.name)
    if a.json:
        print(P.export_json(s, ctx, __version__))
    else:
        print(P.export_text(P.build(s, a.target, a.lang, ctx)))
    return 0
