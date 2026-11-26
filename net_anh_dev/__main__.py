"""python -m net_anh_dev            → app có menu Nhà phát triển
python -m net_anh_dev prompt ...  → in prompt ra màn hình (xem net_anh_dev/cli.py)"""
import sys

import net_anh  # noqa: F401  ép stdout/stderr sang UTF-8

if len(sys.argv) > 1 and sys.argv[1] == "prompt":
    from .cli import main as cli_main

    raise SystemExit(cli_main(sys.argv[2:]))

from net_anh.gui import main

from .gui_dev import install

raise SystemExit(main(setup=install))
