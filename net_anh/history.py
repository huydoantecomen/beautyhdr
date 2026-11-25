"""Lịch sử xử lý trên máy (SQLite): đường dẫn ảnh gốc/ảnh ra, thumbnail, chế độ, thông số, thời gian."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def data_dir() -> Path:
    d = Path.home() / ".net_anh"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Entry:
    id: int
    created: float
    source: str
    output: str
    thumb: str
    mode: str
    settings: dict
    width: int
    height: int
    size_bytes: int
    seconds: float


class History:
    def __init__(self, root: Path | None = None):
        self.root = root or data_dir()
        self.thumbs = self.root / "thumbs"
        self.thumbs.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "history.db", check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, source TEXT, output TEXT, thumb TEXT,
            mode TEXT, settings TEXT, width INTEGER, height INTEGER, size_bytes INTEGER, seconds REAL)""")
        self.db.commit()

    def add(self, source: Path, output: Path, img: np.ndarray, mode: str, settings: dict, seconds: float) -> int:
        h, w = img.shape[:2]
        k = 256 / max(h, w)
        thumb = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
        tpath = self.thumbs / f"{int(time.time() * 1000)}.jpg"
        ok, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            buf.tofile(str(tpath))
        cur = self.db.execute(
            "INSERT INTO history (created, source, output, thumb, mode, settings, width, height, size_bytes, seconds)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (time.time(), str(source), str(output), str(tpath), mode, json.dumps(settings), w, h,
             Path(output).stat().st_size if Path(output).exists() else 0, seconds))
        self.db.commit()
        return int(cur.lastrowid)

    def list(self, limit: int = 200) -> list[Entry]:
        rows = self.db.execute("SELECT id, created, source, output, thumb, mode, settings, width, height, size_bytes,"
                               " seconds FROM history ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [Entry(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[6] or "{}"), r[7], r[8], r[9], r[10])
                for r in rows]

    def delete(self, entry_id: int, delete_output: bool = False) -> None:
        row = self.db.execute("SELECT output, thumb FROM history WHERE id=?", (entry_id,)).fetchone()
        if row:
            Path(row[1]).unlink(missing_ok=True)
            if delete_output:
                Path(row[0]).unlink(missing_ok=True)
        self.db.execute("DELETE FROM history WHERE id=?", (entry_id,))
        self.db.commit()

    def clear(self) -> None:
        for e in self.list(100000):
            self.delete(e.id)

    def close(self) -> None:
        self.db.close()
