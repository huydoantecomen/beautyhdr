"""Nguồn xử lý của app PC (net_anh/sources.py): máy chủ GPU qua API v1, khoá API ngoài, Settings.from_dict."""
from __future__ import annotations

import json
import urllib.parse

import cv2
import numpy as np
import pytest

from net_anh import providers as P
from net_anh import server as S
from net_anh import sources as SRC
from net_anh.settings import Settings


def _img(w=160, h=120):
    rng = np.random.default_rng(0)
    return cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 2)


def test_normalize_url():
    assert SRC.normalize_url("192.168.1.5") == "http://192.168.1.5:8765"
    assert SRC.normalize_url("http://may:9000/") == "http://may:9000"
    assert SRC.normalize_url("https://x.ts.net/api") == "https://x.ts.net"
    with pytest.raises(SRC.SourceError):
        SRC.normalize_url("")
    with pytest.raises(SRC.SourceError):
        SRC.normalize_url("ftp://x")


def test_settings_from_dict_round_trip_and_rejects_bad():
    s = Settings(mode="portrait", upscale=4, deshine=0.6, scene_vivid=1.0, scene_ai=1.0, bg_color=(255, 0, 10),
                 detail_fidelity=0.2, damage_mask="C:/mat_na.png", look={"name": "x", "skin_L": 70.0})
    warn: list[str] = []
    t = Settings.from_dict(json.loads(json.dumps(s.to_dict())), warn)
    assert t.mode == "portrait" and t.upscale == 4 and t.deshine == 0.6 and t.scene_ai == 1.0
    assert t.bg_color == (255, 0, 10) and t.detail_fidelity == 0.2
    assert t.damage_mask == ""  # đường dẫn trên máy gửi không bao giờ được nhận
    assert t.look and t.look["skin_L"] == 70.0
    bad = Settings.from_dict({"mode": "rm -rf", "fidelity": 9, "upscale": "x", "low_light": 1, "la": 1,
                              "bg_color": [1, 2], "sharpness": float("nan")}, warn := [])
    assert bad.mode == Settings.mode and bad.fidelity == 1.0 and bad.upscale == 0 and bad.low_light is False
    assert bad.bg_color is None and bad.sharpness == Settings.sharpness
    assert any("la" in w for w in warn) and any("mode" in w for w in warn)


def test_server_params_full_and_old_server():
    s = Settings(mode="portrait", deshine=0.5, bg_color=(0, 0, 255), repair=1.0, fade=1.0, scene_vivid=1.0,
                 damage_mask="x.png")
    q = SRC.server_params(s, 3, ["restore", "bg", "deshine", "settings"])
    assert q["level"] == "3" and "settings" in q and "restore" not in q
    d = json.loads(q["settings"])
    assert "damage_mask" not in d and d["deshine"] == 0.5
    old = SRC.server_params(s, None, ["restore", "bg", "subject", "styles", "deshine"])
    assert old["level"] == "auto" and old["restore"] == "1" and old["bg"] == "ff0000" and old["deshine"] == "0.50"
    assert "settings" not in old
    assert SRC.lost_on_old_server(s, ["restore"]) == ["nâng màu cảnh", "vùng hỏng tự tô"]


def test_parse_extra_settings():
    s = Settings(mode="quick", tone=0.2, deshine=0.4)
    ex = S.parse_extra({"settings": json.dumps(s.to_dict())})
    assert ex["__settings__"].tone == 0.2 and ex["__settings__"].deshine == 0.4
    with pytest.raises(S.ApiError):
        S.parse_extra({"settings": "{không phải json"})
    with pytest.raises(S.ApiError):
        S.parse_extra({"settings": "x" * (S.MAX_SETTINGS_CHARS + 1)})
    assert "settings" in S.FEATURES


class FakeServer:
    def __init__(self, states, out: bytes, fail_job=None):
        self.states, self.out, self.calls, self.fail_job = list(states), out, [], fail_job

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, headers))
        u = urllib.parse.urlsplit(url)
        if method == "POST" and u.path == "/api/v1/jobs":
            assert headers["Authorization"] == "Bearer tok"
            assert cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR).shape == (120, 160, 3)
            return 202, {}, json.dumps({"id": "j1", "state": "queued", "position": 1}).encode()
        if method == "GET" and u.path == "/api/v1/jobs/j1":
            st = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            d = {"id": "j1", "state": st, "progress": "Đang làm nét mặt", "position": 0}
            if st == "done":
                d["report"] = {"mode": "portrait", "device": "RTX 4090", "seconds": 3.2, "faces_found": 1,
                               "faces_restored": 1, "server": "may-thue", "notes": ["ok"], "face_model": "gfpgan"}
            if st == "error":
                d["error"] = self.fail_job
            return 200, {}, json.dumps(d).encode()
        if method == "GET" and u.path == "/api/v1/jobs/j1/result":
            return 200, {}, self.out
        if method == "DELETE":
            return 200, {}, b"{}"
        raise AssertionError(url)


def test_run_on_server_flow(monkeypatch):
    monkeypatch.setattr(SRC, "POLL_SECONDS", 0.01)
    out = cv2.imencode(".jpg", _img(320, 240))[1].tobytes()
    net = FakeServer(["queued", "running", "done"], out)
    monkeypatch.setattr(P, "transport", net)
    msgs = []
    res, rep = SRC.run_on_server("http://h:8765", "tok", _img(), {"level": "6"}, msgs.append)
    assert res.shape == (240, 320, 3)
    r = SRC.report_from_server(rep, "http://h:8765")
    assert "may-thue" in r.device and r.faces_restored == 1 and r.device_kind == "remote"
    assert any("Đang" in m for m in msgs)


def test_run_on_server_error_and_cancel(monkeypatch):
    monkeypatch.setattr(SRC, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(P, "transport", FakeServer(["running", "error"], b"", fail_job="hết RAM"))
    with pytest.raises(SRC.SourceError, match="hết RAM"):
        SRC.run_on_server("http://h:8765", "tok", _img(), {})
    net = FakeServer(["running"], b"")
    monkeypatch.setattr(P, "transport", net)
    with pytest.raises(P.ProviderCancelled):
        SRC.run_on_server("http://h:8765", "tok", _img(), {}, cancel=lambda: True)
    assert any(m == "DELETE" for m, _, _ in net.calls)


def test_save_api_key(tmp_path):
    p = tmp_path / "providers.json"
    SRC.save_api_key("replicate", "r8_abcdef123456", {"model": "a/b"}, path=p)
    SRC.save_api_key("fal", "fal-key", path=p)
    cfg = P.load_config(p, env={})
    assert cfg["replicate"]["key"] == "r8_abcdef123456" and cfg["replicate"]["model"] == "a/b"
    assert set(P.engines(cfg)) == {"fal-clarity", "replicate"}
    SRC.save_api_key("replicate", None, {"model": "c/d"}, path=p)  # giữ khoá, đổi model
    assert P.load_config(p, env={})["replicate"] == {"key": "r8_abcdef123456", "model": "c/d"}
    SRC.save_api_key("fal", "", path=p)
    assert "fal-clarity" not in P.engines(P.load_config(p, env={}))
    assert SRC.mask_key("r8_abcdef123456").endswith("3456") and "abc" not in SRC.mask_key("r8_abcdef123456")
    with pytest.raises(ValueError):
        SRC.save_api_key("khac", "x", path=p)


def test_run_api_falls_back_when_api_fails(monkeypatch):
    e = P.Engine("fal-clarity", "fal.ai Clarity", "fal", "~0,03 USD", "", "k", {})

    def boom(*a, **k):
        raise P.ProviderError("no_credit", "hết tiền")
    monkeypatch.setattr(P.Engine, "run", boom)
    r = SRC.run_api(None, e, _img(), 2)
    assert not r.used_api and "no_credit" in r.note


def test_run_api_guard(monkeypatch):
    e = P.Engine("fal-clarity", "fal.ai Clarity", "fal", "~0,03 USD", "", "k", {})
    big = _img(320, 240)
    monkeypatch.setattr(P.Engine, "run", lambda self, img, up, progress=None, cancel=None: big)
    monkeypatch.setattr(P, "face_ids", lambda eng, a, b: [0.9, 0.5])
    r = SRC.run_api(None, e, _img(), 2)
    assert not r.used_api and "0.50" in r.note
    monkeypatch.setattr(P, "face_ids", lambda eng, a, b: [0.9])
    r = SRC.run_api(None, e, _img(), 2)
    assert r.used_api and r.img is big
    rep = SRC.report_from_api(e, r, 4.0)
    assert rep.mode == "external" and rep.device_kind == "external"


def test_real_server_with_full_settings(tmp_path, monkeypatch):
    """Máy chủ thật trong tiến trình: app PC gửi đủ thông số (Cải thiện nhanh, phóng 2×) qua tính năng "settings"."""
    monkeypatch.setattr(SRC, "POLL_SECONDS", 0.2)
    pairing = S.Pairing(tmp_path / "server.json", static_tokens=["tok-pc"])
    srv = S.Server("127.0.0.1", 0, "cpu", say=lambda m: None, pairing=pairing, discovery=False).start()
    try:
        url = f"http://127.0.0.1:{srv.port}"
        info = SRC.server_info(url)
        assert "settings" in info["features"]
        s = Settings(mode="quick", upscale=2, tone=0.0, exposure=0.0)
        res, rep = SRC.run_on_server(url, "tok-pc", _img(), SRC.server_params(s, None, info["features"]))
        assert res.shape[:2] == (240, 320)
        assert "thông số do app PC gửi" in rep.get("reasons", [])
        with pytest.raises(SRC.SourceError) as e:
            SRC.run_on_server(url, "sai", _img(), {})
        assert e.value.code == "unauthorized"
    finally:
        srv.stop()
