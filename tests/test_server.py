"""Chế độ máy chủ (net_anh/server.py): ghép nối, xác thực, gửi ảnh, hỏi trạng thái, lấy kết quả, huỷ, tìm máy UDP."""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from net_anh import server as S


def _req(base, method, path, body=None, token=None, ctype="application/json"):
    r = urllib.request.Request(base + path, data=body, method=method)
    if body is not None:
        r.add_header("Content-Type", ctype)
    if token:
        r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", "")


@pytest.fixture()
def srv(tmp_path):
    pairing = S.Pairing(tmp_path / "server.json", static_tokens=["token-co-dinh"])
    s = S.Server("127.0.0.1", 0, "cpu", say=lambda m: None, pairing=pairing, discovery=False).start()
    yield s, f"http://127.0.0.1:{s.port}"
    s.stop()


def _jpeg(w=160, h=120) -> bytes:
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_info_needs_no_token(srv):
    s, base = srv
    code, body, _ = _req(base, "GET", "/api/v1/info")
    info = json.loads(body)
    assert code == 200 and info["app"] == "netanh" and info["api"] == 1 and info["pairing"] is True


def test_pair_once_then_code_changes(srv):
    s, base = srv
    code0 = s.pairing.code
    st, body, _ = _req(base, "POST", "/api/v1/pair", json.dumps({"code": "000000" if code0 != "000000" else "111111",
                                                                  "device": "x"}).encode())
    assert st == 403 and json.loads(body)["code"] == "wrong_code"
    st, body, _ = _req(base, "POST", "/api/v1/pair", json.dumps({"code": code0, "device": "Pixel"}).encode())
    token = json.loads(body)["token"]
    assert st == 200 and len(token) > 30
    assert s.pairing.code != code0  # mã dùng một lần
    st, _, _ = _req(base, "POST", "/api/v1/pair", json.dumps({"code": code0, "device": "khác"}).encode())
    assert st == 403
    assert _req(base, "GET", "/api/v1/ping", token=token)[0] == 200
    # token không lưu dạng rõ
    assert token not in (s.pairing.path.read_text(encoding="utf-8"))
    s.pairing.unpair_all()
    assert _req(base, "GET", "/api/v1/ping", token=token)[0] == 401


def test_bruteforce_locks(srv):
    s, base = srv
    wrong = "000000" if s.pairing.code != "000000" else "111111"
    codes = [_req(base, "POST", "/api/v1/pair", json.dumps({"code": wrong}).encode())[0] for _ in range(S.MAX_FAILS + 1)]
    assert codes[:S.MAX_FAILS] == [403] * S.MAX_FAILS and codes[-1] == 429
    # đang khoá thì mã đúng cũng bị từ chối
    assert _req(base, "POST", "/api/v1/pair", json.dumps({"code": s.pairing.code}).encode())[0] == 429


def test_jobs_require_token_and_reject_non_images(srv):
    s, base = srv
    assert _req(base, "POST", "/api/v1/jobs", _jpeg(), ctype="image/jpeg")[0] == 401
    assert _req(base, "POST", "/api/v1/jobs", b"khong phai anh", token="token-co-dinh")[0] == 415
    assert _req(base, "POST", "/api/v1/jobs?level=9", _jpeg(), token="token-co-dinh")[0] == 400


def _wait(base, jid, token, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        st, body, _ = _req(base, "GET", f"/api/v1/jobs/{jid}", token=token)
        d = json.loads(body)
        if d["state"] not in ("queued", "running"):
            return d
        time.sleep(0.3)
    raise AssertionError("quá lâu")


def test_enhance_level1_roundtrip(srv):
    """Mức 1 (không AI): chạy hết vòng gửi → chờ → tải ảnh, kể cả giới hạn cạnh dài max_side."""
    s, base = srv
    st, body, _ = _req(base, "POST", "/api/v1/jobs?level=1&upscale=2&max_side=200&phone=samsung", _jpeg(),
                       token="token-co-dinh", ctype="image/jpeg")
    assert st == 202
    d = _wait(base, json.loads(body)["id"], "token-co-dinh")
    assert d["state"] == "done", d
    rep = d["report"]
    assert rep["level"] == 1 and rep["phone"] == "samsung" and max(rep["width"], rep["height"]) == 200
    st, img, ctype = _req(base, "GET", f"/api/v1/jobs/{d['id']}/result", token="token-co-dinh")
    out = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    assert st == 200 and ctype == "image/jpeg" and out.shape[:2] == (150, 200)


def test_broken_image_reports_error(srv):
    s, base = srv
    bad = _jpeg()[:200]  # đầu JPEG đúng nhưng cụt
    st, body, _ = _req(base, "POST", "/api/v1/jobs?level=1", bad, token="token-co-dinh")
    d = _wait(base, json.loads(body)["id"], "token-co-dinh")
    assert d["state"] == "error" and d["error"]
    assert _req(base, "GET", f"/api/v1/jobs/{d['id']}/result", token="token-co-dinh")[0] == 409


def test_cancel_queued_job(srv):
    s, base = srv
    with s.worker.cv:  # giữ hàng đợi để việc chắc chắn còn ở trạng thái chờ
        job = S.Job(id="abc", data=_jpeg(), ext=".jpg", level=1, phone="auto", upscale=1, max_side=0, quality=90,
                    device="t")
        s.worker.jobs[job.id] = job
        s.worker.order.append(job.id)
        s.worker.cancel(job)
    st, body, _ = _req(base, "GET", "/api/v1/jobs/abc", token="token-co-dinh")
    assert json.loads(body)["state"] == "cancelled"


def test_queue_full(srv):
    s, base = srv
    with s.worker.cv:
        for i in range(S.MAX_QUEUE):
            jid = f"j{i}"
            s.worker.jobs[jid] = S.Job(id=jid, data=b"", ext=".jpg", level=1, phone="auto", upscale=1, max_side=0,
                                       quality=90, device="t")
            s.worker.order.append(jid)
        try:
            with pytest.raises(OverflowError):
                s.worker.submit(S.Job(id="x", data=b"", ext=".jpg", level=1, phone="auto", upscale=1, max_side=0,
                                      quality=90, device="t"))
        finally:
            for i in range(S.MAX_QUEUE):
                s.worker.order.remove(f"j{i}")


def test_discovery_reply(tmp_path):
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.bind(("", S.DISCOVERY_PORT))
        probe.close()
    except OSError:
        pytest.skip("cổng UDP 8766 đang bận")
    s = S.Server("127.0.0.1", 0, "cpu", say=lambda m: None, pairing=S.Pairing(tmp_path / "p.json"), name="May-test").start()
    try:
        c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        c.settimeout(3)
        c.sendto(S.DISCOVER_MAGIC, ("127.0.0.1", S.DISCOVERY_PORT))
        data, _ = c.recvfrom(2048)
        d = json.loads(data)
        assert d["app"] == "netanh" and d["name"] == "May-test" and d["port"] == s.port
        c.close()
    finally:
        s.stop()


def test_parse_extra_restore_bg_subject_styles():
    """Tham số thêm từ app Android 0.8: khôi phục ảnh cũ, đổi nền, preset đối tượng (kiểu mặt gửi kèm)."""
    assert S.parse_extra({}) == {}
    e = S.parse_extra({"restore": "1", "bg": "4386DB", "subject": "woman",
                       "styles": json.dumps({"woman": {"smooth": 0.9, "warm": 5}, "la": {"keep": 1}})})
    assert e["repair"] == 1.0 and e["fade"] == 1.0 and e["auto_rotate"] is True and e["wb"] == 0.0
    assert e["bg_color"] == (0xDB, 0x86, 0x43) and e["subject"] == "woman"
    assert e["face_styles"]["woman"]["smooth"] == 0.9 and e["face_styles"]["woman"]["warm"] == 1.0  # kẹp
    assert "la" not in e["face_styles"]
    for bad in ({"restore": "2"}, {"bg": "xanh"}, {"subject": "meo"}, {"styles": "{"}, {"styles": "[]"},
                {"styles": "x" * 9000}):
        with pytest.raises(S.ApiError):
            S.parse_extra(bad)


def test_parse_extra_deshine():
    from net_anh import server as S
    assert S.parse_extra({"deshine": "0.6"}) == {"deshine": 0.6}
    assert "deshine" in S.FEATURES
    for bad in ("abc", "1.5", "-0.1"):
        with pytest.raises(S.ApiError):
            S.parse_extra({"deshine": bad})
