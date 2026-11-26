"""Cổng API ngoài (net_anh/providers.py) và nguồn xử lý trên máy chủ: không gọi mạng thật, dùng transport giả."""
from __future__ import annotations

import base64
import json
import time
import urllib.parse

import cv2
import numpy as np
import pytest

from net_anh import providers as P
from net_anh import server as S


def _png(w=120, h=90, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (0, 0), 2)
    return cv2.imencode(".png", img)[1].tobytes()


def _img(w=120, h=90):
    return cv2.imdecode(np.frombuffer(_png(w, h), np.uint8), cv2.IMREAD_COLOR)


class FakeNet:
    """Ghi lại yêu cầu, trả câu trả lời theo hàng đợi khớp (method, phần đầu url)."""

    def __init__(self, routes):
        self.routes = routes  # list of (method, url_prefix, response or callable)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, headers, body))
        for i, (m, pre, resp) in enumerate(self.routes):
            if m == method and url.startswith(pre):
                if isinstance(resp, list):
                    r = resp.pop(0) if len(resp) > 1 else resp[0]
                else:
                    r = resp
                return r(method, url, headers, body) if callable(r) else r
        raise AssertionError(f"không có route cho {method} {url}")


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(P, "POLL_SECONDS", 0.01)


def _j(obj, status=200):
    return status, {"Content-Type": "application/json"}, json.dumps(obj).encode()


def test_fal_clarity_flow(monkeypatch):
    out = _png(240, 180, 1)
    net = FakeNet([
        ("POST", "https://queue.fal.run/fal-ai/clarity-upscaler",
         _j({"request_id": "r1", "status_url": "https://queue.fal.run/fal-ai/clarity-upscaler/requests/r1/status",
             "response_url": "https://queue.fal.run/fal-ai/clarity-upscaler/requests/r1",
             "cancel_url": "https://queue.fal.run/fal-ai/clarity-upscaler/requests/r1/cancel"})),
        ("GET", "https://queue.fal.run/fal-ai/clarity-upscaler/requests/r1/status",
         [_j({"status": "IN_QUEUE", "queue_position": 2}), _j({"status": "IN_PROGRESS"}), _j({"status": "COMPLETED"})]),
        ("GET", "https://queue.fal.run/fal-ai/clarity-upscaler/requests/r1",
         _j({"image": {"url": "https://v3.fal.media/files/x.png", "width": 240, "height": 180}})),
        ("GET", "https://v3.fal.media/", (200, {}, out)),
    ])
    monkeypatch.setattr(P, "transport", net)
    eng = P.engines({"fal": {"key": "k-fal"}, "stability": {}, "replicate": {}})["fal-clarity"]
    msgs = []
    res = eng.run(_img(), 2, progress=msgs.append)
    assert res.shape == (180, 240, 3)
    m, url, hdr, body = net.calls[0]
    assert hdr["Authorization"] == "Key k-fal"
    req = json.loads(body)
    assert req["upscale_factor"] == 2 and req["image_url"].startswith("data:image/")
    sent = cv2.imdecode(np.frombuffer(base64.b64decode(req["image_url"].split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert sent.shape == (90, 120, 3)
    assert any("vị trí 2" in x for x in msgs)


def test_fal_cancel_calls_cancel_url(monkeypatch):
    net = FakeNet([
        ("POST", "https://queue.fal.run/", _j({"status_url": "https://q/s", "response_url": "https://q/r",
                                               "cancel_url": "https://q/c"})),
        ("GET", "https://q/s", _j({"status": "IN_PROGRESS"})),
        ("PUT", "https://q/c", _j({"status": "CANCELLATION_REQUESTED"}, 202)),
    ])
    monkeypatch.setattr(P, "transport", net)
    eng = P.engines({"fal": {"key": "k"}})["fal-clarity"]
    n = {"i": 0}

    def cancel():
        n["i"] += 1
        return n["i"] > 2

    with pytest.raises(P.ProviderCancelled):
        eng.run(_img(), 0, cancel=cancel)
    assert net.calls[-1][0] == "PUT"


def test_stability_fast_resizes_and_maps_errors(monkeypatch):
    seen = {}

    def ok(method, url, headers, body):
        seen.update(headers=headers, body=body, url=url)
        return 200, {"finish-reason": "SUCCESS"}, _png(400, 300)

    monkeypatch.setattr(P, "transport", FakeNet([("POST", "https://api.stability.ai/v2beta/stable-image/upscale/fast", ok)]))
    eng = P.engines({"stability": {"key": "k-st"}})["stability-fast"]
    big = cv2.resize(_img(), (2400, 1800))
    res = eng.run(big)
    assert res.shape == (300, 400, 3)
    assert seen["headers"]["Authorization"] == "Bearer k-st" and seen["headers"]["Accept"] == "image/*"
    assert seen["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
    # ảnh gửi đi đã thu về ≤ 1 MP và cạnh ≤ 1536
    part = seen["body"].split(b"\r\n\r\n", 3)[-1]
    sent = cv2.imdecode(np.frombuffer(part, np.uint8), cv2.IMREAD_COLOR)
    assert sent is not None and sent.shape[0] * sent.shape[1] <= 1_048_576 and max(sent.shape[:2]) <= 1536

    monkeypatch.setattr(P, "transport", FakeNet([("POST", "https://api.stability.ai/", _j({"name": "unauthorized",
                                                                                          "errors": ["bad key"]}, 401))]))
    with pytest.raises(P.ProviderError) as e:
        eng.run(_img())
    assert e.value.code == "bad_key" and "bad key" in str(e.value)


def test_stability_conservative_rejects_extreme_aspect():
    eng = P.engines({"stability": {"key": "k"}})["stability-conservative"]
    with pytest.raises(P.ProviderError) as e:
        eng.run(np.zeros((100, 400, 3), np.uint8))
    assert e.value.code == "bad_aspect"


def test_replicate_model_and_polling(monkeypatch):
    net = FakeNet([
        ("POST", "https://api.replicate.com/v1/models/acme/upscaler/predictions",
         _j({"id": "p1", "status": "processing", "urls": {"get": "https://api.replicate.com/v1/predictions/p1",
                                                         "cancel": "https://api.replicate.com/v1/predictions/p1/cancel"}},
            201)),
        ("GET", "https://api.replicate.com/v1/predictions/p1",
         _j({"id": "p1", "status": "succeeded", "output": ["https://replicate.delivery/a.png"]})),
        ("GET", "https://replicate.delivery/", (200, {}, _png(60, 40))),
    ])
    monkeypatch.setattr(P, "transport", net)
    cfg = {"replicate": {"key": "k-rep", "model": "acme/upscaler", "image_field": "img", "input": {"scale": 2}}}
    eng = P.engines(cfg)["replicate"]
    assert eng.run(_img()).shape == (40, 60, 3)
    body = json.loads(net.calls[0][3])
    assert body["input"]["scale"] == 2 and body["input"]["img"].startswith("data:image/")
    assert net.calls[0][2]["Prefer"].startswith("wait")
    assert net.calls[-1][2]["Authorization"] == "Bearer k-rep"  # tải file kết quả cũng cần token


def test_config_env_disabled_and_no_key_leak(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text(json.dumps({"fal": {"creativity": 0.2}, "disabled": ["stability-fast"], "guard_ids": 0.8}))
    cfg = P.load_config(path, env={"FAL_KEY": "secret-fal", "STABILITY_API_KEY": "secret-st"})
    eng = P.engines(cfg)
    assert set(eng) == {"fal-clarity", "stability-conservative"}
    assert eng["fal-clarity"].cfg["creativity"] == 0.2
    assert P.guard_threshold(cfg) == 0.8
    pub = json.dumps([e.public() for e in eng.values()])
    assert "secret" not in pub
    assert P.engines(P.load_config(tmp_path / "khong-co.json", env={})) == {}


def test_encode_clean_strips_metadata():
    from PIL import Image
    import io
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x010F] = "MayAnhBiMat"
    Image.fromarray(np.full((50, 60, 3), 120, np.uint8)).save(buf, "JPEG", exif=exif)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    data, _ = P.encode_clean(img)
    assert b"MayAnhBiMat" not in data


# ---------------------------------------------------------------- máy chủ

class FakeEngine:
    def __init__(self, id="gia", fail=False):
        self.id, self.label, self.provider, self.price, self.note = id, "API giả", "gia", "0 USD", "thử"
        self.fail = fail
        self.calls = 0

    def public(self):
        return {"id": self.id, "label": self.label, "provider": self.provider, "external": True,
                "price": self.price, "note": self.note}

    def run(self, img, upscale=0, progress=None, cancel=None):
        self.calls += 1
        if self.fail:
            raise P.ProviderError("no_credit", "hết credit", 402)
        return cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)


def _req(base, method, path, body=None, token="token-co-dinh"):
    import urllib.error
    import urllib.request
    r = urllib.request.Request(base + path, data=body, method=method)
    if body is not None:
        r.add_header("Content-Type", "image/png")
    r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _server(tmp_path, engines):
    pairing = S.Pairing(tmp_path / "server.json", static_tokens=["token-co-dinh"])
    s = S.Server("127.0.0.1", 0, "cpu", say=lambda m: None, pairing=pairing, discovery=False, engines=engines).start()
    return s, f"http://127.0.0.1:{s.port}"


def _wait_done(base, jid, limit=300):
    t0 = time.time()
    while time.time() - t0 < limit:
        st = json.loads(_req(base, "GET", f"/api/v1/jobs/{jid}")[1])
        if st["state"] in ("done", "error", "cancelled"):
            return st
        time.sleep(0.2)
    raise AssertionError("quá thời gian")


def test_server_lists_engines_and_requires_consent(tmp_path):
    fake = FakeEngine()
    s, base = _server(tmp_path, {"gia": fake})
    try:
        info = json.loads(_req(base, "GET", "/api/v1/info")[1])
        ids = [e["id"] for e in info["engines"]]
        assert ids == ["local", "gia"] and "engine" in info["features"]
        st, body = _req(base, "POST", "/api/v1/jobs?engine=gia", _png())
        assert st == 400 and json.loads(body)["code"] == "consent_required"
        st, body = _req(base, "POST", "/api/v1/jobs?engine=khong-co&consent=1", _png())
        assert st == 400 and json.loads(body)["code"] == "bad_engine"
        assert fake.calls == 0
    finally:
        s.stop()


def test_server_external_job_done(tmp_path):
    fake = FakeEngine()
    s, base = _server(tmp_path, {"gia": fake})
    try:
        st, body = _req(base, "POST", "/api/v1/jobs?engine=gia&consent=1&level=1", _png())
        assert st == 202
        done = _wait_done(base, json.loads(body)["id"])
        assert done["state"] == "done", done
        rep = done["report"]
        assert rep["engine"] == "gia" and rep["mode"] == "external" and (rep["width"], rep["height"]) == (240, 180)
        assert fake.calls == 1
        st, img = _req(base, "GET", f"/api/v1/jobs/{done['id']}/result")
        assert st == 200 and img[:3] == b"\xff\xd8\xff"
    finally:
        s.stop()


def test_server_falls_back_to_local_when_api_fails(tmp_path):
    s, base = _server(tmp_path, {"gia": FakeEngine(fail=True)})
    try:
        st, body = _req(base, "POST", "/api/v1/jobs?engine=gia&consent=1&level=1", _png())
        done = _wait_done(base, json.loads(body)["id"])
        assert done["state"] == "done", done
        rep = done["report"]
        assert rep["engine"] == "local" and rep["engine_requested"] == "gia"
        assert "hết credit" in rep["notes"][0]
    finally:
        s.stop()


def test_server_guard_rejects_changed_face(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "face_ids", lambda eng, a, b: [0.95, 0.31])
    s, base = _server(tmp_path, {"gia": FakeEngine()})
    try:
        st, body = _req(base, "POST", "/api/v1/jobs?engine=gia&consent=1&level=1", _png())
        done = _wait_done(base, json.loads(body)["id"])
        rep = done["report"]
        assert done["state"] == "done" and rep["engine"] == "local" and rep["ids"] == [0.95, 0.31]
        assert "đổi khuôn mặt" in rep["notes"][0]
        # guard=0: tin kết quả API, không kiểm
        st, body = _req(base, "POST", "/api/v1/jobs?engine=gia&consent=1&level=1&guard=0", _png())
        done = _wait_done(base, json.loads(body)["id"])
        assert done["report"]["engine"] == "gia"
    finally:
        s.stop()


def test_face_ids_real_face():
    """Ảnh astronaut (công cộng, NASA): phóng to giữ mặt → IDS ~1; làm nhoè mặt → IDS rất thấp."""
    data = pytest.importorskip("skimage.data")
    from net_anh.pipeline import Enhancer
    eng = Enhancer("cpu")
    src = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    up = cv2.resize(src, (1024, 1024), interpolation=cv2.INTER_CUBIC)
    assert P.face_ids(eng, src, up)[0] > 0.95
    bad = src.copy()
    bad[30:200, 140:260] = cv2.GaussianBlur(bad[30:200, 140:260], (0, 0), 12)
    assert P.face_ids(eng, src, bad)[0] < P.GUARD_IDS
