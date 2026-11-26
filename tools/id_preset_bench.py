"""Đo bộ preset ảnh thẻ trước/sau: chạy đúng đường CLI (Tự động hoặc --level) với từng bộ trong file preset, rồi so ảnh
ra với ảnh gốc trên mặt đã căn 512 (cùng ma trận căn của ảnh gốc).

    python tools/id_preset_bench.py --preset-file anh-the.netanh.json --sets "Ảnh thẻ · Nam trẻ" "Da đẹp ảnh thẻ" \
        --images nam_tre=a.jpg nu_gia=b.jpg --save /tmp/ra [--cli "--deshine 0.4"]

Số đo cho mỗi (ảnh, bộ):
  tuoi   tuổi ước lượng (kỳ vọng 81 lớp, model Cydral) của ảnh ra trừ ảnh gốc; âm = bị làm trẻ đi. Model lệch trẻ nên
         chỉ dùng hiệu số trên cùng một mặt.
  ids    giống người gốc (ArcFace cos)
  nhan   năng lượng tầng giữa của L trong vùng da (σ 1–4 % mặt: nếp nhăn, rãnh, đốm) ra / gốc; < 1 = bớt nếp nhăn, vết
  van    năng lượng tầng mịn (σ < 1 %: lỗ chân lông, vân da) ra / gốc
  L a b  màu da giữa mặt (trung vị) của ảnh ra
"""
from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from net_anh import cli, look as lookmod, pipeline  # noqa: E402
from net_anh.subject import AgeGender, FFHQ_512  # noqa: E402


class _Capture:
    eng = None
    out = None


def _engine(*a, **k):
    if _Capture.eng is None:
        _Capture.eng = pipeline.Enhancer(*a, **k)
    return _Capture.eng


def _save(img, path, quality=94):
    _Capture.out = img


def run_cli(src: str, argv: list[str]) -> np.ndarray:
    cli.Enhancer = _engine
    cli.save_image = _save
    _Capture.out = None
    code = cli.main([src, "-o", "/tmp/_bench_out.jpg", "--phone", "auto", *argv])
    if code or _Capture.out is None:
        raise RuntimeError(f"CLI lỗi ({code}) với {src}")
    return _Capture.out


def band_energy(L: np.ndarray, m: np.ndarray, s0: float, s1: float | None) -> float:
    lo = cv2.GaussianBlur(L, (0, 0), s0)
    x = L - lo if s1 is None else lo - cv2.GaussianBlur(L, (0, 0), s1)
    return float(np.sqrt(np.mean(x[m] ** 2)))


def face_metrics(eng, ag: AgeGender, ident, face: np.ndarray) -> dict:
    helper = eng._load_helper(1)
    lab_parse = eng._parse(helper, face)
    core = cv2.erode(np.isin(lab_parse, (1,)).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    lab = cv2.cvtColor(face.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    pa, pm = ag.probs(face, [FFHQ_512])
    age = float((pa[0] / pa[0].sum() * np.arange(pa.shape[1])).sum())
    return dict(age=age, male=float(pm[0]), core=core, L=L,
                skin=(float(np.median(L[core])), float(np.median(lab[..., 1][core])), float(np.median(lab[..., 2][core]))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset-file", required=True)
    p.add_argument("--sets", nargs="+", required=True, help="tên bộ trong file; san:<mã> = bộ có sẵn; 'goc' = không preset, không tông")
    p.add_argument("--images", nargs="+", required=True, help="nhom=đường_dẫn")
    p.add_argument("--own", nargs="*", default=[], help="nhom=tên bộ: bộ riêng của nhóm, gọi trong --sets bằng '@'")
    p.add_argument("--cli", default="", help="tham số CLI thêm, vd \"--level 4 --deshine 0.4\"")
    p.add_argument("--genderage", help="InsightFace genderage.onnx (chỉ đo nội bộ, không phát hành): thêm cột tuoi2")
    p.add_argument("--save")
    a = p.parse_args()
    extra = shlex.split(a.cli)
    from net_anh.identity import Identity
    ag = AgeGender.load()
    age2 = None
    if a.genderage:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from age_bench import AgeNet
        net = AgeNet(a.genderage)
        age2 = lambda f: net(f, (110, 110, 402, 430))  # noqa: E731  khung mặt trên mặt căn FFHQ 512
    rows = []
    for item in a.images:
        grp, path = item.split("=", 1)
        img = cv2.imread(path)
        outs = {}
        own = dict(x.split("=", 1) for x in a.own)
        for st in a.sets:
            if st == "@":
                if grp not in own:
                    continue
                st = own[grp]
            if st == "goc":
                argv = ["--subject", "off", "--look", "off"]
            elif st.startswith("san:"):  # bộ có sẵn trong app, vd san:da-dep-the
                argv = ["--preset-set", st[4:]]
            else:
                argv = ["--preset-file", a.preset_file, "--preset-set", st]
            outs[st] = run_cli(path, argv + extra)
        eng = _Capture.eng
        helper = eng._load_helper(1)
        helper.clean_all()
        helper.read_image(img)
        helper.get_face_landmarks_5(only_center_face=True, resize=640, eye_dist_threshold=5)
        helper.align_warp_face()
        mat = helper.affine_matrices[0]
        ident = Identity(eng.dev.device)
        f0 = helper.cropped_faces[0].copy()
        m0 = face_metrics(eng, ag, ident, f0)
        tiles = [f0]
        for st, out in outs.items():
            up = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
            f1 = cv2.warpAffine(up, mat, (512, 512), flags=cv2.INTER_LINEAR, borderValue=(135, 133, 132))
            m1 = face_metrics(eng, ag, ident, f1)
            core = m0["core"] & m1["core"]
            # tông, dải sáng tối kéo giãn L làm mọi tầng to ra: chia cho độ dốc L ra theo L gốc (thang σ 8) để chỉ còn
            # phần nếp nhăn, vân thật sự bị bớt hay thêm
            x0 = cv2.GaussianBlur(m0["L"], (0, 0), 8)[core]
            x1 = cv2.GaussianBlur(m1["L"], (0, 0), 8)[core]
            gain = float(np.clip(np.cov(x0, x1)[0, 1] / max(np.var(x0), 1e-6), 0.5, 2.0)) if x0.size > 50 else 1.0
            r = dict(grp=grp, img=Path(path).stem[-14:], set=st, tuoi=m1["age"] - m0["age"], tuoi0=m0["age"],
                     tuoi2=(age2(f1) - age2(f0)) if age2 else 0.0, ids=ident.similarity(f0, f1),
                     nhan=band_energy(m1["L"], core, 5, 20) / max(band_energy(m0["L"], core, 5, 20), 1e-6) / gain,
                     van=band_energy(m1["L"], core, 5, None) / max(band_energy(m0["L"], core, 5, None), 1e-6) / gain,
                     L=m1["skin"][0], a_=m1["skin"][1], b_=m1["skin"][2], L0=m0["skin"][0], a0=m0["skin"][1], b0=m0["skin"][2])
            rows.append(r)
            tiles.append(f1)
            print(f"{grp:8s} {r["img"]:14s} {st:26s} tuổi {r['tuoi']:+5.1f} (gốc {r['tuoi0']:4.1f}) tuổi2 {r['tuoi2']:+5.1f}  ids {r['ids']:.3f}  "
                  f"nhăn {r['nhan']:.2f}  vân {r['van']:.2f}  da L {r['L0']:.0f}→{r['L']:.0f} a {r['a0']:.1f}→{r['a_']:.1f} "
                  f"b {r['b0']:.1f}→{r['b_']:.1f}", flush=True)
            if a.save:
                Path(a.save).mkdir(parents=True, exist_ok=True)
                safe = "".join(c if c.isalnum() else "_" for c in st)
                cv2.imwrite(f"{a.save}/{grp}_{Path(path).stem}_{safe}.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if a.save:
            cv2.imwrite(f"{a.save}/mat_{grp}_{Path(path).stem}.jpg",
                        np.hstack([cv2.resize(t, (320, 320), interpolation=cv2.INTER_AREA) for t in tiles]))
    print("\nTrung bình theo nhóm, bộ:")
    keys = sorted({(r["grp"], r["set"]) for r in rows})
    for g, st in keys:
        v = [r for r in rows if r["grp"] == g and r["set"] == st]
        f = lambda k: np.mean([r[k] for r in v])  # noqa: E731
        print(f"{g:8s} {st:26s} n={len(v):2d} tuổi {f('tuoi'):+5.1f} tuổi2 {f('tuoi2'):+5.1f}  ids {f('ids'):.3f}  nhăn {f('nhan'):.2f}  "
              f"vân {f('van'):.2f}  da L {f('L'):.1f} a {f('a_'):.1f} b {f('b_'):.1f}")
    if a.save:
        import json
        Path(a.save, "so_do.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
