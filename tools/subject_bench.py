"""Đo bộ đoán tuổi, giới tính (net_anh/subject.py) để tự chọn preset theo đối tượng: trẻ em, nam, nữ, người lớn tuổi.

Ảnh: UTKFace (chỉ dùng đo, nghiên cứu phi thương mại), tên file `tuổi_giới_chủng_thời-gian.jpg.chip.jpg` (giới 0 nam,
1 nữ; chủng 2 là người châu Á). Mỗi ảnh viền thêm (replicate) để bộ dò mặt chạy ổn, đo ở 3 điều kiện:
  - goc: ảnh gốc
  - mo:  mờ Gauss σ2,2 + nhiễu + JPEG 75 (như tools/age_bench.py)
  - nho: thu nhỏ 4× + mờ nhẹ + JPEG 75 (mặt còn ~35 px)
Mặt được dò bằng RetinaFace của app trên ảnh đã làm hỏng (như khi chạy thật). Ghi từng mặt ra CSV, in ma trận nhầm lẫn
nhóm đối tượng, sai số tuổi, độ đúng giới tính theo nhóm tuổi, riêng cho người châu Á.

    python tools/subject_bench.py --dir UTKFace_chon --csv ket_qua_doi_tuong.csv [--cond goc mo nho]
    python tools/subject_bench.py --from-csv ket_qua_doi_tuong.csv   # chỉ tính lại bảng (ví dụ sau khi đổi ngưỡng)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from net_anh import subject  # noqa: E402

PAD = 56
ORDER = ["child", "man", "woman", "elder", "adult"]


def true_subject(age: int, male: bool) -> str:
    if age <= subject.CHILD_MAX:
        return "child"
    if age >= subject.ELDER_MIN:
        return "elder"
    return "man" if male else "woman"


def band(age: int) -> str:
    return "0-12" if age <= 12 else "13-17" if age <= 17 else "18-39" if age <= 39 else "40-59" if age <= 59 else "60+"


def report(rows: list[dict]) -> None:
    for cond in sorted({r["cond"] for r in rows}, key=["goc", "mo", "nho"].index):
        rs = [r for r in rows if r["cond"] == cond]
        for title, sub in (("tất cả", rs), ("người châu Á", [r for r in rs if r["race"] == "2"])):
            print(f"\n== {cond}, {title}: {len(sub)} mặt")
            conf = Counter((r["true"], r["pred"]) for r in sub)
            print("thật \\ đoán " + "".join(f"{subject.SUBJECTS[p]:>15s}" for p in ORDER) + "   đúng")
            for t in ORDER[:4]:
                n = sum(conf[(t, p)] for p in ORDER)
                if not n:
                    continue
                ok = conf[(t, t)] + (conf[(t, "adult")] if t in ("man", "woman") else 0)
                print(f"{subject.SUBJECTS[t]:>12s}" + "".join(f"{conf[(t, p)]:15d}" for p in ORDER)
                      + f"   {ok / n * 100:4.0f} %")
            exact = sum(r["true"] == r["pred"] for r in sub)
            neutral = sum(r["pred"] == "adult" for r in sub)
            wrong = sum(r["pred"] != r["true"] and not (r["pred"] == "adult" and r["true"] in ("man", "woman"))
                        for r in sub)
            print(f"đúng nhóm {exact / len(sub) * 100:.0f} %, trung tính {neutral / len(sub) * 100:.0f} %, "
                  f"sai {wrong / len(sub) * 100:.0f} %")
            by = defaultdict(list)
            for r in sub:
                by[band(int(r["age"]))].append(r)
            print("nhóm tuổi: sai số tuổi TB, đoán tuổi TB, giới tính đúng")
            for b in ("0-12", "13-17", "18-39", "40-59", "60+"):
                g = by.get(b)
                if g:
                    mae = np.mean([abs(float(r["age_pred"]) - int(r["age"])) for r in g])
                    mean_pred = np.mean([float(r["age_pred"]) for r in g])
                    gok = np.mean([(float(r["p_male"]) >= 0.5) == (str(r["male"]) == "1") for r in g])
                    print(f"  {b:>6s}: {mae:5.1f} tuổi, {mean_pred:5.1f}, {gok * 100:4.0f} % ({len(g)} mặt)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--cond", nargs="+", default=["goc", "mo", "nho"])
    ap.add_argument("--csv")
    ap.add_argument("--from-csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.from_csv:
        rows = list(csv.DictReader(open(a.from_csv, encoding="utf-8")))
        for r in rows:  # tính lại nhóm theo ngưỡng hiện tại
            pa = np.array([float(x) for x in r["age_prob"].split()], np.float32)
            att = subject.classify(pa, float(r["p_male"]))
            r["pred"], r["age_pred"] = att.subject, f"{att.age:.2f}"
        report(rows)
        return
    from net_anh.pipeline import Enhancer
    from age_bench import degrade
    eng = Enhancer("cpu")
    ag = subject.AgeGender.load()
    files = sorted(Path(a.dir).glob("*.jpg"))
    if a.limit:
        files = files[: a.limit]
    rows = []
    for ii, f in enumerate(files):
        m = re.match(r"(\d+)_([01])_(\d)_", f.name)
        if not m:
            continue
        age, male, race = int(m[1]), m[2] == "0", m[3]
        gt = cv2.copyMakeBorder(cv2.imread(str(f)), PAD, PAD, PAD, PAD, cv2.BORDER_REPLICATE)
        for ki, cond in enumerate(a.cond):
            img = gt if cond == "goc" else degrade(gt, cond, ii * 10 + ki)
            faces = eng.detect_faces(img)
            if not faces:
                continue
            box, lm = max(faces, key=lambda bl: (bl[0][2] - bl[0][0]) * (bl[0][3] - bl[0][1]))
            pa, pm = ag.probs(img, [lm])
            att = subject.classify(pa[0], float(pm[0]))
            rows.append(dict(file=f.name, age=age, male=int(male), race=race, cond=cond,
                             true=true_subject(age, male), pred=att.subject, age_pred=f"{att.age:.2f}",
                             p_child=f"{att.p_child:.3f}", p_elder=f"{att.p_elder:.3f}", p_male=f"{att.p_male:.3f}",
                             face_px=int(box[2] - box[0]), age_prob=" ".join(f"{x:.4f}" for x in pa[0])))
        if ii % 50 == 0:
            print(f"{ii}/{len(files)}", flush=True)
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    report(rows)


if __name__ == "__main__":
    main()
