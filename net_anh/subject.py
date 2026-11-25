"""Đoán tuổi và giới tính từng khuôn mặt để tự áp preset theo đối tượng (trẻ em, nam, nữ, người lớn tuổi).

Model (11 MB, tải ở lần dùng đầu): bộ đoán tuổi ResNet-10 (81 lớp, 0–80 tuổi) và bộ phân loại giới tính của Cydral
Technology trong github.com/davisking/dlib-models, giấy phép CC0 1.0 (public domain, dùng thương mại được); bản ONNX
lấy từ gói age-and-gender của mowshon (MIT). Chạy bằng OpenCV DNN nên không cần thêm thư viện.

Mặt được căn giống dlib get_face_chip_details (lề 0,2) từ 5 điểm RetinaFace: 64×64 cho tuổi, 32×32 cho giới tính, RGB
trừ trung bình từng kênh rồi chia 256. Kết quả chỉ dùng để chọn preset, không lưu, không gửi đi đâu; đoán sai hoặc
không chắc thì dùng preset trung tính, người dùng chọn tay được.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

AGE_BINS = np.array([0.25] + list(range(1, 81)), np.float32)  # lớp 0 là trẻ dưới 1 tuổi (như mã gốc của dlib)
MEANS_RGB = (122.781998, 117.000999, 104.297997)
AGE_SIZE, GENDER_SIZE, PADDING = 64, 32, 0.2

# Mặt chuẩn của dlib (toạ độ 0..1 trong khung mặt) cho 5 điểm của RetinaFace: tâm mắt trái, tâm mắt phải (trung bình hai
# khoé mắt của mẫu 5 điểm dlib), chóp mũi, khoé miệng trái, khoé miệng phải (điểm 30, 48, 54 của mẫu 68 điểm dlib).
TEMPLATE = np.array([[0.22733006, 0.22139585], [0.75281396, 0.22123280], [0.490127, 0.515625],
                     [0.254149, 0.780233], [0.726104, 0.780233]], np.float64)
# 5 điểm của mặt đã căn 512×512 kiểu FFHQ (facexlib FaceRestoreHelper.face_template)
FFHQ_512 = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], np.float64)

CHILD_MAX = 12  # trẻ em: 0–12 tuổi
ELDER_MIN = 60  # người lớn tuổi: từ 60 tuổi

SUBJECTS = {"child": "Trẻ em", "man": "Nam", "woman": "Nữ", "elder": "Người lớn tuổi", "adult": "Người lớn"}


def _similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Phép đồng dạng (xoay, co giãn đều, dời) khớp src → dst theo bình phương tối thiểu (Umeyama), ma trận 2×3."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    cov = b.T @ a / len(src)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[1, 1] = -1
    r = u @ s @ vt
    scale = np.trace(np.diag(d) @ s) / a.var(0).sum()
    m = np.zeros((2, 3))
    m[:, :2] = scale * r
    m[:, 2] = md - scale * r @ ms
    return m


def face_chip(img_bgr: np.ndarray, lm5: np.ndarray, size: int, padding: float = PADDING) -> np.ndarray:
    """Mặt căn size×size (RGB) như dlib: ảnh thu nhỏ dần theo kim tự tháp trước khi lấy mẫu để khỏi răng cưa."""
    dst = (padding + TEMPLATE) / (2 * padding + 1) * size
    m = _similarity(np.asarray(lm5, np.float64).reshape(5, 2), dst)
    src = img_bgr
    while np.hypot(m[0, 0], m[0, 1]) < 0.5 and min(src.shape[:2]) > 2 * size:
        src = cv2.pyrDown(src)
        m[:, :2] *= 2.0
    chip = cv2.warpAffine(src, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)


def _blob(chips_rgb: list[np.ndarray]) -> np.ndarray:
    x = np.stack(chips_rgb).astype(np.float32)
    x -= np.asarray(MEANS_RGB, np.float32)
    x /= 256.0
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


@dataclass(frozen=True)
class FaceAttr:
    age: float  # tuổi ước lượng (kỳ vọng trên 81 lớp tuổi)
    p_child: float  # xác suất từ CHILD_MAX tuổi trở xuống
    p_elder: float  # xác suất từ ELDER_MIN tuổi trở lên
    p_male: float
    subject: str  # khoá trong SUBJECTS
    confidence: float  # 0..1, xác suất của nhóm được chọn

    @property
    def label(self) -> str:
        return f"{SUBJECTS[self.subject]} (~{self.age:.0f} tuổi)"


# Ngưỡng chọn nhóm, chỉnh bằng tools/subject_bench.py (xem README): xác suất trẻ em / người lớn tuổi từ ngưỡng này
# trở lên thì chọn nhóm đó; giới tính cần chắc hơn GENDER_SURE, không thì preset trung tính "Người lớn".
CHILD_P, ELDER_P, GENDER_SURE = 0.5, 0.5, 0.65


def classify(age_prob: np.ndarray, p_male: float) -> FaceAttr:
    p = np.asarray(age_prob, np.float64).ravel()
    p = p / max(p.sum(), 1e-9)
    age = float((p * AGE_BINS).sum())
    p_child = float(p[: CHILD_MAX + 1].sum())  # lớp i là i tuổi (lớp 0: dưới 1 tuổi)
    p_elder = float(p[ELDER_MIN:].sum())
    if p_child >= CHILD_P:
        subj, conf = "child", p_child
    elif p_elder >= ELDER_P:
        subj, conf = "elder", p_elder
    elif p_male >= GENDER_SURE:
        subj, conf = "man", p_male
    elif 1 - p_male >= GENDER_SURE:
        subj, conf = "woman", 1 - p_male
    else:
        subj, conf = "adult", 1 - p_child - p_elder
    return FaceAttr(age, p_child, p_elder, float(p_male), subj, float(conf))


class AgeGender:
    """Hai mạng ONNX nhỏ chạy bằng OpenCV DNN trên CPU (vài mili giây mỗi mặt)."""

    def __init__(self, age_onnx: str, gender_onnx: str):
        self.age_net = cv2.dnn.readNetFromONNX(str(age_onnx))
        self.gender_net = cv2.dnn.readNetFromONNX(str(gender_onnx))

    @classmethod
    def load(cls, progress=None) -> "AgeGender":
        from . import weights
        a = weights.ensure("age_cydral", progress)
        g = weights.ensure("gender_cydral", progress)
        return cls(str(a), str(g))

    def probs(self, img_bgr: np.ndarray, faces_lm5: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """(xác suất 81 lớp tuổi, xác suất nam) cho từng mặt."""
        if not faces_lm5:
            return np.zeros((0, len(AGE_BINS)), np.float32), np.zeros(0, np.float32)
        self.age_net.setInput(_blob([face_chip(img_bgr, lm, AGE_SIZE) for lm in faces_lm5]))
        pa = self.age_net.forward()
        self.gender_net.setInput(_blob([face_chip(img_bgr, lm, GENDER_SIZE) for lm in faces_lm5]))
        pg = self.gender_net.forward()
        return pa.reshape(len(faces_lm5), -1), pg.reshape(len(faces_lm5), 2)[:, 1]

    def __call__(self, img_bgr: np.ndarray, faces_lm5: list[np.ndarray]) -> list[FaceAttr]:
        pa, pm = self.probs(img_bgr, faces_lm5)
        return [classify(a, float(m)) for a, m in zip(pa, pm)]

    def aligned512(self, faces512_bgr: list[np.ndarray]) -> list[FaceAttr]:
        """Như __call__ nhưng cho mặt đã căn 512×512 kiểu FFHQ (biết sẵn 5 điểm)."""
        out = []
        for f in faces512_bgr:
            k = f.shape[0] / 512.0
            out += self(f, [FFHQ_512 * k])
        return out


__all__ = ["AgeGender", "FaceAttr", "SUBJECTS", "classify", "face_chip", "CHILD_MAX", "ELDER_MIN"]
