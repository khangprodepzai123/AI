"""Trích chọn đặc trưng LBP (uniform) + histogram lưới 4×4, hỗ trợ đa vùng."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist
from skimage.feature import local_binary_pattern

import config


def _n_bins() -> int:
    """Số bin histogram cho uniform LBP với P điểm."""
    return config.LBP_P + 2


def _single_region_dim() -> int:
    gh, gw = config.LBP_GRID
    return _n_bins() * gh * gw


def num_patches() -> int:
    """Số vùng trích đặc trưng khi bật MULTI_PATCH."""
    if not config.MULTI_PATCH:
        return 1
    # full + 4 quadrants + center
    return 6


def feature_dim() -> int:
    """Chiều vector đặc trưng."""
    return num_patches() * _single_region_dim()


def _apply_clahe(gray: np.ndarray) -> np.ndarray:
    if not config.USE_CLAHE:
        return gray
    # clip_limit nhỏ tránh nhiễu quá mức trên ảnh đồng cỏ
    enhanced = equalize_adapthist(gray, clip_limit=0.02)
    return enhanced.astype(np.float32)


def preprocess_image(path: str | Path) -> np.ndarray:
    """
    Đọc ảnh → resize IMAGE_SIZE → grayscale [0, 1] → CLAHE (tùy cấu hình).
    """
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize(config.IMAGE_SIZE, Image.Resampling.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return _apply_clahe(arr)


def _resize_region(region: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize một vùng cắt về kích thước chuẩn cho LBP."""
    if region.size == 0:
        raise ValueError("Vùng ảnh rỗng")
    pil = Image.fromarray((np.clip(region, 0, 1) * 255).astype(np.uint8), mode="L")
    pil = pil.resize(size, Image.Resampling.LANCZOS)
    return np.asarray(pil, dtype=np.float32) / 255.0


def iter_patch_regions(gray: np.ndarray) -> list[np.ndarray]:
    """
    Sinh các vùng: toàn ảnh, 4 góc, 60% vùng giữa.
    Phù hợp ảnh bầy đàn / chụp xa (động vật có thể nằm rải khắp khung).
    """
    if not config.MULTI_PATCH:
        return [gray]

    h, w = gray.shape
    mid_h, mid_w = h // 2, w // 2
    margin_h, margin_w = int(h * 0.2), int(w * 0.2)

    return [
        gray,
        gray[0:mid_h, 0:mid_w],
        gray[0:mid_h, mid_w:w],
        gray[mid_h:h, 0:mid_w],
        gray[mid_h:h, mid_w:w],
        gray[margin_h : h - margin_h, margin_w : w - margin_w],
    ]


def lbp_histogram_grid(gray: np.ndarray) -> np.ndarray:
    """
    Tính LBP uniform (P, R) rồi histogram mật độ trên lưới LBP_GRID.
    """
    p, r = config.LBP_P, config.LBP_R
    gh, gw = config.LBP_GRID
    n_bins = _n_bins()

    # LBP ổn định hơn trên ảnh uint8
    gray_u8 = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray_u8, P=p, R=r, method=config.LBP_METHOD)
    h, w = lbp.shape
    cell_h, cell_w = h // gh, w // gw

    if cell_h == 0 or cell_w == 0:
        raise ValueError(
            f"Kích thước ảnh {gray.shape} quá nhỏ cho lưới {config.LBP_GRID}"
        )

    parts: list[np.ndarray] = []
    for i in range(gh):
        for j in range(gw):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            cell = lbp[y0:y1, x0:x1]
            hist, _ = np.histogram(
                cell.ravel(),
                bins=n_bins,
                range=(0, n_bins),
                density=True,
            )
            parts.append(hist.astype(np.float32))

    return np.concatenate(parts)


def extract_features_from_gray(gray: np.ndarray) -> np.ndarray:
    """Trích vector đặc trưng từ ảnh xám đã preprocess."""
    vectors: list[np.ndarray] = []
    target_size = (
        config.IMAGE_SIZE if not config.MULTI_PATCH else config.PATCH_LBP_SIZE
    )

    for region in iter_patch_regions(gray):
        if region.shape[0] < 8 or region.shape[1] < 8:
            vectors.append(np.zeros(_single_region_dim(), dtype=np.float32))
            continue
        if config.MULTI_PATCH or region.shape != config.IMAGE_SIZE:
            region = _resize_region(region, target_size)
        vectors.append(lbp_histogram_grid(region))

    return np.concatenate(vectors)


def extract_features_from_path(path: str | Path) -> np.ndarray:
    """Pipeline đầy đủ cho một file ảnh."""
    gray = preprocess_image(path)
    return extract_features_from_gray(gray)
