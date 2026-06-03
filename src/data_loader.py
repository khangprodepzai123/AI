"""Quét dataset theo thư mục lớp và chia train/val/test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

import config


def discover_classes(data_dir: Path | None = None) -> list[str]:
    """Liệt kê tên lớp từ thư mục con có ảnh (sắp xếp alphabet)."""
    root = data_dir or config.DATA_DIR
    if not root.is_dir():
        return []

    classes: list[str] = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and _count_images(sub) > 0:
            classes.append(sub.name)
    return classes


def _count_images(folder: Path) -> int:
    n = 0
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS:
            n += 1
    return n


def collect_image_paths(
    data_dir: Path | None = None,
    classes: list[str] | None = None,
) -> tuple[list[Path], np.ndarray, list[str]]:
    """
    Thu thập đường dẫn ảnh và nhãn số.
    Trả về: paths, y (int), class_names (thứ tự label 0..K-1).
    """
    root = data_dir or config.DATA_DIR
    class_names = classes or discover_classes(root)
    if not class_names:
        raise FileNotFoundError(
            f"Không tìm thấy lớp nào trong {root}. "
            f"Tạo thư mục ví dụ: {root / 'dog'}, {root / 'cat'} và thêm ảnh JPG."
        )

    paths: list[Path] = []
    labels: list[int] = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Thiếu thư mục lớp: {class_dir}")

        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in config.IMAGE_EXTENSIONS:
                paths.append(img_path)
                labels.append(label_idx)

    if not paths:
        raise FileNotFoundError(f"Không có ảnh trong {root}.")

    return paths, np.array(labels, dtype=np.int32), class_names


def stratified_split(
    y: np.ndarray,
    train_ratio: float = config.TRAIN_RATIO,
    val_ratio: float = config.VAL_RATIO,
    test_ratio: float = config.TEST_RATIO,
    random_state: int = config.RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chia chỉ số train / val / test (stratified)."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Tỷ lệ chia phải cộng bằng 1, hiện tại: {total}")

    indices = np.arange(len(y))

    idx_train, idx_temp, _, y_temp = train_test_split(
        indices,
        y,
        test_size=(1.0 - train_ratio),
        stratify=y,
        random_state=random_state,
    )

    # idx_temp: 30% còn lại → chia đôi val/test (15% / 15%)
    relative_test = test_ratio / (val_ratio + test_ratio)

    idx_val, idx_test, _, _ = train_test_split(
        idx_temp,
        y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=random_state,
    )

    return idx_train, idx_val, idx_test
