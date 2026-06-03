"""
Tạo ảnh mẫu nhỏ để kiểm tra pipeline (KHÔNG dùng cho báo cáo chính thức).
Chạy: python scripts/create_sample_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

N_PER_CLASS = 40
SIZE = config.IMAGE_SIZE


def _save_class(class_name: str, seed: int) -> None:
    out_dir = config.DATA_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(N_PER_CLASS):
        # Texture giả khác nhau giữa dog/cat để RF học được gì đó
        base = 0.35 if class_name == "dog" else 0.55
        noise = rng.normal(0, 0.08, SIZE).astype(np.float32)
        grad = np.linspace(0, 0.2, SIZE[0], dtype=np.float32)[:, None]
        arr = np.clip((base + grad + noise) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(out_dir / f"sample_{i:04d}.jpg")


def main() -> None:
    _save_class("dog", seed=42)
    _save_class("cat", seed=99)
    print(f"Đã tạo {N_PER_CLASS} ảnh/lớp trong {config.DATA_DIR}")
    print("Chạy tiếp: python -m src.train")


if __name__ == "__main__":
    main()
