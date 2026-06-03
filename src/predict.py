"""Dự đoán một ảnh hoặc thư mục bằng mô hình đã train."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.features import extract_features_from_path


def load_artifact(model_path: Path | None = None) -> dict:
    path = model_path or config.MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Chưa có mô hình tại {path}. Chạy: python -m src.train"
        )
    return joblib.load(path)


def predict_image(
    image_path: str | Path,
    model_path: Path | None = None,
) -> tuple[str, dict[str, float]]:
    """
    Dự đoán một ảnh.
    Trả về: (tên_lớp, dict xác suất theo lớp).
    """
    artifact = load_artifact(model_path)
    pipeline = artifact["pipeline"]
    class_names: list[str] = artifact["class_names"]

    features = extract_features_from_path(image_path).reshape(1, -1)
    proba = pipeline.predict_proba(features)[0]
    pred_idx = int(np.argmax(proba))

    prob_dict = {name: float(p) for name, p in zip(class_names, proba)}
    return class_names[pred_idx], prob_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict dog/cat")
    parser.add_argument("image", type=Path, help="Đường dẫn ảnh")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    args = parser.parse_args()

    label, probs = predict_image(args.image, args.model)
    print(f"Dự đoán: {label}")
    for name, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {name}: {p:.2%}")


if __name__ == "__main__":
    main()
