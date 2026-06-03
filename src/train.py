"""Huấn luyện Random Forest trên đặc trưng LBP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Cho phép chạy: python -m src.train hoặc python src/train.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data_loader import collect_image_paths, stratified_split
from src.features import (
    extract_features_from_path,
    feature_dim,
    num_patches,
)


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=config.RF_N_ESTIMATORS,
                    max_depth=config.RF_MAX_DEPTH,
                    min_samples_split=config.RF_MIN_SAMPLES_SPLIT,
                    class_weight=config.RF_CLASS_WEIGHT,
                    random_state=config.RANDOM_STATE,
                    n_jobs=config.RF_N_JOBS,
                ),
            ),
        ]
    )


def extract_all_features(
    paths: list[Path],
    cache_path: Path | None = None,
    force: bool = False,
) -> np.ndarray:
    """Trích LBP cho toàn bộ ảnh; có thể cache ra .npz."""
    expected_dim = feature_dim()
    if cache_path and cache_path.is_file() and not force:
        data = np.load(cache_path, allow_pickle=True)
        same_paths = len(data["paths"]) == len(paths) and all(
            str(p) == str(sp) for p, sp in zip(paths, data["paths"])
        )
        same_dim = int(data["feature_dim"]) == expected_dim
        if same_paths and same_dim:
            return data["X"]
        if same_paths and not same_dim:
            print(
                f"Stale cache (dim {data['feature_dim']} -> {expected_dim}), "
                "recomputing LBP..."
            )

    X = np.zeros((len(paths), expected_dim), dtype=np.float32)
    for i, path in enumerate(tqdm(paths, desc="LBP features", unit="img")):
        X[i] = extract_features_from_path(path)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X,
            paths=np.array([str(p) for p in paths]),
            feature_dim=np.array(expected_dim),
        )

    return X


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    n_cls = len(class_names)
    fig_size = max(6, min(14, n_cls * 0.9))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted",
        title="Confusion matrix (test)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def train(
    data_dir: Path | None = None,
    use_cache: bool = True,
    force_features: bool = False,
) -> dict:
    paths, y, class_names = collect_image_paths(data_dir)
    print(f"Classes: {class_names}")
    print(f"Total images: {len(paths)}")

    idx_train, idx_val, idx_test = stratified_split(y)
    print(
        f"Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}"
    )

    cache_path = config.FEATURES_CACHE_PATH if use_cache else None
    X = extract_all_features(paths, cache_path=cache_path, force=force_features)

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    pipeline = build_pipeline()
    print("Training Random Forest...")
    pipeline.fit(X_train, y_train)

    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    report = classification_report(
        y_test, y_test_pred, target_names=class_names, output_dict=True
    )

    print(f"Accuracy validation: {val_acc:.4f}")
    print(f"Accuracy test:       {test_acc:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=class_names))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "class_names": class_names,
        "config": {
            "image_size": config.IMAGE_SIZE,
            "patch_lbp_size": config.PATCH_LBP_SIZE,
            "use_clahe": config.USE_CLAHE,
            "multi_patch": config.MULTI_PATCH,
            "num_patches": num_patches(),
            "lbp_p": config.LBP_P,
            "lbp_r": config.LBP_R,
            "lbp_method": config.LBP_METHOD,
            "lbp_grid": config.LBP_GRID,
            "feature_dim": feature_dim(),
            "rf_class_weight": config.RF_CLASS_WEIGHT,
        },
    }
    joblib.dump(artifact, config.MODEL_PATH)

    with open(config.CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    metrics = {
        "n_samples": len(paths),
        "n_train": int(len(idx_train)),
        "n_val": int(len(idx_val)),
        "n_test": int(len(idx_test)),
        "classes": class_names,
        "val_accuracy": float(val_acc),
        "test_accuracy": float(test_acc),
        "classification_report": report,
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm_path = config.REPORTS_DIR / "confusion_matrix_test.png"
    save_confusion_matrix(y_test, y_test_pred, class_names, cm_path)
    print(f"Saved model: {config.MODEL_PATH}")
    print(f"Metrics: {config.METRICS_PATH}")
    print(f"Confusion matrix: {cm_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LBP + Random Forest")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help="Thư mục data/raw (dog/, cat/, ...)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Không đọc/ghi cache đặc trưng",
    )
    parser.add_argument(
        "--force-features",
        action="store_true",
        help="Tính lại LBP bỏ qua cache",
    )
    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        use_cache=not args.no_cache,
        force_features=args.force_features,
    )


if __name__ == "__main__":
    main()
