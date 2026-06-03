"""Cấu hình dự án phân loại động vật (LBP + Random Forest)."""

from pathlib import Path

# Đường dẫn gốc dự án
PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset: mỗi lớp là một thư mục con (dog/, cat/, cow/, ...)
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Artefacts
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
CACHE_DIR = PROJECT_ROOT / "cache"

MODEL_PATH = MODELS_DIR / "pipeline.joblib"
CLASSES_PATH = MODELS_DIR / "classes.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
FEATURES_CACHE_PATH = CACHE_DIR / "features_multipatch_v1.npz"

# Ảnh — kích thước lớn hơn giúp ảnh chụp xa / vật thể nhỏ trong khung
IMAGE_SIZE = (224, 224)
# Mỗi vùng (toàn ảnh / góc / giữa) resize về cỡ này trước LBP
PATCH_LBP_SIZE = (128, 128)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Tiền xử lý: tăng tương phản cục bộ (ảnh xa thường mờ / tối)
USE_CLAHE = True

# Đa vùng: toàn ảnh + 4 góc + vùng giữa → bắt texture khi nhiều con / xa
MULTI_PATCH = True

# LBP
LBP_P = 8
LBP_R = 1
LBP_METHOD = "uniform"
LBP_GRID = (4, 4)  # histogram 4×4 trên mỗi vùng

# Chia dữ liệu (stratified)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42

# Random Forest — cân bằng lớp (pig ít ảnh hơn cat, ...)
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 40
RF_MIN_SAMPLES_SPLIT = 2
RF_CLASS_WEIGHT = "balanced"
RF_N_JOBS = -1
