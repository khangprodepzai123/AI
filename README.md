# Phân loại động vật — LBP + Random Forest

Dự án báo cáo cuối kỳ: **LBP** + **Random Forest**, hỗ trợ nhiều lớp (dog, cat, cow, …) và ảnh **chụp xa / bầy đàn**.

## Cấu hình hiện tại

| Thông số | Giá trị |
|----------|---------|
| Resize | **224×224**, grayscale + **CLAHE** |
| Đa vùng | Toàn ảnh + 4 góc + vùng giữa (mỗi vùng LBP 128×128, grid 4×4) |
| Vector đặc trưng | **960** chiều (6 vùng × 160) |
| LBP | uniform, P=8, R=1 |
| Augmentation | Không |
| Chia dữ liệu | 70% / 15% / 15%, stratified |
| Random Forest | 300 cây, `class_weight=balanced` (lớp ít ảnh vẫn được coi trọng) |

### Vì sao đổi pipeline?

Ảnh **cận mặt** → LBP trên 128×128 đủ. Ảnh **xa / nhiều con** → vật thể nhỏ, texture pha nền → dùng ảnh lớn hơn, CLAHE, và **trích LBP từ nhiều vùng** rồi ghép vector.

## Cấu trúc thư mục

```
data/raw/
  dog/
  cat/
  cow/
  ...
models/pipeline.joblib
reports/metrics.json
reports/confusion_matrix_test.png
cache/features_multipatch_v1.npz
```

## Cài đặt

```powershell
cd d:\work\AIcuoiky
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Kiểm tra dataset

```powershell
python scripts\validate_dataset.py
```

## Huấn luyện (sau khi thêm ảnh / lớp mới)

```powershell
python -m src.train --force-features
```

Lần đầu với ~20k ảnh có thể mất **30–90 phút** (CPU). Lần sau nhanh hơn nhờ cache.

## Giao diện

`run_app.bat` hoặc:

```powershell
python -m streamlit run app.py
```

## Thêm lớp hoặc đổi ảnh

1. Thêm/sửa thư mục trong `data/raw/<ten_lop>/`
2. `python -m src.train --force-features`
3. Chạy lại Streamlit

## Báo cáo

- `reports/metrics.json`
- `reports/confusion_matrix_test.png`

## Giới hạn (nên ghi trong báo cáo)

- Không phát hiện từng con trong ảnh bầy đàn (không dùng object detection).
- Ảnh nhiều loài lẫn lộn có thể nhầm lớp.
- LBP + RF phù hợp báo cáo cổ điển; accuracy có thể thấp hơn CNN trên ảnh rất khó.
# AI
