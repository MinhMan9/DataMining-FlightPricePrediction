# Flight Price Prediction - Data Mining Project

## Giới thiệu

Dự án dự đoán giá vé máy bay sử dụng các kỹ thuật Data Mining và Machine Learning.  
Pipeline bao gồm: tiền xử lý dữ liệu → tạo đặc trưng → huấn luyện mô hình baseline → cải tiến mô hình → đánh giá và trực quan hóa.

## Dataset

- **File dữ liệu:** `data/raw/Data_Train.xlsx`
- **Số dòng:** 10,683
- **Số cột:** 11
- **Biến mục tiêu:** `Price` (giá vé máy bay, đơn vị INR)

## Mục tiêu

- Phân tích các yếu tố ảnh hưởng đến giá vé máy bay
- Xây dựng mô hình baseline (Linear Regression, Decision Tree)
- Cải tiến bằng mô hình ensemble (Random Forest)
- So sánh và đánh giá mô hình tốt nhất

## Kết quả

| Model             | MAE      | RMSE     | R²     |
|-------------------|----------|----------|--------|
| Random Forest     | 1,063.45 | 1,526.98 | 0.8328 |
| Decision Tree     | 1,076.84 | 1,739.16 | 0.7831 |
| Linear Regression | 1,804.90 | 2,302.48 | 0.6198 |

**Mô hình tốt nhất:** Random Forest (R² = 0.8328)

---

## Cài đặt và chạy dự án

### Yêu cầu

- Python 3.11+
- Conda (khuyến nghị) hoặc pip

### Bước 1: Clone dự án

```bash
git clone <repository-url>
cd DataMining-FlightPricePrediction
```

### Bước 2: Tạo môi trường và cài thư viện

```bash
conda create -n flightprice python=3.11 -y
conda activate flightprice
pip install -r requirements.txt
```

### Bước 3: Đặt file dữ liệu

Đặt file `Data_Train.xlsx` vào thư mục `data/raw/`.

### Bước 4: Chạy toàn bộ pipeline

```bash
python main.py
```

Pipeline sẽ tự động thực hiện:
1. **Preprocessing** — Làm sạch dữ liệu, xử lý missing values, duplicate
2. **Feature Engineering** — Tạo đặc trưng từ thời gian, thời lượng bay, số điểm dừng, one-hot encoding, tách train/test
3. **Baseline Models** — Train Linear Regression + Decision Tree
4. **Improved Models** — Train Random Forest
5. **Export** — Lưu bảng so sánh mô hình vào `reports/tables/model_comparison.csv`

### Chạy từng bước qua Notebook

Nếu muốn xem chi tiết từng bước, mở Jupyter Notebook:

```bash
jupyter notebook
```

Chạy lần lượt các notebook trong thư mục `notebooks/`:

| Thứ tự | Notebook                               | Mô tả                                                            |
|--------|----------------------------------------|-------------------------------------------------------------------|
| 1      | `01_data_understanding.ipynb`          | Khám phá dữ liệu, phân phối giá, phân tích theo hãng/điểm dừng  |
| 2      | `02_data_cleaning_feature_engineering.ipynb` | Làm sạch, tạo features, one-hot encoding, tách train/test   |
| 3      | `03_baseline_models.ipynb`             | Train baseline: Linear Regression & Decision Tree                 |
| 3b     | `03b_improved_models.ipynb`            | Train improved: Random Forest (ensemble từ Decision Tree)         |
| 4      | `04_final_evaluation_visualization.ipynb` | Đánh giá cuối, biểu đồ Actual vs Predicted, Residuals, Feature Importance |

---

## Cấu trúc thư mục

```text
DataMining-FlightPricePrediction/
│
├── main.py                          # Chạy toàn bộ pipeline
├── requirements.txt                 # Thư viện cần cài
├── README.md
│
├── data/
│   ├── raw/
│   │   └── Data_Train.xlsx          # Dữ liệu gốc
│   └── processed/                   # Dữ liệu sau xử lý (tự sinh khi chạy)
│       ├── train_clean.csv
│       ├── train_features.csv
│       ├── train_model_ready.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 03b_improved_models.ipynb
│   └── 04_final_evaluation_visualization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                    # Đường dẫn, hằng số cấu hình
│   ├── utils.py                     # Hàm tiện ích (save/load model, ensure_dir)
│   ├── data_preprocessing.py        # Làm sạch dữ liệu
│   ├── feature_engineering.py       # Tạo đặc trưng
│   ├── train.py                     # Huấn luyện mô hình
│   └── evaluate.py                  # Đánh giá và trực quan hóa
│
├── models/
│   ├── baseline/                    # Linear Regression, Decision Tree
│   └── improved/                    # Random Forest
│
└── reports/
    ├── figures/                     # Biểu đồ (tự sinh khi chạy notebook 04)
    └── tables/
        └── model_comparison.csv     # Bảng so sánh kết quả
```

---

## Mô tả các module trong `src/`

| Module                  | Chức năng                                                                 |
|-------------------------|---------------------------------------------------------------------------|
| `config.py`             | Khai báo đường dẫn dữ liệu, tên biến mục tiêu, các tham số cấu hình     |
| `utils.py`              | Hàm tiện ích: save/load model, tạo thư mục, save DataFrame               |
| `data_preprocessing.py` | Làm sạch dữ liệu, xử lý missing values, duplicate, chuẩn hóa định dạng  |
| `feature_engineering.py`| Tạo đặc trưng từ Date_of_Journey, Duration, Total_Stops, one-hot encoding|
| `train.py`              | Huấn luyện baseline (LR + DT) và improved (RF), lưu model                |
| `evaluate.py`           | Tính metrics (MAE, RMSE, R²), vẽ biểu đồ, export kết quả                |