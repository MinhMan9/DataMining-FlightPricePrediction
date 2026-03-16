# Flight Price Prediction - Data Mining Project

## Giới thiệu
Dự án này thực hiện bài toán dự đoán giá vé máy bay bằng các kỹ thuật Data Mining và Machine Learning.

## Dataset
- File dữ liệu: `Data_Train.xlsx`
- Số dòng: 10,683
- Số cột: 11
- Biến mục tiêu: `Price`

## Mục tiêu
- Phân tích các yếu tố ảnh hưởng đến giá vé máy bay
- Xây dựng nhiều mô hình hồi quy để dự đoán giá vé
- So sánh và lựa chọn mô hình tốt nhất


## Cài đặt môi trường

- Bước 1: Tạo môi trường conda

```bash
conda create -n flightprice python=3.11 -y
```

- Bước 2: Kích hoạt môi trường

```bash
conda activate flightprice
```

- Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

- Bước 4: Mở Jupyter Notebook

```bash
jupyter notebook
```

## Thao tác trên Jupyter Notebook

- Bước 1: Kích hoạt môi trường

```bash
conda activate flightprice
```

- Bước 2: Mở Jupyter Notebook

```bash
jupyter notebook
```

## Cấu trúc thư mục dự án

```text
DataMining-FlightPricePrediction/
│
├── data/
│   ├── raw/
│   │   └── Data_Train.xlsx
│   ├── processed/
│   │   ├── train_clean.csv
│   │   ├── train_fe.csv
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── external/
│
├── notebooks/
│   ├── 01_data_understanding_eda.ipynb
│   ├── 02_data_cleaning_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_advanced_models_tuning.ipynb
│   └── 05_final_evaluation_visualization.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   ├── baseline/
│   ├── tuned/
│   └── final_model.joblib
│
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── slides/
│   └── final_report.pdf
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

### Mô tả các thư mục và tệp chính

data/
```
Thư mục chứa toàn bộ dữ liệu của dự án.
	•	data/raw/: chứa dữ liệu gốc, chưa qua xử lý.
	•	data/processed/: chứa dữ liệu sau khi làm sạch, tạo đặc trưng và tách train/test.
	•	data/external/: dùng để lưu dữ liệu bổ sung từ bên ngoài nếu nhóm mở rộng dự án sau này.
```

notebooks/
```
Chứa các file Jupyter Notebook phục vụ cho quá trình phân tích và thực nghiệm theo từng giai đoạn.
	•	01_data_understanding_eda.ipynb: đọc dữ liệu, khám phá dữ liệu, kiểm tra missing values, duplicate, trực quan hóa ban đầu.
	•	02_data_cleaning_feature_engineering.ipynb: làm sạch dữ liệu và tạo các đặc trưng mới từ thời gian, thời lượng bay, số điểm dừng.
	•	03_baseline_models.ipynb: huấn luyện và so sánh các mô hình cơ bản.
	•	04_advanced_models_tuning.ipynb: thử các mô hình nâng cao và tinh chỉnh siêu tham số.
	•	05_final_evaluation_visualization.ipynb: đánh giá mô hình cuối cùng và trực quan hóa kết quả.
```

src/
```
Chứa mã nguồn Python chính của dự án, được tổ chức thành các module để tái sử dụng.
	•	config.py: khai báo đường dẫn dữ liệu, tên biến mục tiêu, các cấu hình chung.
	•	utils.py: các hàm tiện ích như đọc/lưu dữ liệu, in thông tin cơ bản, hỗ trợ xử lý file.
	•	data_preprocessing.py: các hàm làm sạch dữ liệu, xử lý missing values, duplicate, chuẩn hóa định dạng cột.
	•	feature_engineering.py: các hàm tạo đặc trưng mới từ Date_of_Journey, Dep_Time, Arrival_Time, Duration, Total_Stops, …
	•	train.py: chứa logic huấn luyện mô hình.
	•	evaluate.py: chứa các hàm đánh giá mô hình bằng các chỉ số như MAE, RMSE, R², MAPE.
```

models/
```
Thư mục lưu các mô hình đã huấn luyện.
	•	models/baseline/: lưu các mô hình cơ bản.
	•	models/tuned/: lưu các mô hình sau khi tuning.
	•	final_model.joblib: mô hình cuối cùng được chọn để sử dụng và báo cáo kết quả.
```

reports/
```
Chứa các sản phẩm phục vụ báo cáo và thuyết trình.
	•	reports/figures/: lưu hình ảnh biểu đồ dùng trong báo cáo và slide.
	•	reports/tables/: lưu các bảng kết quả, bảng so sánh mô hình.
	•	reports/slides/: lưu file slide thuyết trình.
	•	final_report.pdf: báo cáo hoàn chỉnh của nhóm.
```

main.py
```
Tệp chạy chính của dự án, có thể dùng để gọi pipeline tổng quát khi cần.
```