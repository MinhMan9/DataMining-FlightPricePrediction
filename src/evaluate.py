from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import FEATURE_FILE, MODELS_DIR, TARGET_COL, TABLES_DIR
from src.utils import load_model, save_dataframe



def load_test_like_data(path=FEATURE_FILE) -> pd.DataFrame:
    """Đọc dữ liệu feature để phục vụ đánh giá.

    Hiện tại đây chỉ là placeholder. Khi nhóm hoàn thiện pipeline,
    nên đọc đúng test set đã tách riêng.
    """
    return pd.read_csv(path)



def compute_metrics(y_true, y_pred) -> dict:
    """Tính metric hồi quy."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
    }



def evaluate_saved_model(model_path: Path | None = None) -> dict:
    """Đánh giá một model đã lưu."""
    # TODO: sửa lại để dùng đúng X_test và y_test đã lưu ra file
    if model_path is None:
        model_path = MODELS_DIR / "baseline" / "random_forest.joblib"

    df = load_test_like_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    model = load_model(model_path)
    preds = model.predict(X)
    return compute_metrics(y, preds)



def export_model_comparison(results_df: pd.DataFrame) -> None:
    """Lưu bảng so sánh model."""
    save_dataframe(results_df, TABLES_DIR / "model_comparison.csv")


# TODO: thêm các hàm vẽ actual vs predicted, residual plot, feature importance
