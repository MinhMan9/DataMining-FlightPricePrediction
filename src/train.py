from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_COL,
    X_TRAIN_FILE,
    X_TEST_FILE,
    Y_TRAIN_FILE,
    Y_TEST_FILE,
)
from src.utils import save_model


def load_train_test_split() -> dict:
    """Đọc dữ liệu train/test đã tách sẵn từ bước feature engineering."""
    X_train = pd.read_csv(X_TRAIN_FILE)
    X_test = pd.read_csv(X_TEST_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE)[TARGET_COL]
    y_test = pd.read_csv(Y_TEST_FILE)[TARGET_COL]
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def build_models() -> dict:
    """Khai báo các baseline model."""
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_regression(y_true, y_pred) -> dict:
    """Tính metric cơ bản cho hồi quy."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def train_baseline_models() -> pd.DataFrame:
    """Train các baseline models trên dữ liệu đã tách sẵn và trả về bảng kết quả."""
    data = load_train_test_split()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    models = build_models()
    results = []

    for model_name, model in models.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate_regression(y_test, preds)
        metrics["model"] = model_name
        results.append(metrics)
        save_model(pipe, MODELS_DIR / "baseline" / f"{model_name}.joblib")

    return pd.DataFrame(results)[["model", "MAE", "RMSE", "R2"]].sort_values("RMSE")


def run_cross_validation(X, y, model_name: str, model, cv: int = 5) -> dict:
    """Chạy cross-validation cho một model."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "model": model_name,
        "CV_MAE": round(-scores["test_mae"].mean(), 2),
        "CV_RMSE": round(-scores["test_rmse"].mean(), 2),
        "CV_R2": round(scores["test_r2"].mean(), 4),
    }



