from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    CV_FOLDS,
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


def tune_random_forest(X_train=None, y_train=None, n_iter: int = 30) -> tuple:
    """Tuning Random Forest bằng RandomizedSearchCV."""
    if X_train is None or y_train is None:
        data = load_train_test_split()
        X_train, y_train = data["X_train"], data["y_train"]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    param_dist = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [10, 15, 20, 25, 30, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.3, 0.5],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    save_model(best_pipe, MODELS_DIR / "tuned" / "random_forest_tuned.joblib")
    print(f"Best params: {search.best_params_}")
    print(f"Best CV RMSE: {-search.best_score_:.2f}")

    return best_pipe, search.best_params_, -search.best_score_
