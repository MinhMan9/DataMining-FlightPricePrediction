from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_FILE,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from src.utils import save_model



def load_feature_data(path=FEATURE_FILE) -> pd.DataFrame:
    """Đọc dữ liệu sau feature engineering."""
    return pd.read_csv(path)



def get_feature_lists(df: pd.DataFrame):
    """Lấy danh sách feature thực sự có trong DataFrame."""
    numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    # TODO: thêm logic nếu nhóm tạo thêm feature mới
    return numeric, categorical



def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """Tạo preprocessor chung cho numeric + categorical."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )



def build_models() -> dict:
    """Khai báo các baseline model."""
    return {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        # TODO: thêm XGBoost / LightGBM / CatBoost nếu nhóm phụ trách modeling muốn bổ sung
    }



def split_data(df: pd.DataFrame):
    """Tách train/test."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)



def evaluate_regression(y_true, y_pred) -> dict:
    """Tính metric cơ bản cho hồi quy."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred),
    }



def train_baseline_models() -> pd.DataFrame:
    """Train các baseline models và trả về bảng kết quả."""
    df = load_feature_data()
    numeric_features, categorical_features = get_feature_lists(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models()
    X_train, X_test, y_train, y_test = split_data(df)

    results = []
    for model_name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate_regression(y_test, preds)
        metrics["model"] = model_name
        results.append(metrics)
        save_model(pipe, MODELS_DIR / "baseline" / f"{model_name}.joblib")

    return pd.DataFrame(results)



def run_cross_validation(df: pd.DataFrame, model_name: str, model, cv: int = 5) -> dict:
    """Chạy cross-validation cho một model."""
    numeric_features, categorical_features = get_feature_lists(df)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
    # TODO: tổng hợp scores thành dict dễ đọc hơn
    return {"model": model_name, "cv_scores": scores}


# TODO: thêm hàm tuning cho RandomForest / XGBoost / LightGBM
