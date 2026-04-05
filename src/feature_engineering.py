from __future__ import annotations

import re
from pathlib import Path

import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLEAN_FILE,
    FEATURE_FILE,
    MODEL_READY_FILE,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    X_TEST_FILE,
    X_TRAIN_FILE,
    Y_TEST_FILE,
    Y_TRAIN_FILE,
)
from src.utils import save_dataframe



def load_clean_data(path=CLEAN_FILE) -> pd.DataFrame:
    return pd.read_csv(path)



def extract_journey_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tách ngày, tháng và thứ từ Date_of_Journey."""
    fe = df.copy()
    journey_dt = pd.to_datetime(fe["Date_of_Journey"], format="%d/%m/%Y", errors="coerce")
    
    fe["journey_day"] = journey_dt.dt.day
    fe["journey_month"] = journey_dt.dt.month
    
    # Thêm thuộc tính thứ trong tuần (0 = Thứ Hai, 6 = Chủ Nhật)
    fe["day_of_week"] = journey_dt.dt.dayofweek 
    
    return fe



def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tách giờ/phút từ Dep_Time và Arrival_Time."""
    fe = df.copy()
    dep_dt = pd.to_datetime(fe["Dep_Time"], format="%H:%M", errors="coerce")

    # Arrival_Time có thể có thêm ngày như "01:10 22 Mar", chỉ lấy phần HH:MM đầu tiên.
    arrival_hhmm = fe["Arrival_Time"].astype(str).str.extract(r"(\d{1,2}:\d{2})", expand=False)
    arr_dt = pd.to_datetime(arrival_hhmm, format="%H:%M", errors="coerce")

    fe["dep_hour"] = dep_dt.dt.hour
    fe["arrival_hour"] = arr_dt.dt.hour
    return fe



def duration_to_minutes(duration_text: str) -> float:
    """Chuyển chuỗi Duration như '2h 50m' thành số phút."""
    if pd.isna(duration_text):
        return np.nan
    hours = 0
    minutes = 0
    hour_match = re.search(r"(\d+)h", str(duration_text))
    minute_match = re.search(r"(\d+)m", str(duration_text))
    if hour_match:
        hours = int(hour_match.group(1))
    if minute_match:
        minutes = int(minute_match.group(1))
    return hours * 60 + minutes



def extract_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo duration_hour, duration_minutes từ Duration."""
    fe = df.copy()
    fe["duration_minutes"] = fe["Duration"].apply(duration_to_minutes)
    return fe



def map_total_stops(df: pd.DataFrame) -> pd.DataFrame:
    """Map Total_Stops sang total_stops_num."""
    fe = df.copy()
    stop_map = {
        "non-stop": 0,
        "1 stop": 1,
        "2 stops": 2,
        "3 stops": 3,
        "4 stops": 4,
    }
    normalized = fe["Total_Stops"].astype("string").str.strip().str.lower()
    fe["total_stops_num"] = normalized.map(stop_map)
    return fe


def binary_encode_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Binary Encoding cho các cột categorical bằng category_encoders."""
    fe = df.copy()
    cols = [col for col in columns if col in fe.columns]
    if cols:
        encoder = ce.BinaryEncoder(cols=cols, return_df=True)
        fe = encoder.fit_transform(fe)
    return fe



def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Chạy toàn bộ feature engineering."""
    # Lấy thứ, ngày, tháng từ Date_of_Journey
    fe = extract_journey_features(df)

    # Tách giờ từ Dep_Time và Arrival_Time
    fe = extract_time_features(fe)

    # Chuyển Duration thành số phút
    fe = extract_duration_features(fe)

    # Map Total_Stops sang số lượng dừng
    fe = map_total_stops(fe)

    # Binary encode các cột Airline, Source, Destination
    fe = binary_encode_categorical(fe, columns=["Airline", "Source", "Destination"])
    return fe



def save_feature_data(df: pd.DataFrame, path=FEATURE_FILE) -> None:
    """Lưu dữ liệu sau feature engineering."""
    save_dataframe(df, path)


def build_model_ready_dataframe(
    feature_df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Tạo bản dữ liệu model-ready cho huấn luyện."""
    drop_cols = ["Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration", "Route", "Total_Stops"]
    model_ready_df = feature_df.drop(columns=drop_cols, errors="ignore")

    object_cols = [
        col
        for col in model_ready_df.select_dtypes(include=["object", "string"]).columns
        if col != target_col
    ]
    if object_cols:
        model_ready_df = pd.get_dummies(model_ready_df, columns=object_cols, drop_first=False, dtype=int)

    return model_ready_df


def save_model_ready_data(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Lưu dữ liệu model-ready."""
    output_path = path or MODEL_READY_FILE
    save_dataframe(df, output_path)
    return output_path


def split_and_save_train_test(
    model_ready_df: pd.DataFrame,
    time_reference_df: pd.DataFrame | None = None,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    """Tách train/test theo thời gian (quá khứ -> tương lai) và lưu các file riêng.

    Nếu không có cột thời gian hợp lệ trong ``time_reference_df``, sẽ fallback sang random split.
    """
    if target_col not in model_ready_df.columns:
        raise ValueError(f"Không tìm thấy cột target '{target_col}' trong model_ready_df")

    X = model_ready_df.drop(columns=[target_col])
    y = model_ready_df[target_col]

    use_time_split = (
        time_reference_df is not None
        and "Date_of_Journey" in time_reference_df.columns
        and len(time_reference_df) == len(model_ready_df)
    )

    if use_time_split:
        journey_dt = pd.to_datetime(
            time_reference_df["Date_of_Journey"], format="%d/%m/%Y", errors="coerce"
        )

        if "Dep_Time" in time_reference_df.columns:
            dep_dt = pd.to_datetime(time_reference_df["Dep_Time"], format="%H:%M", errors="coerce")
            dep_minutes = dep_dt.dt.hour.fillna(0) * 60 + dep_dt.dt.minute.fillna(0)
            timestamp = journey_dt + pd.to_timedelta(dep_minutes, unit="m")
        else:
            timestamp = journey_dt

        # NaT được đẩy về đầu để tránh rò rỉ dữ liệu tương lai vào test set.
        ordered_index = timestamp.sort_values(na_position="first").index
        if len(ordered_index) < 2:
            raise ValueError("Không đủ dữ liệu để tách train/test")

        split_idx = int(len(ordered_index) * (1 - test_size))
        split_idx = max(1, min(split_idx, len(ordered_index) - 1))

        train_idx = ordered_index[:split_idx]
        test_idx = ordered_index[split_idx:]

        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    save_dataframe(X_train, X_TRAIN_FILE)
    save_dataframe(X_test, X_TEST_FILE)
    y_train.to_frame(name=target_col).to_csv(Y_TRAIN_FILE, index=False)
    y_test.to_frame(name=target_col).to_csv(Y_TEST_FILE, index=False)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def run_step2_pipeline() -> dict[str, pd.DataFrame | Path]:
    """Chạy bước 2: feature, model-ready, split train/test."""
    clean_df = load_clean_data()
    feature_df = build_feature_dataframe(clean_df)
    save_feature_data(feature_df)

    model_ready_df = build_model_ready_dataframe(feature_df)
    model_ready_path = save_model_ready_data(model_ready_df)
    split_dict = split_and_save_train_test(
        model_ready_df,
        time_reference_df=feature_df,
    )

    return {
        "clean_df": clean_df,
        "feature_df": feature_df,
        "model_ready_df": model_ready_df,
        "model_ready_path": model_ready_path,
        **split_dict,
    }
