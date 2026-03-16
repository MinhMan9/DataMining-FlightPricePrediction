from __future__ import annotations

import re
import pandas as pd

from src.config import CLEAN_FILE, FEATURE_FILE
from src.utils import save_dataframe



def load_clean_data(path=CLEAN_FILE) -> pd.DataFrame:
    """Đọc dữ liệu sạch."""
    return pd.read_csv(path)



def extract_journey_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tách ngày và tháng từ Date_of_Journey."""
    fe = df.copy()
    # TODO: parse cột Date_of_Journey sang datetime và tạo journey_day, journey_month
    return fe



def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tách giờ/phút từ Dep_Time và Arrival_Time."""
    fe = df.copy()
    # TODO: tạo dep_hour, dep_minute, arrival_hour, arrival_minute
    return fe



def duration_to_minutes(duration_text: str) -> int:
    """Chuyển chuỗi Duration như '2h 50m' thành số phút."""
    if pd.isna(duration_text):
        return None
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
    # TODO: apply duration_to_minutes cho cột Duration
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
    # TODO: map cột Total_Stops bằng stop_map
    return fe



# def optional_route_features(df: pd.DataFrame) -> pd.DataFrame:
#     """Feature nâng cao từ Route nếu nhóm muốn làm thêm."""
#     fe = df.copy()
#     # TODO: có thể tạo số chặng, số sân bay trung gian, route_simple...
#     return fe


def one_hot_encode_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    fe = df.copy()
    # TODO: áp dụng one-hot encoding cho các cột categorical nếu cần
    return fe


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Chạy toàn bộ feature engineering."""
    fe = extract_journey_features(df)
    fe = extract_time_features(fe)
    fe = extract_duration_features(fe)
    fe = map_total_stops(fe)
    fe = one_hot_encode_categorical(fe, columns=["Airline", "Source", "Destination"])
    # fe = optional_route_features(fe)
    # TODO: chọn drop các cột gốc nào sau khi tạo feature
    return fe



def save_feature_data(df: pd.DataFrame, path=FEATURE_FILE) -> None:
    """Lưu dữ liệu sau feature engineering."""
    save_dataframe(df, path)



def run_feature_engineering() -> pd.DataFrame:
    """Đọc clean data, tạo feature, lưu file."""
    df = load_clean_data()
    fe = build_feature_dataframe(df)
    save_feature_data(fe)
    return fe


if __name__ == "__main__":
    run_feature_engineering()
