from __future__ import annotations

import pandas as pd

from src.config import RAW_FILE, CLEAN_FILE
from src.utils import save_dataframe



def load_raw_data(path=RAW_FILE) -> pd.DataFrame:
    """Đọc dữ liệu gốc từ Excel."""
    return pd.read_excel(path)



def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột nếu nhóm muốn chuyển về snake_case."""
    # TODO: quyết định có đổi tên cột hay giữ nguyên tên gốc
    return df.copy()


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Xóa dòng trùng lặp."""
    return df.drop_duplicates().copy()



def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý missing values cho Route và Total_Stops."""
    cleaned = df.copy()
    # TODO: thống nhất chiến lược xử lý missing (drop hay fill)
    return cleaned



def basic_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline làm sạch cơ bản."""
    cleaned = standardize_column_names(df)
    cleaned = remove_duplicates(cleaned)
    cleaned = handle_missing_values(cleaned)
    # TODO: thêm kiểm tra giá trị bất thường nếu cần
    return cleaned



def save_clean_data(df: pd.DataFrame, path=CLEAN_FILE) -> None:
    """Lưu dữ liệu sạch ra CSV."""
    save_dataframe(df, path)



def run_preprocessing() -> pd.DataFrame:
    """Chạy toàn bộ bước preprocessing cơ bản."""
    df = load_raw_data()
    cleaned = basic_cleaning_pipeline(df)
    save_clean_data(cleaned)
    return cleaned


if __name__ == "__main__":
    run_preprocessing()
