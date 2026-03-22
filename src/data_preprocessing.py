from __future__ import annotations

import pandas as pd

from src.config import RAW_FILE, CLEAN_FILE
from src.utils import save_dataframe


TEXT_COLUMNS = ["Airline", "Source", "Destination", "Total_Stops"]
REQUIRED_COLUMNS = ["Total_Stops"]



def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột ở mức an toàn (strip khoảng trắng)."""
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Xóa dòng trùng lặp."""
    return df.drop_duplicates().copy()


def normalize_text_columns(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Chuẩn hóa dữ liệu text để hạn chế mismatch category."""
    cleaned = df.copy()
    target_columns = columns or TEXT_COLUMNS

    for col in target_columns:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype("string").str.strip()

    if "Total_Stops" in cleaned.columns:
        cleaned["Total_Stops"] = cleaned["Total_Stops"].str.lower()

    return cleaned


def validate_required_columns(df: pd.DataFrame, required_columns: list[str] | None = None) -> None:
    """Kiểm tra các cột bắt buộc cho bước clean dữ liệu."""
    required = required_columns or REQUIRED_COLUMNS
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu cột bắt buộc trong dữ liệu: {missing_cols}")



def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Xử lý missing values và loại bỏ các cột không cần thiết."""
    cleaned = df.copy()
    validate_required_columns(cleaned)
    cleaned = cleaned.dropna(subset=["Total_Stops"])
    cleaned = cleaned.drop(columns=["Route", "Additional_Info"], errors="ignore")

    return cleaned



def basic_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline làm sạch cơ bản."""
    cleaned = standardize_column_names(df)
    cleaned = remove_duplicates(cleaned)
    cleaned = normalize_text_columns(cleaned)
    cleaned = handle_missing_values(cleaned)
    return cleaned



def save_clean_data(df: pd.DataFrame, path=CLEAN_FILE) -> None:
    """Lưu dữ liệu sạch ra CSV."""
    save_dataframe(df, path)


def run_preprocessing(path=RAW_FILE) -> pd.DataFrame:
    """Đọc dữ liệu thô, chạy cleaning pipeline và lưu dữ liệu sạch."""
    raw_df = pd.read_excel(path)
    clean_df = basic_cleaning_pipeline(raw_df)
    save_clean_data(clean_df)
    return clean_df

