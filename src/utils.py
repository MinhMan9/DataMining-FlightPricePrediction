from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd


def ensure_dir(path: Path) -> None:
    """Tạo thư mục nếu chưa tồn tại."""
    path.mkdir(parents=True, exist_ok=True)



def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Lưu DataFrame ra CSV."""
    ensure_dir(path.parent)
    df.to_csv(path, index=index)



def save_model(model, path: Path) -> None:
    """Lưu model ra file joblib."""
    ensure_dir(path.parent)
    joblib.dump(model, path)



def load_model(path: Path):
    """Đọc model từ file joblib."""
    return joblib.load(path)



def print_section(title: str) -> None:
    """In tiêu đề section cho dễ đọc log."""
    print(f"{'=' * 20} {title} {'=' * 20}")


# TODO: thêm các hàm tiện ích như logging, timer, export metrics nếu cần
