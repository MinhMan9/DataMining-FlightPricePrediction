from __future__ import annotations

from src.data_preprocessing import run_preprocessing
from src.evaluate import export_model_comparison
from src.feature_engineering import run_step2_pipeline as run_feature_engineering
from src.train import train_baseline_models, train_improved_models

import pandas as pd


def run_pipeline() -> None:
    """Chạy toàn bộ pipeline từ data cleaning đến model training."""
    print("=== STEP 1: Preprocessing ===")
    clean_df = run_preprocessing()
    print(f"Clean data shape: {clean_df.shape}")

    print("\n=== STEP 2: Feature engineering ===")
    feature_df = run_feature_engineering()
    print(f"Feature engineering completed. Train shape: {feature_df['X_train'].shape}, Test shape: {feature_df['X_test'].shape}")

    print("\n=== STEP 3: Train baseline models (Linear Regression + Decision Tree) ===")
    baseline_df = train_baseline_models()
    print(baseline_df.to_string(index=False))

    print("\n=== STEP 4: Train improved models (Random Forest) ===")
    improved_df = train_improved_models()
    print(improved_df.to_string(index=False))

    print("\n=== KẾT QUẢ CUỐI CÙNG ===")
    all_results = pd.concat([baseline_df, improved_df], ignore_index=True).sort_values("RMSE")
    export_model_comparison(all_results)
    print(all_results.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
