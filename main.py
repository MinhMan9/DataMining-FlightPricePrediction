from __future__ import annotations

from src.data_preprocessing import run_preprocessing
from src.evaluate import export_model_comparison
from src.feature_engineering import run_step2_pipeline as run_feature_engineering
from src.train import train_baseline_models


def run_pipeline() -> None:
    """Chạy toàn bộ pipeline từ data cleaning đến baseline training."""
    print("=== STEP 1: Preprocessing ===")
    clean_df = run_preprocessing()
    print(f"Clean data shape: {clean_df.shape}")

    print("\n=== STEP 2: Feature engineering ===")
    feature_df = run_feature_engineering()
    # print(f"Feature data shape: {feature_df.shape}")

    print("\n=== STEP 3: Train baseline models ===")
    results_df = train_baseline_models()
    results_df = results_df[["model", "MAE", "RMSE", "R2"]].sort_values("RMSE")
    export_model_comparison(results_df)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
