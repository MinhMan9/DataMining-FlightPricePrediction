from __future__ import annotations

from src.data_preprocessing import run_preprocessing
from src.evaluate import export_model_comparison
from src.feature_engineering import run_step2_pipeline as run_feature_engineering
from src.train import train_baseline_models, tune_random_forest, load_train_test_split, evaluate_regression


def run_pipeline() -> None:
    """Chạy toàn bộ pipeline từ data cleaning đến baseline training và tuning."""
    print("=== STEP 1: Preprocessing ===")
    clean_df = run_preprocessing()
    print(f"Clean data shape: {clean_df.shape}")

    print("\n=== STEP 2: Feature engineering ===")
    feature_df = run_feature_engineering()

    print("\n=== STEP 3: Train baseline models (Linear Regression + Random Forest) ===")
    results_df = train_baseline_models()
    print(results_df.to_string(index=False))

    print("\n=== STEP 4: Tune Random Forest (RandomizedSearchCV) ===")
    data = load_train_test_split()
    best_pipe, best_params, best_rmse = tune_random_forest(
        data["X_train"], data["y_train"]
    )
    # Đánh giá tuned RF trên test set
    preds = best_pipe.predict(data["X_test"])
    tuned_metrics = evaluate_regression(data["y_test"], preds)
    tuned_metrics["model"] = "random_forest_tuned"
    import pandas as pd
    tuned_row = pd.DataFrame([tuned_metrics])
    results_df = pd.concat([results_df, tuned_row], ignore_index=True)
    results_df = results_df[["model", "MAE", "RMSE", "R2"]].sort_values("RMSE")
    export_model_comparison(results_df)
    print("\n=== KẾT QUẢ CUỐI CÙNG ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_pipeline()
