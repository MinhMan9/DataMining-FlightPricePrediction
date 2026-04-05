from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    MODELS_DIR, TARGET_COL, TABLES_DIR, FIGURES_DIR,
    X_TEST_FILE, Y_TEST_FILE,
)
from src.utils import load_model, save_dataframe, ensure_dir


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Đọc X_test và y_test đã tách sẵn."""
    X_test = pd.read_csv(X_TEST_FILE)
    y_test = pd.read_csv(Y_TEST_FILE)[TARGET_COL]
    return X_test, y_test


def compute_metrics(y_true, y_pred) -> dict:
    """Tính metric hồi quy."""
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2_score(y_true, y_pred), 4),
    }


def evaluate_saved_model(model_path: Path, X_test=None, y_test=None) -> dict:
    """Đánh giá một model đã lưu trên test set."""
    if X_test is None or y_test is None:
        X_test, y_test = load_test_data()
    model = load_model(model_path)
    preds = model.predict(X_test)
    return compute_metrics(y_test, preds)


def evaluate_all_models(model_dir: Path, X_test=None, y_test=None) -> pd.DataFrame:
    """Đánh giá tất cả model .joblib trong một thư mục."""
    if X_test is None or y_test is None:
        X_test, y_test = load_test_data()

    results = []
    for model_path in sorted(model_dir.glob("*.joblib")):
        metrics = evaluate_saved_model(model_path, X_test, y_test)
        metrics["model"] = model_path.stem
        results.append(metrics)

    return pd.DataFrame(results)[["model", "MAE", "RMSE", "R2"]].sort_values("R2", ascending=False)


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted", save_path=None):
    """Vẽ scatter plot actual vs predicted."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color="steelblue")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Đường lý tưởng")
    ax.set_xlabel("Giá thực tế")
    ax.set_ylabel("Giá dự đoán")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_residuals(y_true, y_pred, title="Residual Plot", save_path=None):
    """Vẽ residual plot."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_pred, residuals, alpha=0.4, s=15, color="coral")
    axes[0].axhline(y=0, color="black", linestyle="--")
    axes[0].set_xlabel("Giá dự đoán")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"{title} - Residuals vs Predicted")

    axes[1].hist(residuals, bins=50, color="coral", edgecolor="white")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"{title} - Phân phối Residual")

    plt.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=15, title="Feature Importance", save_path=None):
    """Vẽ feature importance (cho tree-based models)."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices], color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, metric="R2", save_path=None):
    """Vẽ bar chart so sánh các model."""
    df = results_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("viridis", len(df))
    ax.barh(df["model"], df[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f"So sánh các mô hình theo {metric}")
    for i, v in enumerate(df[metric]):
        ax.text(v + 0.005, i, f"{v:.4f}", va="center")
    plt.tight_layout()
    if save_path:
        ensure_dir(Path(save_path).parent)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def export_model_comparison(results_df: pd.DataFrame) -> None:
    """Lưu bảng so sánh model."""
    save_dataframe(results_df, TABLES_DIR / "model_comparison.csv")
