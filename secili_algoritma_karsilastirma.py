# -*- coding: utf-8 -*-
"""Ridge, XGBoost, Random Forest ve Gradient Boosting karşılaştırması.

Bu betik mevcut modeli veya arayüzü değiştirmez. Sadece tez/sunum için
seçili algoritmaların aynı test setindeki performansını grafikler klasörüne
yazar.
"""

from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from train import RANDOM_STATE, ozellik_ekle, veriyi_yukle


OUT_DIR = Path("grafikler")
MODEL_FILE = Path("egitilmis_model.pkl")
BLUE_DARK = "#1e3a8a"
BLUE = "#2563eb"
BLUE_MID = "#3b82f6"
BLUE_LIGHT = "#93c5fd"


def clipped_duration(df: pd.DataFrame) -> pd.Series:
    q1, q3 = df["Süre_Dk"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = max(1.0, q1 - 1.5 * iqr)
    upper = q3 + 2.0 * iqr
    return df["Süre_Dk"].clip(lower=lower, upper=upper)


def build_features(df: pd.DataFrame):
    vectorizer = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    max_features=1500,
                    ngram_range=(1, 3),
                    min_df=1,
                    sublinear_tf=True,
                    token_pattern=r"\b\w\w+\b",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    max_features=2500,
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    x_tfidf = vectorizer.fit_transform(df["Temiz"])
    kmeans = KMeans(n_clusters=30, random_state=RANDOM_STATE, n_init=20)
    df["Grup"] = kmeans.fit_predict(x_tfidf).astype(str)
    machine_dummy = pd.get_dummies(df["Makine_Tipi"], prefix="mak")
    x_sparse = sp.hstack([sp.csr_matrix(machine_dummy.values.astype(float)), x_tfidf], format="csr")
    return x_sparse


def summarize(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1.0, None)
    errors = np.abs(y_true - y_pred)
    return {
        "Model": name,
        "MAE_min": float(mean_absolute_error(y_true, y_pred)),
        "Median_AE_min": float(np.median(errors)),
        "RMSE_min": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "P80_AE_min": float(np.percentile(errors, 80)),
        "P90_AE_min": float(np.percentile(errors, 90)),
    }


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(exist_ok=True)

    package = joblib.load(MODEL_FILE)
    data_file = package.get("data_file", "verileriniz_normalize.xlsx")

    df = ozellik_ekle(veriyi_yukle(data_file))
    df["Süre_Dk_M"] = clipped_duration(df)

    x_sparse = build_features(df)
    y_log = np.log1p(df["Süre_Dk"].values)
    y_true = df["Süre_Dk_M"].values

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    x_train = x_sparse[train_idx]
    x_test = x_sparse[test_idx]
    y_train_log = y_log[train_idx]
    y_test = y_true[test_idx]

    dense_components = min(120, x_train.shape[0] - 1, x_train.shape[1] - 1)
    models = {
        "Ridge": Ridge(alpha=30.0),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            tree_method="hist",
        ),
        "Random Forest": make_pipeline(
            TruncatedSVD(n_components=dense_components, random_state=RANDOM_STATE),
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=250,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        "Gradient Boosting": make_pipeline(
            TruncatedSVD(n_components=dense_components, random_state=RANDOM_STATE),
            StandardScaler(),
            GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.04,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(x_train, y_train_log)
        pred = np.expm1(model.predict(x_test))
        results.append(summarize(name, y_test, pred))

    comparison = pd.DataFrame(results).sort_values("MAE_min", ascending=True).reset_index(drop=True)
    comparison.to_csv(OUT_DIR / "selected_models_comparison.csv", index=False, encoding="utf-8-sig")

    colors = [BLUE_DARK if model == "Gradient Boosting" else BLUE for model in comparison["Model"]]
    plt.figure(figsize=(8.5, 5))
    bars = plt.bar(comparison["Model"], comparison["MAE_min"], color=colors)
    plt.title("Selected Model Comparison by MAE")
    plt.ylabel("MAE (min)")
    plt.xlabel("Model")
    plt.grid(axis="y", alpha=0.25)
    for bar in bars:
        value = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{value:.2f}", ha="center", fontsize=9)
    save_fig(OUT_DIR / "11_selected_models_mae_comparison.png")

    metric_view = comparison.set_index("Model")[["MAE_min", "Median_AE_min", "RMSE_min"]]
    plt.figure(figsize=(9, 5))
    metric_view.plot(kind="bar", color=[BLUE_DARK, BLUE, BLUE_LIGHT], ax=plt.gca())
    plt.title("Selected Models: Error Metrics")
    plt.ylabel("Minutes")
    plt.xlabel("Model")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(["MAE", "Median AE", "RMSE"])
    save_fig(OUT_DIR / "12_selected_models_error_metrics.png")

    print(comparison.to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
