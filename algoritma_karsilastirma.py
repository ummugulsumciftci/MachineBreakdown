# -*- coding: utf-8 -*-
"""Farklı regresyon algoritmalarını aynı test setinde karşılaştırır.

Bu betik mevcut modeli değiştirmez. Tez/sunum için algoritma karşılaştırma
tablosu ve grafikleri üretir.
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
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from train import (
    RANDOM_STATE,
    gecmis_ozellik_haritalari,
    gecmis_ozellikleri_olustur,
    ozellik_ekle,
    veriyi_yukle,
)


OUT_DIR = Path("grafikler")
MODEL_FILE = Path("egitilmis_model.pkl")
BLUE_DARK = "#1e3a8a"
BLUE = "#2563eb"
BLUE_MID = "#3b82f6"
BLUE_LIGHT = "#93c5fd"
BLUE_PALE = "#dbeafe"
BLUE_GRAY = "#64748b"


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
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1.0, None)
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


def add_labels(ax):
    for patch in ax.patches:
        value = patch.get_width()
        ax.annotate(
            f"{value:.1f}",
            (value, patch.get_y() + patch.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )


def main():
    warnings.filterwarnings("ignore")
    OUT_DIR.mkdir(exist_ok=True)

    package = joblib.load(MODEL_FILE)
    data_file = package.get("data_file", "verileriniz_normalize.xlsx")
    df = ozellik_ekle(veriyi_yukle(data_file))
    df["Süre_Dk_M"] = clipped_duration(df)

    x_sparse = build_features(df)
    y_log = np.log1p(df["Süre_Dk"].values)
    y_true_clipped = df["Süre_Dk_M"].values

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    x_train = x_sparse[train_idx]
    x_test = x_sparse[test_idx]
    y_train_log = y_log[train_idx]
    y_test = y_true_clipped[test_idx]

    history_maps = gecmis_ozellik_haritalari(df.iloc[train_idx])
    x_train_support = sp.hstack(
        [
            x_train,
            sp.csr_matrix(gecmis_ozellikleri_olustur(df.iloc[train_idx], history_maps)),
        ],
        format="csr",
    )
    x_test_support = sp.hstack(
        [
            x_test,
            sp.csr_matrix(gecmis_ozellikleri_olustur(df.iloc[test_idx], history_maps)),
        ],
        format="csr",
    )

    results = []
    predictions = pd.DataFrame(
        {
            "Machine": df.iloc[test_idx]["Makine_Tipi"].values,
            "Failure_Description": df.iloc[test_idx]["Ariza_Aciklamasi"].values,
            "Actual_Duration_Min": y_test,
        }
    )

    machine_mean = df.iloc[train_idx].groupby("Makine_Tipi")["Süre_Dk_M"].mean()
    global_mean = float(df.iloc[train_idx]["Süre_Dk_M"].mean())
    pred_machine = df.iloc[test_idx]["Makine_Tipi"].map(machine_mean).fillna(global_mean).values
    results.append(summarize("Machine Average Baseline", y_test, pred_machine))
    predictions["Machine_Average_Baseline"] = pred_machine

    history_lookup = (
        df.iloc[train_idx].groupby(["Makine_Tipi", "Temiz"])["Süre_Dk_M"].median().to_dict()
    )
    global_median = float(df.iloc[train_idx]["Süre_Dk_M"].median())
    pred_history = np.array(
        [
            history_lookup.get((row.Makine_Tipi, row.Temiz), global_median)
            for row in df.iloc[test_idx].itertuples(index=False)
        ]
    )
    results.append(summarize("Similar History Median", y_test, pred_history))
    predictions["Similar_History_Median"] = pred_history

    sparse_models = {
        "Ridge Regression": Ridge(alpha=30.0),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.15, max_iter=5000, random_state=RANDOM_STATE),
        "Linear SVR": LinearSVR(C=0.35, epsilon=0.05, random_state=RANDOM_STATE, max_iter=6000),
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
    }

    ridge_pred = None
    xgb_pred = None
    for name, model in sparse_models.items():
        if name == "XGBoost":
            model.fit(x_train_support, y_train_log)
            pred = np.expm1(model.predict(x_test_support))
            xgb_pred = pred
        else:
            model.fit(x_train, y_train_log)
            pred = np.expm1(model.predict(x_test))
            if name == "Ridge Regression":
                ridge_pred = pred
        results.append(summarize(name, y_test, pred))
        predictions[name.replace(" ", "_")] = pred

    if ridge_pred is not None and xgb_pred is not None:
        ensemble_pred = 0.65 * ridge_pred + 0.35 * xgb_pred
        results.append(summarize("Final Ensemble (Ridge + XGBoost)", y_test, ensemble_pred))
        predictions["Final_Ensemble"] = ensemble_pred

    n_components = min(120, x_train.shape[0] - 1, x_train.shape[1] - 1)
    dense_models = {
        "KNN Regression": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.04,
            max_leaf_nodes=16,
            random_state=RANDOM_STATE,
        ),
    }

    for name, estimator in dense_models.items():
        model = make_pipeline(
            TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE),
            StandardScaler(),
            estimator,
        )
        model.fit(x_train, y_train_log)
        pred = np.expm1(model.predict(x_test))
        results.append(summarize(name, y_test, pred))
        predictions[name.replace(" ", "_")] = pred

    comparison = pd.DataFrame(results).sort_values("MAE_min", ascending=True).reset_index(drop=True)
    comparison.to_csv(OUT_DIR / "algorithm_model_comparison.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT_DIR / "algorithm_test_predictions.csv", index=False, encoding="utf-8-sig")

    colors = [BLUE_DARK if i == 0 else BLUE if i < 4 else BLUE_MID for i in range(len(comparison))]
    plt.figure(figsize=(10, 6))
    ax = plt.barh(comparison["Model"], comparison["MAE_min"], color=colors)
    plt.gca().invert_yaxis()
    plt.title("Algorithm Comparison by Mean Absolute Error")
    plt.xlabel("MAE (min)")
    plt.ylabel("Model")
    plt.grid(axis="x", alpha=0.25)
    for patch in ax:
        width = patch.get_width()
        plt.text(width + 0.2, patch.get_y() + patch.get_height() / 2, f"{width:.2f}", va="center", fontsize=9)
    save_fig(OUT_DIR / "09_algorithm_comparison_mae.png")

    metric_view = comparison.head(8).set_index("Model")[["MAE_min", "Median_AE_min", "RMSE_min", "P80_AE_min"]]
    plt.figure(figsize=(11, 6))
    metric_view.plot(kind="bar", color=[BLUE_DARK, BLUE, BLUE_MID, BLUE_LIGHT], ax=plt.gca())
    plt.title("Top Algorithms: Error Metrics")
    plt.ylabel("Minutes")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(["MAE", "Median AE", "RMSE", "P80 AE"])
    save_fig(OUT_DIR / "10_algorithm_comparison_error_metrics.png")

    best = comparison.iloc[0]
    readme = f"""Algorithm comparison outputs

Training data: {data_file}
Test set size: {len(test_idx)}
Best model by MAE: {best['Model']}
Best MAE: {best['MAE_min']:.2f} min
Best Median AE: {best['Median_AE_min']:.2f} min

Generated files:
- algorithm_model_comparison.csv
- algorithm_test_predictions.csv
- 09_algorithm_comparison_mae.png
- 10_algorithm_comparison_error_metrics.png
"""
    (OUT_DIR / "algorithm_comparison_README.txt").write_text(readme, encoding="utf-8")

    print(comparison[["Model", "MAE_min", "Median_AE_min", "RMSE_min", "R2"]].to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
