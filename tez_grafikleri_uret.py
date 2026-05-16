# -*- coding: utf-8 -*-
"""Tez için model analiz grafiklerini üretir.

Bu dosya modeli yeniden kaydetmez; yalnızca mevcut veri/model ayarlarıyla
analiz çıktıları oluşturur ve grafikler klasörüne yazar.
"""

from pathlib import Path
import shutil

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from xgboost import XGBRegressor

from train import (
    LONG_DURATION_THRESHOLD,
    RANDOM_STATE,
    en_iyi_kume_sayisi_bul,
    gecmis_ozellik_haritalari,
    gecmis_ozellikleri_olustur,
    ozellik_ekle,
    veriyi_yukle,
)


OUT_DIR = Path("grafikler")
MODEL_FILE = Path("egitilmis_model.pkl")
BLUE = "#2563eb"
BLUE_DARK = "#1e3a8a"
BLUE_MID = "#3b82f6"
BLUE_LIGHT = "#93c5fd"
BLUE_PALE = "#dbeafe"
BLUE_GRAY = "#64748b"


def save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def add_value_labels(ax, fmt="{:.1f}"):
    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def reset_output_dir():
    OUT_DIR.mkdir(exist_ok=True)
    for child in OUT_DIR.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def load_and_prepare():
    paket = joblib.load(MODEL_FILE)
    data_file = paket.get("data_file", "verileriniz_normalize.xlsx")
    df_raw = veriyi_yukle(data_file)
    df = ozellik_ekle(df_raw)

    q1, q3 = df["Süre_Dk"].quantile([0.25, 0.75])
    iqr = q3 - q1
    alt = max(1.0, q1 - 1.5 * iqr)
    ust = q3 + 2.0 * iqr
    df["Süre_Dk_M"] = df["Süre_Dk"].clip(lower=alt, upper=ust)
    return paket, data_file, df


def holdout_predictions(df: pd.DataFrame):
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
    best_k = en_iyi_kume_sayisi_bul(x_tfidf)
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    df["Grup"] = kmeans.fit_predict(x_tfidf).astype(str)

    makine_dummy = pd.get_dummies(df["Makine_Tipi"], prefix="mak")
    x = sp.hstack([sp.csr_matrix(makine_dummy.values.astype(float)), x_tfidf], format="csr")
    y = np.log1p(df["Süre_Dk"].values)
    y_long = (df["Süre_Dk"].values > LONG_DURATION_THRESHOLD).astype(int)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y[train_idx]
    y_long_train = y_long[train_idx]
    y_long_test = y_long[test_idx]
    y_true = df["Süre_Dk_M"].values[test_idx]
    y_true_original = df["Süre_Dk"].values[test_idx]

    eval_history_maps = gecmis_ozellik_haritalari(df.iloc[train_idx])
    x_train_support = sp.hstack(
        [
            x_train,
            sp.csr_matrix(gecmis_ozellikleri_olustur(df.iloc[train_idx], eval_history_maps)),
        ],
        format="csr",
    )
    x_test_support = sp.hstack(
        [
            x_test,
            sp.csr_matrix(gecmis_ozellikleri_olustur(df.iloc[test_idx], eval_history_maps)),
        ],
        format="csr",
    )

    ridge = Ridge(alpha=30.0)
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        tree_method="hist",
    )
    risk_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    ridge.fit(x_train, y_train)
    xgb.fit(x_train_support, y_train)
    risk_model.fit(x_train, y_long_train)

    ridge_pred = np.expm1(ridge.predict(x_test))
    xgb_pred = np.expm1(xgb.predict(x_test_support))
    ensemble_weight = 0.65
    y_pred = ensemble_weight * ridge_pred + (1.0 - ensemble_weight) * xgb_pred
    risk_prob = risk_model.predict_proba(x_test)[:, 1]
    risk_class = (risk_prob >= 0.5).astype(int)

    makine_baseline = df.groupby("Makine_Tipi")["Süre_Dk_M"].transform("mean").values[test_idx]
    history_lookup = (
        df.iloc[train_idx]
        .groupby(["Makine_Tipi", "Temiz"])["Süre_Dk_M"]
        .median()
        .to_dict()
    )
    global_median = float(df.iloc[train_idx]["Süre_Dk_M"].median())
    history_pred = np.array(
        [
            history_lookup.get((row.Makine_Tipi, row.Temiz), global_median)
            for row in df.iloc[test_idx].itertuples(index=False)
        ]
    )

    pred_df = df.iloc[test_idx][["Makine_Tipi", "Ariza_Aciklamasi", "Temiz", "Grup"]].copy()
    pred_df["Gercek_Sure_Dk"] = y_true
    pred_df["Gercek_Orijinal_Sure_Dk"] = y_true_original
    pred_df["Tahmin_Dk"] = y_pred
    pred_df["Mutlak_Hata_Dk"] = np.abs(y_true - y_pred)
    pred_df["Ridge_Tahmin_Dk"] = ridge_pred
    pred_df["XGBoost_Tahmin_Dk"] = xgb_pred
    pred_df["Uzun_Durus_Gercek"] = y_long_test
    pred_df["Uzun_Durus_Risk_Skoru"] = risk_prob
    pred_df["Uzun_Durus_Tahmin"] = risk_class
    pred_df = pred_df.rename(
        columns={
            "Makine_Tipi": "Machine",
            "Ariza_Aciklamasi": "Failure_Description",
            "Temiz": "Cleaned_Text",
            "Grup": "Cluster",
            "Gercek_Sure_Dk": "Actual_Duration_Min",
            "Gercek_Orijinal_Sure_Dk": "Original_Actual_Duration_Min",
            "Tahmin_Dk": "Predicted_Duration_Min",
            "Mutlak_Hata_Dk": "Absolute_Error_Min",
            "Ridge_Tahmin_Dk": "Ridge_Prediction_Min",
            "XGBoost_Tahmin_Dk": "XGBoost_Prediction_Min",
            "Uzun_Durus_Gercek": "Actual_Long_Downtime",
            "Uzun_Durus_Risk_Skoru": "Long_Downtime_Risk_Score",
            "Uzun_Durus_Tahmin": "Predicted_Long_Downtime",
        }
    )

    metrics = {
        "Ensemble MAE": mean_absolute_error(y_true, y_pred),
        "Ridge MAE": mean_absolute_error(y_true, ridge_pred),
        "XGBoost MAE": mean_absolute_error(y_true, xgb_pred),
        "Machine Average MAE": mean_absolute_error(y_true, makine_baseline),
        "Similar History Median MAE": mean_absolute_error(y_true, history_pred),
        "Original Duration MAE": mean_absolute_error(y_true_original, y_pred),
        "Median Absolute Error": float(np.median(np.abs(y_true - y_pred))),
        "Risk Accuracy": accuracy_score(y_long_test, risk_class),
        "Risk AUC": roc_auc_score(y_long_test, risk_prob),
        "Test Record Count": len(test_idx),
        "Cluster Count": best_k,
    }

    return pred_df, metrics, y_true, y_pred, ridge_pred, xgb_pred, y_long_test, risk_prob, risk_class


def grafikler_uret():
    reset_output_dir()
    paket, data_file, df = load_and_prepare()
    pred_df, metrics, y_true, y_pred, ridge_pred, xgb_pred, y_long_test, risk_prob, risk_class = holdout_predictions(df)

    pred_df.to_csv(OUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{"Metric": key, "Value": value} for key, value in {**paket.get("metrics", {}), **metrics}.items()]
    ).to_csv(OUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    makine_ozet = (
        df.groupby("Makine_Tipi")["Süre_Dk"]
        .agg(Record_Count="size", Mean="mean", Median="median", Minimum="min", Maximum="max")
        .sort_values("Mean", ascending=False)
    )
    makine_ozet.to_csv(OUT_DIR / "machine_summary.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.72, color=BLUE_MID, edgecolor="white", linewidth=0.35)
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[0].plot([0, max_val], [0, max_val], "--", color=BLUE_DARK, linewidth=1.2)
    axes[0].set_title("Actual vs Predicted Downtime")
    axes[0].set_xlabel("Actual duration (min)")
    axes[0].set_ylabel("Predicted duration (min)")
    axes[0].grid(alpha=0.25)
    axes[1].hist(np.abs(y_true - y_pred), bins=22, color=BLUE, edgecolor="white")
    axes[1].axvline(metrics["Ensemble MAE"], color=BLUE_DARK, linestyle="--", label=f"MAE: {metrics['Ensemble MAE']:.1f} min")
    axes[1].set_title("Absolute Error Distribution")
    axes[1].set_xlabel("Absolute error (min)")
    axes[1].set_ylabel("Record count")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Model Performance Overview", fontsize=14)
    save_fig(OUT_DIR / "00_model_performance_overview.png")

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.72, color=BLUE_MID, edgecolor="white", linewidth=0.35)
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([0, max_val], [0, max_val], "--", color=BLUE_DARK, linewidth=1.2)
    plt.title("Actual Downtime vs Model Prediction")
    plt.xlabel("Actual duration (min)")
    plt.ylabel("Predicted duration (min)")
    plt.grid(alpha=0.25)
    save_fig(OUT_DIR / "01_actual_vs_predicted_scatter.png")

    plt.figure(figsize=(8, 5))
    plt.hist(np.abs(y_true - y_pred), bins=22, color=BLUE, edgecolor="white")
    plt.axvline(metrics["Ensemble MAE"], color=BLUE_DARK, linestyle="--", label=f"MAE: {metrics['Ensemble MAE']:.1f} min")
    plt.title("Absolute Error Distribution")
    plt.xlabel("Absolute error (min)")
    plt.ylabel("Record count")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_fig(OUT_DIR / "02_absolute_error_distribution.png")

    comparison = pd.Series(
        {
            "Machine\naverage": metrics["Machine Average MAE"],
            "Similar history\nmedian": metrics["Similar History Median MAE"],
            "Ridge": metrics["Ridge MAE"],
            "XGBoost": metrics["XGBoost MAE"],
            "Final model": metrics["Ensemble MAE"],
        }
    )
    plt.figure(figsize=(9, 5))
    ax = comparison.plot(kind="bar", color=[BLUE_PALE, BLUE_LIGHT, BLUE_MID, BLUE, BLUE_DARK])
    plt.title("Model Comparison by MAE")
    plt.ylabel("MAE (min)")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.25)
    add_value_labels(ax)
    save_fig(OUT_DIR / "03_model_comparison_mae.png")

    top_machines = makine_ozet.sort_values("Record_Count", ascending=False).head(12).sort_values("Mean")
    plt.figure(figsize=(9, 5))
    top_machines["Mean"].plot(kind="barh", color=BLUE)
    plt.title("Average Downtime by Machine")
    plt.xlabel("Average duration (min)")
    plt.ylabel("Machine")
    plt.grid(axis="x", alpha=0.25)
    save_fig(OUT_DIR / "04_average_downtime_by_machine.png")

    grup_counts = df["Grup"].value_counts().head(15).sort_values()
    plt.figure(figsize=(9, 6))
    grup_counts.plot(kind="barh", color=BLUE_MID)
    plt.title("Most Frequent Failure Clusters")
    plt.xlabel("Record count")
    plt.ylabel("Cluster")
    plt.grid(axis="x", alpha=0.25)
    save_fig(OUT_DIR / "05_failure_cluster_distribution.png")

    cm = confusion_matrix(y_long_test, risk_class)
    plt.figure(figsize=(5.8, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Long Downtime Risk Model - Confusion Matrix")
    plt.xticks([0, 1], ["Predicted short", "Predicted long"])
    plt.yticks([0, 1], ["Actual short", "Actual long"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13)
    plt.colorbar(fraction=0.046, pad=0.04)
    save_fig(OUT_DIR / "06_long_downtime_confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_long_test, risk_prob)
    plt.figure(figsize=(6.5, 5))
    plt.plot(fpr, tpr, label=f"AUC: {metrics['Risk AUC']:.3f}", color=BLUE_DARK, linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color=BLUE_GRAY)
    plt.title("Long Downtime Risk Model - ROC Curve")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.grid(alpha=0.25)
    save_fig(OUT_DIR / "07_long_downtime_roc_curve.png")

    df_excel = pd.read_excel(data_file)
    if "Sure_Dk_Orijinal" in df_excel.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df_excel["Sure_Dk_Orijinal"], bins=30, alpha=0.50, color=BLUE_LIGHT, label="Original duration")
        plt.hist(df_excel["Süre_Dk"], bins=30, alpha=0.65, color=BLUE_DARK, label="Normalized duration")
        plt.title("Duration Distribution Before and After Normalization")
        plt.xlabel("Duration (min)")
        plt.ylabel("Record count")
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
        save_fig(OUT_DIR / "08_duration_distribution_normalization.png")

    readme = f"""Thesis model analysis outputs

Training data: {data_file}
Record count: {len(df)}
Test record count: {int(metrics['Test Record Count'])}
Final model MAE: {metrics['Ensemble MAE']:.2f} min
Median Absolute Error: {metrics['Median Absolute Error']:.2f} min
Risk model AUC: {metrics['Risk AUC']:.3f}
Risk model accuracy: {metrics['Risk Accuracy'] * 100:.1f}%

Note: These outputs were generated for thesis reporting with the current model configuration.
Because normalized scenario data is used, this should be stated clearly in the methodology section.
"""
    (OUT_DIR / "README.txt").write_text(readme, encoding="utf-8")

    print(f"{OUT_DIR.resolve()} klasörüne {len(list(OUT_DIR.glob('*')))} analiz çıktısı yazıldı.")


if __name__ == "__main__":
    grafikler_uret()
