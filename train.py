# -*- coding: utf-8 -*-
import os
import re
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor


DATA_FILE = "verileriniz.xlsx"
MODEL_FILE = "egitilmis_model.pkl"
ANALYSIS_FILE = "model_analiz.png"
RANDOM_STATE = 42
SURE_WEIGHT = 2.0


ANLAMSIZ_TOKENLAR = {
    "zli", "lmesi", "ldi", "lmis", "lmak", "masi",
    "yardim", "lar", "ler", "dan", "den", "nin", "nun",
    "bir", "ile"
}


def temizle(text: str) -> str:
    text = str(text).lower()

    terimler = {
        "kafa temizligi": "kafa_temizligi",
        "kafa temizliği": "kafa_temizligi",
        "sensor arizasi": "sensor_arizasi",
        "sensör arızası": "sensor_arizasi",
        "parametre ayari": "parametre_ayari",
        "parametre ayarı": "parametre_ayari",
        "eksen ayarı": "eksen_ayari",
        "eksen ayari": "eksen_ayari",
    }

    for eski, yeni in terimler.items():
        text = text.replace(eski, yeni)

    text = re.sub(r"\d+\s*(dk|dakika|dak|saat|sa|sn|saniye)\.?", " ", text)
    text = re.sub(r"\d+", " ", text)

    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    text = text.translate(tr_map)
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = [
        token for token in text.split()
        if token not in ANLAMSIZ_TOKENLAR and len(token) >= 3
    ]

    return " ".join(tokens)


def veriyi_yukle(dosya_yolu: str) -> pd.DataFrame:
    if not os.path.exists(dosya_yolu):
        raise FileNotFoundError(f"{dosya_yolu} bulunamadı.")

    df = pd.read_excel(dosya_yolu)
    df.columns = df.columns.astype(str).str.strip()

    gerekli = {"Süre_Dk", "Ariza_Aciklamasi", "Makine_Tipi"}

    if not gerekli.issubset(set(df.columns)) and len(df.columns) > 4:
        aday = df.iloc[:, 2:].copy()
        aday.columns = aday.iloc[0].astype(str).str.strip()
        aday = aday.drop(0).reset_index(drop=True)

        if gerekli.issubset(set(aday.columns)):
            df = aday

    eksik = gerekli - set(df.columns)
    if eksik:
        raise ValueError(f"Excel içinde şu kolonlar eksik: {', '.join(sorted(eksik))}")

    df["Süre_Dk"] = pd.to_numeric(df["Süre_Dk"], errors="coerce")
    df = df.dropna(subset=["Süre_Dk", "Ariza_Aciklamasi", "Makine_Tipi"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("Temizleme sonrası eğitim için veri kalmadı.")

    return df


def ozellik_ekle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Ariza_Aciklamasi"] = df["Ariza_Aciklamasi"].astype(str)
    df["Makine_Tipi"] = df["Makine_Tipi"].astype(str).str.strip()
    df["Temiz"] = df["Ariza_Aciklamasi"].apply(temizle)
    return df


def en_iyi_kume_sayisi_bul(X_tfidf, min_k=2, max_k=30) -> int:
    kayit_sayisi = X_tfidf.shape[0]
    max_k = min(max_k, kayit_sayisi - 1)

    if max_k < min_k:
        return 2

    from sklearn.metrics import silhouette_score

    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        kmeans_tmp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans_tmp.fit_predict(X_tfidf)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(X_tfidf, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def grafik_kaydet(y_true, y_pred, baseline_mae, mae):
    hatalar = np.abs(y_true - y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, y_pred, alpha=0.75)
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    axes[0].plot([0, max_val], [0, max_val], "--")
    axes[0].set_title("Gerçek vs Tahmin")
    axes[0].set_xlabel("Gerçek Süre (dk)")
    axes[0].set_ylabel("Tahmin Süre (dk)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(hatalar, bins=20)
    axes[1].set_title("Mutlak Hata Dağılımı")
    axes[1].set_xlabel("Hata (dk)")
    axes[1].set_ylabel("Kayıt Sayısı")
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Model Analizi | MAE: {mae:.2f} dk | Baseline: {baseline_mae:.2f} dk")
    plt.tight_layout()
    plt.savefig(ANALYSIS_FILE, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Veri yükleniyor...")
    df = veriyi_yukle(DATA_FILE)
    df = ozellik_ekle(df)

    print(f"Kayıt sayısı: {len(df)}")

    Q1, Q3 = df["Süre_Dk"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    alt = max(1.0, Q1 - 1.5 * IQR)
    ust = Q3 + 2.0 * IQR

    df["Süre_Dk_M"] = df["Süre_Dk"].clip(lower=alt, upper=ust)

    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 3),
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"\b\w\w+\b",
    )

    X_tfidf = vectorizer.fit_transform(df["Temiz"])

    sure_log = np.log1p(df["Süre_Dk_M"].values)
    sure_log_mean = float(sure_log.mean())
    sure_log_std = float(sure_log.std())

    if sure_log_std == 0:
        sure_log_std = 1.0

    sure_scaled = ((sure_log - sure_log_mean) / sure_log_std).reshape(-1, 1)
    X_cluster = sp.hstack(
        [X_tfidf, sp.csr_matrix(sure_scaled * SURE_WEIGHT)],
        format="csr"
    )

    print("Küme sayısı seçiliyor...")
    best_k = en_iyi_kume_sayisi_bul(X_tfidf)
    print(f"Seçilen küme sayısı: {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    df["Grup"] = kmeans.fit_predict(X_cluster).astype(str)

    df["Süre_Dk_M_log"] = np.log1p(df["Süre_Dk_M"])
    grup_mean_map = df.groupby("Grup")["Süre_Dk_M"].mean().apply(np.log1p).to_dict()
    makine_mean_map = df.groupby("Makine_Tipi")["Süre_Dk_M"].mean().apply(np.log1p).to_dict()
    global_mean = float(np.log1p(df["Süre_Dk_M"].mean()))

    df["Grup_MeanEnc"] = df["Grup"].map(grup_mean_map).fillna(global_mean).astype(float)
    df["Makine_MeanEnc"] = df["Makine_Tipi"].map(makine_mean_map).fillna(global_mean).astype(float)

    makine_dummy = pd.get_dummies(df["Makine_Tipi"], prefix="mak")
    grup_dummy = pd.get_dummies(df["Grup"], prefix="grup")

    MAKINE_KOLONLARI = makine_dummy.columns.tolist()
    GRUP_KOLONLARI = grup_dummy.columns.tolist()

    X = sp.hstack(
        [
            sp.csr_matrix(makine_dummy.values.astype(float)),
            sp.csr_matrix(grup_dummy.values.astype(float)),
            sp.csr_matrix(df[["Grup_MeanEnc", "Makine_MeanEnc"]].values.astype(float)),
            X_tfidf,
        ],
        format="csr",
    )

    y = np.log1p(df["Süre_Dk_M"].values)

    makine_ort = df.groupby("Makine_Tipi")["Süre_Dk_M"].transform("mean")
    baseline_mae = mean_absolute_error(df["Süre_Dk_M"], makine_ort)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.8,
        reg_lambda=2.5,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        tree_method="hist",
        early_stopping_rounds=100,
    )

    print("Model eğitiliyor...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )

    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    medae = float(np.median(np.abs(y_true - y_pred)))

    print("\n" + "=" * 60)
    print(f"TEST MAE     : {mae:.2f} dk")
    print(f"MEDIAN AE    : {medae:.2f} dk")
    print(f"BASELINE MAE : {baseline_mae:.2f} dk")
    print("=" * 60)

    grafik_kaydet(y_true, y_pred, baseline_mae, mae)

    paket = {
        "model": model,
        "vectorizer": vectorizer,
        "kmeans": kmeans,
        "grup_mean_map": grup_mean_map,
        "makine_mean_map": makine_mean_map,
        "global_mean": global_mean,
        "MAKINE_KOLONLARI": MAKINE_KOLONLARI,
        "GRUP_KOLONLARI": GRUP_KOLONLARI,
        "sure_log_mean": sure_log_mean,
        "sure_log_std": sure_log_std,
        "sure_weight": SURE_WEIGHT,
        "temizle_min_len": 3,
        "metrics": {
            "mae": float(mae),
            "median_absolute_error": medae,
            "baseline_mae": float(baseline_mae),
            "cluster_count": int(best_k),
            "record_count": int(len(df)),
        },
    }

    joblib.dump(paket, MODEL_FILE)

    print(f"\nModel kaydedildi: {MODEL_FILE}")
    print(f"Analiz grafiği kaydedildi: {ANALYSIS_FILE}")


if __name__ == "__main__":
    main()
