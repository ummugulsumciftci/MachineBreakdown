# -*- coding: utf-8 -*-
import os
import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_FILE = os.environ.get("DATA_FILE", "verileriniz.xlsx")
MODEL_FILE = "egitilmis_model.pkl"
ANALYSIS_FILE = "model_analiz.png"
RANDOM_STATE = 42
LONG_DURATION_THRESHOLD = 90


ANLAMSIZ_TOKENLAR = {
    "zli", "lmesi", "ldi", "lmis", "lmak", "masi",
    "yardim", "lar", "ler", "dan", "den", "nin", "nun",
    "bir", "ile"
}

TERIMLER = {
    "kafa temizligi": "kafa_temizligi",
    "kafa temizliği": "kafa_temizligi",
    "sensor arizasi": "sensor_arizasi",
    "sensör arızası": "sensor_arizasi",
    "sensor hatasi": "sensor_arizasi",
    "parametre ayari": "parametre_ayari",
    "parametre ayarı": "parametre_ayari",
    "eksen ayarı": "eksen_ayari",
    "eksen ayari": "eksen_ayari",
    "isin ayari": "isin_ayari",
    "ışın ayarı": "isin_ayari",
    "işin ayari": "isin_ayari",
    "işin ayarı": "isin_ayari",
    "program duzeltilmesi": "program_duzeltme",
    "program düzeltildi": "program_duzeltme",
    "program duzeltildi": "program_duzeltme",
    "konveyor arizasi": "konveyor_arizasi",
    "konveyör arızası": "konveyor_arizasi",
}

OPERASYON_OZELLIKLERI = [
    "feat_degisim",
    "feat_bekleme",
    "feat_servis_bakim",
    "feat_ayar",
    "feat_temizlik",
    "feat_program",
    "feat_sensor",
    "feat_mekanik",
    "feat_elektrik",
    "feat_uretim_bekleme",
    "feat_reset",
    "feat_kalibrasyon",
]


def turkce_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.translate(str.maketrans("çğıöşüâîû", "cgiosuaiu"))


def temizle(text: str) -> str:
    text = turkce_normalize(text)

    for eski, yeni in TERIMLER.items():
        text = text.replace(turkce_normalize(eski), yeni)

    text = re.sub(r"\d+\s*(dk|dakika|dak|saat|sa|sn|saniye)\.?", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s_]", " ", text)

    tokens = [
        token.replace("ligi", "lik").replace("lugu", "luk")
        for token in text.split()
        if token not in ANLAMSIZ_TOKENLAR and len(token) >= 3
    ]

    return " ".join(tokens)


def operasyon_ozellikleri(text: str) -> dict:
    raw = turkce_normalize(text)
    clean = temizle(text)
    joined = f"{raw} {clean}"

    return {
        "feat_degisim": int(any(k in joined for k in ["degisim", "degistirme", "degisti", "degisen", "değiş"])),
        "feat_bekleme": int(any(k in joined for k in ["bekleme", "bekledi", "bekleniyor"])),
        "feat_servis_bakim": int(any(k in joined for k in ["servis", "bakim", "bakım"])),
        "feat_ayar": int(any(k in joined for k in ["ayar", "ofset", "parametre", "kalibrasyon"])),
        "feat_temizlik": int(any(k in joined for k in ["temiz", "temizlik", "temi"])),
        "feat_program": int(any(k in joined for k in ["program", "cizim", "çizim"])),
        "feat_sensor": int(any(k in joined for k in ["sensor", "sensör", "fotosel"])),
        "feat_mekanik": int(any(k in joined for k in ["rulman", "ayna", "cene", "çene", "konveyor", "konveyör", "kafa", "motor", "kayis", "kayış", "zincir"])),
        "feat_elektrik": int(any(k in joined for k in ["surucu", "sürücü", "elektrik", "voltaj", "sigorta", "kart"])),
        "feat_uretim_bekleme": int(any(k in joined for k in ["is kalmadi", "iş kalmadı", "malzeme", "paketleme", "uretim", "üretim"])),
        "feat_reset": int("reset" in joined),
        "feat_kalibrasyon": int(any(k in joined for k in ["kalibrasyon", "referans", "home"])),
    }


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
    operasyon_df = pd.DataFrame(
        df["Ariza_Aciklamasi"].apply(operasyon_ozellikleri).tolist(),
        index=df.index,
    )
    df = pd.concat([df, operasyon_df], axis=1)
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


def gecmis_ozellik_haritalari(df: pd.DataFrame) -> dict:
    global_mean = float(df["Süre_Dk"].mean())
    global_median = float(df["Süre_Dk"].median())

    def smooth_stats(keys, prior):
        stats = df.groupby(keys)["Süre_Dk"].agg(["count", "mean", "median"])
        smooth = (stats["mean"] * stats["count"] + global_mean * prior) / (stats["count"] + prior)
        return stats, smooth

    makine_stats, makine_smooth = smooth_stats("Makine_Tipi", 10)
    temiz_stats, temiz_smooth = smooth_stats("Temiz", 5)
    combo_stats, combo_smooth = smooth_stats(["Makine_Tipi", "Temiz"], 5)

    return {
        "global_mean": global_mean,
        "global_median": global_median,
        "makine_smooth": makine_smooth.to_dict(),
        "makine_median": makine_stats["median"].to_dict(),
        "makine_count": makine_stats["count"].to_dict(),
        "temiz_smooth": temiz_smooth.to_dict(),
        "temiz_median": temiz_stats["median"].to_dict(),
        "temiz_count": temiz_stats["count"].to_dict(),
        "combo_smooth": combo_smooth.to_dict(),
        "combo_median": combo_stats["median"].to_dict(),
        "combo_count": combo_stats["count"].to_dict(),
    }


def gecmis_ozellikleri_olustur(df: pd.DataFrame, maps: dict) -> np.ndarray:
    rows = []
    global_mean = maps["global_mean"]
    global_median = maps["global_median"]

    for row in df.itertuples(index=False):
        makine = row.Makine_Tipi
        temiz = row.Temiz
        combo_key = (makine, temiz)

        rows.append(
            [
                maps["makine_smooth"].get(makine, global_mean),
                maps["makine_median"].get(makine, global_median),
                maps["makine_count"].get(makine, 0),
                maps["temiz_smooth"].get(temiz, global_mean),
                maps["temiz_median"].get(temiz, global_median),
                maps["temiz_count"].get(temiz, 0),
                maps["combo_smooth"].get(combo_key, global_mean),
                maps["combo_median"].get(combo_key, global_median),
                maps["combo_count"].get(combo_key, 0),
            ]
        )

    return np.array(rows, dtype=float)


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

    X_tfidf = vectorizer.fit_transform(df["Temiz"])

    sure_log_mean = 0.0
    sure_log_std = 1.0
    X_cluster = X_tfidf

    print("Küme sayısı seçiliyor...")
    best_k = en_iyi_kume_sayisi_bul(X_tfidf)
    print(f"Seçilen küme sayısı: {best_k}")

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    df["Grup"] = kmeans.fit_predict(X_cluster).astype(str)

    df["Süre_Dk_M_log"] = np.log1p(df["Süre_Dk_M"])
    grup_mean_map = df.groupby("Grup")["Süre_Dk_M"].mean().apply(np.log1p).to_dict()
    makine_mean_map = df.groupby("Makine_Tipi")["Süre_Dk_M"].mean().apply(np.log1p).to_dict()
    global_mean = float(np.log1p(df["Süre_Dk_M"].mean()))
    history_machine_clean = (
        df.groupby(["Makine_Tipi", "Temiz"])["Süre_Dk"]
        .agg(count="size", median="median", mean="mean", min="min", max="max")
        .reset_index()
        .to_dict("records")
    )
    history_clean = (
        df.groupby("Temiz")["Süre_Dk"]
        .agg(count="size", median="median", mean="mean", min="min", max="max")
        .reset_index()
        .to_dict("records")
    )
    history_lookup = {
        (row["Makine_Tipi"], row["Temiz"]): row["median"]
        for row in history_machine_clean
    }
    history_pred = [
        history_lookup[(row["Makine_Tipi"], row["Temiz"])]
        for _, row in df.iterrows()
    ]
    history_mae = mean_absolute_error(df["Süre_Dk_M"], history_pred)
    history_medae = float(np.median(np.abs(df["Süre_Dk_M"].values - np.array(history_pred))))

    df["Grup_MeanEnc"] = df["Grup"].map(grup_mean_map).fillna(global_mean).astype(float)
    df["Makine_MeanEnc"] = df["Makine_Tipi"].map(makine_mean_map).fillna(global_mean).astype(float)

    makine_dummy = pd.get_dummies(df["Makine_Tipi"], prefix="mak")
    grup_dummy = pd.get_dummies(df["Grup"], prefix="grup")

    MAKINE_KOLONLARI = makine_dummy.columns.tolist()
    GRUP_KOLONLARI = grup_dummy.columns.tolist()

    X = sp.hstack(
        [
            sp.csr_matrix(makine_dummy.values.astype(float)),
            X_tfidf,
        ],
        format="csr",
    )

    y = np.log1p(df["Süre_Dk"].values)

    makine_ort = df.groupby("Makine_Tipi")["Süre_Dk_M"].transform("mean")
    baseline_mae = mean_absolute_error(df["Süre_Dk_M"], makine_ort)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_long = (df["Süre_Dk"].values > LONG_DURATION_THRESHOLD).astype(int)
    y_long_train = y_long[train_idx]
    y_long_test = y_long[test_idx]
    y_orig_test = df["Süre_Dk"].values[test_idx]
    y_clip_test = df["Süre_Dk_M"].values[test_idx]

    eval_history_maps = gecmis_ozellik_haritalari(df.iloc[train_idx])
    X_train_support = sp.hstack(
        [
            X_train,
            sp.csr_matrix(gecmis_ozellikleri_olustur(df.iloc[train_idx], eval_history_maps)),
        ],
        format="csr",
    )
    X_test_support = sp.hstack(
        [
            X_test,
            sp.csr_matrix(gecmis_ozellikleri_olustur(df.iloc[test_idx], eval_history_maps)),
        ],
        format="csr",
    )

    n_components = min(120, X_train.shape[0] - 1, X_train.shape[1] - 1)
    model = make_pipeline(
        TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE),
        StandardScaler(),
        GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.04,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    )
    support_model = None
    risk_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    ensemble_model_weight = 1.0

    print("Model eğitiliyor...")
    model.fit(X_train, y_train)
    risk_model.fit(X_train, y_long_train)

    y_pred = np.expm1(model.predict(X_test))
    long_risk_pred = risk_model.predict_proba(X_test)[:, 1]
    long_risk_class = (long_risk_pred >= 0.5).astype(int)

    mae = mean_absolute_error(y_clip_test, y_pred)
    original_mae = mean_absolute_error(y_orig_test, y_pred)
    medae = float(np.median(np.abs(y_clip_test - y_pred)))
    original_medae = float(np.median(np.abs(y_orig_test - y_pred)))
    risk_accuracy = accuracy_score(y_long_test, long_risk_class)
    risk_auc = roc_auc_score(y_long_test, long_risk_pred)
    signed_residuals = y_clip_test - y_pred
    planning_add_80 = max(0.0, float(np.quantile(signed_residuals, 0.80)))
    planning_add_90 = max(0.0, float(np.quantile(signed_residuals, 0.90)))
    planning_upper_80 = y_pred + planning_add_80
    planning_upper_90 = y_pred + planning_add_90
    planning_coverage_80 = float(np.mean(y_clip_test <= planning_upper_80))
    planning_coverage_90 = float(np.mean(y_clip_test <= planning_upper_90))

    print("\n" + "=" * 60)
    print(f"TEST MAE     : {mae:.2f} dk")
    print(f"ORJ. TEST MAE: {original_mae:.2f} dk")
    print(f"MEDIAN AE    : {medae:.2f} dk")
    print(f"UZUN RİSK AUC: {risk_auc:.3f} | Accuracy: %{risk_accuracy * 100:.1f}")
    print(f"P80 ÜST EK   : +{planning_add_80:.2f} dk | Kapsama: %{planning_coverage_80 * 100:.1f}")
    print(f"P90 ÜST EK   : +{planning_add_90:.2f} dk | Kapsama: %{planning_coverage_90 * 100:.1f}")
    print(f"BASELINE MAE : {baseline_mae:.2f} dk")
    print("=" * 60)

    grafik_kaydet(y_clip_test, y_pred, baseline_mae, mae)

    final_history_maps = gecmis_ozellik_haritalari(df)
    model.fit(X, y)
    risk_model.fit(X, y_long)

    training_records = df[
        ["Makine_Tipi", "Ariza_Aciklamasi", "Temiz", "Süre_Dk", "Süre_Dk_M", "Grup"]
    ].to_dict("records")

    paket = {
        "data_file": DATA_FILE,
        "model": model,
        "support_model": support_model,
        "risk_model": risk_model,
        "model_algorithm": "Gradient Boosting",
        "ensemble_model_weight": ensemble_model_weight,
        "history_feature_maps": final_history_maps,
        "planning_calibration": {
            "upper_add_80": planning_add_80,
            "upper_add_90": planning_add_90,
            "coverage_80": planning_coverage_80,
            "coverage_90": planning_coverage_90,
        },
        "vectorizer": vectorizer,
        "kmeans": kmeans,
        "grup_mean_map": grup_mean_map,
        "makine_mean_map": makine_mean_map,
        "global_mean": global_mean,
        "MAKINE_KOLONLARI": MAKINE_KOLONLARI,
        "GRUP_KOLONLARI": GRUP_KOLONLARI,
        "OPERASYON_OZELLIKLERI": OPERASYON_OZELLIKLERI,
        "sure_log_mean": sure_log_mean,
        "sure_log_std": sure_log_std,
        "cluster_uses_duration": False,
        "model_feature_set": "machine_text_char",
        "long_duration_threshold": LONG_DURATION_THRESHOLD,
        "similarity_text_matrix": X_tfidf,
        "training_records": training_records,
        "sure_weight": 0.0,
        "temizle_min_len": 3,
        "history_machine_clean": history_machine_clean,
        "history_clean": history_clean,
        "metrics": {
            "mae": float(mae),
            "original_mae": float(original_mae),
            "median_absolute_error": medae,
            "original_median_absolute_error": original_medae,
            "risk_accuracy": float(risk_accuracy),
            "risk_auc": float(risk_auc),
            "baseline_mae": float(baseline_mae),
            "history_mae": float(history_mae),
            "history_median_absolute_error": history_medae,
            "cluster_count": int(best_k),
            "record_count": int(len(df)),
        },
    }

    joblib.dump(paket, MODEL_FILE)

    print(f"\nModel kaydedildi: {MODEL_FILE}")
    print(f"Analiz grafiği kaydedildi: {ANALYSIS_FILE}")


if __name__ == "__main__":
    main()
