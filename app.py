import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import unicodedata
import scipy.sparse as sp
from pathlib import Path

# 1. SAYFA AYARLARI VE MODEL YÜKLEME
st.set_page_config(page_title="Hakan Sac Metal | DSS", layout="wide")

@st.cache_resource
def modeli_yukle():
    return joblib.load("egitilmis_model.pkl")

paket = modeli_yukle()
model = paket["model"]
support_model = paket.get("support_model")
risk_model = paket.get("risk_model")
ensemble_model_weight = paket.get("ensemble_model_weight", 1.0)
vectorizer = paket["vectorizer"]
kmeans = paket["kmeans"]
g_map = paket["grup_mean_map"]
m_map = paket["makine_mean_map"]
g_mean = paket["global_mean"]
MAK_KOL = paket["MAKINE_KOLONLARI"]
GRP_KOL = paket["GRUP_KOLONLARI"]
OPERASYON_OZELLIKLERI = paket.get("OPERASYON_OZELLIKLERI", [])
s_mean = paket["sure_log_mean"]
s_std = paket["sure_log_std"]
sure_weight = paket.get("sure_weight", 2.0)
temizle_min_len = paket.get("temizle_min_len", 3)
metrics = paket.get("metrics", {})
cluster_uses_duration = paket.get("cluster_uses_duration", True)
model_feature_set = paket.get("model_feature_set", "full")
history_feature_maps = paket.get("history_feature_maps")
planning_calibration = paket.get("planning_calibration", {})
long_duration_threshold = paket.get("long_duration_threshold", 90)
similarity_text_matrix = paket.get("similarity_text_matrix")
training_records = paket.get("training_records", [])
data_file = paket.get("data_file", "verileriniz.xlsx")
analysis_path = Path("model_analiz.png")
data_path = Path(data_file)
history_machine_clean = {
    (row["Makine_Tipi"], row["Temiz"]): row
    for row in paket.get("history_machine_clean", [])
}
history_clean = {
    row["Temiz"]: row
    for row in paket.get("history_clean", [])
}


def veri_dosyasini_yukle():
    if not data_path.exists():
        return pd.DataFrame(columns=["Tarih", "Makine_Tipi", "Ariza_Aciklamasi", "Süre_Dk"])

    df = pd.read_excel(data_path)
    df.columns = df.columns.astype(str).str.strip()
    return df


def veri_dosyasini_kaydet(df: pd.DataFrame):
    df.to_excel(data_path, index=False)


def veri_satiri_olustur(kolonlar, tarih, makine, ariza_aciklamasi, sure_dk):
    temiz_ariza = temizle(ariza_aciklamasi)
    satir = {kolon: np.nan for kolon in kolonlar}

    if "Tarih" in satir:
        satir["Tarih"] = pd.Timestamp(tarih)
    if "Makine_Tipi" in satir:
        satir["Makine_Tipi"] = makine
    if "Ariza_Aciklamasi" in satir:
        satir["Ariza_Aciklamasi"] = ariza_aciklamasi
    if "Süre_Dk" in satir:
        satir["Süre_Dk"] = float(sure_dk)
    if "Sure_Dk_Orijinal" in satir:
        satir["Sure_Dk_Orijinal"] = float(sure_dk)
    if "Normalize_Ariza_Grubu" in satir:
        satir["Normalize_Ariza_Grubu"] = temiz_ariza
    if "Normalize_Notu" in satir:
        satir["Normalize_Notu"] = "Arayüzden eklenen/düzenlenen kayıt."

    return satir

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

OPERASYON_ADLARI = {
    "feat_degisim": "Parça/değişim",
    "feat_bekleme": "Bekleme",
    "feat_servis_bakim": "Servis/bakım",
    "feat_ayar": "Ayar",
    "feat_temizlik": "Temizlik",
    "feat_program": "Program/çizim",
    "feat_sensor": "Sensör",
    "feat_mekanik": "Mekanik",
    "feat_elektrik": "Elektrik",
    "feat_uretim_bekleme": "Üretim/malzeme",
    "feat_reset": "Reset",
    "feat_kalibrasyon": "Kalibrasyon",
}


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
        if token not in ANLAMSIZ_TOKENLAR and len(token) >= temizle_min_len
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


def gecmis_tahmini_bul(makine: str, temiz_metin: str):
    makine_eslesme = history_machine_clean.get((makine, temiz_metin))
    if makine_eslesme:
        return makine_eslesme, "Aynı makine ve aynı arıza geçmişi"

    metin_eslesme = history_clean.get(temiz_metin)
    if metin_eslesme and metin_eslesme.get("count", 0) >= 3:
        return metin_eslesme, "Aynı arıza geçmişi"

    return None, None


def gecmis_ozellik_satiri(makine: str, temiz_metin: str):
    if not history_feature_maps:
        return None

    global_mean = history_feature_maps["global_mean"]
    global_median = history_feature_maps["global_median"]
    combo_key = (makine, temiz_metin)

    return np.array(
        [
            history_feature_maps["makine_smooth"].get(makine, global_mean),
            history_feature_maps["makine_median"].get(makine, global_median),
            history_feature_maps["makine_count"].get(makine, 0),
            history_feature_maps["temiz_smooth"].get(temiz_metin, global_mean),
            history_feature_maps["temiz_median"].get(temiz_metin, global_median),
            history_feature_maps["temiz_count"].get(temiz_metin, 0),
            history_feature_maps["combo_smooth"].get(combo_key, global_mean),
            history_feature_maps["combo_median"].get(combo_key, global_median),
            history_feature_maps["combo_count"].get(combo_key, 0),
        ],
        dtype=float,
    ).reshape(1, -1)


def kritik_etiketler(temiz_metin: str) -> set:
    etiketler = set()
    tokens = set(temiz_metin.split())

    if any(kelime in temiz_metin for kelime in ["rulman", "bearing"]):
        etiketler.add("rulman")
    if any(kelime in temiz_metin for kelime in ["isindi", "isiniyor", "isinma", "sicak", "sicaklik", "hararet"]):
        etiketler.add("isinma")
    if "isin_ayari" in tokens or any(
        ifade in temiz_metin
        for ifade in ["isin ayari", "lazer_isin", "lazer isin"]
    ):
        etiketler.add("isin_ayari")
    if any(kelime in temiz_metin for kelime in ["tasima", "tasinma", "malzeme_tasima"]):
        etiketler.add("tasima")
    if any(kelime in temiz_metin for kelime in ["konveyor", "konveyor_arizasi"]):
        etiketler.add("konveyor")
    if any(kelime in temiz_metin for kelime in ["sensor", "sensor_arizasi"]):
        etiketler.add("sensor")
    if any(kelime in temiz_metin for kelime in ["ayna"]):
        etiketler.add("ayna")
    if any(kelime in temiz_metin for kelime in ["kafa"]):
        etiketler.add("kafa")
    if any(kelime in temiz_metin for kelime in ["program", "program_duzeltme"]):
        etiketler.add("program")

    return etiketler


def kritik_uyumlu_mu(sorgu_etiketleri: set, aday_metin: str) -> bool:
    if not sorgu_etiketleri:
        return True

    aday_etiketleri = kritik_etiketler(aday_metin)

    if "rulman" in sorgu_etiketleri and "rulman" not in aday_etiketleri:
        return False
    if "isinma" in sorgu_etiketleri and not ({"isinma", "rulman"} & aday_etiketleri):
        return False
    if "isinma" in sorgu_etiketleri and {"isin_ayari", "tasima"} & aday_etiketleri:
        return False
    if "isin_ayari" in sorgu_etiketleri and "isinma" in aday_etiketleri:
        return False

    ortak = sorgu_etiketleri & aday_etiketleri
    return bool(ortak) or not aday_etiketleri


def benzer_kayitlari_bul(v_tfidf, makine: str, temiz_metin: str, adet: int = 6):
    if similarity_text_matrix is None or not training_records:
        return [], 0.0

    skorlar = (v_tfidf @ similarity_text_matrix.T).toarray()[0]
    sorgu_etiketleri = kritik_etiketler(temiz_metin)
    makineler = np.array([row["Makine_Tipi"] for row in training_records])
    skorlar = skorlar + np.where(makineler == makine, 0.08, 0.0)
    temizler = [row["Temiz"] for row in training_records]

    for idx, aday_metin in enumerate(temizler):
        if not kritik_uyumlu_mu(sorgu_etiketleri, aday_metin):
            skorlar[idx] = -1.0

    sirali = [idx for idx in np.argsort(-skorlar) if skorlar[idx] > 0][:adet]

    kayitlar = []
    for idx in sirali:
        row = training_records[int(idx)]
        kayitlar.append(
            {
                "Benzerlik": round(float(skorlar[idx]), 3),
                "Makine": row["Makine_Tipi"],
                "Arıza": row["Ariza_Aciklamasi"],
                "Süre (dk)": float(row["Süre_Dk"]),
            }
        )

    return kayitlar, float(skorlar[sirali[0]]) if len(sirali) else 0.0


def dinamik_planlama_suresi(tahmin_dk: float, uzun_risk: float, benzer_kayitlar: list):
    benzer_sureler = [
        row["Süre (dk)"]
        for row in benzer_kayitlar
        if row["Benzerlik"] >= 0.30
    ]

    if len(benzer_sureler) >= 3:
        p80 = float(np.percentile(benzer_sureler, 80))
        p90 = float(np.percentile(benzer_sureler, 90))
        risk_carpani = 0.35 + (uzun_risk * 0.65)
        plan_sure = max(tahmin_dk + 10.0, p80 + (p90 - p80) * risk_carpani)
        return plan_sure, "Benzer geçmiş P80/P90 + risk ayarı"

    if uzun_risk < 0.30:
        ek = 15.0
    elif uzun_risk < 0.55:
        ek = 30.0
    elif uzun_risk < 0.75:
        ek = 50.0
    else:
        ek = planning_calibration.get("upper_add_90", 80.0)

    return tahmin_dk + ek, "Risk seviyesine göre güvenlik payı"


def benzer_gecmis_ozeti(benzer_kayitlar: list):
    guvenilir_sureler = [
        row["Süre (dk)"]
        for row in benzer_kayitlar
        if row["Benzerlik"] >= 0.25
    ]

    if len(guvenilir_sureler) < 3 and len(benzer_kayitlar) >= 5:
        aday_sureler = [row["Süre (dk)"] for row in benzer_kayitlar if row["Benzerlik"] > 0]
        if aday_sureler and (max(aday_sureler) - min(aday_sureler) <= 40):
            guvenilir_sureler = aday_sureler

    if not guvenilir_sureler:
        return None

    return {
        "count": len(guvenilir_sureler),
        "median": float(np.median(guvenilir_sureler)),
        "p80": float(np.percentile(guvenilir_sureler, 80)),
        "max": float(np.max(guvenilir_sureler)),
        "min": float(np.min(guvenilir_sureler)),
    }


def guven_seviyesi(gecmis, en_yakin_skor: float, uzun_risk: float, benzer_ozet=None, benzer_kayitlar=None):
    puan = 0
    if gecmis and gecmis.get("count", 0) >= 8:
        puan += 2
    elif gecmis and gecmis.get("count", 0) >= 3:
        puan += 1

    benzer_sayi = benzer_ozet.get("count", 0) if benzer_ozet else 0
    if benzer_sayi >= 6:
        puan += 2
    elif benzer_sayi >= 3:
        puan += 1
    elif benzer_kayitlar and len(benzer_kayitlar) >= 5 and en_yakin_skor >= 0.25:
        puan += 1

    if en_yakin_skor >= 0.75:
        puan += 2
    elif en_yakin_skor >= 0.45:
        puan += 1
    elif en_yakin_skor >= 0.25 and benzer_sayi >= 5:
        puan += 1

    if 0.35 <= uzun_risk <= 0.65:
        puan -= 1

    if puan >= 3:
        return "Yüksek"
    if puan >= 1:
        return "Orta"
    return "Düşük"

# 3. ARAYÜZ TASARIMI
st.title("🏭 Arıza Bakım Karar Destek Sistemi")
st.markdown("---")

col_girdi, col_cikti = st.columns([1, 1.5], gap="large")

with col_girdi:
    st.subheader("📋 Arıza Giriş Formu")
    secilen_makine = st.selectbox("Makine Ünitesi", [m.replace("mak_", "") for m in MAK_KOL])
    ariza_notu = st.text_area("Arıza Açıklaması (Detaylı yazınız)", height=150, placeholder="Örn: Rulmanlarda aşırı ısınma ve yüksek ses var...")
    
    hesapla = st.button("ANALİZ ET VE TAHMİN ET", use_container_width=True, type="primary")

with col_cikti:
    if hesapla and ariza_notu:
        # --- İÇERİDE DÖNEN HESAPLAMALAR ---
        t_metin = temizle(ariza_notu)
        v_tfidf = vectorizer.transform([t_metin])
        
        # Kümeleme
        if cluster_uses_duration:
            s_ph = (g_mean - s_mean) / s_std
            X_clust = sp.hstack([v_tfidf, sp.csr_matrix([[s_ph * sure_weight]])], format="csr")
        else:
            X_clust = v_tfidf
        g_no = str(kmeans.predict(X_clust)[0])

        # Encoding
        ge = g_map.get(g_no, g_mean)
        me = m_map.get(secilen_makine, g_mean)
        
        # Dummy Hazırlığı
        mk_row = np.zeros((1, len(MAK_KOL)))
        gr_row = np.zeros((1, len(GRP_KOL)))
        if f"mak_{secilen_makine}" in MAK_KOL: mk_row[0, MAK_KOL.index(f"mak_{secilen_makine}")] = 1
        if f"grup_{g_no}" in GRP_KOL: gr_row[0, GRP_KOL.index(f"grup_{g_no}")] = 1
        operasyon_map = operasyon_ozellikleri(ariza_notu)
        operasyon_row = np.array(
            [[operasyon_map.get(kolon, 0) for kolon in OPERASYON_OZELLIKLERI]],
            dtype=float,
        )

        if model_feature_set == "machine_text_char":
            X_final = sp.hstack([sp.csr_matrix(mk_row), v_tfidf], format="csr")
        else:
            X_final = sp.hstack([sp.csr_matrix(mk_row), sp.csr_matrix(gr_row), sp.csr_matrix([[ge, me]]), v_tfidf], format="csr")
        
        ridge_tahmin_dk = float(np.expm1(model.predict(X_final)[0]))
        model_tahmin_dk = ridge_tahmin_dk

        gecmis_ozellikleri = gecmis_ozellik_satiri(secilen_makine, t_metin)
        if support_model is not None and gecmis_ozellikleri is not None:
            X_support = sp.hstack(
                [X_final, sp.csr_matrix(gecmis_ozellikleri)],
                format="csr",
            )
            destek_tahmin_dk = float(np.expm1(support_model.predict(X_support)[0]))
            model_tahmin_dk = (
                ensemble_model_weight * ridge_tahmin_dk
                + (1.0 - ensemble_model_weight) * destek_tahmin_dk
            )
        else:
            destek_tahmin_dk = None

        if risk_model is not None:
            uzun_risk = float(risk_model.predict_proba(X_final)[0, 1])
        else:
            uzun_risk = 1.0 if model_tahmin_dk > long_duration_threshold else 0.0

        gecmis, gecmis_kaynak = gecmis_tahmini_bul(secilen_makine, t_metin)
        benzer_kayitlar, en_yakin_skor = benzer_kayitlari_bul(v_tfidf, secilen_makine, t_metin)
        benzer_ozet = benzer_gecmis_ozeti(benzer_kayitlar)
        guven = guven_seviyesi(gecmis, en_yakin_skor, uzun_risk, benzer_ozet, benzer_kayitlar)
        tahmin_dk = model_tahmin_dk
        planlama_p80 = tahmin_dk + planning_calibration.get("upper_add_80", 0.0)
        planlama_p90_global = tahmin_dk + planning_calibration.get("upper_add_90", 0.0)
        planlama_suresi, planlama_kaynagi = dinamik_planlama_suresi(
            tahmin_dk,
            uzun_risk,
            benzer_kayitlar,
        )

        # --- GÖRSEL SONUÇLAR ---
        st.subheader("🎯 Tahmin Sonuçları")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Beklenen Süre", f"{tahmin_dk:.1f} dk")
        c2.metric("Planlama Süresi", f"{planlama_suresi:.1f} dk")
        c3.metric("Uzun Duruş Riski", f"%{uzun_risk * 100:.0f}")
        c4.metric("Güven", guven)

        if uzun_risk >= 0.65 or planlama_suresi > 120:
            st.error(f"🔴 KRİTİK: Üretim planında {planlama_suresi/60:.1f} saate kadar duruş payı ayırın.")
        elif uzun_risk >= 0.35 or planlama_suresi > 60:
            st.warning(f"🟡 DİKKAT: Planlama için {planlama_suresi:.0f} dk güvenli süre kullanın.")
        else:
            st.success("🟢 NORMAL: Kısa süreli müdahale olasılığı daha yüksek.")

        if benzer_kayitlar:
            st.markdown("#### Benzer Geçmiş Özeti")
            if benzer_ozet:
                o1, o2, o3, o4 = st.columns(4)
                o1.metric("Benzer Kayıt", int(benzer_ozet["count"]))
                o2.metric("Medyan", f"{benzer_ozet['median']:.1f} dk")
                o3.metric("P80", f"{benzer_ozet['p80']:.1f} dk")
                o4.metric("Maksimum", f"{benzer_ozet['max']:.1f} dk")
            else:
                st.info("Güvenilir benzer geçmiş özeti üretmek için yeterli eşleşme yok.")

            st.markdown("#### Benzer Geçmiş Kayıtlar")
            st.dataframe(pd.DataFrame(benzer_kayitlar), use_container_width=True, hide_index=True)

        # JÜRİ İÇİN TEKNİK DETAY (Expander)
        with st.expander("🔍 Model Nasıl Karar Verdi? (Teknik Detay)"):
            st.write(f"**İşlenen Kelimeler:** `{t_metin}`")
            st.write(f"**Model Tahmini:** {model_tahmin_dk:.1f} dk")
            st.write(f"**Ridge Tahmini:** {ridge_tahmin_dk:.1f} dk")
            if destek_tahmin_dk is not None:
                st.write(f"**XGBoost Destek Tahmini:** {destek_tahmin_dk:.1f} dk")
            st.write(f"**Planlama Kaynağı:** {planlama_kaynagi}")
            st.write(f"**Planlama P80 Üst Süre:** {planlama_p80:.1f} dk")
            st.write(f"**Global P90 Üst Süre:** {planlama_p90_global:.1f} dk")
            st.write(f"**Dinamik Planlama Süresi:** {planlama_suresi:.1f} dk")
            st.write(f"**Uzun Duruş Eşiği:** {long_duration_threshold} dk")
            st.write(f"**Uzun Duruş Riski:** %{uzun_risk * 100:.1f}")
            st.write(f"**En Yakın Benzerlik Skoru:** {en_yakin_skor:.3f}")
            aktif_operasyonlar = [
                OPERASYON_ADLARI.get(kolon, kolon)
                for kolon in OPERASYON_OZELLIKLERI
                if operasyon_map.get(kolon, 0) == 1
            ]
            st.write("**Algılanan Operasyon Özellikleri:** " + (", ".join(aktif_operasyonlar) if aktif_operasyonlar else "Yok"))
            if benzer_ozet:
                st.write(
                    f"**Benzer Geçmiş Özeti:** medyan {benzer_ozet['median']:.1f} dk, "
                    f"P80 {benzer_ozet['p80']:.1f} dk, maksimum {benzer_ozet['max']:.1f} dk"
                )
            if gecmis:
                st.write(f"**Kullanılan Geçmiş:** {gecmis_kaynak}")
                st.write(
                    f"**Geçmiş Aralık:** {gecmis['min']:.1f} - {gecmis['max']:.1f} dk "
                    f"({int(gecmis['count'])} kayıt)"
                )
            st.write(f"**Grup Etkisi:** {np.expm1(ge):.1f} dk (Ortalama)")
            st.write(f"**Makine Etkisi:** {np.expm1(me):.1f} dk (Ortalama)")
            if not t_metin:
                st.warning("⚠️ Uyarı: Açıklama çok kısa olduğu için model metin analizi yapamadı, sadece genel ortalamayı kullanıyor.")

    else:
        st.info("Analiz sonuçlarını görmek için lütfen sol taraftaki formu doldurup butona basın.")

st.markdown("---")
with st.expander("🗂️ Veri Yönetimi (Yeni Kayıt / Kayıt Düzenleme)", expanded=False):
    veri_df = veri_dosyasini_yukle()
    makine_secenekleri = [m.replace("mak_", "") for m in MAK_KOL]
    veri_makineleri = []
    if "Makine_Tipi" in veri_df.columns:
        veri_makineleri = sorted(veri_df["Makine_Tipi"].dropna().astype(str).unique().tolist())
    makine_secenekleri = sorted(set(makine_secenekleri + veri_makineleri))

    st.caption(f"Veri dosyası: `{data_file}` | Toplam kayıt: {len(veri_df)}")
    st.info("Bu bölüm Excel verisini günceller. Yeni kayıtların model tahminlerine yansıması için modelin yeniden eğitilmesi gerekir.")

    tab_ekle, tab_duzenle = st.tabs(["Yeni Kayıt Ekle", "Kayıt Düzenle"])

    with tab_ekle:
        with st.form("yeni_kayit_formu", clear_on_submit=True):
            yeni_tarih = st.date_input("Tarih")
            yeni_makine = st.selectbox("Makine Ünitesi", makine_secenekleri, key="yeni_makine")
            yeni_ariza = st.text_area("Arıza Açıklaması", height=110, key="yeni_ariza")
            yeni_sure = st.number_input("Süre (dk)", min_value=1, max_value=10000, value=30, step=1, key="yeni_sure")
            yeni_kaydet = st.form_submit_button("Kaydı Ekle", type="primary")

        if yeni_kaydet:
            if not yeni_ariza.strip():
                st.error("Arıza açıklaması boş olamaz.")
            else:
                guncel_df = veri_dosyasini_yukle()
                if guncel_df.empty:
                    guncel_df = pd.DataFrame(columns=["Tarih", "Makine_Tipi", "Ariza_Aciklamasi", "Süre_Dk"])

                yeni_satir = veri_satiri_olustur(
                    guncel_df.columns,
                    yeni_tarih,
                    yeni_makine,
                    yeni_ariza.strip(),
                    yeni_sure,
                )
                guncel_df = pd.concat(
                    [guncel_df, pd.DataFrame([yeni_satir], columns=guncel_df.columns)],
                    ignore_index=True,
                )
                veri_dosyasini_kaydet(guncel_df)
                st.success(f"Kayıt eklendi. Güncel kayıt sayısı: {len(guncel_df)}")

    with tab_duzenle:
        if veri_df.empty:
            st.warning("Düzenlenecek kayıt bulunamadı.")
        else:
            arama = st.text_input("Kayıt ara (makine veya arıza açıklaması)", key="kayit_arama")
            gorunum = veri_df.copy()
            if arama.strip():
                arama_metni = arama.strip().lower()
                bos_seri = pd.Series("", index=gorunum.index)
                makine_mask = gorunum.get("Makine_Tipi", bos_seri).astype(str).str.lower().str.contains(arama_metni, na=False)
                ariza_mask = gorunum.get("Ariza_Aciklamasi", bos_seri).astype(str).str.lower().str.contains(arama_metni, na=False)
                gorunum = gorunum[makine_mask | ariza_mask]

            gorunum = gorunum.copy()
            gorunum.insert(0, "Kayıt_No", gorunum.index.astype(int))
            gosterilecek_kolonlar = [
                kolon
                for kolon in ["Kayıt_No", "Tarih", "Makine_Tipi", "Ariza_Aciklamasi", "Süre_Dk"]
                if kolon in gorunum.columns
            ]
            st.dataframe(
                gorunum.sort_values("Kayıt_No", ascending=False).head(100)[gosterilecek_kolonlar],
                use_container_width=True,
                hide_index=True,
            )

            kayit_no = st.number_input(
                "Düzenlenecek Kayıt No",
                min_value=0,
                max_value=max(0, len(veri_df) - 1),
                value=max(0, len(veri_df) - 1),
                step=1,
            )
            secili_satir = veri_df.iloc[int(kayit_no)]
            tarih_degeri = pd.to_datetime(secili_satir.get("Tarih", pd.Timestamp.today()), errors="coerce")
            if pd.isna(tarih_degeri):
                tarih_degeri = pd.Timestamp.today()
            makine_degeri = str(secili_satir.get("Makine_Tipi", makine_secenekleri[0] if makine_secenekleri else ""))
            if makine_degeri not in makine_secenekleri:
                makine_secenekleri.append(makine_degeri)
            sure_degeri = pd.to_numeric(secili_satir.get("Süre_Dk", 30), errors="coerce")
            if pd.isna(sure_degeri) or sure_degeri < 1:
                sure_degeri = 30

            with st.form("kayit_duzenleme_formu"):
                duzenle_tarih = st.date_input("Tarih", value=tarih_degeri.date(), key="duzenle_tarih")
                duzenle_makine = st.selectbox(
                    "Makine Ünitesi",
                    makine_secenekleri,
                    index=makine_secenekleri.index(makine_degeri),
                    key="duzenle_makine",
                )
                duzenle_ariza = st.text_area(
                    "Arıza Açıklaması",
                    value=str(secili_satir.get("Ariza_Aciklamasi", "")),
                    height=110,
                    key="duzenle_ariza",
                )
                duzenle_sure = st.number_input(
                    "Süre (dk)",
                    min_value=1,
                    max_value=10000,
                    value=int(round(float(sure_degeri))),
                    step=1,
                    key="duzenle_sure",
                )
                guncelle = st.form_submit_button("Kaydı Güncelle", type="primary")

            if guncelle:
                if not duzenle_ariza.strip():
                    st.error("Arıza açıklaması boş olamaz.")
                else:
                    guncel_df = veri_dosyasini_yukle()
                    hedef_index = int(kayit_no)
                    if "Tarih" in guncel_df.columns:
                        guncel_df.at[hedef_index, "Tarih"] = pd.Timestamp(duzenle_tarih)
                    if "Makine_Tipi" in guncel_df.columns:
                        guncel_df.at[hedef_index, "Makine_Tipi"] = duzenle_makine
                    if "Ariza_Aciklamasi" in guncel_df.columns:
                        guncel_df.at[hedef_index, "Ariza_Aciklamasi"] = duzenle_ariza.strip()
                    if "Süre_Dk" in guncel_df.columns:
                        guncel_df.at[hedef_index, "Süre_Dk"] = float(duzenle_sure)
                    if "Sure_Dk_Orijinal" in guncel_df.columns:
                        guncel_df.at[hedef_index, "Sure_Dk_Orijinal"] = float(duzenle_sure)
                    if "Normalize_Ariza_Grubu" in guncel_df.columns:
                        guncel_df.at[hedef_index, "Normalize_Ariza_Grubu"] = temizle(duzenle_ariza)
                    if "Normalize_Notu" in guncel_df.columns:
                        guncel_df.at[hedef_index, "Normalize_Notu"] = "Arayüzden düzenlenen kayıt."

                    veri_dosyasini_kaydet(guncel_df)
                    st.success(f"{hedef_index} numaralı kayıt güncellendi.")
