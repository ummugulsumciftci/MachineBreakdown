import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
import scipy.sparse as sp

# 1. SAYFA AYARLARI VE MODEL YÜKLEME
st.set_page_config(page_title="Hakan Sac Metal | DSS", layout="wide")

@st.cache_resource
def modeli_yukle():
    return joblib.load("egitilmis_model.pkl")

paket = modeli_yukle()
model = paket["model"]
vectorizer = paket["vectorizer"]
kmeans = paket["kmeans"]
g_map = paket["grup_mean_map"]
m_map = paket["makine_mean_map"]
g_mean = paket["global_mean"]
MAK_KOL = paket["MAKINE_KOLONLARI"]
GRP_KOL = paket["GRUP_KOLONLARI"]
s_mean = paket["sure_log_mean"]
s_std = paket["sure_log_std"]

ANLAMSIZ_TOKENLAR = {
    "zli", "lmesi", "ldi", "lmis", "lmak", "masi",
    "yardim", "arizasi", "ariza",
    "lar", "ler", "dan", "den", "nin", "nun", "bir", "ile"
}

def temizle(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'\d+\s*(dk|dakika|dak|saat|sa|sn|saniye)\.?', '', text)
    text = re.sub(r'\d+', '', text)
    tr_map = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    text = text.translate(tr_map)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [t for t in text.split() if t not in ANLAMSIZ_TOKENLAR and len(t) >= 4]
    return " ".join(tokens)

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
        s_ph = (g_mean - s_mean) / s_std
        X_clust = sp.hstack([v_tfidf, sp.csr_matrix([[s_ph * 2.0]])], format="csr")
        g_no = str(kmeans.predict(X_clust)[0])

        # Encoding
        ge = g_map.get(g_no, g_mean)
        me = m_map.get(secilen_makine, g_mean)
        
        # Dummy Hazırlığı
        mk_row = np.zeros((1, len(MAK_KOL)))
        gr_row = np.zeros((1, len(GRP_KOL)))
        if f"mak_{secilen_makine}" in MAK_KOL: mk_row[0, MAK_KOL.index(f"mak_{secilen_makine}")] = 1
        if f"grup_{g_no}" in GRP_KOL: gr_row[0, GRP_KOL.index(f"grup_{g_no}")] = 1

        X_final = sp.hstack([sp.csr_matrix(mk_row), sp.csr_matrix(gr_row), sp.csr_matrix([[ge, me]]), v_tfidf], format="csr")
        
        tahmin_dk = np.expm1(model.predict(X_final)[0])

        # --- GÖRSEL SONUÇLAR ---
        st.subheader("🎯 Tahmin Sonuçları")
        
        c1, c2 = st.columns(2)
        c1.metric("Tahmini Onarım Süresi", f"{tahmin_dk:.1f} Dakika")
        c2.metric("Atanan Arıza Grubu", f"Grup {g_no}")

        if tahmin_dk > 60:
            st.error(f"🔴 KRİTİK: Yaklaşık {tahmin_dk/60:.1f} saat duruş öngörülüyor. Bakım ekibini acil çağırın.")
        else:
            st.success("🟢 NORMAL: Kısa süreli müdahale ile çözülebilir görünüyor.")

        # JÜRİ İÇİN TEKNİK DETAY (Expander)
        with st.expander("🔍 Model Nasıl Karar Verdi? (Teknik Detay)"):
            st.write(f"**İşlenen Kelimeler:** `{t_metin}`")
            st.write(f"**Grup Etkisi:** {np.expm1(ge):.1f} dk (Ortalama)")
            st.write(f"**Makine Etkisi:** {np.expm1(me):.1f} dk (Ortalama)")
            if not t_metin:
                st.warning("⚠️ Uyarı: Açıklama çok kısa olduğu için model metin analizi yapamadı, sadece genel ortalamayı kullanıyor.")

    else:
        st.info("Analiz sonuçlarını görmek için lütfen sol taraftaki formu doldurup butona basın.")