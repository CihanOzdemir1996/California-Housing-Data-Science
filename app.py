import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing

# 1. Sayfa Konfigürasyonu
st.set_page_config(
    page_title="California Emlak Analiz & Tahmin",
    page_icon="🏠",
    layout="wide"
)

# 2. Model ve Veri Yükleme (Önbelleğe alarak hızı artırıyoruz)
@st.cache_resource
def setup_model():
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = housing.target
    
    # Modelimizi eğitiyoruz
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, housing

model, raw_data = setup_model()

# 3. Başlık ve Giriş
st.title("🏠 California Konut Piyasası Analiz Paneli")
st.markdown("""
Bu profesyonel panel, **Makine Öğrenmesi (Random Forest)** kullanarak California'daki ev fiyatlarını tahmin eder 
ve verilerin arkasındaki derin analizleri sunar.
""")

# 4. Sekmeli Yapı Oluşturma
tab1, tab2 = st.tabs(["🎯 Fiyat Tahmin Aracı", "📊 Model Analizleri & Yorumlar"])

# --- SEKME 1: TAHMİN ARACI ---
with tab1:
    st.header("Konut Değerleme Simülatörü")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Ev Özelliklerini Girin")
        med_inc = st.slider("Bölge Gelir Düzeyi (MedInc)", 0.5, 15.0, 3.5, help="Bölgedeki ortalama gelir düzeyi (10.000$)")
        house_age = st.slider("Bina Yaşı", 1, 52, 28)
        ave_rooms = st.slider("Ortalama Oda Sayısı", 1, 10, 5)
        ave_occup = st.slider("Hanehalkı Sayısı", 1, 6, 3)
        lat = st.slider("Enlem (Latitude)", 32.0, 42.0, 35.0)
        lon = st.slider("Boylam (Longitude)", -124.0, -114.0, -119.0)
        
        # Giriş verilerini DataFrame yapalım
        input_data = pd.DataFrame({
            'MedInc': [med_inc], 'HouseAge': [house_age], 'AveRooms': [ave_rooms],
            'AveBedrms': [1.0], 'Population': [1500], 'AveOccup': [ave_occup],
            'Latitude': [lat], 'Longitude': [lon]
        })

    with col2:
        st.subheader("Tahmin Sonucu")
        if st.button("Piyasa Değerini Hesapla"):
            prediction = model.predict(input_data)[0]
            st.metric(label="Tahmini Ev Değeri", value=f"${prediction * 100000:,.0f}")
            st.success("Model başarıyla hesapladı! Bu fiyat, girdiğiniz kriterlere göre piyasa ortalamasını yansıtır.")
            st.balloons()
        else:
            st.info("Hesaplama yapmak için soldaki butonları kullanın.")

# --- SEKME 2: ANALİZ VE YORUMLAR ---
with tab2:
    st.header("Veri Bilimi Çıktıları")
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.subheader("1. Hangi Özellik Fiyatı Daha Çok Etkiliyor?")
        st.image("california_housing_feature_importance.png")
        st.write("""
        **Analiz Yorumu:** Grafikte görüldüğü üzere, bir evin fiyatını belirleyen en güçlü faktör **MedInc (Bölge Gelir Düzeyi)**. 
        Bu, konut piyasasında lokasyonun ve sosyo-ekonomik durumun önemini kanıtlar.
        """)

    with col_img2:
        st.subheader("2. Model Ne Kadar Başarılı?")
        st.image("california_housing_predictions.png")
        st.write("""
        **Analiz Yorumu:** Gerçek fiyatlar ile tahminler arasındaki korelasyon, modelin piyasa trendlerini 
        doğru yakaladığını gösteriyor. Çizgiye yakın noktalar isabetli tahminleri temsil eder.
        """)

    st.divider()
    st.subheader("3. Veri Dağılımı ve Korelasyon")
    st.image("california_housing_eda.png", use_container_width=True)
    st.write("**Genel Değerlendirme:** Veri setindeki evlerin yaş, konum ve gelir değişkenlerine göre dağılımı, modelin öğrenme sürecindeki temel dayanak noktasıdır.")

# Alt Bilgi
st.sidebar.markdown("---")
st.sidebar.write("💻 Geliştiren: Cihan Özdemir")
st.sidebar.info("Bu proje bir Portfolyo çalışmasıdır.")