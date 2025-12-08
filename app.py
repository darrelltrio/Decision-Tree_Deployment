import streamlit as st
import pandas as pd
import joblib

# Fungsi untuk menentukan kategori umur (REPLIKA DARI model.py)
# Logika: <=10 TAHUN = New, 11-30 TAHUN = Modern, >30 TAHUN = Old
def categorize_age(age):
    if age <= 10:
        return 'New'
    elif age <= 30:
        return 'Modern'
    else:
        return 'Old'

# Load Model DAN Metrics
try:
    model = joblib.load('model_decision_tree.pkl')
    metrics = joblib.load('model_metrics.pkl')
except FileNotFoundError:
    st.error("File model atau metrics tidak ditemukan. Jalankan model.py terlebih dahulu.")
    st.stop()

# Judul
st.title("🏠 Prediksi Harga Rumah")
st.caption("Dibuat menggunakan Algoritma Decision Tree Regressor")

# Tampilkan Performa Model Secara Dinamis
st.subheader("Performa Model Saat Ini")
col1, col2 = st.columns(2)

with col1:
    st.metric("Akurasi (R² Score)", f"{metrics['r2']:.2%}")

with col2:
    st.metric("Rata-rata Error (RMSE)", f"${metrics['rmse']:,.0f}")

st.markdown("---")

# Form Input User
st.header("Masukkan Detail Rumah")
c1, c2 = st.columns(2)

# --- KOLOM 1 ---
with c1:
    sq_feet = st.number_input("Luas Tanah (sq feet)", min_value=500, value=1500)
    num_rooms = st.number_input("Jumlah Kamar", min_value=1, max_value=10, value=3)

# --- KOLOM 2 ---
with c2:
    age = st.number_input("Umur Bangunan (Tahun)", min_value=0, value=10)
    
    # --- BAGIAN BARU: Tampilkan Kategori Umur Secara Dinamis ---
    age_category = categorize_age(age)
    if age_category == 'New':
        st.info(f"Kategori: **{age_category}** (Maksimal 10 Tahun)")
    elif age_category == 'Modern':
        st.warning(f"Kategori: **{age_category}** (11 - 30 Tahun)")
    else:
        st.error(f"Kategori: **{age_category}** (Di atas 30 Tahun)")
    
    dist = st.number_input("Jarak ke Kota (km)", min_value=0.0, value=5.5)


if st.button("Estimasi Harga"):
    
    # Feature Engineering Manual (Menggunakan hasil categorize_age)
    age_cat_for_model = categorize_age(age)
    
    # Buat DataFrame
    input_data = pd.DataFrame({
        'square_feet': [sq_feet],
        'num_rooms': [num_rooms],
        'age': [age],
        'distance_to_city(km)': [dist],
        'Age_Category': [age_cat_for_model]
    })
    
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimasi Harga: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")