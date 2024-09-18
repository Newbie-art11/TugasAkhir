import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Prediksi EWMA ETH-USD")

# Memuat model EWMA dari file
with open('models/ewma_model_ETH-USD.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Dummy data test sebagai contoh (ini harus diganti dengan data asli kamu)
# Contoh: test_data = pd.read_csv('test_data.csv', index_col='Date', parse_dates=True)
test_data = pd.DataFrame({'ETH-USD': [2000, 2100, 2200, 2300, 2250, 2350]}, 
                         index=pd.date_range(start='2024-07-31', periods=6, freq='W'))

# Menghitung prediksi EWMA
ewma_test_loaded = test_data.ewm(span=30, adjust=False).mean()

# Membuat plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(test_data.index, test_data, label='Data Uji')
ax.plot(test_data.index, ewma_test_loaded, label='Prediksi EWMA dari Model yang Dimuat', color='green')
ax.legend()

# Menampilkan plot di Streamlit
st.pyplot(fig)

# Menampilkan data uji dan hasil prediksi di Streamlit
st.write("Data Uji:", test_data)
st.write("Prediksi EWMA:", ewma_test_loaded)
