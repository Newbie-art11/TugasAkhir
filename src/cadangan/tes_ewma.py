import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
# Fungsi untuk memuat model dari file pickle
def load_model():
    model_filename = os.path.join('../models/new_model_ewma/', f'ewma_model_XRP-USD.pkl')
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Fungsi untuk melakukan forecasting
def forecast(ewma_model, n_forecast):
    last_value = ewma_model.iloc[-1]
    forecast = [last_value] * n_forecast
    future_dates = pd.date_range(start=ewma_model.index[-1] + pd.Timedelta(days=1), periods=n_forecast, freq='D')
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_dates)
    return forecast_df

# Memuat model yang telah disimpan
ewma_model = load_model()

# Sidebar untuk input jumlah hari forecast
st.sidebar.title("Forecasting Parameters")
n_forecast = st.sidebar.number_input("Number of Days to Forecast", min_value=1, max_value=30, value=5)

# Menampilkan data EWMA dan forecasting di aplikasi
st.title("EWMA Forecasting")
st.write("Model EWMA yang dimuat:")

# Tampilkan tabel model EWMA
st.line_chart(ewma_model)

# Melakukan forecasting berdasarkan input user
forecast_df = forecast(ewma_model, n_forecast)
print(forecast_df.info())
# Gabungkan model asli dan hasil forecast
st.write(f"Forecasting untuk {n_forecast} hari ke depan:")
st.line_chart(forecast_df)

# Visualisasi gabungan EWMA dan Forecast
st.write("Visualisasi Data EWMA dan Forecast:")
fig, ax = plt.subplots()
ewma_model.plot(ax=ax, label='EWMA (Loaded Model)', color='orange')
forecast_df.plot(ax=ax, label='Forecast', color='green', linestyle='dashed')
ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.legend()
st.pyplot(fig)
