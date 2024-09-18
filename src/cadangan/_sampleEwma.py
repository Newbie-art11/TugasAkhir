import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Setup tampilan Streamlit
st.title('Crypto Forecasting dengan EWMA dan Conditional Formatting')
st.write('Aplikasi ini melakukan forecasting harga crypto dengan menggunakan metode Exponential Weighted Moving Average (EWMA), serta memberikan warna pada angka optimal.')

cryptos = ["BTC-USD", "ETH-USD","USDT-USD","BNB-USD", "SOL-USD","USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]

symbol_stock = st.sidebar.selectbox("Pilih Crypto", cryptos)
# Download data dari yfinance
data = yf.download(symbol_stock, start='2021-01-01', end='2024-09-30', interval='1d')

# Cek jika data tidak kosong
if not data.empty:
    # Ambil harga penutupan
    data_daily = data['Close']

    # Konversi data harian menjadi data mingguan
    data_weekly = data_daily.resample('W').std()
    open_prices = data['Open'].resample('W').std()
    high_prices = data['High'].resample('W').std()
    close_prices = data['Close'].resample('W').std()
    volume_prices = data['Volume'].resample('W').std()
    combineData = pd.DataFrame({
        'HargaBuka': open_prices,
        'HargaTertingi': high_prices,
        'HargaTutup': close_prices,
        'HargaVolume': volume_prices
    })
    # Melakukan Min-Max Scaling
    min_val = data_weekly.min()
    max_val = data_weekly.max()
    minvalueOpen = open_prices.min()
    maxvalueOpen = open_prices.max()
    minvalueHigh = high_prices.min()
    maxvalueHigh = high_prices.max()
    minvalueClose = close_prices.min()
    maxvalueClose = close_prices.max()
    minvalueVolume = volume_prices.min()
    maxvalueVolume = volume_prices.max()
    
    scaledDataOpen = (open_prices - minvalueOpen) / (maxvalueOpen - minvalueOpen) 
    scaledDataHigh = (high_prices - minvalueHigh) / (maxvalueHigh - minvalueHigh) 
    scaledDataClose = (close_prices - minvalueClose) / (maxvalueClose - maxvalueClose) 
    scaledDataVolume = (volume_prices - minvalueVolume) / (maxvalueVolume - minvalueVolume) 
    scaled_data = (data_weekly - min_val) / (max_val - min_val)
    combineDataScale = pd.DataFrame({
        'HargaBuka': scaledDataOpen,
        'HargaTertingi': scaledDataHigh,
        'HargaTutup': scaledDataClose,
        'HargaVolume': scaledDataVolume
    })
    # Tampilkan tabel sebelum dan sesudah Min-Max Scaling
    st.write('### Data Standar Deviasi Mingguan')
    st.dataframe(data_weekly)
    st.dataframe(combineData)

    st.write('### Data Setelah Min-Max Scaling')
    st.dataframe(combineDataScale)
    
    # Data training (Januari 2021 - Juli 2024)
    training_data = data_weekly.loc['2021-01-01':'2024-07-31']

    # Menggunakan metode EWMA (Exponential Weighted Moving Average) pada data training
    alpha = st.slider('Pilih nilai alpha (smoothing factor)', 0.0, 1.0, 0.3)
    ewma_model = training_data.ewm(alpha=alpha, adjust=False).mean()

    # Prediksi untuk data testing (Agustus - September 2024)
    future_dates = pd.date_range(start='2024-08-01', end='2024-09-30', freq='W')
    predictions = []

    last_ewma_value = ewma_model.iloc[-1]  # Nilai EWMA terakhir dari training data
    for _ in future_dates:
        predictions.append(last_ewma_value)
        last_ewma_value = alpha * last_ewma_value + (1 - alpha) * last_ewma_value

    # Buat DataFrame untuk hasil prediksi
    df_predictions = pd.DataFrame(predictions, index=future_dates, columns=['Predictions'])

    # Ambil data aktual untuk periode testing (Agustus - September 2024)
    actual_data = data_weekly.loc['2024-08-01':'2024-09-30']

    # Menggabungkan data prediksi dan aktual untuk evaluasi
    df_evaluation = pd.concat([actual_data, df_predictions], axis=1)
    df_evaluation.columns = ['Actual', 'Predictions']

    # Hapus baris dengan missing values (jika ada)
    df_evaluation.dropna(inplace=True)

    # Menghitung Loss dan Selisih (EWMA - Actual)
    df_evaluation['Loss'] = df_evaluation['Predictions'] - df_evaluation['Actual']
    df_evaluation['Selisih'] = np.abs(df_evaluation['Actual'] - df_evaluation['Predictions'])
    df_evaluation['Accuracy (%)'] = 100 - (np.abs((df_evaluation['Actual'] - df_evaluation['Predictions']) / df_evaluation['Actual']) * 100)

   
       # Menghitung MAPE
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Menghitung MSE
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    # Hitung MAPE dan MSE
    mape = mean_absolute_percentage_error(df_evaluation['Actual'], df_evaluation['Predictions'])
    mse = mean_squared_error(df_evaluation['Actual'], df_evaluation['Predictions'])

    # Menentukan warna berdasarkan apakah nilai optimal atau tidak
    mape_color = 'green' if mape < 50 else 'red'  # Anggap optimal jika MAPE < 10%
    mse_color = 'green' if mse < 1000 else 'red'  # Anggap optimal jika MSE < 1000

    # Tampilkan MAPE dan MSE di Streamlit dengan pewarnaan
    st.markdown(f'MAPE : <p style="color:{mape_color};"> {mape:.2f}%</p>', unsafe_allow_html=True)
    st.markdown(f'MSE : <p style="color:{mse_color};"> {mse:.2f}</p>', unsafe_allow_html=True)


    # Fungsi untuk memberikan warna hijau untuk optimal dan merah untuk tidak optimal
    def highlight_values(val):
        if val >= 1000:  # Misal angka optimal jika Accuracy >= 90
            color = 'green'
        else:
            color = 'red'
        return f'color: {color}'

    # Terapkan styling pada kolom 'Accuracy (%)', 'Selisih', dan 'Loss' menggunakan Pandas Styler
    styled_df = df_evaluation.style.applymap(highlight_values, subset=['Accuracy (%)', 'Selisih', 'Loss'])

    # Menampilkan tabel evaluasi dengan styling
    st.write('### Tabel Evaluasi Prediksi')
    st.dataframe(styled_df)

        # Menampilkan grafik prediksi vs data aktual
    # Membuat figure
    fig = go.Figure()

    # Menambahkan data aktual
    fig.add_trace(go.Scatter(x=df_evaluation.index, y=df_evaluation['Actual'],
                            mode='lines+markers',
                            name='Actual Data',
                            line=dict(color='blue'),
                            marker=dict(symbol='circle')))

    # Menambahkan data prediksi
    fig.add_trace(go.Scatter(x=df_evaluation.index, y=df_evaluation['Predictions'],
                            mode='lines+markers',
                            name='Predicted Data (EWMA)',
                            line=dict(color='green', dash='dash'),
                            marker=dict(symbol='x')))

    # Menambahkan judul dan label
    fig.update_layout(title=f'{symbol_stock} - Actual vs Predicted (EWMA)',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    legend_title='Legend')

    # Menampilkan plot di Streamlit
    st.write('### Grafik Prediksi vs Data Aktual')
    st.plotly_chart(fig)

else:
    st.write(f"Tidak ada data yang tersedia untuk simbol {symbol_stock}. Coba simbol lain.")
