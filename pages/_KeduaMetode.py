import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import pickle
import joblib
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


#List 10 cryptocurrency 

cryptosList = ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", "USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]

for model in cryptosList:
    with open(f'models/ewma_model_{model}.pkl', 'rb') as f:  # f-string digunakan di sini
        loaded_model = pickle.load(f)
        # Lakukan sesuatu dengan loaded_model
def main():
    st.title("Selamat Datang di Top 10 Crptocurrency Prediksi Untuk Kedua Metode")
    st.write("Aplikasi ini memprediksi harga mata uang kripto menggunakan Exponentially Weighted Moving Average (EWMA) dan Triple Exponential Smoothing (TES). Model tersebut dilatih pada data historis untuk prediksi yang lebih cepat.")   

    #Input Sidebar
    st.sidebar.header("Data Unduhan")
    symbol_stock = st.sidebar.selectbox("Pilih Cryptocurrency:",cryptosList)
    start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("Tanggal Terakhir",pd.to_datetime("2024-07-31"))
    
    data = yf.download(symbol_stock, start=start_date, end=end_date)

    #tab1,tab2,tab3
    tab1, tab2, tab3, tab4 = st.tabs(["Open Price","High Price","Close Price","Volume Price"])
    with tab1:
        st.header(f"Hasil Harga Open {symbol_stock} untuk EWMA dan TES")
        st.write(f"Exponential Weighted Moving Average - MSE:")
        st.write(f"Exponential Weighted Moving Average - MAPE:")
        st.write(f"Triple Exponential Smoothing - MSE: ", )
        st.write(f"Triple Exponential Smoothing - MAPE: ", )
        # visualize_predictions(data, trainsize, test_open, forecast_ewma_open, forecast_tes_open, 'Open', )
    with tab2:
        st.header(f"Hasil Harga High {symbol_stock} untuk EWMA dan TES")
        st.write(f"Exponential Weighted Moving Average - MSE:")
        st.write(f"Exponential Weighted Moving Average - MAPE:")
        st.write(f"Triple Exponential Smoothing - MSE: ")
        st.write(f"Triple Exponential Smoothing - MAPE: ")
    with tab3:
        st.header(f"Hasil Harga Close {symbol_stock}  Untuk EWMA dan TES")
        st.write(f"Exponential Weighted Moving Average - MSE:")
        st.write(f"Exponential Weighted Moving Average - MAPE:")
        st.write(f"Triple Exponential Smoothing - MSE: ")
        st.write(f"Triple Exponential Smoothing - MAPE: ")
    with tab4:
        st.header(f"Hasil Harga Volume {symbol_stock} Untuk EWMA dan TES")   
        st.write(f"Exponential Weighted Moving Average - MSE:")
        st.write(f"Exponential Weighted Moving Average - MAPE:")
        st.write(f"Triple Exponential Smoothing - MSE: ")
        st.write(f"Triple Exponential Smoothing - MAPE: ")
        
def visualize_predictions(data, train_size, y_test, y_pred_tes, y_pred_arima, price_type):
    fig = go.Figure()

    # Add training data
    fig.add_trace(go.Scatter(x=data.index[:train_size],
                             y=data[price_type][:train_size],
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    # Add actual stock prices
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_test,
                             mode='lines',
                             name="Actual Prices",
                             line=dict(color='blue')))

    # Add TES predictions
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_pred_tes,
                             mode='lines',
                             name="EWMA Predictions",
                             line=dict(color='red')))

    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_pred_arima,
                             mode='lines',
                             name="TES Predictions",
                             line=dict(color='green')))

    fig.update_layout(title=f"{price_type} Price Prediction for EWMA & TES",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template='plotly_dark')

    st.plotly_chart(fig)
    
    st.divider()
    st.write("EWMA_Prediction :")
    st.write("TES_Pediction :")
    st.write("EWMA_Accuraccy Average :")
    st.write("TES_Accuraccy Average :")
    st.write("EWMA_Loss Average :")
    st.write("TES_Loss Average :")
if __name__ == "__main__":
    main()
