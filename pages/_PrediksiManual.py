import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing 
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error 
import statsmodels.api as sm 
import math


# List 10 cryptocurrency 
cryptosList = ["BTC-USD", "ETH-USD","USDT-USD","BNB-USD", "SOL-USD","USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]

# Function for downloading data 
def download_data(crypto, start_date, end_date):
    data = yf.download(crypto, start=start_date, end=end_date)
    return data

# Function to calculate evaluation metrics
def metrik_calculating(actual, forecast):
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    mse = mean_squared_error(actual,forecast)
    return mape, mse

# Main app function
def main():
    st.title("Manual Memprediksi Harga Top 10 Cryptocurrency")
    st.write("Aplikasi ini memungkinkan penyetelan parameter manual untuk model Expoenentially Weighted Moving Average  dan Triple Exponential Smoothing.")
    
    # Sidebar for user input
    st.sidebar.header("Data Unduhan")
    stock_symbol = st.sidebar.selectbox("Pilih Cryptocurrency:", cryptosList)
    start_date = st.sidebar.date_input("Tanggal Mulai :", pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("Tanggal Akhir  :", pd.to_datetime("2024-07-31"))
    modelchoice = st.sidebar.radio("Select Model:", ["Exponentially Weighted Moving Average (EWMA)", "Triple Exponential Smoothing (TES)"])
    
    # Title for selected model and stock
    st.title(f"{modelchoice} Prediksi untuk {stock_symbol}")
    
    # Download data
    data = download_data(stock_symbol, start_date, end_date)
    
    # Process data
    open_price = data['Open'].resample('W').std()
    high_price = data['High'].resample('W').std()
    close_price = data['Close'].resample('W').std()
    volume_price = data['Volume'].resample('W').std()

    train_size = int(len(close_price) * 0.8)
    train_close = close_price[:train_size]
    test_close = close_price[train_size:]
    train_open = open_price[:train_size]
    test_open = open_price[train_size:]
    train_high = high_price[:train_size]
    test_high = high_price[train_size:]
    train_volume = volume_price[:train_size]
    test_volume = volume_price[train_size:]

    # Parameter Selection
    st.header("Manual Parameter Selection")
    feature_for_choice = st.selectbox("Pilih Parameter untuk di Prediksi", ["Open", "High", "Close", "Volume"])
    
    if modelchoice == "Exponentially Weighted Moving Average (EWMA)":
        alpha = st.slider("Parameter Smoothing factor EWMA (@)", min_value=0.01, max_value=1.00, value=0.3, step=0.01)
    else:
        alpha = st.slider("Alpha (smoothing level)", 0.01, 1.0, 0.5, 0.01)
        beta = st.slider("Beta (smoothing slope)", 0.01, 1.0, 0.5, 0.01)
        gamma = st.slider("Gamma (smoothing seasonal)", 0.01, 1.0, 0.5, 0.01)
        seasonal_periods = st.slider("Seasonal Periods", 7, 365, 12)
    
    if st.button("Apply Manual Parameter"):
        if modelchoice == "Exponentially Weighted Moving Average (EWMA)":
            if feature_for_choice == "Open":
                ewma_model = test_open.ewm(alpha=alpha, adjust=False).mean()
                forecast = [ewma_model.iloc[-1]] * len(test_open)
                actual = test_open
            elif feature_for_choice == "High":
                ewma_model = test_high.ewm(alpha=alpha, adjust=False).mean()
                forecast = [ewma_model.iloc[-1]] * len(test_high)
                actual = test_high
            elif feature_for_choice == "Close":
                ewma_model = test_close.ewm(alpha=alpha, adjust=False).mean()
                forecast = [ewma_model.iloc[-1]] * len(test_close)
                actual = test_close
            elif feature_for_choice == "Volume":
                ewma_model = test_volume.ewm(alpha=alpha, adjust=False).mean()
                forecast = [ewma_model.iloc[-1]] * len(test_volume)
                actual = test_volume
        else:
            if feature_for_choice == "Open":
                model_tes = ExponentialSmoothing(train_open, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
                forecast = model_tes.forecast(steps=len(test_open))
                actual = test_open
            elif feature_for_choice == "High":
                model_tes = ExponentialSmoothing(train_high, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
                forecast = model_tes.forecast(steps=len(test_high))
                actual = test_high
            elif feature_for_choice == "Close":
                model_tes = ExponentialSmoothing(train_close, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
                forecast = model_tes.forecast(steps=len(test_close))
                actual = test_close
            elif feature_for_choice == "Volume":
                model_tes = ExponentialSmoothing(train_volume, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
                forecast = model_tes.forecast(steps=len(test_volume))
                actual = test_volume
        
        # Function to apply conditional coloring on metrics
        def colorMetric(value, threshold, metricname):
            colorCondisition = 'green' if value <= threshold else 'red'
            return f"<span style='color:{colorCondisition}'> {metricname} : {value: .4f}</span>" if metricname == "MSE" else f"<span style='color:{colorCondisition}'>{metricname} : {value:.2f}</span>"
        
        # Metrics for evaluation
        mse, mape = metrik_calculating(actual, forecast)

        # Display evaluation metrics
        st.subheader("Metriks Evaluasi")
        mse_threshold = 500_000_000  # Batas bawah 500 juta
        mape_threshold = 600_000_000 
        
        st.markdown(f"Nilai MSE : {colorMetric(mse, mse_threshold, "")}", unsafe_allow_html=True)
        st.markdown(f"Nilai MAPE : {colorMetric(mape, mape_threshold, "")} %", unsafe_allow_html=True)
        
        # Visualization using Line chart
        st.subheader(f"Harga {feature_for_choice}")
        fig = go.Figure()
        if feature_for_choice == "Open":
            fig.add_trace(go.Scatter(x=open_price.index, y=open_price, mode='lines', name='Actual (Train)', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=test_open.index, y=test_open, mode='lines', name='Actual (Test)', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=test_open.index, y=forecast, mode='lines', name='Data Forecast', line=dict(color='red')))
        elif feature_for_choice == "High":
            fig.add_trace(go.Scatter(x=high_price.index, y=high_price, mode='lines', name='Actual (Train)', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=test_high.index, y=test_high, mode='lines', name='Actual (Test)', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=test_high.index, y=forecast, mode='lines', name='Data Forecast', line=dict(color='red')))
        elif feature_for_choice == "Close":
            fig.add_trace(go.Scatter(x=close_price.index, y=close_price, mode='lines', name='Actual (Train)', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=test_close.index, y=test_close, mode='lines', name='Actual (Test)', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=test_close.index, y=forecast, mode='lines', name='Data Forecast', line=dict(color='red')))
        elif feature_for_choice == "Volume":
            fig.add_trace(go.Scatter(x=volume_price.index, y=volume_price, mode='lines', name='Actual (Train)', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=test_volume.index, y=test_volume, mode='lines', name='Actual (Test)', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=test_volume.index, y=forecast, mode='lines', name='Data Forecast', line=dict(color='red')))
        
        st.plotly_chart(fig)
        st.write("Forecast Table:")

        # Menghitung akurasi dan loss
        accuracy = 100 - (np.abs((actual - forecast) / actual) * 100)
        loss = np.abs(actual - forecast)

        # Buat DataFrame dari nilai actual, forecast, accuracy, dan loss
        forecast_df = pd.DataFrame({
            "Actual Price": actual,
            "Forecast": forecast,
            "Accuracy (%)": accuracy,
            "Loss": loss
        }).set_index(actual.index)

        # Format the index to remove time component
        forecast_df.index = forecast_df.index.strftime('%Y-%m-%d')

        # Function to apply color formatting
        def color_accuracy(val):
            color = 'green' if val >= 95 else 'red'  # Set threshold for accuracy, e.g., 95% as stable
            return f'color: {color}'

        def color_loss(val):
            color = 'green' if val <= 100000 else 'red'  # Set threshold for loss, e.g., below 100,000 is stable
            return f'color: {color}'

        # Apply conditional formatting
        styled_forecast_df = forecast_df.style.applymap(color_accuracy, subset=['Accuracy (%)' ,'Loss'])

        # Display DataFrame in Streamlit with conditional formatting
        st.write(styled_forecast_df)


        
if __name__ == "__main__":
    main()
