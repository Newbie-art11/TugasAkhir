import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import pickle
import joblib
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# List 10 cryptocurrency 
cryptosList = ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", "USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]

def ewma_forecasting(train_data, test_data, span):
    # Train model with train data
    ewma_model = pd.Series(train_data.flatten()).ewm(span=span, adjust=False).mean()
    
    # Forecast for test data (using the last observed value in train as the forecast)
    ewma_forecast = [ewma_model.iloc[-1]] * len(test_data)
    
    return ewma_forecast

def tes_forecasting(train_data, test_data, seasonal_periods):
    tes_model = ExponentialSmoothing(train_data, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit()
    
    # Forecast the future values
    tes_forecast = tes_model.forecast(len(test_data))
    
    return tes_forecast
def main():
    st.title("Selamat Datang di Top 10 Cryptocurrency Prediksi Untuk Kedua Metode")
    st.write("Aplikasi ini memprediksi harga mata uang kripto menggunakan Exponentially Weighted Moving Average (EWMA) dan Triple Exponential Smoothing (TES). Model tersebut dilatih pada data historis untuk prediksi yang lebih cepat.")   

    # Input Sidebar
    st.sidebar.header("Data Unduhan")
    symbol_stock = st.sidebar.selectbox("Pilih Cryptocurrency:", cryptosList)
    start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("Tanggal Terakhir", pd.to_datetime("2024-07-31"))
    
    data = yf.download(symbol_stock, start=start_date, end=end_date)

    # Preprocess data (use Open prices for example)
    open_prices = data['Open'].values
    high_prices = data['High'].values
    close_prices = data['Close'].values
    volume_prices = data['Volume'].values
    
    train_size = int(len(open_prices) * 0.8)
    
    # Split the data into train and test sets
    train_open = open_prices[:train_size]
    test_open = open_prices[train_size:]
    
    train_high = high_prices[:train_size]
    test_high = high_prices[train_size:]
    
    train_close = close_prices[:train_size]
    test_close = close_prices[train_size:]
    
    train_volume = volume_prices[:train_size]
    test_volume = volume_prices[train_size:]

    # Perform EWMA and TES predictions
    span = 3  # Example span for EWMA
    seasonal_periods = 12  # Example seasonal period for TES
    
    # Forecasts for Open prices
    forecast_ewma_open = ewma_forecasting(train_open, test_open, span)
    forecast_tes_open = tes_forecasting(train_open, test_open, seasonal_periods)

    # Forecasts for High prices
    forecast_ewma_high = ewma_forecasting(train_high, test_high, span)
    forecast_tes_high = tes_forecasting(train_high, test_high, seasonal_periods)

    # Forecasts for Close prices
    forecast_ewma_close = ewma_forecasting(train_close, test_close, span)
    forecast_tes_close = tes_forecasting(train_close, test_close, seasonal_periods)

    # Forecasts for Volume prices
    forecast_ewma_volume = ewma_forecasting(train_volume, test_volume, span)
    forecast_tes_volume = tes_forecasting(train_volume, test_volume, seasonal_periods)

    # Calculate errors (MAPE and MSE)
    mse_ewma_open = mean_squared_error(test_open, forecast_ewma_open)
    mape_ewma_open = mean_absolute_percentage_error(test_open, forecast_ewma_open)
    mse_tes_open = mean_squared_error(test_open, forecast_tes_open)
    mape_tes_open = mean_absolute_percentage_error(test_open, forecast_tes_open)

    mse_ewma_high = mean_squared_error(test_high, forecast_ewma_high)
    mape_ewma_high = mean_absolute_percentage_error(test_high, forecast_ewma_high)
    mse_tes_high = mean_squared_error(test_high, forecast_tes_high)
    mape_tes_high = mean_absolute_percentage_error(test_high, forecast_tes_high)

    mse_ewma_close = mean_squared_error(test_close, forecast_ewma_close)
    mape_ewma_close = mean_absolute_percentage_error(test_close, forecast_ewma_close)
    mse_tes_close = mean_squared_error(test_close, forecast_tes_close)
    mape_tes_close = mean_absolute_percentage_error(test_close, forecast_tes_close)

    mse_ewma_volume = mean_squared_error(test_volume, forecast_ewma_volume)
    mape_ewma_volume = mean_absolute_percentage_error(test_volume, forecast_ewma_volume)
    mse_tes_volume = mean_squared_error(test_volume, forecast_tes_volume)
    mape_tes_volume = mean_absolute_percentage_error(test_volume, forecast_tes_volume)

    # Create tabs for each price type
    tab1, tab2, tab3, tab4 = st.tabs(["Open Price", "High Price", "Close Price", "Volume Price"])
    
    with tab1:
        st.header(f"Hasil Harga Open {symbol_stock} untuk EWMA dan TES")
        st.write(f"Exponential Weighted Moving Average - MSE: {mse_ewma_open}")
        st.write(f"Exponential Weighted Moving Average - MAPE: {mape_ewma_open}")
        st.write(f"Triple Exponential Smoothing - MSE: {mse_tes_open}")
        st.write(f"Triple Exponential Smoothing - MAPE: {mape_tes_open}")
        visualize_predictions(data, train_size, test_open, forecast_tes_open, forecast_ewma_open, 'Open')
        
        # Add table for actual data, forecasting, loss, and accuracy
        open_price_table = pd.DataFrame({
            'Date': data.index[train_size:],
            'Actual Data': test_open,
            'EWMA Forecast': forecast_ewma_open,
            'TES Forecast': forecast_tes_open,
            'EWMA Loss (MSE)': [mse_ewma_open] * len(test_open),
            'TES Loss (MSE)': [mse_tes_open] * len(test_open),
            'EWMA Accuracy (MAPE)': [mape_ewma_open] * len(test_open),
            'TES Accuracy (MAPE)': [mape_tes_open] * len(test_open)
        })
        st.dataframe(open_price_table)

    # Repeat for other tabs similarly...
    with tab2:
        st.header(f"Hasil Harga Open {symbol_stock} untuk EWMA dan TES")
        st.write(f"Exponential Weighted Moving Average - MSE: {mse_ewma_high}")
        st.write(f"Exponential Weighted Moving Average - MAPE: {mape_ewma_high}")
        st.write(f"Triple Exponential Smoothing - MSE: {mse_tes_high}")
        st.write(f"Triple Exponential Smoothing - MAPE: {mape_tes_high}")
        visualize_predictions(data, train_size, test_open, forecast_tes_high, forecast_ewma_high, 'Open')
        
        # Add table for actual data, forecasting, loss, and accuracy
        high_price_table = pd.DataFrame({
            'Date': data.index[train_size:],
            'Actual Data': test_high,
            'EWMA Forecast': forecast_ewma_high,
            'TES Forecast': forecast_tes_high,
            'EWMA Loss (MSE)': [mse_ewma_high] * len(test_high),
            'TES Loss (MSE)': [mse_tes_high] * len(test_high),
            'EWMA Accuracy (MAPE)': [mape_ewma_high] * len(test_high),
            'TES Accuracy (MAPE)': [mape_tes_high] * len(test_high)
        })
        st.dataframe(high_price_table)
    with tab3:
        st.header(f"Hasil Harga Close {symbol_stock} untuk EWMA dan TES")
        st.write(f"Exponential Weighted Moving Average - MSE: {mse_ewma_close}")
        st.write(f"Exponential Weighted Moving Average - MAPE: {mape_ewma_close}")
        st.write(f"Triple Exponential Smoothing - MSE: {mse_tes_close}")
        st.write(f"Triple Exponential Smoothing - MAPE: {mape_tes_close}")
        visualize_predictions(data, train_size, test_close, forecast_tes_close, forecast_ewma_close, 'Close')
        
        # Add table for actual data, forecasting, loss, and accuracy
        close_price_table = pd.DataFrame({
            'Date': data.index[train_size:],
            'Actual Data': test_close,
            'EWMA Forecast': forecast_ewma_close,
            'TES Forecast': forecast_tes_close,
            'EWMA Loss (MSE)': [mse_ewma_close] * len(test_close),
            'TES Loss (MSE)': [mse_tes_close] * len(test_close),
            'EWMA Accuracy (MAPE)': [mape_ewma_close] * len(test_close),
            'TES Accuracy (MAPE)': [mape_tes_close] * len(test_close)
        })
        st.dataframe(close_price_table)
    with tab4:
        st.header(f"Hasil Harga Volume {symbol_stock} untuk EWMA dan TES")
        st.write(f"Exponential Weighted Moving Average - MSE: {mse_ewma_volume}")
        st.write(f"Exponential Weighted Moving Average - MAPE: {mape_ewma_volume}")
        st.write(f"Triple Exponential Smoothing - MSE: {mse_tes_volume}")
        st.write(f"Triple Exponential Smoothing - MAPE: {mape_tes_volume}")
        visualize_predictions(data, train_size, test_volume, forecast_tes_volume, forecast_ewma_volume, 'Volume')
        
        # Add table for actual data, forecasting, loss, and accuracy
        volume_price_table = pd.DataFrame({
            'Date': data.index[train_size:],
            'Actual Data': test_volume,
            'EWMA Forecast': forecast_ewma_volume,
            'TES Forecast': forecast_tes_volume,
            'EWMA Loss (MSE)': [mse_ewma_volume] * len(test_volume),
            'TES Loss (MSE)': [mse_tes_volume] * len(test_volume),
            'EWMA Accuracy (MAPE)': [mape_ewma_volume] * len(test_volume),
            'TES Accuracy (MAPE)': [mape_tes_volume] * len(test_volume)
        })
        st.dataframe(volume_price_table)




def visualize_predictions(data, train_size, y_test, y_pred_tes, y_pred_ewma, price_type):
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
                             name="TES Predictions",
                             line=dict(color='green')))

    # Add EWMA predictions
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_pred_ewma,
                             mode='lines',
                             name="EWMA Predictions",
                             line=dict(color='red')))

    fig.update_layout(title=f"{price_type} Price Prediction for EWMA & TES",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template='plotly_dark')

    st.plotly_chart(fig)
if __name__ == "__main__":
    main()
