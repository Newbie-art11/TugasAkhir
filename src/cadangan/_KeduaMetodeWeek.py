import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# List 10 cryptocurrency 
cryptosList = ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", "USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]

def ewma_forecasting(train_data, test_data, span):
    ewma_model = pd.Series(train_data.flatten()).ewm(span=span, adjust=False).mean()
    ewma_forecast = [ewma_model.iloc[-1]] * len(test_data)
    return ewma_forecast

def tes_forecasting(train_data, test_data, seasonal_periods):
    tes_model = ExponentialSmoothing(train_data, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit()
    tes_forecast = tes_model.forecast(len(test_data))
    return tes_forecast

def main():
    st.title("Cryptocurrency Forecasting per Minggu Menggunakan EWMA dan TES")
    
    # Input Sidebar
    st.sidebar.header("Data Unduhan")
    symbol_stock = st.sidebar.selectbox("Pilih Cryptocurrency:", cryptosList)
    start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("Tanggal Terakhir", pd.to_datetime("2024-07-31"))
    
    # Download data dari yfinance
    data = yf.download(symbol_stock, start=start_date, end=end_date)
    
    # Resampling data per minggu menggunakan standar deviasi untuk Open, High, Close, dan Volume
    weekly_data = data.resample('W').agg({
        'Open': 'std', 
        'High': 'std', 
        'Close': 'std', 
        'Volume': 'std'
    })

    # Normalisasi data menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(weekly_data), columns=weekly_data.columns, index=weekly_data.index)

    # Split train dan test set
    train_size = int(len(scaled_data) * 0.8)
    train_open = scaled_data['Open'][:train_size].values
    test_open = scaled_data['Open'][train_size:].values
    train_high = scaled_data['High'][:train_size].values
    test_high = scaled_data['High'][train_size:].values
    train_close = scaled_data['Close'][:train_size].values
    test_close = scaled_data['Close'][train_size:].values
    train_volume = scaled_data['Volume'][:train_size].values
    test_volume = scaled_data['Volume'][train_size:].values
    

    # EWMA dan TES untuk harga Open
    span = 3  # Span untuk EWMA
    seasonal_periods = 8  # Periode musiman untuk TES
    
    
    forecast_ewma_open = ewma_forecasting(train_open, test_open, span)
    forecast_tes_open = tes_forecasting(train_open, test_open, seasonal_periods)
    
    forecast_ewma_high = ewma_forecasting(train_high, test_high, span)
    forecast_tes_high = tes_forecasting(train_high, test_high, seasonal_periods)
    
    
    forecast_ewma_close = ewma_forecasting(train_close,test_close,span)
    forecast_tes_close = tes_forecasting(train_close,test_close,seasonal_periods)
    
    forecast_ewma_volume = ewma_forecasting(train_volume,test_volume,span)
    forecast_tes_volume = tes_forecasting(train_volume,test_volume,seasonal_periods)

    # Hitung MAPE dan MSE
    mse_ewma_open = mean_squared_error(test_open, forecast_ewma_open)
    mape_ewma_open = mean_absolute_percentage_error(test_open, forecast_ewma_open)
    mse_tes_open = mean_squared_error(test_open, forecast_tes_open)
    mape_tes_open = mean_absolute_percentage_error(test_open, forecast_tes_open)
    # Hitung MAPE dan MSE
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
        # Tampilkan hasil
        st.header(f"Hasil Prediksi Open Price {symbol_stock} Per Minggu untuk EWMA dan TES")
        st.write(f"Exponential Weighted Moving Average (EWMA) - MSE: {mse_ewma_open}")
        st.write(f"Exponential Weighted Moving Average (EWMA) - MAPE: {mape_ewma_open} %")
        st.write(f"Triple Exponential Smoothing (TES) - MSE: {mse_tes_open}")
        st.write(f"Triple Exponential Smoothing (TES) - MAPE: {mape_tes_open} %")
        
        # Visualisasi
        visualize_predictions(scaled_data, train_size, test_open, forecast_tes_open, forecast_ewma_open, 'Open')

        # Tampilkan data dalam tabel
        weekly_open_price_table = pd.DataFrame({
            'Date': scaled_data.index[train_size:],
            'Actual Data': test_open,
            'EWMA Forecast': forecast_ewma_open,
            'TES Forecast': forecast_tes_open
        })
        st.dataframe(weekly_open_price_table)
    with tab2:
        st.header(f"Hasil Prediksi High Price {symbol_stock} Per Minggu untuk EWMA dan TES")
        st.write(f"Exponential Weighted Moving Average (EWMA) - MSE: {mse_ewma_high} ")
        st.write(f"Exponential Weighted Moving Average (EWMA) - MAPE: {mape_ewma_high} %")
        st.write(f"Triple Exponential Smoothing (TES) - MSE: {mse_tes_high}")
        st.write(f"Triple Exponential Smoothing (TES) - MAPE: {mape_tes_high} %")
        # Visualisasi
        visualize_predictions(scaled_data, train_size, test_high, forecast_tes_high, forecast_ewma_high, 'High')

        # Tampilkan data dalam tabel
        weekly_high_price_table = pd.DataFrame({
            'Date': scaled_data.index[train_size:],
            'Actual Data': test_high,
            'EWMA Forecast': forecast_ewma_high,
            'TES Forecast': forecast_tes_high
        })
        st.dataframe(weekly_high_price_table)
    with tab3:
        st.header(f"Hasil Prediksi Close Price {symbol_stock} Per Minggu untuk EWMA dan TES")
        st.write(f"Exponential Weighted Moving Average (EWMA) - MSE: {mse_ewma_close} ")
        st.write(f"Exponential Weighted Moving Average (EWMA) - MAPE: {mape_ewma_close} %")
        st.write(f"Triple Exponential Smoothing (TES) - MSE: {mse_tes_close}")
        st.write(f"Triple Exponential Smoothing (TES) - MAPE: {mape_tes_close} %")
        # Visualisasi
        visualize_predictions(scaled_data, train_size, test_close, forecast_tes_close, forecast_ewma_close, 'Close')

        # Tampilkan data dalam tabel
        weekly_close_price_table = pd.DataFrame({
            'Date': scaled_data.index[train_size:],
            'Actual Data': test_close,
            'EWMA Forecast': forecast_ewma_close,
            'TES Forecast': forecast_tes_close
        })
        st.dataframe(weekly_close_price_table)
    with tab4:
        st.header(f"Hasil Prediksi Volume Price {symbol_stock} Per Minggu untuk EWMA dan TES")
        st.write(f"Exponential Weighted Moving Average (EWMA) - MSE: {mse_ewma_volume} ")
        st.write(f"Exponential Weighted Moving Average (EWMA) - MAPE:  {mape_ewma_volume} %")
        st.write(f"Triple Exponential Smoothing (TES) - MSE: {mse_tes_volume}")
        st.write(f"Triple Exponential Smoothing (TES) - MAPE: {mape_tes_volume} %")
        # Visualisasi
        visualize_predictions(scaled_data, train_size, test_volume, forecast_tes_volume, forecast_ewma_volume, 'Volume')

        # Tampilkan data dalam tabel
        weekly_volume_price_table = pd.DataFrame({
            'Date': scaled_data.index[train_size:],
            'Actual Data': test_volume,
            'EWMA Forecast': forecast_ewma_volume,
            'TES Forecast': forecast_tes_volume
        })
        st.dataframe(weekly_volume_price_table)

def visualize_predictions(data, train_size, y_test, y_pred_tes, y_pred_ewma, price_type):
    fig = go.Figure()

    # Tambahkan data training
    fig.add_trace(go.Scatter(x=data.index[:train_size],
                             y=data[price_type][:train_size],
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    # Tambahkan harga aktual
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_test,
                             mode='lines',
                             name="Actual Prices",
                             line=dict(color='blue')))

    # Tambahkan prediksi TES
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_pred_tes,
                             mode='lines',
                             name="TES Predictions",
                             line=dict(color='green')))

    # Tambahkan prediksi EWMA
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_pred_ewma,
                             mode='lines',
                             name="EWMA Predictions",
                             line=dict(color='red')))

    fig.update_layout(title=f"{price_type} Price Prediction for EWMA & TES",
                      xaxis_title="Date",
                      yaxis_title="Normalized Price",
                      template='plotly_dark')

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
