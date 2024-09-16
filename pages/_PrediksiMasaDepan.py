import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import pickle

cryptos = ["BTC-USD", "ETH-USD","USDT-USD","BNB-USD", "SOL-USD","USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]
with open('models/ewma_model_ETH-USD.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
def main():
    st.title("Selamat Datang di Top 10 Cryptocurrency Prediksi di Masa Depan")
    st.write("This app predicts cryptocurrency prices EWMA dan TES")
        # Memuat model EWMA dari file
    with open('models/ewma_model_ETH-USD.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    #Sidebar Input Data
    st.sidebar.header("Data Unduhan")
    stock_symbol = st.sidebar.selectbox("Pilih Cryptocurrenccy:", cryptos)
    
    
    #Download Stock price data 
    start = "2021-01-01"
    end = "2024-07-30"
    data = yf.download(stock_symbol,start=start,end=end)
    
    #Prosess Close Price 
    open_price = data['Open'].resample('W').std()
    high_price = data['High'].resample('W').std()
    close_price = data['Close'].resample('W').std()
    volume_price = data['Volume'].resample('W').std()
    
    # print(open_price)
    # print(high_price)
    # print(close_price)
    # print(volume_price)
    
    #Splitting data 
    train_size = int(len(close_price) * 0.8)
    train_close = close_price[:train_size]
    test_close = close_price[train_size:]
    train_open = open_price[:train_size]
    test_open = open_price[train_size:]
    train_high = high_price[:train_size]
    test_high = high_price[train_size:]
    train_volume = volume_price[:train_size]
    test_volume = volume_price[train_size]


    #Input peramalan 
    st.header("Parameter peramalan")
    start_date_forecast = st.date_input("Mulai Tanggal Peramalan", pd.to_datetime("2024-10-01"))
    end_date_forecast = st.date_input("Akhir Tanggal Peramalan", pd.to_datetime("2024-12-30"))
    priode_forcast = st.slider("Forcast priode (days)",1,365,30) 
    
    
    # Menghitung selisih waktu dalam minggu
    if (end_date_forecast - start_date_forecast).days < forecast_period * 7:
        forecast_period = (end_date_forecast - start_date_forecast).days // 7  # Konversi ke minggu

    forecast_steps = forecast_period

    #Forecasting
    
    
    #Tab fiture 
    tab1, tab2,tab3,tab4 = st.tabs(["Open Prices","High Prices","Close Prices", "volume prices"]) 
    with tab1:
        st.header(f"Hasil harga Open {stock_symbol} untuk EWMA dan TES")
        st.write("Exponentially Weighted Moving Average - MSE:")
        st.write("Exponentially Weighted Moving Average - MAPE:")
        st.write("Triple Exponential Smoothing - MSE:")
        st.write("Triple Exponential Smoothing - MAPE:")
        visualize_predictions()
        st.subheader("Prediksi Masa Depan Untuk Harga Open")
        
    with tab2:
        st.header(f"Hasil harga High {stock_symbol} untuk EWMA dan TES")
        st.write("Exponentially Weighted Moving Average - MSE:")
        st.write("Exponentially Weighted Moving Average - MAPE:")
        st.write("Triple Exponential Smoothing - MSE:")
        st.write("Triple Exponential Smoothing - MAPE:")
        
        st.subheader("Prediksi Masa Depan Untuk Harga High")
        
    with tab3:
        st.header(f"Hasil harga Close {stock_symbol} untuk EWMA dan TES")
        st.write("Exponentially Weighted Moving Average - MSE:")
        st.write("Exponentially Weighted Moving Average - MAPE:")
        st.write("Triple Exponential Smoothing - MSE:")
        st.write("Triple Exponential Smoothing - MAPE:")
        
        st.subheader("Prediksi Masa Depan Harga Close")
        
    with tab4:
        st.header(f"Hasil harga Volume {stock_symbol} untuk EWMA dan TES")  
        st.write("Exponentially Weighted Moving Average - MSE:")
        st.write("Exponentially Weighted Moving Average - MAPE:")
        st.write("Triple Exponential Smoothing - MSE:")
        st.write("Triple Exponential Smoothing - MAPE:")
        
        st.subheader("Prediksi Masa Depan Untuk Volume")



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

    # Add ARIMA predictions
    fig.add_trace(go.Scatter(x=data.index[train_size:],
                             y=y_pred_arima,
                             mode='lines',
                             name="TES Predictions",
                             line=dict(color='green')))

    fig.update_layout(title=f"{price_type} Harga Prediksi Untuk EWMA & TES",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template='plotly_dark')

    st.plotly_chart(fig)

def visualize_future_predictions(dates, y_pred_tes, y_pred_arima, price_type):
    fig = go.Figure()

    # Add future dates
    fig.add_trace(go.Scatter(x=dates,
                             y=y_pred_tes,
                             mode='lines',
                             name="EWMA Future Predictions",
                             line=dict(color='red')))

    # Add ARIMA future predictions
    fig.add_trace(go.Scatter(x=dates,
                             y=y_pred_arima,
                             mode='lines',
                             name="TES Future Predictions",
                             line=dict(color='green')))

    fig.update_layout(title=f"Future {price_type} Price Prediction for EWMA & TES",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template='plotly_dark')

    st.plotly_chart(fig)

def display_future_table(dates, y_pred_tes, y_pred_arima, price_type, x_accuracy,y_loss):
    # Display table
    st.write(f"Table Future Predictions for {price_type} Prices")
    # Create DataFrame for future predictions
    df_future = pd.DataFrame({
        'Date': dates.date,
        'EWMA Prediction': y_pred_tes,
        'TES Prediction': y_pred_arima,
        'Accuracy': x_accuracy,
        'Loss': y_loss
    })
        # Display the combined data table
    st.table(df_future.reset_index(drop=True))
            
if __name__ == "__main__":
    main()