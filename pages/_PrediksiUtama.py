import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

cryptos = ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", "USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]
model_folder_tes = 'models/new_model_tes/'
model_folder_ewma = 'models/new_model_ewma/'
def main():
    st.title("Prediksi Harga Top 10 Cryptocurrency")
    
    # Sidebar Input
    st.sidebar.header("Data Unduhan")
    symbol_stock = st.sidebar.selectbox("Pilih Crypto", cryptos)
    model_choice = st.sidebar.radio("Pilih Model:", ["Exponentially Weighted Moving Average (EWMA)", "Triple Exponential Smoothing (TES)"])
    start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("Tanggal Terakhir", pd.to_datetime("2024-07-30"))

    if end_date < start_date:
        st.sidebar.error("End date must be after start date")
        return

    # Mengunduh data menggunakan yfinance
    data = yf.download(symbol_stock, start=start_date, end=end_date)
    if data.empty:
        st.error("Data tidak ditemukan")
        return

    # Agregasi mingguan dengan standar deviasi
    open_price = data['Open'].resample('W').std()
    high_price = data['High'].resample('W').std()
    close_price = data['Close'].resample('W').std()
    volume_price = data['Volume'].resample('W').std()
        

    # Buat satu instance MinMaxScaler
    scaler = MinMaxScaler()

    # Gabungkan semua kolom harga menjadi satu array, lalu terapkan scaler sekali saja
    week_prices = pd.DataFrame({
        'Harga Buka': open_price.values,
        'Harga Tertinggi': high_price.values,
        'Harga Penutup': close_price.values,
        'Harga Volume': volume_price.values
    })

    # Terapkan MinMaxScaler ke seluruh dataframe sekaligus
    scaled_week_prices = scaler.fit_transform(week_prices)

    # Konversi kembali menjadi DataFrame
    combine_week = pd.DataFrame(scaled_week_prices, columns=['Harga Buka', 'Harga Tertinggi', 'Harga Penutup', 'Harga Volume'])

    # Tampilkan hasil normalisasi
    st.subheader("Normalization Data dengan rentang nilai 0 dan 1")
    st.dataframe(combine_week)
    # st.table(combine_week)
    # Memproses prediksi berdasarkan model pilihan
    if model_choice == "Exponentially Weighted Moving Average (EWMA)":
        model_filename_ewma = os.path.join(model_folder_ewma, f'ewma_model_{symbol_stock}.pkl')
        with open(model_filename_ewma, 'rb') as f:
            loadmodel = pickle.load(f)        
        st.write(f'Model untuk {symbol_stock} berhasil dimuat dari {model_filename_ewma}.')
            
         # Melakukan prediksi dengan model TES
        steps = len(open_price)
        tes_open = loadmodel
        tes_high = loadmodel
        tes_close = loadmodel
        tes_volume = loadmodel
        # span = 30
        # ewma_open = open_price.ewm(span=span, adjust=False).mean()
        # ewma_high = high_price.ewm(span=span, adjust=False).mean()
        # ewma_close = close_price.ewm(span=span, adjust=False).mean()
        # ewma_volume = volume_price.ewm(span=span, adjust=False).mean()

        # predictions = {
        #     'Open': ewma_open,
        #     'High': ewma_high,
        #     'Close': ewma_close,
        #     'Volume': ewma_volume
        # }
    
    elif model_choice == "Triple Exponential Smoothing (TES)":
        model_filename = os.path.join(model_folder_tes, f'tes_model_{symbol_stock}.pkl')
        try:
            # Memuat model TES dari file Pickle
            with open(model_filename, 'rb') as file:
                loaded_model = pickle.load(file)
            st.write(f'Model untuk {symbol_stock} berhasil dimuat dari {model_filename}.')
            
            # Melakukan prediksi dengan model TES
            steps = len(open_price)
            tes_open = loaded_model.forecast(steps)
            tes_high = loaded_model.forecast(steps)
            tes_close = loaded_model.forecast(steps)
            tes_volume = loaded_model.forecast(steps)

            predictions = {
                'Open': tes_open,
                'High': tes_high,
                'Close': tes_close,
                'Volume': tes_volume
            }

        except FileNotFoundError:
            st.error(f"Model untuk {symbol_stock} tidak ditemukan di {model_filename}.")
            return
    
    # Membuat tab untuk setiap variabel harga
    tab1, tab2, tab3, tab4 = st.tabs(["Open Price", "High Price", "Close Price", "Volume Price"])

    with tab1:
        st.header(f"Hasil Harga Open {symbol_stock} untuk model {model_choice}")
        
    with tab2:
        st.header(f"Hasil Harga High {symbol_stock} untuk model {model_choice}")
        # visualize_predictions(high_price, predictions['High'], 'High')
        # display_metrics(high_price, predictions['High'], 'High')

    with tab3:
        st.header(f"Hasil Harga Close {symbol_stock} untuk model {model_choice}")
        # visualize_predictions(close_price, predictions['Close'], 'Close')
        # display_metrics(close_price, predictions['Close'], 'Close')

    with tab4:
        st.header(f"Hasil Harga Volume {symbol_stock} untuk model {model_choice}")
        # visualize_predictions(volume_price, predictions['Volume'], 'Volume')
        # display_metrics(volume_price, predictions['Volume'], 'Volume')

def forecast_ewma(ewma_model,n_forecast):
    last_value = ewma_model.iloc[-1]
    forecast = [last_value] * n_forecast
    future_dates = pd.date_range(start=ewma_model.index[-1] + pd.Timedelta(days=1), periods=n_forecast, freq='D')
    forecast_df = pd.DataFrame({'Forecast': forecast}, index=future_dates)
    return forecast_df

# Fungsi untuk menampilkan grafik prediksi
def visualize_predictions(actual_data, predicted_data, price_type):
    fig = go.Figure()

    # Menampilkan data aktual
    fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data, mode='lines', name="Actual Data", line=dict(color='blue')))

    # Menampilkan prediksi
    fig.add_trace(go.Scatter(x=actual_data.index, y=predicted_data, mode='lines', name="Predicted Data", line=dict(color='red')))

    fig.update_layout(title=f"{price_type} Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Price (USD)",
                      template='plotly_dark')

    st.plotly_chart(fig)


# Fungsi untuk menampilkan tabel dan menghitung metrik
def display_metrics(actual_data, predicted_data, price_type):
    # Calculate metrics
    mse = mean_squared_error(actual_data, predicted_data)
    mape = mean_absolute_percentage_error(actual_data, predicted_data)

    # Hitung selisih prediksi
    difference = predicted_data - actual_data

    # Buat DataFrame untuk tabel
    metrics_table = pd.DataFrame({
        'Date': actual_data.index,
        'Actual Data': actual_data,
        'Predicted Data': predicted_data,
        'Difference': difference
    })

    # Tampilkan tabel dengan metrik
    st.write(f"Tabel {price_type} Data (Actual vs Prediksi)")
    st.dataframe(metrics_table)

    # Tampilkan MSE dan MAPE
    st.write(f"**Mean Squared Error (MSE)** untuk {price_type}: {mse}")
    st.write(f"**Mean Absolute Percentage Error (MAPE)** untuk {price_type}: {mape * 100:.2f}%")

if __name__ == "__main__":
    main()


























































# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import plotly.graph_objects as go
# import joblib
# import pickle
# from sklearn.metrics import mean_absolute_percentage_error
# from sklearn.metrics import mean_squared_error
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from sklearn.preprocessing import MinMaxScaler
# import os

# cryptos = ["BTC-USD", "ETH-USD","USDT-USD","BNB-USD", "SOL-USD","USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]
# model_folder_tes = 'models/new_model_tes/'
# model_folder_ewma = 'models/new_model_ewma/'
# def main():
#     st.title("Prediksi Harga Top 10 Cryptocurrency")
#     st.write("")
#     # Sidebar Input
#     st.sidebar.header("Data Unduhan")
#     symbol_stock = st.sidebar.selectbox("Pilih Crypto", cryptos)
#     model_choice = st.sidebar.radio("Pilih Model:", ["Exponentially Weighted Moving Average (EWMA)", "Triple Exponential Smoothing (TES)"])
#     start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime("2021-01-01"))
#     end_date = st.sidebar.date_input("Tanggal Terakhir", pd.to_datetime("2024-07-30"))

#     if end_date < start_date:
#         st.sidebar.error("End date must be after start date")
#         return

#     # Mengunduh data menggunakan yfinance
#     data = yf.download(symbol_stock, start=start_date, end=end_date)
#     if data.empty:
#         st.error("Data tidak ditemukan")
#         return

#     # Memproses harga penutupan (Close Price)
#     open_price = data['Open'].resample('W').std()
#     high_price = data['High'].resample('W').std()
#     close_price = data['Close'].resample('W').std()
#     volume_price = data['Volume'].resample('W').std()

 
#     if model_choice == "Exponentially Weighted Moving Average (EWMA)":
#         model_filename = os.path.join(model_folder_ewma, f'ewma_model_{symbol_stock}.pkl')
#         try:
#             with open(model_filename, 'rb') as file:
#                 loaded_model = pickle.load(file)    
#             print(f'Model untuk {symbol_stock} berhasil dimuat dari {model_filename}.')
#         except FileNotFoundError:
#             st.error(f"Model untuk {symbol_stock} tidak ditemukan di {model_filename}.")
#             return
        
        
        
        
#     elif model_choice == "Triple Exponential Smoothing (TES)":
#         model_filename = os.path.join(model_folder_tes, f'tes_model_{symbol_stock}.pkl')
        
#         try:
#             # Memuat model TES dari file Pickle
#             with open(model_filename, 'rb') as file:
#                 loaded_model = pickle.load(file)
#             print(f'Model untuk {symbol_stock} berhasil dimuat dari {model_filename}.')
#             # Menggunakan model untuk prediksi
#             tes_open = loaded_model.predict(start=len(open_price), end=len(open_price) + len(open_price) - 1)
#             tes_high = loaded_model.predict(start=len(high_price), end=len(high_price) + len(high_price) - 1)
#             tes_close = loaded_model.predict(start=len(close_price), end=len(close_price) + len(close_price) - 1)
#             tes_volume = loaded_model.predict(start=len(volume_price), end=len(volume_price) + len(volume_price) - 1)
          
#         except FileNotFoundError:
#             st.error(f"Model untuk {symbol_stock} tidak ditemukan di {model_filename}.")
#             return
        
        
        
        
#     tab1, tab2, tab3, tab4 = st.tabs(["Open Price", "High Price", "Close Price", "Volume Price"])

#     with tab1:
#         st.header(f"Hasil Harga Open {symbol_stock} untuk model {model_choice}")
#     with tab2:
#         st.header(f"Hasil Harga High {symbol_stock} untuk model {model_choice}")
#     with tab3:
#         st.header(f"Hasil Harga Close {symbol_stock} untuk model {model_choice}")
#     with tab4:
#         st.header(f"Hasil Harga Volume {symbol_stock} untuk model {model_choice}")
#     # Tab lainnya dapat ditambahkan seperti tab1 untuk high, close, dan volume


# # Fungsi untuk menampilkan grafik prediksi
# def visualize_predictions(actual_data, predicted_data, price_type):
#     fig = go.Figure()

#     # Menampilkan data aktual
#     fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data, mode='lines', name="Actual Data", line=dict(color='blue')))

#     # Menampilkan prediksi
#     fig.add_trace(go.Scatter(x=predicted_data.index, y=predicted_data, mode='lines', name="Predicted Data", line=dict(color='red')))

#     fig.update_layout(title=f"{price_type} Price Prediction",
#                       xaxis_title="Date",
#                       yaxis_title="Price (USD)",
#                       template='plotly_dark')

#     st.plotly_chart(fig)
# def visualize_predictions(data, train_size, y_test, y_pred, price_type):
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(x=data.index[:train_size],
#                              y=data[price_type][:train_size],
#                              mode='lines',
#                              name="Training Data",
#                              line=dict(color='gray')))

#     fig.add_trace(go.Scatter(x=data.index[train_size:],
#                              y=y_test,
#                              mode='lines',
#                              name="Actual Prices",
#                              line=dict(color='blue')))

#     fig.add_trace(go.Scatter(x=data.index[train_size:],
#                              y=y_pred,
#                              mode='lines',
#                              name="Predicted Prices",
#                              line=dict(color='red')))

#     fig.update_layout(title=f"{price_type} Price Prediction",
#                       xaxis_title="Date",
#                       yaxis_title="Price (USD)",
#                       template='plotly_dark')

#     st.plotly_chart(fig)


# if __name__ == "__main__":
#     main()






































# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# import plotly.graph_objects as go
# import joblib
# from sklearn.metrics import mean_absolute_percentage_error
# from sklearn.metrics import mean_squared_error
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from sklearn.preprocessing import MinMaxScaler
# import os
# import pickle

# cryptos = ["BTC-USD", "ETH-USD","USDT-USD","BNB-USD", "SOL-USD","USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]

# models_ewma = {}
# models_tes = {}

# for crypto in cryptos:
#     models_ewma[crypto] = joblib.load(f"ewma_model_{crypto}.pkl")
#     models_tes[crypto] = joblib.load(f"tes_model_{crypto}.pkl")

# def main():
#         title = "Selamat Datang di Prediksi Utama pada penelitian Ini Top 10 Cryptocurrency Price Prediction"
#         st.title(title)
#         st.write("")
#         crypto_symbols = ["BTC-USD", "ETH-USD","USDT-USD","BNB-USD", "SOL-USD",
#                     "USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]
    
#         #Sidebar Input
#         st.sidebar.header("Data Unduhan")
#         symbol_stock = st.sidebar.selectbox("Pilih Crypto",crypto_symbols)
#         modelchoise = st.sidebar.radio("Pilih Model:",["Exponentially Weighted Moving Average (EWMA)","Triple Exponential Smoothing (TES)"])
#         start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime("2021-01-01"))
#         end_date = st.sidebar.date_input("Tanggal Terakhir",pd.to_datetime("2024-07-30"))

#         if end_date < start_date:
#             st.sidebar.error("End date must be after start date")
#             return
        
#         #Dowloading Data using yahoofinance
#         data = yf.download(symbol_stock,start=start_date,end=end_date)
#         if data.empty:
#             st.error("Data tidak ditemukan")
#             return
#         #Prosessing close price 
#         open_price = data['Open'].resample('W').std()
#         high_price = data['High'].resample('W').std()
#         close_price = data['Close'].resample('W').std()
#         volume_price = data['Volume'].resample('W').std()
        
#         #Agregasi data 
#         combinedAgregasi = pd.DataFrame({
#             'Harga_Open': open_price,
#             'Harga_High': high_price,
#             'Harga_Penutupan': close_price,
#             'Harga_Volume': volume_price
#         })
#         st.subheader("Data Diagregasi Ke menggunakan Standar Deiviasi")
#         st.table(combinedAgregasi.head(10))
#         # MinMax Scaling untuk setiap kolom yang dihasilkan dari agregasi
#         scaler = MinMaxScaler()
#         scaled_data = scaler.fit_transform(combinedAgregasi)
#         # Menyimpan hasil scaling dalam DataFrame baru dengan kolom yang sama
#         combinedAgregasi_scaled = pd.DataFrame(scaled_data, columns=combinedAgregasi.columns, index=combinedAgregasi.index)
#         # Menampilkan hasil scaling
#         st.subheader("Data dilakukan Minmax scaller jika dilakukan standar deviasi nilai eksrim dan terlalu besar")
#         st.table(combinedAgregasi_scaled.head(10))
#         st.divider()
        
#         # Split data
#         train_size = int(len(close_price) * 0.8)
#         train_close = close_price[:train_size]
#         test_close = close_price[train_size:]
#         train_open = open_price[:train_size]
#         test_open = open_price[train_size:]
#         train_high = high_price[:train_size]
#         test_high = high_price[train_size:]
#         train_low = volume_price[:train_size]
#         test_low = volume_price[train_size:]
        
#         #Forecasting data 
#         if modelchoise == "Exponentially Weighted Moving Average (EWMA)":
#             # Tentukan smoothing level alpha
#             alpha = 0.2
#             model_ewma = ExponentialSmoothing(train_open, trend=None, seasonal=None).fit(smoothing_level=alpha)
#             forecast_open = model_ewma.forecast(steps=len(test_open))
#         else:
#             forecast_open = ExponentialSmoothing(train_open, trend='add', seasonal='add', seasonal_periods=12).fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).forecast(steps=len(test_open))
      

#         tab1, tab2, tab3, tab4 = st.tabs(["Open Price","High Price","Close Price","Volume Price"])
        
#         with tab1:
#             st.header(f"Hasil Harga Open {symbol_stock} untuk model {modelchoise}")
#             st.write(f"{modelchoise} - MSE :")
#             st.write(f"{modelchoise} - MAPE :", "%")
#             visualize_predictions(data, train_size,test_open,forecast_open, 'Open')
#             display_forecast_table(f"Table {modelchoise} Model - Close Predicted Prices ", data.index[train_size:], test_close, forecast_close, key='close')

#         with tab2:
#             st.header(f"Hasil Harga High {symbol_stock} untuk model {modelchoise}")
#             st.write(f"{modelchoise} - MSE :")
#             st.write(f"{modelchoise} - MAPE :")
#         with tab3:
#             st.header(f"Hasil Harga Close {symbol_stock} untuk model {modelchoise}")
#             st.write(f"{modelchoise} - MSE :")
#             st.write(f"{modelchoise} - MAPE :")
#         with tab4:
#             st.header(f"Hasil Harga Volume {symbol_stock} untuk model {modelchoise}")
#             st.write(f"{modelchoise} - MSE :")
#             st.write(f"{modelchoise} - MAPE :")

# def visualize_predictions(data, train_size, y_test, y_pred, price_type):
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(x=data.index[:train_size],
#                              y=data[price_type][:train_size],
#                              mode='lines',
#                              name="Training Data",
#                              line=dict(color='gray')))

#     fig.add_trace(go.Scatter(x=data.index[train_size:],
#                              y=y_test,
#                              mode='lines',
#                              name="Actual Prices",
#                              line=dict(color='blue')))

#     fig.add_trace(go.Scatter(x=data.index[train_size:],
#                              y=y_pred,
#                              mode='lines',
#                              name="Predicted Prices",
#                              line=dict(color='red')))

#     fig.update_layout(title=f"{price_type} Price Prediction",
#                       xaxis_title="Date",
#                       yaxis_title="Price (USD)",
#                       template='plotly_dark')

#     st.plotly_chart(fig)
        

# def display_forecast_table(title, dates, actual, predicted, key):
#     st.write(f"### {title}")

#     min_len = min(len(dates), len(actual), len(predicted))
#     dates = dates[:min_len]
#     actual = actual[:min_len]
#     predicted = predicted[:min_len]

#     price_difference = actual - predicted
#     percentage_difference = (price_difference / actual) * 100

#     combined_data = pd.DataFrame({
#         'Tanggal': dates.date,
#         'Actual_Prices': actual,
#         'Predicted_Prices': predicted,
#         'Price_Diference': price_difference.abs(),
#         'Precentage_diferent': percentage_difference.abs().map("{:.2f}%".format),
#         'Accuracy': actual,
#         'Loss':predicted
#     })

#     st.write("Data range:", dates.min(), "to", dates.max())
#     st.table(combined_data.reset_index(drop=True))

#     average_actual_prices = combined_data['Actual_Prices'].mean()
#     average_price_difference = combined_data['Price_Difference'].mean()
#     average_percentage_difference = combined_data['Percentage_Difference'].str.rstrip('%').astype('float').mean()

#     st.write("Average Actual Prices:", average_actual_prices)
#     st.write("Average Price Difference:", average_price_difference)
#     st.write("Accuracy Price: {:.2f}%".format(average_percentage_difference))
#     st.write("Loss Price: {:.2f}%".format(average_percentage_difference))

#     csv = combined_data.to_csv(index=False).encode('utf-8')
#     st.download_button(
#         label="Download data as CSV",
#         data=csv,
#         file_name='forecast_data.csv',
#         mime='text/csv',
#         key=key
#     )     
        


# if __name__ == "__main__":
#     main()