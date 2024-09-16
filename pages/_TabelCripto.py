import streamlit as st
import pandas as pd
import yfinance as yf


def line(n=1):
    for i in range(n):
        st.write("\n")

def getCrptodata(coins):
    #menyiapkan data ditampilkan ditable
    list_crypto = []
    for coin in coins:
        try:
            crypto_info = yf.Ticker(coin)
            name_crypto = crypto_info.info.get('longName', 'N/A')
            symbol_crypto = coin
            price_crypto = get_crypto_price(crypto_info)
            # Tambahkan informasi ke dalam list
            list_crypto.append({
                'Nama': name_crypto,
                'Kode': symbol_crypto,
                'Harga': price_crypto,
            })
        except Exception as e:
            print(f"Error processing {coin}: {e}")
            list_crypto.append({
                'Nama': 'Error',
                'Kode': coin,
                'Harga': None,
            })
    return pd.DataFrame(list_crypto)       

def get_crypto_price(crypto_info):
    try:
        # Mendapatkan harga koin kripto dari yfinance
        crypto_data = crypto_info.history(period="1d")
        if not crypto_data.empty:
            return crypto_data['Close'][0]
    except Exception as e:
        print(f"Error getting price for {crypto_info}: {e}")
    return None 

def get_crypto_prices_weekly(crypto_info):
    try:
        # Mendapatkan data harga koin kripto per minggu dari yfinance
        crypto_data = crypto_info.history(period="1mo", interval="1wk")
        if not crypto_data.empty:
            return crypto_data['Close']
    except Exception as e:
        print(f"Error getting price for {crypto_info}: {e}")
    return None

def normalize_prices(prices):
    if prices is not None and not prices.empty:
        mean_price = prices.mean()
        return prices / mean_price  # Normalisasi dengan membagi setiap harga dengan mean
    return None


def crypto_data_week(coins):
    # Menyiapkan data untuk ditampilkan dalam tabel mingguan
    list_crypto = []
    for coin in coins:
        try:
            crypto_info = yf.Ticker(coin)
            name_crypto = crypto_info.info.get('longName', 'N/A')
            symbol_crypto = coin
            weekly_prices = get_crypto_prices_weekly(crypto_info)
            normalized_prices = normalize_prices(weekly_prices)
            total_price = weekly_prices.sum() if weekly_prices is not None else None
            
            if normalized_prices is not None:
                # Mengubah data harga menjadi string yang mudah dibaca
                normalized_prices_str = "; ".join([f"{date.strftime('%Y-%m-%d')}: {price:.4f}" for date, price in normalized_prices.items()])
                
                # Tambahkan informasi ke dalam list
                list_crypto.append({
                    'Nama': name_crypto,
                    'Kode': symbol_crypto,
                    'Harga Ternormalisasi': normalized_prices_str,
                    'Total Harga Asli': total_price,
                })
        except Exception as e:
            print(f"Error processing {coin}: {e}")
            list_crypto.append({
                'Nama': 'Error',
                'Kode': coin,
                'Harga Ternormalisasi': None,
                'Total Harga Asli': None,
            })
    return pd.DataFrame(list_crypto)
def main():
    st.title('Table ini Mereseprentasikan 10 Top Cryptocurrency sebagai Objek Penelitian')
    pathImage = "src/images/buble.jpg"
    st.image(pathImage,width=600)
    st.write("Di era digital yang semakin maju ini, cryptocurrency telah menjelma menjadi salah satu aset investasi yang paling diminati oleh berbagai kalangan. Tingginya volatilitas harga membuat prediksi harga cryptocurrency menjadi sebuah tantangan yang sangat menarik untuk dieksplorasi lebih lanjut.")
    st.divider()
    crypto_symbols = ["BTC-USD", "ETH-USD","USDT-USD","BNB-USD", "SOL-USD",
                  "USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]
    df_data_Day = getCrptodata(crypto_symbols)
    st.title("Tabel Crypto")
    st.table(df_data_Day)
    st.divider()
    # Judul aplikasi
    st.title("Profile TOP 10 Data Cryptocurrency")
    cryptos = ["BTC-USD", "ETH-USD","USDT-USD","BNB-USD", "SOL-USD","USDC-USD", "XRP-USD", "STETH-USD", "TON11419-USD", "DOGE-USD"]
    selected_crypto = st.selectbox("Pilih Cryptocurrency", cryptos)
    start_date = st.date_input("Tanggal Mulai", pd.to_datetime("2021-01-01"))
    end_date = st.date_input("Tanggal Akhir", pd.to_datetime("2024-06-30"))
    if st.button("Download Data"):
        with st.spinner("Mengunduh data..."):
            df = yf.download(selected_crypto, start=start_date, end=end_date)
            st.write(f"Data untuk {selected_crypto} dari {start_date} hingga {end_date}")
            st.write(df)
if __name__ == "__main__":
    main()
#Line 
