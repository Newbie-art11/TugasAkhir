import streamlit as st

#Line 
def line(n=1):
    for i in range(n):
        st.write("\n")

def main():
    # Path relatif ke file logo
    image_url = "src/images/logo.png"
    # Display the image with custom alignment
    alignment = "center"  # Change to "left", "center", or "right"
    width = 100  
    # Set the desired height
    height = 200  # Adjust the height of the image (in pixels)

    # Apply the CSS for alignment
    st.markdown(
        f"""
        <style>
        .stImage > img {{
            display: flex;
            margin-left: auto;
            margin-right: auto;
            text-align: {alignment};
            width: {width}px;
            height: {height}px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the image
    st.image(image_url, caption='Unimal Hebat', use_column_width=False)

    # Dataframe selection
    st.markdown("<h2 align='center'> <b>PERBANDINGAN METODE <i>EXPONENTIALLY WEIGHTED MOVING AVERAGE </i>(EWMA) DAN METODE <i>TRIPLE EXPONENTIAL SMOOTHING </i>(TES) UNTUK PERAMALAN CRYPTOCURRENCY</b></h2>", unsafe_allow_html=True)
    line(1)
    st.markdown("Selamat datang! Aplikasi ini dirancang untuk membandingkan metode <i>Exponentially Weighted Moving Average</i> (EWMA) dan metode <i>Triple Exponential Smoothing</i> (TES) dalam memprediksi harga <i>cryptocurrency</i>. Platform ini diharapkan menjadi fondasi bagi strategi perdagangan yang lebih cerdas dan keputusan investasi yang lebih tajam dalam dunia cryptocurrency, membuka peluang untuk keuntungan yang lebih optimal.", unsafe_allow_html=True)
    line()
    
    #EWMA EXPLAINED
    
    st.header("Exponentially Weighted Moving Average (EWMA)")
    st.subheader("Keterangan")
    line(1)
    st.markdown("Exponentially Weighted Moving Average (EWMA) adalah metode smoothing yang digunakan untuk menghaluskan data time series, di mana data terbaru diberikan bobot yang lebih besar. Berbeda dengan simple moving average yang memberikan bobot yang sama pada setiap data, EWMA menggunakan bobot eksponensial yang menurun seiring waktu.")
    #TES Explained
    st.header("Triple Exponential Smoothing (TES)")
    st.subheader("Keterangan")
    line(1)
    st.markdown("Triple Exponential Smoothing (TES), juga dikenal sebagai metode Holt-Winters, adalah metode yang lebih kompleks yang digunakan untuk memodelkan data time series yang memiliki tren dan musiman (seasonality). TES adalah pengembangan dari Single Exponential Smoothing (SES) dan Double Exponential Smoothing (DES) yang menambahkan komponen musiman ke dalam model.")
    st.divider()
if __name__ == "__main__":
        main()
