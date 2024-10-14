import streamlit as st
import pandas as pd 
from io import BytesIO


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data
def main():
    st.title("Input Data Manual dan unduh sebagai excel dan CSV")
    date_input= st.date_input("Pilih tanggal")
    st.subheader("Input Manual")
    open_value = st.number_input("Masukkan nilai Open", format="%.8f")
    high_value = st.number_input("Masukkan nilai High", format="%.8f")
    close_value = st.number_input("Masukkan nilai Close", format="%.8f")
    volume_value = st.number_input("Masukkan nilai Volume", format="%.8f")
    data = {
        "Tanggal": [],
        "Open": [],
        "High": [],
        "Close": [],
        "Volume": []
    }

    if st.button("Tambahkan Data"):
        data["Tanggal"].append(date_input)
        data["Open"].append(open_value)
        data["High"].append(high_value)
        data["Close"].append(close_value)
        data["Volume"].append(volume_value)
        st.success("Data berhasil ditambahkan!")
        df = pd.DataFrame(data)
        st.subheader("Data yang telah diinput:")
        st.write(df)
        st.subheader("Unduh Data")
        if not df.empty:
            csv = convert_df_to_csv(df)
            st.download_button(label="Unduh sebagai CSV", data=csv, file_name="data.csv", mime="text/csv")
            excel = convert_df_to_excel(df)
            st.download_button(label="Unduh sebagai Excel", data=excel, file_name="data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
if __name__ == "__main__":
    main()
