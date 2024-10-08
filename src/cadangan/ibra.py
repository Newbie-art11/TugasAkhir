import streamlit as st
import numpy as np
from sklearn.cluster import AgglomerativeClustering as aa
import matplotlib.pyplot as plt

# Judul aplikasi
st.title("Agglomerative Hierarchical Clustering with 8 Clusters")

# Membuat data contoh secara acak (bisa juga diganti data lain)
np.random.seed(42)
data = np.random.rand(100, 2) * 10  # 100 titik acak dalam rentang 0-10

# Menampilkan data
st.write("Data sample (koordinat x, y):")
st.write(data)

# Membuat model clustering dengan 8 cluster
model = aa(n_clusters=2, affinity='euclidean', linkage='average')

# Fit model dan prediksi label cluster
labels = model.fit_predict(data)

# Menampilkan label cluster hasil prediksi
st.write("Cluster Labels for Each Data Point:")
st.write(labels)

# Membuat plot data dengan warna berbeda untuk setiap cluster
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='rainbow', s=50)
plt.title('Agglomerative Clustering with 8 Clusters')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

# Tampilkan plot ke Streamlit
st.pyplot(plt)
