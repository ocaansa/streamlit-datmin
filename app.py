import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

st.set_page_config(page_title="Prediksi Kelayakan Pendidikan", layout="wide")
st.title("🏫 Prediksi Kelayakan Provinsi dalam Sektor Pendidikan")

st.markdown("""
Aplikasi ini menggunakan *Decision Tree Classifier* untuk menilai kelayakan provinsi dalam sektor pendidikan
berdasarkan indikator seperti jumlah guru S1, kondisi ruang kelas, dan jumlah siswa.
""")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/kelayakan-pendidikan-indonesia.csv") 
    df = df.drop(columns=['Unnamed: 14'], errors='ignore')
    df['Persen Guru S1'] = df['Kepala Sekolah dan Guru(≥ S1)'] / (
        df['Kepala Sekolah dan Guru(≥ S1)'] + df['Kepala Sekolah dan Guru(<S1)'])
    df['Total Kelas Rusak'] = df['Ruang kelas(rusak ringan)'] + df['Ruang kelas(rusak sedang)'] + df['Ruang kelas(rusak berat)']
    df['Kelayakan'] = np.where(
        (df['Persen Guru S1'] > 0.5) & (df['Ruang kelas(baik)'] > df['Total Kelas Rusak']),
        'Layak', 'Tidak Layak')
    return df

df = load_data()
st.subheader("🗂️ Data Pendidikan Provinsi")
st.dataframe(df[['Provinsi', 'Kepala Sekolah dan Guru(≥ S1)', 'Ruang kelas(baik)', 'Total Kelas Rusak', 'Kelayakan']], use_container_width=True)

# Encode label
le = LabelEncoder()
df['Target'] = le.fit_transform(df['Kelayakan'])

# Fitur dan Target
features = ['Sekolah', 'Siswa', 'Putus Sekolah', 'Kepala Sekolah dan Guru(≥ S1)', 'Tenaga Kependidikan(>SM)',
            'Ruang kelas(baik)', 'Total Kelas Rusak']
X = df[features]
y = df['Target']

# Split dan Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

st.subheader("📊 Evaluasi Model")
st.write(f"**Akurasi Model:** {acc:.2%}")
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

# Visualisasi
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(X_test['Ruang kelas(baik)'], X_test['Total Kelas Rusak'], c=y_pred, cmap='coolwarm')
ax.set_xlabel("Ruang Kelas Baik")
ax.set_ylabel("Total Kelas Rusak")
ax.set_title("Visualisasi Prediksi Kelayakan")
cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(le.classes_)))
cbar.ax.set_yticklabels(le.classes_)
st.pyplot(fig)

st.caption("Developed by Anisa Ulfadilah with ❤️ using Streamlit")
