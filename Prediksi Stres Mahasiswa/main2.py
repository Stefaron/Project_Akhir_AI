import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Baca data dari file CSV
data = pd.read_csv("data_stres.csv")

# Pisahkan fitur dan label
fitur = data.drop("tingkat_stres", axis=1)
label = data["tingkat_stres"]

# Inisialisasi LabelEncoder
label_encoder = LabelEncoder()
label = label_encoder.fit_transform(label)

# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(fitur, label, test_size=0.2, random_state=42)

# Skala fitur menggunakan StandardScaler dengan menyertakan nama fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Berikan nama fitur kembali setelah transformasi
X_train = pd.DataFrame(X_train, columns=fitur.columns)
X_test = pd.DataFrame(X_test, columns=fitur.columns)

# Latih model RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Lakukan prediksi pada data uji
prediksi = model.predict(X_test)

# Hitung akurasi model
akurasi = accuracy_score(y_test, prediksi)
st.write("Akurasi model: {:.2f}%".format(akurasi * 100))

# User input data baru
st.subheader("Masukkan Data Baru")
lama_belajar = st.number_input("Lama Belajar:")
intensitas_tugas = st.number_input("Intensitas Tugas:")
jam_kuliah = st.number_input("Jam Kuliah:")
jam_tidur = st.number_input("Jam Tidur:")
jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
perantau = st.selectbox("Perantau", ["Bukan Perantau", "Perantau"])
jml_sks = st.number_input("Jumlah SKS:")
uang_bulanan = st.number_input("Uang Bulanan:")

# Tampilkan tombol Predict
predict_button = st.button("Predict")

# Saat tombol Predict ditekan
if predict_button:
    # Lakukan prediksi pada data baru
    data_baru = [[lama_belajar, intensitas_tugas, jam_kuliah, jam_tidur, jenis_kelamin, perantau, jml_sks, uang_bulanan]]
    data_baru = pd.DataFrame(data_baru, columns=fitur.columns)

    # One-hot encoding untuk jenis kelamin
    data_baru["jenis_kelamin"] = 0
    if jenis_kelamin == "Perempuan":
        data_baru["jenis_kelamin"] = 1

    # One-hot encoding untuk perantau
    data_baru["perantau"] = 0
    if perantau == "Perantau":
        data_baru["perantau"] = 1

    # Skala fitur data baru menggunakan StandardScaler
    data_baru = scaler.transform(data_baru)

    # Lakukan prediksi pada data baru yang telah di-transform
    hasil_prediksi = model.predict(data_baru)

    # Konversi hasil prediksi kembali ke label kelas
    hasil_prediksi_label = label_encoder.inverse_transform(hasil_prediksi)[0]
    st.write("Prediksi tingkat stres mahasiswa: " + hasil_prediksi_label)
