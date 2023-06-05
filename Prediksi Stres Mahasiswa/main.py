# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder

# # Baca data dari file CSV
# data = pd.read_csv("data_stres.csv")

# # Pisahkan fitur dan label
# fitur = data.drop("tingkat_stres", axis=1)
# label = data["tingkat_stres"]

# # Inisialisasi LabelEncoder
# label_encoder = LabelEncoder()
# label = label_encoder.fit_transform(label)

# # Bagi data menjadi data latih dan data uji
# X_train, X_test, y_train, y_test = train_test_split(fitur, label, test_size=0.2, random_state=42)

# # Skala fitur menggunakan StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Berikan nama fitur kembali setelah transformasi
# X_train = pd.DataFrame(X_train, columns=fitur.columns)
# X_test = pd.DataFrame(X_test, columns=fitur.columns)

# # Latih model RandomForestClassifier
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Lakukan prediksi pada data uji
# prediksi = model.predict(X_test)

# # Hitung akurasi model
# akurasi = accuracy_score(y_test, prediksi)
# print("Akurasi model: {:.2f}%".format(akurasi * 100))

# # User input data baru
# lama_belajar = int(input("Lama Belajar: "))
# intensitas_tugas = int(input("Intensitas Tugas: "))
# jam_kuliah = int(input("Jam Kuliah: "))
# jam_tidur = int(input("Jam Tidur: "))
# jenis_kelamin = int(input("Jenis Kelamin (0: Laki-laki, 1: Perempuan): "))
# perantau = int(input("Perantau (0: Bukan Perantau, 1: Perantau): "))
# jml_sks = int(input("Jumlah SKS: "))
# uang_bulanan = int(input("Uang Bulanan: "))

# # Lakukan prediksi pada data baru
# data_baru = [[lama_belajar, intensitas_tugas, jam_kuliah, jam_tidur, jenis_kelamin, perantau, jml_sks, uang_bulanan]]
# data_baru = pd.DataFrame(data_baru, columns=fitur.columns)
# data_baru = scaler.transform(data_baru)
# hasil_prediksi = model.predict(data_baru)

# # Konversi hasil prediksi kembali ke label kelas
# hasil_prediksi_label = label_encoder.inverse_transform(hasil_prediksi)[0]
# print("Prediksi tingkat stres mahasiswa: " + hasil_prediksi_label)


import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

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
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)


# Lakukan prediksi pada data uji
prediksi = model.predict(X_test)

# Hitung akurasi model
akurasi = accuracy_score(y_test, prediksi)
print("Akurasi model: {:.2f}%".format(akurasi * 100))

# User input data baru
lama_belajar = int(input("Lama Belajar: "))
intensitas_tugas = int(input("Intensitas Tugas: "))
jam_kuliah = int(input("Jam Kuliah: "))
jam_tidur = int(input("Jam Tidur: "))
jenis_kelamin = int(input("Jenis Kelamin (0: Laki-laki, 1: Perempuan): "))
perantau = int(input("Perantau (0: Bukan Perantau, 1: Perantau): "))
jml_sks = int(input("Jumlah SKS: "))
uang_bulanan = int(input("Uang Bulanan: "))

# Lakukan prediksi pada data baru
data_baru = [[lama_belajar, intensitas_tugas, jam_kuliah, jam_tidur, jenis_kelamin, perantau, jml_sks, uang_bulanan]]
data_baru = pd.DataFrame(data_baru, columns=fitur.columns)
data_baru = scaler.transform(data_baru)

# Lakukan prediksi pada data baru yang telah di-transform
hasil_prediksi = model.predict(data_baru)

# Konversi hasil prediksi kembali ke label kelas
hasil_prediksi_label = label_encoder.inverse_transform(hasil_prediksi)[0]
print("Prediksi tingkat stres mahasiswa: " + hasil_prediksi_label)
