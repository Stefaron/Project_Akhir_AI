import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load data
data = pd.read_csv("data_stress.csv")

# Preprocessing data
le_jk = LabelEncoder()
le_perantau = LabelEncoder()
data['jenis_kelamin'] = le_jk.fit_transform(data['jenis_kelamin'])
data['perantau'] = le_perantau.fit_transform(data['perantau'])
X = data.drop('tingkat_stres', axis=1)
y = data['tingkat_stres']

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat Model Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Membuat list untuk nama feature
feature_names = list(X.columns)

# Menerima input dari pengguna
lama_belajar = int(input("Lama belajar (jam): "))
intensitas_tugas = int(input("Intensitas tugas: "))
jam_kuliah = int(input("Jam kuliah (jam): "))
jam_tidur = int(input("Jam tidur (jam): "))
jenis_kelamin = int(input("Jenis kelamin (0=Laki-laki, 1=Perempuan): "))
perantau = int(input("Mahasiswa perantau (0=Tidak, 1=Ya): "))
jml_sks = int(input("Jumlah SKS yang diambil: "))
uang_bulanan = float(input("Uang bulanan: "))

# Create input data as a list
input_data = [[lama_belajar, intensitas_tugas, jam_kuliah, jam_tidur, jenis_kelamin, perantau, jml_sks, uang_bulanan]]

# Create dataframe from input data
input_df = pd.DataFrame(input_data, columns=feature_names)

# Preprocessing input data
input_df['jenis_kelamin'] = le_jk.transform(input_df['jenis_kelamin'])
input_df['perantau'] = le_perantau.transform(input_df['perantau'])

# Make prediction using the model
prediksi = model.predict(input_df)

# Display the prediction result
if prediksi[0] == "Tinggi":
    print("Tingkat stres Anda tinggi.")
else:
    print("Tingkat stres Anda rendah.")
