import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, request, jsonify

app = Flask(__name__)

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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Membuat Model Decision Tree
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Membuat list untuk nama feature
    feature_names = list(X.columns)

    # Preprocessing input data
    input_data = pd.DataFrame(data, columns=feature_names)
    input_data['jenis_kelamin'] = le_jk.transform(input_data['jenis_kelamin'])
    input_data['perantau'] = le_perantau.transform(input_data['perantau'])

    # Make prediction using the model
    prediksi = model.predict(input_data)

    # Mengembalikan hasil prediksi sebagai respons JSON
    return jsonify({'prediction': prediksi})

if __name__ == '__main__':
    app.run()
