import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from flask import Flask, render_template, request
import os

base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "../pickle-models/iris-model.pkl")

app = Flask(__name__, template_folder=os.path.join(base_dir, '../templates'))

with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['SepalLength'])
        sepal_width = float(request.form['SepalWidth'])
        petal_length = float(request.form['PetalLength'])
        petal_width = float(request.form['PetalWidth'])

        # Use the exact feature names from training
        feature_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
        features_df = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)

        # Predict using DataFrame
        prediction = model.predict(features_df)

        species = ['setosa', 'versicolor', 'virginica']
        predicted_species = species[prediction[0]]

        return render_template("index.html", prediction=f'Predicted Iris Species: {predicted_species}')

    except Exception as e:
        return render_template("index.html", prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)