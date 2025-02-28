from flask import Flask, render_template, request
import pickle
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import CONFIG


app = Flask(__name__, template_folder='../templates')

with open(CONFIG['MODEL_PATH'], 'rb') as file:
    model = pickle.load(file)
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
    
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        features_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=feature_names)

        prediction = model.predict(features_df)
        
        heart_condition = ['No Heart Disease', 'Have Heart Disease']
        prediction = heart_condition[int(prediction[0])]

        return render_template("index.html", prediction=f'Heart Condition: {prediction}')

    except Exception as e:
        return render_template("index.html", prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
        