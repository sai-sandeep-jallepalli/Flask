from flask import Flask, request, render_template
import pickle
import os
import pandas as pd

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "../templates"))

# Load model, label encoders, and column names
base_path = os.path.abspath(os.path.dirname(__file__))  # Get the base directory
model_path = os.path.join(base_path, "../pickle-models/train.pkl")
encoders_path = os.path.join(base_path, "../pickle-models/label_encoders.pkl")
column_names_path = os.path.join(base_path, "../pickle-models/column_names.pkl")

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(encoders_path, 'rb') as file:
    label_encoders = pickle.load(file)

with open(column_names_path, 'rb') as file:
    column_names = pickle.load(file)
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Create a dictionary to store user inputs
        input_data = {}

        for column in column_names:
            if column in label_encoders:
                value = request.form.get(column, "missing")
                try:
                    input_data[column] = label_encoders[column].transform([value])[0]  # Transform categorical data
                except ValueError:
                    return render_template('index.html', prediction=f"Invalid input for column: {column}")
            else:
                value = request.form.get(column, 0)  # Default value for missing numeric fields
                try:
                    input_data[column] = float(value)  # Convert to float
                except ValueError:
                    return render_template('index.html', prediction=f"Invalid numeric input for column: {column}")

        # Convert input dictionary to DataFrame with exact feature names
        features_df = pd.DataFrame([input_data], columns=column_names)

        # Make prediction
        prediction = model.predict(features_df)[0]

        return render_template('predict.html', prediction=f'Prediction: {"Survived" if prediction == 1 else "Did not survive"}')

    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)