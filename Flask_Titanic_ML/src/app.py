from flask import Flask, request, render_template
import pickle
import os

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
    features = []
    for column in column_names:
        if column in label_encoders:
            value = request.form.get(column, "missing")
            try:
                features.append(label_encoders[column].transform([value])[0])
            except ValueError:
                return render_template('index.html', prediction="Invalid input for column: " + column)
        else:
            value = request.form.get(column, 0)  # Default value for missing numeric fields
            try:
                features.append(float(value))
            except ValueError:
                return render_template('index.html', prediction="Invalid numeric input for column: " + column)

    prediction = model.predict([features])[0]
    return render_template('predict.html', prediction=f'Prediction: {"Survived" if prediction == 1 else "Did not survive"}')

if __name__ == "__main__":
    app.run(debug=True)