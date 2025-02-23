import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import pickle

base_dir = os.path.abspath(os.path.dirname(__file__))
dataset_path = os.path.join(base_dir, "../datasets/iris-dataset.csv")
model_path = os.path.join(base_dir, "../pickle-models/iris-model.pkl")

data = pd.read_csv(dataset_path)

X = data[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
y = data["Species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

with open(model_path, 'wb') as file:
    pickle.dump(model, file)
    
print("Model trained and saved successfully!")



