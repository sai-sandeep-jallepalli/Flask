import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.preprocessing import StandardScaler

from config.config import CONFIG


data = pd.read_csv(CONFIG["DATASET_PATH"])

data = data.dropna()

scale_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

scaler = StandardScaler()
data[scale_features] = scaler.fit_transform(data[scale_features])

X = data.drop(columns=['target'], axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy * 100)

with open(CONFIG['MODEL_PATH'], 'wb') as file:
    pickle.dump(model, file)

print("Model training completed and saved.")

