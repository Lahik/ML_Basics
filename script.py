import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data from a CSV file
data = pd.read_csv('diabetes.csv')

# Show the first 5 rows
# print(data.head())

# X = all features (inputs), y = label (output)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict using the test data
y_pred = model.predict(X_test)

# Measure accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, 'diabetes_model.pkl')
