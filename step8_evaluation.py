import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

df = pd.read_csv("encoded_data.csv")

X = df.drop("air_quality_status", axis=1)
y = df["air_quality_status"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = joblib.load("aqi_model.pkl")

predictions = model.predict(X_test)

# ── Load label names ──────────────────────────────────────────────────────────
encoders = joblib.load("encoders.pkl")
label_names = encoders["air_quality_status"].classes_

print("Accuracy:")
print(round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=label_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))