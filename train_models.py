import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(
    "synthetic_aqi_dataset_20000.csv"
)

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

df.drop("date",axis=1,inplace=True)

encoders = {}

for col in [
    "state",
    "area",
    "prominent_pollutants",
    "air_quality_status"
]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col]) # type: ignore
    encoders[col] = le

joblib.dump(
    encoders,
    "encoders.pkl"
)

X = df.drop(
    "air_quality_status",
    axis=1
)

y = df["air_quality_status"]

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

gb = GradientBoostingClassifier()
gb.fit(X_train,y_train)

joblib.dump(
    rf,
    "models/random_forest.pkl"
)

joblib.dump(
    dt,
    "models/decision_tree.pkl"
)

joblib.dump(
    gb,
    "models/gradient_boost.pkl"
)

print("Models Saved")