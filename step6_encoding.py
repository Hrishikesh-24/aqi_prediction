import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# ─────────────────────────────────────────────
# 1. LOAD CLEANED DATA
# ─────────────────────────────────────────────
df = pd.read_csv("cleaned_aqi.csv")
print("=" * 55)
print("ENCODING PIPELINE")
print("=" * 55)
print(f"Loaded cleaned_aqi.csv → Shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. ORDINAL ENCODINGS (order matters)
# ─────────────────────────────────────────────
ordinal_maps = {
    "AQI_Category": {
        "Good": 0, "Moderate": 1, "Unhealthy_Sensitive": 2,
        "Unhealthy": 3, "Very_Unhealthy": 4, "Hazardous": 5
    },
    "PM25_Category_India": {
        "Good": 0, "Satisfactory": 1, "Moderate": 2,
        "Poor": 3, "Very_Poor": 4, "Severe": 5
    },
}

for col, mapping in ordinal_maps.items():
    df[col + "_Enc"] = df[col].map(mapping)
    print(f"  [Ordinal] {col} → {mapping}")

# ─────────────────────────────────────────────
# 3. LABEL ENCODINGS (nominal categoricals)
# ─────────────────────────────────────────────
label_cols = [
    "City", "State",
    "Season", "Time_of_Day",
    "Humidity_Category", "Wind_Category",
]

encoders = {}   # saved for inference / inverse_transform later

for col in label_cols:
    le = LabelEncoder()
    df[col + "_Enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"  [Label]   {col} → {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ─────────────────────────────────────────────
# 4. STANDARD SCALER (continuous numerics)
# ─────────────────────────────────────────────
skip_scaling = [
    "US_AQI",                                       # target
    "AQI_Category_Enc", "PM25_India_Enc",
    "Is_Weekend", "Is_Raining", "Heavy_Rain",
    "Is_Daytime", "Wind_Stagnation",
    "Festival_Period", "Crop_Burning_Season",
    "Year", "Month", "Day", "Hour",
    "Day_of_Week", "Week_of_Year", "Quarter",
    "City_Enc", "State_Enc",
    "Season_Enc", "Time_of_Day_Enc",
    "Humidity_Category_Enc", "Wind_Category_Enc",
]

# Only scale columns that exist in df after encoding
numeric_cols = df.select_dtypes(include="number").columns.tolist()
scale_cols   = [c for c in numeric_cols if c not in skip_scaling]

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
print(f"\n  [Scaler]  StandardScaler fitted on {len(scale_cols)} columns")

# ─────────────────────────────────────────────
# 5. DROP ORIGINAL STRING COLUMNS
# ─────────────────────────────────────────────
drop_cols = list(ordinal_maps.keys()) + label_cols
df.drop(columns=drop_cols, inplace=True, errors="ignore")
print(f"\nDropped original string columns: {drop_cols}")

# ─────────────────────────────────────────────
# 6. SAVE ENCODERS + SCALER AS PKL
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(encoders,     "models/label_encoders.pkl")
joblib.dump(ordinal_maps, "models/ordinal_maps.pkl")
joblib.dump(scaler,       "models/scaler.pkl")

print("\nSaved:")
print("  models/label_encoders.pkl  ← LabelEncoder objects for City, State, Season, etc.")
print("  models/ordinal_maps.pkl    ← Ordinal dicts for AQI_Category & PM25_Category_India")
print("  models/scaler.pkl          ← StandardScaler fitted on numeric features")

# ─────────────────────────────────────────────
# 7. SAVE ENCODED DATAFRAME
# ─────────────────────────────────────────────
df.to_csv("encoded_data.csv", index=False)

print("\n" + "=" * 55)
print("ENCODING COMPLETE")
print("=" * 55)
print(f"Shape        : {df.shape}")
print(f"Missing vals : {df.isnull().sum().sum()}")
print("Saved        : encoded_data.csv")