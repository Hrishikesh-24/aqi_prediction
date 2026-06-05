import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

# ─────────────────────────────────────────────
# 1. LOAD CLEANED DATA
# ─────────────────────────────────────────────
df = pd.read_csv("cleaned_aqi.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

print("=" * 60)
print("ORIGINAL SHAPE:", df.shape)
print("=" * 60)

# ─────────────────────────────────────────────
# 2. DROP LEAKAGE & REDUNDANT COLUMNS
#    US_AQI_* and EU_AQI_* are sub-components
#    used to compute the target (US_AQI) — keeping
#    them would cause data leakage in any model.
#    Rain_mm == Precipitation_mm (duplicate).
#    Cloud_Low/Mid/High are captured by Cloud_Cover.
#    Datetime, Day_Name are replaced by numeric features.
# ─────────────────────────────────────────────
leakage_cols = [
    "US_AQI_PM25", "US_AQI_PM10", "US_AQI_NO2", "US_AQI_O3", "US_AQI_CO",
    "EU_AQI", "EU_AQI_PM25", "EU_AQI_PM10",
]
redundant_cols = [
    "Rain_mm",                                    # identical to Precipitation_mm
    "Cloud_Low_Percent", "Cloud_Mid_Percent", "Cloud_High_Percent",  # captured by Cloud_Cover_Percent
    "Day_Name",                                   # Day_of_Week (int) already present
    "Datetime",                                   # replaced by Year/Month/Day/Hour below
]
df.drop(columns=leakage_cols + redundant_cols, inplace=True, errors="ignore")
print(f"Dropped {len(leakage_cols)} leakage cols and {len(redundant_cols)} redundant cols")

# ─────────────────────────────────────────────
# 3. NEW FEATURES
# ─────────────────────────────────────────────

# 3a. Cyclical encoding for circular numeric features
#     Wind direction (1–360) and Hour (0–23) are circular:
#     degree 1 and degree 360 are neighbours. Sin/cos
#     encoding preserves this topology.
df["Wind_Dir_Sin"] = np.sin(np.radians(df["Wind_Dir_10m"]))
df["Wind_Dir_Cos"] = np.cos(np.radians(df["Wind_Dir_10m"]))
df["Hour_Sin"]     = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_Cos"]     = np.cos(2 * np.pi * df["Hour"] / 24)
df["Month_Sin"]    = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_Cos"]    = np.cos(2 * np.pi * df["Month"] / 12)
df.drop(columns=["Wind_Dir_10m"], inplace=True)   # raw direction now encoded
print("Added cyclical features: Wind_Dir, Hour, Month (sin/cos)")

# 3b. Pollution interaction features
df["PM_Load"]        = df["PM2_5_ugm3"] + df["PM10_ugm3"]          # total particulate burden
df["PM_Ratio"]       = df.get("PM_Ratio", df["PM2_5_ugm3"] / (df["PM10_ugm3"] + 1e-6))  # keep if exists
df["NOx_SO2_Index"]  = df["NO2_ugm3"] + df["SO2_ugm3"]             # combined secondary pollutants
df["O3_NO2_Ratio"]   = df["O3_ugm3"]  / (df["NO2_ugm3"] + 1e-6)   # photochemical smog indicator
df["Heat_Humidity"]  = df["Temp_2m_C"] * df["Humidity_Percent"] / 100  # apparent heat stress

print("Added interaction features: PM_Load, NOx_SO2_Index, O3_NO2_Ratio, Heat_Humidity")

# 3c. Rolling lag features (city-level, sorted by time proxy: Month+Day+Hour)
#     Gives model short-term memory of recent AQI conditions.
df_sorted = df.sort_values(["City", "Month", "Day", "Hour"]).copy()
for lag in [1, 3]:
    df_sorted[f"PM25_lag_{lag}h"] = (
        df_sorted.groupby("City")["PM2_5_ugm3"]
        .shift(lag)
        .bfill()
    )
    df_sorted[f"AQI_lag_{lag}h"] = (
        df_sorted.groupby("City")["US_AQI"]
        .shift(lag)
        .bfill()
    )
df = df_sorted.reset_index(drop=True)
print("Added lag features: PM25_lag_1h, PM25_lag_3h, AQI_lag_1h, AQI_lag_3h")

# ─────────────────────────────────────────────
# 4. ENCODE CATEGORICALS
# ─────────────────────────────────────────────

# 4a. Target: AQI_Category (ordinal — order matters for models)
aqi_order = {
    "Good": 0, "Moderate": 1, "Unhealthy_Sensitive": 2,
    "Unhealthy": 3, "Very_Unhealthy": 4, "Hazardous": 5
}
df["AQI_Category_Enc"] = df["AQI_Category"].map(aqi_order)
print("\nAQI_Category encoding:", aqi_order)

# 4b. PM25 India category (ordinal)
india_order = {
    "Good": 0, "Satisfactory": 1, "Moderate": 2,
    "Poor": 3, "Very_Poor": 4, "Severe": 5
}
df["PM25_India_Enc"] = df["PM25_Category_India"].map(india_order)
print("PM25_Category_India encoding:", india_order)

# 4c. Low-cardinality nominals → Label Encode
label_cols = ["Season", "Time_of_Day", "Humidity_Category", "Wind_Category"]
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col + "_Enc"] = le.fit_transform(df[col].astype(str))
    le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {col} → {le_dict[col]}")

# 4d. High-cardinality nominals (City, State) → Label Encode
#     For tree models this works well; for linear models
#     consider target encoding instead.
for col in ["City", "State"]:
    le = LabelEncoder()
    df[col + "_Enc"] = le.fit_transform(df[col].astype(str))
    le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))

print(f"\nCity encoded ({df['City'].nunique()} cities), State encoded ({df['State'].nunique()} states)")

# ─────────────────────────────────────────────
# 5. DROP ORIGINAL STRING COLUMNS
#    (replaced by encoded versions above)
# ─────────────────────────────────────────────
str_cols = ["City", "State", "Season", "Time_of_Day",
            "Humidity_Category", "Wind_Category",
            "AQI_Category", "PM25_Category_India"]
df.drop(columns=str_cols, inplace=True)
print(f"\nDropped original string columns: {str_cols}")

# ─────────────────────────────────────────────
# 6. SCALE NUMERIC FEATURES
#    StandardScaler for continuous variables;
#    leave binary/encoded/target columns unscaled.
# ─────────────────────────────────────────────
skip_scaling = [
    "US_AQI",                                      # target — do not scale
    "AQI_Category_Enc", "PM25_India_Enc",          # ordinal targets
    "Is_Weekend", "Is_Raining", "Heavy_Rain",
    "Is_Daytime", "Wind_Stagnation",
    "Festival_Period", "Crop_Burning_Season",
    "Year", "Month", "Day", "Hour", "Day_of_Week",
    "Week_of_Year", "Quarter",
    "City_Enc", "State_Enc",
    "Season_Enc", "Time_of_Day_Enc",
    "Humidity_Category_Enc", "Wind_Category_Enc",
]
scale_cols = [c for c in df.select_dtypes(include="number").columns
              if c not in skip_scaling]

scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
print(f"\nStandardScaled {len(scale_cols)} numeric columns")

# ─────────────────────────────────────────────
# 7. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FEATURE-ENGINEERED DATASET SUMMARY")
print("=" * 60)
print(f"Shape              : {df.shape}")
print(f"Missing Values     : {df.isnull().sum().sum()}")
print(f"\nAll columns ({len(df.columns)}):")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

# ─────────────────────────────────────────────
# 8. SAVE
# ─────────────────────────────────────────────
df.to_csv("model_ready.csv", index=False)
print("\nSaved → model_ready.csv")