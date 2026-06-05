import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────
df = pd.read_csv("aqi_sampled_dataset.xls")          # file is CSV despite .xls extension

print("=" * 55)
print("ORIGINAL DATASET")
print("=" * 55)
print(f"Shape          : {df.shape}")
print(f"Duplicate Rows : {df.duplicated().sum()}")
print(f"\nMissing Values per Column:")
missing = df.isnull().sum()
print(missing[missing > 0])

# ─────────────────────────────────────────────
# 2. REMOVE DUPLICATES
# ─────────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed : {before - len(df)}")

# ─────────────────────────────────────────────
# 3. DROP FULLY NULL COLUMNS (100 % missing)
#    These columns have no data in this sample
#    and cannot be imputed or used.
# ─────────────────────────────────────────────
all_null_cols = [c for c in df.columns if df[c].isnull().all()]
df.drop(columns=all_null_cols, inplace=True)
print(f"\nDropped 100%-null columns ({len(all_null_cols)}) :")
print(" ", all_null_cols)

# Temp_Inversion is always 0 with no variation — also drop
if "Temp_Inversion" in df.columns and df["Temp_Inversion"].nunique() == 1:
    df.drop(columns=["Temp_Inversion"], inplace=True)
    print("  Temp_Inversion (constant zero — no inversion events in sample)")

# ─────────────────────────────────────────────
# 4. PARSE DATETIME
# ─────────────────────────────────────────────
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
bad_dt = df["Datetime"].isna().sum()
if bad_dt:
    print(f"\nUnparseable Datetime rows dropped: {bad_dt}")
    df.dropna(subset=["Datetime"], inplace=True)

# ─────────────────────────────────────────────
# 5. FIX PHYSICALLY IMPOSSIBLE VALUES
#    O3 concentration cannot be negative.
#    Clip to 0 to preserve the rows.
# ─────────────────────────────────────────────
neg_o3 = (df["O3_ugm3"] < 0).sum()
if neg_o3:
    df["O3_ugm3"] = df["O3_ugm3"].clip(lower=0)
    print(f"\nO3_ugm3 negative values clipped to 0 : {neg_o3} rows")

# ─────────────────────────────────────────────
# 6. FILL MISSING CATEGORICAL (AQI_Category)
#    Derive missing labels from US_AQI breakpoints.
# ─────────────────────────────────────────────
def aqi_label(aqi):
    if pd.isna(aqi):
        return np.nan
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy_Sensitive"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very_Unhealthy"
    return "Hazardous"

missing_cat = df["AQI_Category"].isna().sum()
if missing_cat:
    mask = df["AQI_Category"].isna()
    df.loc[mask, "AQI_Category"] = df.loc[mask, "US_AQI"].apply(aqi_label)
    print(f"\nAQI_Category filled from US_AQI    : {missing_cat} rows")

# ─────────────────────────────────────────────
# 7. STANDARDISE COLUMN TYPES
# ─────────────────────────────────────────────
bool_cols = [c for c in ["Is_Weekend", "Is_Raining", "Heavy_Rain",
                          "Is_Daytime", "Festival_Period", "Crop_Burning_Season"]
             if c in df.columns]
for col in bool_cols:
    df[col] = df[col].astype(bool)

category_cols = [c for c in ["City", "State", "Season", "Time_of_Day", "Day_Name",
                               "Humidity_Category", "Wind_Category", "Wind_Stagnation",
                               "AQI_Category", "PM25_Category_India"]
                 if c in df.columns]
for col in category_cols:
    df[col] = df[col].astype("category")

# ─────────────────────────────────────────────
# 8. FINAL REPORT
# ─────────────────────────────────────────────
remaining_null = df.isnull().sum()
print("\n" + "=" * 55)
print("CLEANED DATASET")
print("=" * 55)
print(f"Shape              : {df.shape}")
print(f"Remaining Nulls    : {df.isnull().sum().sum()}")
if remaining_null.sum() > 0:
    print("Columns still with nulls:")
    print(remaining_null[remaining_null > 0])
print(f"Duplicate Rows     : {df.duplicated().sum()}")

# ─────────────────────────────────────────────
# 9. SAVE
# ─────────────────────────────────────────────
df.to_csv("cleaned_aqi.csv", index=False)
print("\nCleaned dataset saved → cleaned_aqi.csv")