import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# 1. LOAD CLEANED DATA
# ─────────────────────────────────────────────
df = pd.read_csv("cleaned_aqi.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
df.info()

# ─────────────────────────────────────────────
# 2. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS — POLLUTANTS & AQI")
print("=" * 60)
pollutant_cols = ["PM2_5_ugm3", "PM10_ugm3", "CO_ugm3", "NO2_ugm3",
                  "SO2_ugm3", "O3_ugm3", "Dust_ugm3", "AOD",
                  "US_AQI", "EU_AQI"]
print(df[pollutant_cols].describe().round(2))

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS — WEATHER")
print("=" * 60)
weather_cols = ["Temp_2m_C", "Humidity_Percent", "Wind_Speed_10m_kmh",
                "Wind_Gusts_kmh", "Precipitation_mm", "Pressure_MSL_hPa",
                "Solar_Radiation_Wm2", "Cloud_Cover_Percent"]
print(df[weather_cols].describe().round(2))

# ─────────────────────────────────────────────
# 3. CATEGORICAL DISTRIBUTIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("AQI CATEGORY DISTRIBUTION (US Standard)")
print("=" * 60)
print(df["AQI_Category"].value_counts())
print(f"\n  (% share)\n{df['AQI_Category'].value_counts(normalize=True).mul(100).round(1)}")

print("\n" + "=" * 60)
print("PM2.5 CATEGORY DISTRIBUTION (India Standard)")
print("=" * 60)
print(df["PM25_Category_India"].value_counts())

print("\n" + "=" * 60)
print("STATE-WISE RECORD COUNT")
print("=" * 60)
print(df["State"].value_counts())

print("\n" + "=" * 60)
print("CITY COUNT  |  UNIQUE CITIES:", df["City"].nunique())
print("=" * 60)
print(df["City"].value_counts())

# ─────────────────────────────────────────────
# 4. AQI AGGREGATIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("AVERAGE US_AQI BY STATE  (sorted worst → best)")
print("=" * 60)
print(df.groupby("State")["US_AQI"].mean().sort_values(ascending=False).round(2))

print("\n" + "=" * 60)
print("AVERAGE US_AQI BY CITY  (sorted worst → best)")
print("=" * 60)
print(df.groupby("City")["US_AQI"].mean().sort_values(ascending=False).round(2))

print("\n" + "=" * 60)
print("AVERAGE US_AQI BY SEASON")
print("=" * 60)
print(df.groupby("Season")["US_AQI"].mean().sort_values(ascending=False).round(2))

print("\n" + "=" * 60)
print("AVERAGE US_AQI BY TIME OF DAY")
print("=" * 60)
print(df.groupby("Time_of_Day")["US_AQI"].mean().sort_values(ascending=False).round(2))

print("\n" + "=" * 60)
print("AVERAGE US_AQI BY MONTH")
print("=" * 60)
print(df.groupby("Month")["US_AQI"].mean().sort_values(ascending=False).round(2))

print("\n" + "=" * 60)
print("AVERAGE US_AQI — FESTIVAL PERIOD vs NORMAL")
print("=" * 60)
print(df.groupby("Festival_Period")["US_AQI"].mean().round(2))

print("\n" + "=" * 60)
print("AVERAGE US_AQI — CROP BURNING SEASON vs NORMAL")
print("=" * 60)
print(df.groupby("Crop_Burning_Season")["US_AQI"].mean().round(2))

# ─────────────────────────────────────────────
# 5. POLLUTANT-LEVEL STATS BY CATEGORY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MEAN POLLUTANT LEVELS BY AQI CATEGORY")
print("=" * 60)
aqi_order = ["Good", "Moderate", "Unhealthy_Sensitive",
             "Unhealthy", "Very_Unhealthy", "Hazardous"]
grp = df.groupby("AQI_Category")[
    ["PM2_5_ugm3", "PM10_ugm3", "NO2_ugm3", "SO2_ugm3",
     "O3_ugm3", "CO_ugm3"]
].mean().round(2)
grp = grp.reindex([c for c in aqi_order if c in grp.index])
print(grp)

# ─────────────────────────────────────────────
# 6. CORRELATION — WEATHER vs US_AQI
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("TOP CORRELATIONS WITH US_AQI")
print("=" * 60)
num_df = df.select_dtypes(include="number")
corr = num_df.corr()["US_AQI"].drop("US_AQI").abs().sort_values(ascending=False)
print("Top 15 (absolute correlation):")
print(corr.head(15).round(4))

# ─────────────────────────────────────────────
# 7. WEEKEND vs WEEKDAY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("AVERAGE US_AQI — WEEKEND vs WEEKDAY")
print("=" * 60)
print(df.groupby("Is_Weekend")["US_AQI"].mean().rename({False: "Weekday", True: "Weekend"}).round(2))

# ─────────────────────────────────────────────
# 8. MISSING VALUE SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("REMAINING MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
missing = missing[missing > 0]
if missing.empty:
    print("  No missing values.")
else:
    print(missing)

print("\nEDA Complete.")