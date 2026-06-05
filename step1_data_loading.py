import pandas as pd

# Load Dataset
df = pd.read_csv(r"C:\AQI-Project\aqi_sampled_dataset.xls")

print("Dataset Loaded Successfully")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())