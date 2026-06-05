import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE
# ─────────────────────────────────────────────
df = pd.read_csv("encoded_data.csv")

print("=" * 60)
print("MODEL TRAINING PIPELINE")
print("=" * 60)
print(f"Loaded encoded_data.csv → Shape: {df.shape}")

# Drop leakage columns (sub-AQI computed from target)
leakage_cols = ["US_AQI_PM25", "US_AQI_PM10", "US_AQI_NO2",
                "US_AQI_O3", "US_AQI_CO", "EU_AQI",
                "EU_AQI_PM25", "EU_AQI_PM10"]

# Drop non-numeric / identifier columns
drop_cols = ["Datetime", "Day_Name", "Rain_mm",
             "Cloud_Low_Percent", "Cloud_Mid_Percent", "Cloud_High_Percent",
             "PM25_Category_India_Enc"]   # secondary target, not a feature

drop_cols = [c for c in leakage_cols + drop_cols if c in df.columns]
df.drop(columns=drop_cols, inplace=True)

# Drop rows with any remaining nulls (only 8 rows)
df.dropna(inplace=True)

# Cast bool columns to int (RF needs numeric)
bool_cols = df.select_dtypes(include="bool").columns.tolist()
df[bool_cols] = df[bool_cols].astype(int)

print(f"After cleanup → Shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. FEATURES & TARGET
#    Target  : AQI_Category_Enc (0=Good … 5=Hazardous)
#    Features: everything else except US_AQI is fine —
#              US_AQI is the raw score that drives AQI_Category,
#              but in real inference you'd have it; keep it.
# ─────────────────────────────────────────────
TARGET = "AQI_Category_Enc"
X = df.drop(columns=[TARGET])
y = df[TARGET]

CLASS_NAMES = ["Good", "Moderate", "Unhealthy_Sensitive",
               "Unhealthy", "Very_Unhealthy", "Hazardous"]

print(f"\nFeatures : {X.shape[1]}")
print(f"Target   : {TARGET}")
print("\nClass distribution:")
for enc, name in enumerate(CLASS_NAMES):
    count = (y == enc).sum()
    pct   = count / len(y) * 100
    print(f"  {enc} {name:<22} : {count:>6,}  ({pct:.1f}%)")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# 4. TRAIN RANDOM FOREST
# ─────────────────────────────────────────────
print("\nTraining Random Forest …")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",   # handles class imbalance (Very_Unhealthy/Hazardous rare)
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy : {rf_acc * 100:.2f}%")

# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

# Only include class names for labels that appear in y_test
present_labels = sorted(y_test.unique())
present_names  = [CLASS_NAMES[i] for i in present_labels]

print(classification_report(y_test, rf_pred,
                             labels=present_labels,
                             target_names=present_names,
                             zero_division=0))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, rf_pred, labels=present_labels)
print(cm)

# ─────────────────────────────────────────────
# 6. CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\nRunning 5-Fold Cross-Validation …")
cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
print(f"CV Accuracy : {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")
print(f"CV Scores   : {[round(s*100,2) for s in cv_scores]}")

# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
importance = pd.Series(rf.feature_importances_, index=X.columns)
top20 = importance.sort_values(ascending=False).head(20)

print(f"\nTop 20 Feature Importances:")
for feat, score in top20.items():
    print(f"  {feat:<30} : {score:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Feature importance bar
axes[0].barh(top20.index[::-1], top20.values[::-1],
             color=plt.cm.RdYlGn_r(np.linspace(0, 1, 20)))
axes[0].set_title("Top 20 Feature Importances (Random Forest)", fontweight="bold")
axes[0].set_xlabel("Importance Score")

# Confusion matrix heatmap
import seaborn as sns
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=present_names, yticklabels=present_names, ax=axes[1])
axes[1].set_title("Confusion Matrix", fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].tick_params(axis="x", rotation=30)
axes[1].tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=120, bbox_inches="tight")
print("\nSaved: model_evaluation.png")

# ─────────────────────────────────────────────
# 8. SAVE MODEL & METADATA
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/aqi_model.pkl")
joblib.dump(list(X.columns), "models/feature_columns.pkl")   # column order for inference

print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)
print("  models/aqi_model.pkl        ← Trained RandomForestClassifier")
print("  models/feature_columns.pkl  ← Feature column order for prediction")
print("  model_evaluation.png        ← Feature importance + confusion matrix")
print(f"\nFinal Test Accuracy : {rf_acc * 100:.2f}%")