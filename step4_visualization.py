import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
df = pd.read_csv("cleaned_aqi.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

AQI_ORDER    = ["Good", "Moderate", "Unhealthy_Sensitive", "Unhealthy", "Very_Unhealthy", "Hazardous"]
AQI_COLORS   = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#9b59b6", "#7f0000"]
INDIA_ORDER  = ["Good", "Satisfactory", "Moderate", "Poor", "Very_Poor", "Severe"]
INDIA_COLORS = ["#27ae60", "#2ecc71", "#f39c12", "#e74c3c", "#8e44ad", "#7f0000"]
MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ─────────────────────────────────────────────
# FIGURE 1 — AQI DISTRIBUTION OVERVIEW (2×2)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("AQI Distribution Overview", fontsize=15, fontweight="bold", y=1.01)

# 1a. US_AQI histogram (cap at 500 for readability; ~99.5% of data)
ax = axes[0, 0]
plot_data = df[df["US_AQI"] <= 500]["US_AQI"]
sns.histplot(plot_data, bins=50, color="#3498db", edgecolor="white", ax=ax) # type: ignore
ax.axvline(plot_data.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {plot_data.mean():.0f}")
ax.axvline(plot_data.median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median: {plot_data.median():.0f}")
ax.set_title("US AQI Distribution (≤ 500)")
ax.set_xlabel("US AQI")
ax.set_ylabel("Count")
ax.legend()

# 1b. AQI Category count (US standard)
ax = axes[0, 1]
order = [c for c in AQI_ORDER if c in df["AQI_Category"].unique()]
colors = [AQI_COLORS[AQI_ORDER.index(c)] for c in order]
counts = df["AQI_Category"].value_counts().reindex(order)
bars = ax.bar(order, counts.values, color=colors, edgecolor="white")
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f"{val:,}", ha="center", va="bottom", fontsize=9)
ax.set_title("AQI Category Distribution (US Standard)")
ax.set_xlabel("AQI Category")
ax.set_ylabel("Count")
ax.set_xticklabels(order, rotation=30, ha="right")

# 1c. PM2.5 Category (India standard)
ax = axes[1, 0]
order_in = [c for c in INDIA_ORDER if c in df["PM25_Category_India"].unique()]
colors_in = [INDIA_COLORS[INDIA_ORDER.index(c)] for c in order_in]
counts_in = df["PM25_Category_India"].value_counts().reindex(order_in)
bars = ax.bar(order_in, counts_in.values, color=colors_in, edgecolor="white")
for bar, val in zip(bars, counts_in.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
            f"{val:,}", ha="center", va="bottom", fontsize=9)
ax.set_title("PM2.5 Category Distribution (India Standard)")
ax.set_xlabel("PM2.5 Category")
ax.set_ylabel("Count")
ax.set_xticklabels(order_in, rotation=30, ha="right")

# 1d. EU_AQI histogram
ax = axes[1, 1]
plot_eu = df[df["EU_AQI"] <= 300]["EU_AQI"]
sns.histplot(plot_eu, bins=50, color="#9b59b6", edgecolor="white", ax=ax) # type: ignore
ax.axvline(plot_eu.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {plot_eu.mean():.0f}")
ax.axvline(plot_eu.median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median: {plot_eu.median():.0f}")
ax.set_title("EU AQI Distribution (≤ 300)")
ax.set_xlabel("EU AQI")
ax.set_ylabel("Count")
ax.legend()

plt.tight_layout()
plt.savefig("fig1_aqi_distribution.png", bbox_inches="tight")
plt.show()
print("Saved: fig1_aqi_distribution.png")

# ─────────────────────────────────────────────
# FIGURE 2 — POLLUTANT ANALYSIS (2×2)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Pollutant Analysis", fontsize=15, fontweight="bold")

pollutants    = ["PM2_5_ugm3", "PM10_ugm3", "NO2_ugm3", "SO2_ugm3", "O3_ugm3", "CO_ugm3", "Dust_ugm3", "AOD"]
poll_labels   = ["PM2.5", "PM10", "NO₂", "SO₂", "O₃", "CO", "Dust", "AOD"]
poll_colors   = sns.color_palette("tab10", len(pollutants))

# 2a. Mean pollutant concentrations bar chart
ax = axes[0, 0]
means = df[pollutants].mean()
# Normalise to 0–100 scale for visual comparison
norm_means = (means / means.max()) * 100
bars = ax.barh(poll_labels, norm_means.values, color=poll_colors)
for bar, raw in zip(bars, means.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{raw:.1f}", va="center", fontsize=9)
ax.set_title("Relative Mean Pollutant Levels (normalised)")
ax.set_xlabel("Normalised Mean (0–100)")
ax.invert_yaxis()

# 2b. Correlation of pollutants with US_AQI
ax = axes[0, 1]
corr = df[pollutants].corrwith(df["US_AQI"]).sort_values(ascending=True)
colors_corr = ["#e74c3c" if v > 0 else "#3498db" for v in corr.values]
bars = ax.barh(
    [poll_labels[pollutants.index(p)] for p in corr.index],
    corr.values, color=colors_corr
)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Pollutant Correlation with US AQI")
ax.set_xlabel("Pearson Correlation")

# 2c. PM2.5 box plot by AQI Category
ax = axes[1, 0]
order = [c for c in AQI_ORDER if c in df["AQI_Category"].unique()]
sns.boxplot(data=df[df["PM2_5_ugm3"] <= 200],
            x="AQI_Category", y="PM2_5_ugm3",
            order=order, palette=AQI_COLORS[:len(order)], ax=ax)
ax.set_title("PM2.5 Distribution by AQI Category")
ax.set_xlabel("AQI Category")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_xticklabels(order, rotation=30, ha="right")

# 2d. PM10 box plot by Season
ax = axes[1, 1]
season_order = ["Winter", "Summer", "Monsoon", "Post_Monsoon"]
season_colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c"]
sns.boxplot(data=df[df["PM10_ugm3"] <= 500],
            x="Season", y="PM10_ugm3",
            order=season_order, palette=season_colors, ax=ax)
ax.set_title("PM10 Distribution by Season")
ax.set_xlabel("Season")
ax.set_ylabel("PM10 (µg/m³)")

plt.tight_layout()
plt.savefig("fig2_pollutants.png", bbox_inches="tight")
plt.show()
print("Saved: fig2_pollutants.png")

# ─────────────────────────────────────────────
# FIGURE 3 — TEMPORAL TRENDS (2×2)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Temporal AQI Trends", fontsize=15, fontweight="bold")

# 3a. Monthly average US_AQI
ax = axes[0, 0]
monthly = df.groupby("Month")["US_AQI"].mean()
ax.plot(monthly.index, monthly.values, marker="o", color="#e74c3c", linewidth=2)
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#e74c3c")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(MONTH_LABELS)
ax.set_title("Monthly Average US AQI")
ax.set_xlabel("Month")
ax.set_ylabel("Mean US AQI")
ax.axhline(monthly.mean(), color="gray", linestyle="--", linewidth=1, label=f"Annual mean: {monthly.mean():.0f}")
ax.legend()

# 3b. AQI by Season (box)
ax = axes[0, 1]
season_order = ["Winter", "Summer", "Post_Monsoon", "Monsoon"]
sns.boxplot(data=df[df["US_AQI"] <= 500],
            x="Season", y="US_AQI",
            order=season_order,
            palette=["#3498db", "#e67e22", "#e74c3c", "#2ecc71"], ax=ax)
ax.set_title("US AQI Distribution by Season")
ax.set_xlabel("Season")
ax.set_ylabel("US AQI")

# 3c. AQI by Time of Day
ax = axes[1, 0]
tod_order = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Night_Late"]
tod_mean = df.groupby("Time_of_Day")["US_AQI"].mean().reindex(tod_order)
bars = ax.bar(tod_order, tod_mean.values,
              color=sns.color_palette("coolwarm", len(tod_order)), edgecolor="white")
for bar, val in zip(bars, tod_mean.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.0f}", ha="center", va="bottom", fontsize=9)
ax.set_title("Average US AQI by Time of Day")
ax.set_xlabel("Time of Day")
ax.set_ylabel("Mean US AQI")
ax.set_xticklabels(tod_order, rotation=30, ha="right")

# 3d. Weekend vs Weekday by Season
ax = axes[1, 1]
wk = df[df["US_AQI"] <= 500].groupby(["Season", "Is_Weekend"])["US_AQI"].mean().unstack()
wk.columns = ["Weekday", "Weekend"]
wk = wk.reindex(["Winter", "Summer", "Post_Monsoon", "Monsoon"])
x = np.arange(len(wk))
w = 0.35
ax.bar(x - w/2, wk["Weekday"], width=w, label="Weekday", color="#3498db", edgecolor="white")
ax.bar(x + w/2, wk["Weekend"], width=w, label="Weekend", color="#e74c3c", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(wk.index)
ax.set_title("Weekday vs Weekend AQI by Season")
ax.set_ylabel("Mean US AQI")
ax.legend()

plt.tight_layout()
plt.savefig("fig3_temporal.png", bbox_inches="tight")
plt.show()
print("Saved: fig3_temporal.png")

# ─────────────────────────────────────────────
# FIGURE 4 — GEOGRAPHIC & CATEGORICAL (2×2)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("Geographic & Categorical Analysis", fontsize=15, fontweight="bold")

# 4a. Top 10 cities by mean US_AQI
ax = axes[0, 0]
city_aqi = df.groupby("City")["US_AQI"].mean().sort_values(ascending=True).tail(10)
colors_city = sns.color_palette("Reds_r", len(city_aqi))
bars = ax.barh(city_aqi.index, city_aqi.values, color=colors_city, edgecolor="white")
for bar, val in zip(bars, city_aqi.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f"{val:.0f}", va="center", fontsize=9)
ax.set_title("Top 10 Most Polluted Cities (Mean US AQI)")
ax.set_xlabel("Mean US AQI")
ax.axvline(100, color="gray", linestyle="--", linewidth=1, label="AQI 100 threshold")
ax.legend()

# 4b. State-wise mean AQI (all states)
ax = axes[0, 1]
state_aqi = df.groupby("State")["US_AQI"].mean().sort_values(ascending=True)
colors_state = sns.color_palette("RdYlGn_r", len(state_aqi))
ax.barh(state_aqi.index, state_aqi.values, color=colors_state, edgecolor="white")
ax.set_title("State-wise Mean US AQI")
ax.set_xlabel("Mean US AQI")
ax.axvline(100, color="gray", linestyle="--", linewidth=1)

# 4c. AQI during Festival vs Non-Festival
ax = axes[1, 0]
festival_data = df[df["US_AQI"] <= 500].copy()
festival_data["Period"] = festival_data["Festival_Period"].map({True: "Festival", False: "Normal", 1: "Festival", 0: "Normal"})
crop_data = df[df["US_AQI"] <= 500].copy()
crop_data["Period"] = crop_data["Crop_Burning_Season"].map({True: "Crop Burning", False: "Normal", 1: "Crop Burning", 0: "Normal"})

combined = pd.concat([
    festival_data[["Period","US_AQI"]].assign(Type="Festival"),
    crop_data[["Period","US_AQI"]].assign(Type="Crop Burning")
])
means_f = festival_data.groupby("Period")["US_AQI"].mean()
means_c = crop_data.groupby("Period")["US_AQI"].mean()

x = np.arange(2)
labels = ["Normal", "Active Period"]
ax.bar(x[0] - 0.2, means_f.get("Normal", 0), width=0.35, label="Festival: Normal", color="#3498db", edgecolor="white")
ax.bar(x[0] + 0.2, means_f.get("Festival", 0), width=0.35, label="Festival: Active", color="#e74c3c", edgecolor="white")
ax.bar(x[1] - 0.2, means_c.get("Normal", 0), width=0.35, label="Crop: Normal", color="#2ecc71", edgecolor="white")
ax.bar(x[1] + 0.2, means_c.get("Crop Burning", 0), width=0.35, label="Crop: Active", color="#e67e22", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(["Festival Period", "Crop Burning Season"])
ax.set_title("AQI Impact: Festival & Crop Burning Events")
ax.set_ylabel("Mean US AQI")
ax.legend(fontsize=8)

# 4d. Humidity vs AQI scatter (sampled for speed)
ax = axes[1, 1]
sample = df[df["US_AQI"] <= 500].sample(3000, random_state=42)
sc = ax.scatter(sample["Humidity_Percent"], sample["US_AQI"],
                c=sample["Temp_2m_C"], cmap="RdYlBu_r",
                alpha=0.4, s=15)
plt.colorbar(sc, ax=ax, label="Temp (°C)")
ax.set_title("Humidity vs US AQI (coloured by Temperature)")
ax.set_xlabel("Humidity (%)")
ax.set_ylabel("US AQI")

plt.tight_layout()
plt.savefig("fig4_geographic_categorical.png", bbox_inches="tight")
plt.show()
print("Saved: fig4_geographic_categorical.png")

# ─────────────────────────────────────────────
# FIGURE 5 — CORRELATION HEATMAP
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))
key_cols = ["US_AQI", "EU_AQI", "PM2_5_ugm3", "PM10_ugm3", "CO_ugm3",
            "NO2_ugm3", "SO2_ugm3", "O3_ugm3", "Dust_ugm3", "AOD",
            "Temp_2m_C", "Humidity_Percent", "Wind_Speed_10m_kmh",
            "Precipitation_mm", "Solar_Radiation_Wm2", "Cloud_Cover_Percent"]
corr_matrix = df[key_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 8}, ax=ax)
ax.set_title("Correlation Heatmap — Pollutants & Weather vs AQI", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig5_correlation_heatmap.png", bbox_inches="tight")
plt.show()
print("Saved: fig5_correlation_heatmap.png")

print("\nAll visualizations complete.")