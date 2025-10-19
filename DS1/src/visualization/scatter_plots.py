import matplotlib.pyplot as plt
import pandas as pd

# Import data
path = "/Users/kayttaja/Desktop/DS1/data/processed/full_data.csv"
df = pd.read_csv(path, parse_dates=True, index_col="Date")
print(df.head())

# In-sample split
TRAIN_START = "1926-07-01"
TRAIN_END = "1994-12-01"

# Relevant variables
relevant_cols = [
    "5yr_excess_return",  # Target
    "CAPE3",
    "CAPE5",
    "CAPE7",
    "CAPE10",
    "EY_CAPE3",
    "EY_CAPE5",
    "EY_CAPE7",
    "EY_CAPE10",
    "ECY3",
    "ECY5",
    "ECY7",  # Plot
    "ECY10",
    "DFY",  # Plot
    "DFY_delta_12m",  # Plot
    "DFY_ma_12m",
    "DFY_vol_12m",
    "infl_yoy",
    "infl_trend_5y",  # Plot
    "infl_gap",
    "infl_vol_12m",  # Plot
    "term_spread",  # Plot
]
df = df[relevant_cols].loc[TRAIN_START:TRAIN_END]

# Set the subplot
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 8))
ax1, ax2, ax3, ax4, ax5, ax6 = axes.ravel()

# CAPE10 against target
ax1.scatter(df["ECY7"], df["5yr_excess_return"])
ax1.set_title("ECY7 vs target (1926–1994)")
ax1.set_xlabel("ECY7 (%)")
ax1.set_ylabel("5-year ahead excess returns (%)")
ax1.grid(True)

# EY_CAPE10 against target
ax2.scatter(df["DFY"], df["5yr_excess_return"])
ax2.set_title("DFY vs target (1926–1994)")
ax2.set_xlabel("DFY (%)")
ax2.set_ylabel("5-year ahead excess returns (%)")
ax2.grid(True)

# ECY10 against target
ax3.scatter(df["DFY_delta_12m"], df["5yr_excess_return"])
ax3.set_title("DFY_delta_12m vs target (1926–1994)")
ax3.set_xlabel("DFY_delta_12m (%)")
ax3.set_ylabel("5-year ahead excess returns (%)")
ax3.grid(True)

# DFY against target
ax4.scatter(df["infl_trend_5y"] * 100, df["5yr_excess_return"])
ax4.set_title("infl_trend_5y vs target (1926–1994)")
ax4.set_xlabel("infl_trend_5y (%)")
ax4.set_ylabel("5-year ahead excess returns (%)")
ax4.grid(True)

# Infl_yoy against target  (ensure column name matches your df exactly)
ax5.scatter(df["infl_vol_12m"] * 100, df["5yr_excess_return"])
ax5.set_title("infl_vol_12m vs target (1926–1994)")
ax5.set_xlabel("infl_vol_12m (%)")
ax5.set_ylabel("5-year ahead excess returns (%)")
ax5.grid(True)

# Term_spread against target
ax6.scatter(df["term_spread"], df["5yr_excess_return"])
ax6.set_title("Term_spread vs target (1926–1994)")
ax6.set_xlabel("Term_spread (%)")
ax6.set_ylabel("5-year ahead excess returns (%)")
ax6.grid(True)

plt.tight_layout()
plt.show()
