import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# Time split
TRAIN_START = "1926-07-01"
TEST_START = "2000-01-01"
GAP_MONTHS = 60  # 5-year embargo
VAL_SIZE = 48  # months in each validation fold
N_SPLITS = 7

# In-sample ends 60 months before TEST_START
in_sample_end = (
    pd.to_datetime(TEST_START) - pd.DateOffset(months=GAP_MONTHS)
).strftime("%Y-%m-%d")

# Monthly date index over the in-sample era (only need dates; data not required)
dates = pd.date_range(start=TRAIN_START, end=in_sample_end, freq="MS")
n_dates = len(dates)

# Time series split
tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=VAL_SIZE, gap=GAP_MONTHS)

# Plot
fig, ax = plt.subplots(figsize=(12, 4.5))

for fold, (train_idx, val_idx) in enumerate(tscv.split(np.arange(n_dates)), start=1):
    y = fold - 0.5  # vertical position for this fold

    # Train (blue squares)
    ax.scatter(
        dates[train_idx],
        np.full_like(train_idx, y, dtype=float),
        color="royalblue",
        marker="s",
        s=10,
        label="Train" if fold == 1 else "",
    )

    # Gap (black squares): indices between last train and first val
    gap_start = train_idx[-1] + 1
    gap_end = val_idx[0] - 1
    if gap_end >= gap_start:
        gap_idx = np.arange(gap_start, gap_end + 1)
        ax.scatter(
            dates[gap_idx],
            np.full_like(gap_idx, y, dtype=float),
            color="black",
            marker="s",
            s=10,
            label="Gap (5 yr)" if fold == 1 else "",
        )

    # Validate (orange squares)
    ax.scatter(
        dates[val_idx],
        np.full_like(val_idx, y, dtype=float),
        color="darkorange",
        marker="s",
        s=10,
        label="Validate" if fold == 1 else "",
    )

# Cosmetics
ax.set_yticks(np.arange(0.5, N_SPLITS + 0.5))
ax.set_yticklabels([f"Fold {i}" for i in range(1, N_SPLITS + 1)])
ax.set_xlabel("Date")
ax.set_title("Time Series Split Cross-Validation (1926-1994)")
ax.legend(loc="lower right")
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/kayttaja/Desktop/DS1/reports/figures/CV_split.png", dpi=200)
plt.show()
