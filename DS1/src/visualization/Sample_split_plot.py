import pandas as pd
import matplotlib.pyplot as plt

# Path
path = "/Users/kayttaja/Desktop/DS1/data/processed/model_data.csv"
target_col = "5yr_excess_return"  # already in %, per your dataset

# Date ranges
TRAIN_START = "1926-07-01"
TEST_START = "2000-01-01"
TEST_END = "2020-06-01"

# Import data
df = pd.read_csv(path, index_col="Date", parse_dates=True)
y = df[target_col]

# Compute the 5-year gap
ts_test_start = pd.to_datetime(TEST_START)
gap_end = ts_test_start - pd.DateOffset(months=60)
gap_start = gap_end

y_train = y.loc[TRAIN_START:gap_end]
y_gap = y.loc[gap_start : ts_test_start - pd.DateOffset(days=1)]
y_test = y.loc[TEST_START:TEST_END] if TEST_END else y.loc[TEST_START:]

# Plot
plt.figure(figsize=(12, 4.5))
plt.plot(y_train.index, y_train.values, label="In-sample", linewidth=2, color="#1f77b4")
plt.plot(y_gap.index, y_gap.values, label="5y gap", linewidth=2.2, color="black")
plt.plot(
    y_test.index, y_test.values, label="Out-of-sample", linewidth=2, color="#ff7f0e"
)

plt.title("5y Ahead Annualized Real Excess Return: In-sample vs Out-of-sample")
plt.xlabel("Date")
plt.ylabel("Return (%)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# (optional) save
plt.savefig(
    "/Users/kayttaja/Desktop/DS1/reports/figures/plot_train_test_split.png", dpi=200
)
plt.show()
