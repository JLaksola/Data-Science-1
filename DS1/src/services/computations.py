import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import the datasets
path1 = "/Users/kayttaja/Desktop/DS1/data/raw/F-F_Research_Data_Factors.csv"
path2 = "/Users/kayttaja/Desktop/DS1/data/raw/ie_data-2.xls"
path3 = "/Users/kayttaja/Desktop/DS1/data/raw/AAA-2.csv"
path4 = "/Users/kayttaja/Desktop/DS1/data/raw/BAA-2.csv"
ff_factors = pd.read_csv(path1)
shiller_data = pd.read_excel(path2, engine="calamine", skiprows=7)
shiller_data.drop(columns=["Unnamed: 13", "Unnamed: 15"], inplace=True)
AAA = pd.read_csv(path3)
BAA = pd.read_csv(path4)

print(ff_factors.head())
print(shiller_data.head())
print(AAA.head())
print(BAA.head())

# Let's make the date columns datetime objects
ff_factors["Date"] = pd.to_datetime(ff_factors["Date"], format="%Y%m")
shiller_data["Date"] = (shiller_data["Date"].astype(float) * 100).astype(
    int
)  # Convert to integer format YYYYMM
shiller_data["Date"] = shiller_data["Date"].astype(
    str
)  # Convert to string format YYYYMM
print(shiller_data["Date"].head())
shiller_data["Date"] = pd.to_datetime(shiller_data["Date"], format="%Y%m")
AAA["observation_date"] = pd.to_datetime(AAA["observation_date"])
BAA["observation_date"] = pd.to_datetime(BAA["observation_date"])
# Set the date columns as index
ff_factors.set_index("Date", inplace=True)
shiller_data.set_index("Date", inplace=True)
AAA.set_index("observation_date", inplace=True)
BAA.set_index("observation_date", inplace=True)

# Compute the target variable: 5-year-ahead excess return
# Let's first compute the 5-year-ahead return
shiller_data["5yr_annualized_return"] = (
    (
        shiller_data["Real Total Returns Price"].shift(-60)
        / shiller_data["Real Total Returns Price"]
    )
    ** (1 / 5)
    - 1
) * 100
# Let's compute the compounded risk-free rate over the next 5 years
# Let's compute an inflation factor for the next 5 years
shiller_data["5yr_inflation_factor"] = (
    shiller_data["CPI"].shift(-60) / shiller_data["CPI"]
)
# Merge the risk-free rate from ff_factors into shiller_data
shiller_data = shiller_data.merge(
    ff_factors["RF"] / 100, left_index=True, right_index=True, how="left"
)
# Compute the 5-year annualized risk-free rate
shiller_data["5yr_annualized_RF"] = (
    (
        (1 + shiller_data["RF"])
        .shift(-1)
        .rolling(60, min_periods=60)
        .apply(np.prod, raw=True)
        .shift(-59)
        / shiller_data["5yr_inflation_factor"]
    )
    ** (1 / 5)
    - 1
) * 100
# Finally, compute the 5-year-ahead excess return
shiller_data["5yr_excess_return"] = (
    shiller_data["5yr_annualized_return"] - shiller_data["5yr_annualized_RF"]
)


# Let's merge the credit spreads into shiller_data
shiller_data = shiller_data.merge(
    AAA["AAA"], left_index=True, right_index=True, how="left"
)
shiller_data = shiller_data.merge(
    BAA["BAA"], left_index=True, right_index=True, how="left"
)


# Let's then compute the explanatory variables
# Rename the CAPE column to CAPE10
shiller_data.rename({"CAPE": "CAPE10"}, axis=1, inplace=True)
# Compute CAPE with 3-year moving average earnings as the denominator
shiller_data["CAPE3"] = (
    shiller_data["Real Price"]
    / shiller_data["Real Earnings"].rolling(window=36, min_periods=36).mean()
)
# Compute CAPE with 5-year moving average earnings as the denominator
shiller_data["CAPE5"] = (
    shiller_data["Real Price"]
    / shiller_data["Real Earnings"].rolling(window=60, min_periods=60).mean()
)
# Compute CAPE with 7-year moving average earnings as the denominator
shiller_data["CAPE7"] = (
    shiller_data["Real Price"]
    / shiller_data["Real Earnings"].rolling(window=84, min_periods=84).mean()
)
# Compute DFY (default yield spread between BAA and AAA)
shiller_data["DFY"] = shiller_data["BAA"] - shiller_data["AAA"]

# Let's also make other computations from the default yield spread
# 3-month and 12-month changes in DFY
shiller_data["DFY_delta_3m"] = shiller_data["DFY"] - shiller_data["DFY"].shift(3)
shiller_data["DFY_delta_12m"] = shiller_data["DFY"] - shiller_data["DFY"].shift(12)
# Smoothing (if you want a slow-moving level)
shiller_data["DFY_ma_12m"] = shiller_data["DFY"].rolling(12, min_periods=12).mean()
# Volatility of the spread (credit uncertainty)
shiller_data["DFY_vol_12m"] = (
    shiller_data["DFY"].diff().rolling(12, min_periods=12).std()
)


# Compute earnings yields (should work better for linear models)
for col in ["CAPE3", "CAPE5", "CAPE7", "CAPE10"]:
    shiller_data[f"EY_{col}"] = 1.0 / shiller_data[col] * 100

# Let's compute also different excess CAPE yields with different earnings yields
# 10y geometric-mean inflation in %
shiller_data["infl_10y_pct"] = (
    (shiller_data["CPI"] / shiller_data["CPI"].shift(120)) ** (1 / 10) - 1
) * 100
# ECY = earnings yield (%) − real 10y yield (%)
# real 10y = GS10 (%) − infl_10y (%)
realGS10 = shiller_data["GS10"] - shiller_data["infl_10y_pct"]
shiller_data["ECY3"] = shiller_data["EY_CAPE3"] - realGS10
shiller_data["ECY5"] = shiller_data["EY_CAPE5"] - realGS10
shiller_data["ECY7"] = shiller_data["EY_CAPE7"] - realGS10
shiller_data.rename({"ECY": "ECY10"}, axis=1, inplace=True)

# Monthly inflation rate and YoY CPI inflation
shiller_data["infl_1m"] = shiller_data["CPI"].pct_change()
shiller_data["infl_yoy"] = shiller_data["CPI"].pct_change(12)
# Trend inflation proxy: 5y moving average of YoY
shiller_data["infl_trend_5y"] = (
    shiller_data["infl_yoy"].rolling(60, min_periods=60).mean()
)
# Inflation gap (Phillips curve-style): YoY minus its 5y trend
shiller_data["infl_gap"] = (
    shiller_data["infl_yoy"] - shiller_data["infl_trend_5y"]
) * 100
# Inflation volatility (realized): 12m std of monthly inflation
shiller_data["infl_vol_12m"] = shiller_data["infl_1m"].rolling(12, min_periods=12).std()

# annualized cash in %, then real short rate
shiller_data["rf_ann_pct"] = ((1 + shiller_data["RF"]) ** 12 - 1) * 100
shiller_data["term_spread"] = shiller_data["GS10"] - shiller_data["rf_ann_pct"]

# Let's inspect the columns of shiller_data
print(shiller_data.columns)


# Let's make a correlation matrix and heatmap to inspect the relationships between the variables
# Let's first select the relevant columns
corr_matrix_cols = [
    "5yr_excess_return",
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
    "ECY7",
    "ECY10",
    "DFY",
    "DFY_delta_12m",
    "DFY_ma_12m",
    "DFY_vol_12m",
    "infl_yoy",
    "infl_trend_5y",
    "infl_gap",
    "infl_vol_12m",
    "term_spread",
]
corr_matrix = shiller_data[corr_matrix_cols].loc["1926-07-01":"1994-12-01"].corr()
# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Computed Variables")
plt.show()


# Let's further cut out some variables based on the correlation matrix
relevant_vars = [
    "5yr_excess_return",
    "ECY7",
    "DFY_vol_12m",
    "infl_gap",
    "term_spread",
]
corr_matrix2 = shiller_data[relevant_vars].loc["1926-07-01":"1994-12-01"].corr()
# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix2, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Chosen Variables")
plt.show()

# Final model data
model_data = shiller_data[relevant_vars].dropna()
print(model_data.head())
print(model_data.tail())

# Save also the full dataset with all computed variables for further analysis
shiller_data.to_csv("/Users/kayttaja/Desktop/DS1/data/processed/full_data.csv")

# Save the final model data to a CSV file
model_data.to_csv("/Users/kayttaja/Desktop/DS1/data/processed/model_data.csv")
