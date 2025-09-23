from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np

# Load the processed model data
data = pd.read_csv(
    "/Users/kayttaja/Desktop/DS1/data/processed/full_data.csv",
    parse_dates=["Date"],
    index_col="Date",
)
print(data.head())

# Candidate features and target variable
candidates = [
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
target = "5yr_excess_return"

# Define in-sample period
IS_start = "1926-07-01"
IS_end = "1994-12-01"
IS_data = data.loc[IS_start:IS_end, candidates + [target]]
# Split into features and target
X_is = IS_data[candidates]
y_is = IS_data[target]

# Fit Random Forest model
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=5,  # keep it shallow
    min_samples_leaf=10,  # avoid overfitting
    random_state=0,
    n_jobs=-1,
).fit(X_is, y_is)

# Get feature importances
# Gini importance
imp_gini = np.array(rf.feature_importances_)
# Permutation importance
perm = permutation_importance(
    rf, X_is, y_is, n_repeats=50, random_state=1, scoring="r2"
)
imp_perm_mean = perm.importances_mean
imp_perm_std = perm.importances_std

# Combine into a DataFrame and sort by permutation importance
rank_tbl = pd.DataFrame(
    {
        "imp_perm_mean": imp_perm_mean,
        "imp_perm_std": imp_perm_std,
        "imp_gini": imp_gini,
    },
    index=candidates,
).sort_values("imp_perm_mean", ascending=False)

print(rank_tbl)  # term_spread, ECY7, infl_trend_5y/infl_vol_12m

# Save the chosen features
final_data = data[
    ["5yr_excess_return", "ECY7", "term_spread", "infl_vol_12m", "infl_trend_5y"]
].loc["1926-07-01":]

final_data.to_csv("/Users/kayttaja/Desktop/DS1/data/processed/model_data.csv")
