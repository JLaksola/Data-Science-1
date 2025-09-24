import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Import data
path = "/Users/kayttaja/Desktop/DS1/data/processed/model_data.csv"
df = pd.read_csv(path, index_col="Date", parse_dates=True)


# Feature and target
X = ["ECY7", "term_spread", "infl_vol_12m", "infl_trend_5y"]
y = "5yr_excess_return"

# Set the training and testing periods
TRAIN_START = "1926-07-01"
TEST_START = "2000-01-01"
TEST_END = "2020-06-01"
# Inspect train and test set sizes
print(len(df.loc[TRAIN_START:"1999-12-01"]))
print(len(df.loc[TEST_START:TEST_END]))


# Recursive Training Initialization
rmse_list = []
predictions = []
actuals = []
dates = []
features = []
best_params = []

alphas = [0.0001, 0.001, 0.01, 0.1, 1]  # Range of alphas to test
# Time series split for feature selection
RFECV_split = TimeSeriesSplit(n_splits=5, test_size=48, gap=60)
# Time series split for hyperparameter tuning
GridSearch_split = TimeSeriesSplit(n_splits=7, test_size=48, gap=60)

# Rolling Forecast Loop
for date in pd.date_range(TEST_START, TEST_END, freq="MS"):  # Monthly rolling forecast
    train_end = (pd.Timestamp(date) - pd.DateOffset(years=5, months=1)).strftime(
        "%Y-%m-%d"
    )
    train = df.loc[TRAIN_START:train_end]  # Use only past data
    test_sample = df.loc[date:date]  # Predict one step ahead

    # Split Features and Target
    X_train, y_train = train[X], train[y]
    X_test, y_test = test_sample[X], test_sample[y]

    # Initiate the pipeline
    rfecv = RFECV(
        estimator=Lasso(),
        step=1,
        min_features_to_select=2,
        scoring="r2",
        cv=RFECV_split,
        n_jobs=-1,
    )

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("rfecv", rfecv),
        ]
    )

    # Initiate grid search
    param_grid = {
        "rfecv__estimator__alpha": alphas,
        "rfecv__min_features_to_select": [2, 3],
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="r2",
        cv=GridSearch_split,
        n_jobs=-1,
    )

    gs.fit(X_train, y_train)

    best_pipe = gs.best_estimator_
    mask = best_pipe.named_steps["rfecv"].support_
    kept = [c for c, m in zip(X, mask) if m]
    features.append(kept)
    best_params.append(gs.best_params_)

    # predict one step
    y_pred = float(best_pipe.predict(X_test)[0])

    # record
    predictions.append(y_pred)
    actuals.append(float(y_test))
    dates.append(pd.Timestamp(date))
    rmse_list.append(np.sqrt(mean_squared_error([y_test], [y_pred])))
    print(f"Date: {date}, RMSE: {np.sqrt(mean_squared_error([y_test], [y_pred]))}")

# Convert Results to DataFrame
results_df = pd.DataFrame(
    {
        "Date": dates,
        "Actual": actuals,
        "Predicted": predictions,
        "RMSE": rmse_list,
        "Best Parameters": best_params,
        "Features": features,
    }
)
results_df.set_index("Date", inplace=True)
print(best_params[-5:])
print(features[-5:])

# Save results
results_df.to_csv("/Users/kayttaja/Desktop/DS1/reports/results_lasso.csv", index=True)

# Import results
results_df = pd.read_csv(
    "/Users/kayttaja/Desktop/DS1/reports/results_lasso.csv",
    index_col="Date",
    parse_dates=True,
)

# Compute RMSE
y_true = results_df["Actual"].to_numpy()
y_pred = results_df["Predicted"].to_numpy()
oos_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
oos_r2 = r2_score(y_true, y_pred)
print(f"OOS RMSE: {oos_rmse:.4f}")
print(f"OOS R^2: {oos_r2:.4f}")

# Plot actuals vs predicted
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(results_df.index, results_df["Actual"], label="Actual", linewidth=1.5)
ax.plot(results_df.index, results_df["Predicted"], label="Predicted", linewidth=1.5)
ax.set_title("5y Ahead Annualized Real Excess Return: Actual vs Predicted")
ax.set_xlabel("Date")
ax.set_ylabel("Return (%)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
