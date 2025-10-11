import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Import data
path = "/Users/kayttaja/Desktop/DS1/data/processed/model_data.csv"
df = pd.read_csv(path, index_col="Date", parse_dates=True)


# Feature and target
X = ["ECY7", "term_spread", "infl_vol_12m"]
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

# Initialize lists to store CV results
cv_train_r2_list = []
cv_valid_r2_list = []
cv_train_rmse_list = []
cv_valid_rmse_list = []

# Scoring metrics
scorers = {
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
}

# Time series split for hyperparameter tuning
tscv = TimeSeriesSplit(n_splits=7, test_size=48, gap=60)

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

    # Set the hyperparameter grid for rfecv and random forest
    param_grid = {
        # For random forests
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [5, 10, 20],
        "max_features": ["sqrt", 0.5],
    }

    # Initiate the gridsearch for optimal features and random forest parameters
    gs = GridSearchCV(
        estimator=RandomForestRegressor(random_state=1, n_jobs=-1),
        param_grid=param_grid,
        scoring=scorers,
        refit="r2",
        return_train_score=True,
        cv=tscv,
        n_jobs=-1,
    )

    gs.fit(X_train, y_train)

    # Extract and store CV train/valid R2
    res = pd.DataFrame(gs.cv_results_)
    best_row = res.loc[res["rank_test_r2"].idxmin()]

    # Store CV results
    cv_train_r2_list.append(float(best_row["mean_train_r2"]))
    cv_valid_r2_list.append(float(best_row["mean_test_r2"]))
    cv_train_rmse_list.append(float(-best_row["mean_train_rmse"]))
    cv_valid_rmse_list.append(float(-best_row["mean_test_rmse"]))

    best_rf = gs.best_estimator_
    best_params.append(gs.best_params_)

    # Predict
    y_pred = float(best_rf.predict(X_test)[0])

    # Record
    predictions.append(y_pred)
    actuals.append(float(y_test))
    dates.append(pd.Timestamp(date))
    rmse_list.append(np.sqrt(mean_squared_error([y_test], [y_pred])))
    print(f"Date: {date}, RMSE: {np.sqrt(mean_squared_error([y_test], [y_pred]))}")


# Save the results
results_df = pd.DataFrame(
    {
        "Date": dates,
        "Actual": actuals,
        "Predicted": predictions,
        "RMSE": rmse_list,
        "Best Parameters": best_params,
        "CV_Train_R2": cv_train_r2_list,
        "CV_Valid_R2": cv_valid_r2_list,
        "CV_Train_RMSE": cv_train_rmse_list,
        "CV_Valid_RMSE": cv_valid_rmse_list,
    }
).set_index("Date")

# Save results
results_df.to_csv("/Users/kayttaja/Desktop/DS1/reports/results_rf.csv", index=True)

# Import results
results_df = pd.read_csv(
    "/Users/kayttaja/Desktop/DS1/reports/results_rf.csv",
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

# Compute the mean CV R2 and RMSE
mean_cv_train_r2 = np.mean(results_df["CV_Train_R2"])
mean_cv_valid_r2 = np.mean(results_df["CV_Valid_R2"])
mean_cv_valid_rmse = np.mean(results_df["CV_Valid_RMSE"])
mean_cv_train_rmse = np.mean(results_df["CV_Train_RMSE"])
print(f"Mean CV Train R²: {mean_cv_train_r2}")
print(f"Mean CV Valid R²: {mean_cv_valid_r2}")
print(f"Mean CV Train RMSE: {mean_cv_train_rmse}")
print(f"Mean CV Valid RMSE: {mean_cv_valid_rmse}")

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

# Plot the feature importances
importances = best_rf.feature_importances_
feature_names = X
indices = np.argsort(importances)
plt.figure(figsize=(8, 4))
plt.title("Feature Importances in Random Forest")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.show()

# Plot the chosen hyperparameters as subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

# Plot max_depth selection frequency
max_depth_values = [params["max_depth"] for params in best_params]
max_depth_counts = pd.Series(max_depth_values).value_counts().sort_index()
axes[0].bar(max_depth_counts.index, max_depth_counts.values, color="orange")
axes[0].set_title("Max Depth Selection Frequency in Random Forest")
axes[0].set_xlabel("Max Depth")
axes[0].set_ylabel("Frequency")
axes[0].grid(axis="y", alpha=0.3)

# Plot min_samples_leaf selection frequency
min_samples_values = [params["min_samples_leaf"] for params in best_params]
min_samples_counts = pd.Series(min_samples_values).value_counts().sort_index()
axes[1].bar(min_samples_counts.index, min_samples_counts.values, color="orange")
axes[1].set_title("Min Samples Leaf Selection Frequency in Random Forest")
axes[1].set_xlabel("Min Samples Leaf")
axes[1].set_ylabel("Frequency")
axes[1].grid(axis="y", alpha=0.3)

# Plot max_features selection frequency
max_features_values = [str(params["max_features"]) for params in best_params]
max_features_counts = pd.Series(max_features_values).value_counts().sort_index()
axes[2].bar(max_features_counts.index, max_features_counts.values, color="orange")
axes[2].set_title("Max Features Selection Frequency in Random Forest")
axes[2].set_xlabel("Max Features")
axes[2].set_ylabel("Frequency")
axes[2].grid(axis="y", alpha=0.3)

plt.show()
