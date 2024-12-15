# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:32:52 2024

@author: VEDANT SHINDE
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate 365 days of synthetic cash flow data
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]
cash_flows = np.random.uniform(1000, 5000, size=365)  # Random values between 1000 and 5000

# Create DataFrame
data = pd.DataFrame({
    "Date": dates,
    "Daily Cash Flow": cash_flows
})

# Save to CSV
data.to_csv("preprocessed_cash_flow.csv", index=False)
print("Synthetic dataset created: preprocessed_cash_flow.csv")


# Load the preprocessed data
data = pd.read_csv("preprocessed_cash_flow.csv")
data["Date"] = pd.to_datetime(data["Date"])

# Use only the Daily Cash Flow column for forecasting
cash_flow = data[["Daily Cash Flow"]].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
cash_flow_scaled = scaler.fit_transform(cash_flow)

# Create sequences for the model
sequence_length = 30  # Use the past 30 days to predict the next day
X, y = [], []
for i in range(sequence_length, len(cash_flow_scaled)):
    X.append(cash_flow_scaled[i-sequence_length:i, 0])
    y.append(cash_flow_scaled[i, 0])
X, y = np.array(X), np.array(y)

# Split into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Flatten X for RandomForest input
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train_flat, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Train the best model
best_model.fit(X_train_flat, y_train)

# Predict on the test set
predicted_cash_flow = best_model.predict(X_test_flat)
predicted_cash_flow = scaler.inverse_transform(predicted_cash_flow.reshape(-1, 1))

# Invert scaling for actual test data
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate and display mean squared error
mse = mean_squared_error(y_test_actual, predicted_cash_flow)
print(f"Mean Squared Error: {mse}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, color='blue', label='Actual Cash Flow')
plt.plot(predicted_cash_flow, color='red', label='Predicted Cash Flow')
plt.title('Cash Flow Prediction')
plt.xlabel('Time')
plt.ylabel('Cash Flow')
plt.legend()
plt.show()

# Feature importance visualization
feature_importances = best_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances)
plt.title('Feature Importance')
plt.xlabel('Feature (Day Lag)')
plt.ylabel('Importance')
plt.show()

print("Model training, predictions, and feature importance visualization completed.")

