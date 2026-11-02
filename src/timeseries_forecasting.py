"""
Triage AI - Time Series Forecasting (GitHub Issue #3)
Forecasts weekly emergency department patient admissions to optimize 
staff scheduling and bed allocation.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

print("--- 1. Generating Historical Hospital Admission Time-Series ---")
# 104 weeks (2 years) of hospital admission records
np.random.seed(42)
date_range = pd.date_range(start="2024-01-01", periods=104, freq="W")

# Base demand + weekly seasonality + winter surge + random noise
trend = np.linspace(120, 160, 104)
seasonality = 25 * np.sin(np.linspace(0, 4 * np.pi, 104))
noise = np.random.normal(0, 8, 104)
weekly_admissions = np.round(trend + seasonality + noise).astype(int)

ts_df = pd.DataFrame({"Admissions": weekly_admissions}, index=date_range)
print(f"Dataset: {len(ts_df)} weeks of historical admission records.")
print(ts_df.head())

print("\n--- 2. Fitting SARIMAX Forecasting Model ---")
# Train on first 90 weeks, test on last 14 weeks
train = ts_df.iloc[:90]
test = ts_df.iloc[90:]

# SARIMAX(1, 1, 1) configuration
model = SARIMAX(train["Admissions"], order=(1, 1, 1), seasonal_order=(1, 0, 0, 12))
results = model.fit(disp=False)

# Forecast the next 14 weeks
forecast = results.get_forecast(steps=14)
forecast_mean = np.round(forecast.predicted_mean).astype(int)
conf_int = np.round(forecast.conf_int()).astype(int)

# Evaluation (Mean Absolute Error)
mae = np.mean(np.abs(test["Admissions"].values - forecast_mean.values))
print(f"Model Mean Absolute Error (MAE): {mae:.2f} patients/week")

print("\n--- 3. 14-Week Projected Patient Surge Forecast ---")
comparison = pd.DataFrame({
    "Actual Admissions": test["Admissions"],
    "Forecasted Admissions": forecast_mean,
    "Lower Bound (95%)": conf_int.iloc[:, 0],
    "Upper Bound (95%)": conf_int.iloc[:, 1]
})
print(comparison.head(7))
print("\n[SUCCESS] Time Series Forecasting module executed successfully.")