import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from math import sqrt

# Part 1: Data Preparation and Exploration
# 1. Data Loading
df = pd.read_csv('exchange_rate.csv', parse_dates=['date'], dayfirst=True)
df.set_index('date', inplace=True)
exchange_rate = df['Ex_rate']

# 2. Initial Exploration
plt.figure(figsize=(12, 6))
plt.plot(exchange_rate, label="Exchange Rate USD to AUD")
plt.title("USD to AUD Exchange Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.show()

# 3. Data Preprocessing
exchange_rate = exchange_rate.fillna(method='ffill')  # Forward fill any missing values

# Part 2: Model Building - ARIMA
# 1. Parameter Selection for ARIMA using ACF and PACF
plt.figure(figsize=(12, 6))
plot_acf(exchange_rate)
plot_pacf(exchange_rate)
plt.show()

# 2. Model Fitting
# Set p, d, q based on ACF and PACF plots (example values used here)
p, d, q = 1, 1, 1  
arima_model = ARIMA(exchange_rate, order=(p, d, q))
arima_fit = arima_model.fit()

# 3. Diagnostics
plt.figure(figsize=(12, 6))
plt.plot(arima_fit.resid)
plt.title("Residuals of ARIMA Model")
plt.show()

# 4. Forecasting
arima_forecast = arima_fit.forecast(steps=30)
plt.figure(figsize=(12, 6))
plt.plot(exchange_rate, label="Actual")
plt.plot(arima_forecast, label="ARIMA Forecast", color='orange')
plt.title("ARIMA Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.show()

# Part 3: Model Building - Exponential Smoothing
# 1. Model Selection and Fitting
exp_smooth_model = ExponentialSmoothing(exchange_rate, trend='add', seasonal=None)
exp_smooth_fit = exp_smooth_model.fit()

# 2. Forecasting
exp_smooth_forecast = exp_smooth_fit.forecast(steps=30)
plt.figure(figsize=(12, 6))
plt.plot(exchange_rate, label="Actual")
plt.plot(exp_smooth_forecast, label="Exponential Smoothing Forecast", color='green')
plt.title("Exponential Smoothing Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.show()

# Part 4: Evaluation and Comparison
# Compute Error Metrics
arima_mae = mean_absolute_error(exchange_rate[-30:], arima_forecast)
arima_rmse = sqrt(mean_squared_error(exchange_rate[-30:], arima_forecast))
arima_mape = mean_absolute_percentage_error(exchange_rate[-30:], arima_forecast)

exp_mae = mean_absolute_error(exchange_rate[-30:], exp_smooth_forecast)
exp_rmse = sqrt(mean_squared_error(exchange_rate[-30:], exp_smooth_forecast))
exp_mape = mean_absolute_percentage_error(exchange_rate[-30:], exp_smooth_forecast)

print("ARIMA - MAE:", arima_mae, "RMSE:", arima_rmse, "MAPE:", arima_mape)
print("Exponential Smoothing - MAE:", exp_mae, "RMSE:", exp_rmse, "MAPE:", exp_mape)

# Conclusion: Comparing error metrics and summarizing model performance
if arima_mape < exp_mape:
    print("ARIMA model performed better.")
else:
    print("Exponential Smoothing model performed better.")
