# NETWORK TRAFFIC TIME SERIES ANALYSIS (CESNET DATASET)

# 0. LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler



# 1. LOAD DATA
# Load CESNET dataset (1-hour interval)
df = pd.read_csv("data//cesnet_1h.csv")

# 2. BASIC PREPROCESSING
# Sort by time index and reset index
df = df.sort_values("id_time")
df.reset_index(drop=True, inplace=True)

# 3. EXPLORATORY DATA ANALYSIS
# Network traffic evolution (bytes)
plt.figure(figsize=(12,6))
plt.plot(df['n_bytes'], color='#1f77b4', linewidth=2.5, alpha=0.9,
         label='Network Traffic (Bytes)')
plt.title("Network Traffic Evolution Over Time",
          fontsize=16, fontweight='bold', color='#444444')
plt.xlabel("Time (Hours)", fontsize=13, color='#555555')
plt.ylabel("Number of Bytes", fontsize=13, color='#333333')
plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("figures/traffic_evolution.png", dpi=300, bbox_inches='tight')
plt.close()

# Multi-metric visualization (normalized)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['n_bytes', 'n_packets', 'n_flows']])
df_scaled = pd.DataFrame(scaled,
                         columns=['n_bytes', 'n_packets', 'n_flows'])

subset = df_scaled.iloc[:600]  # zoom on first 600 hours
plt.figure(figsize=(12,6))
plt.plot(subset['n_bytes'], label='Bytes (scaled)', linewidth=3)
plt.plot(subset['n_packets'], label='Packets (scaled)')
plt.plot(subset['n_flows'], label='Flows (scaled)')
plt.title("Normalized Network Metrics Comparison",
          fontsize=16, fontweight='bold')
plt.xlabel("Time (Hours)", fontsize=13)
plt.ylabel("Normalized Value", fontsize=13)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("figures/multi_metric_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 4. CORRELATION ANALYSIS
cols = ['n_bytes', 'n_packets', 'n_flows']
corr = df[cols].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Between Network Metrics")
plt.tight_layout()
plt.savefig("figures/correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. STATIONARITY TEST (ADF)
# Augmented Dickey-Fuller test
adf_result = adfuller(df['n_bytes'])
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:")
for key, value in adf_result[4].items():
    print(f"   {key}: {value}")

# 6. ANOMALY DETECTION (Z-score)
metric = df['n_bytes']
mean_val = metric.mean()
std_val = metric.std()
z_scores = (metric - mean_val) / std_val

threshold = 3
anomalies = np.where(np.abs(z_scores) > threshold)[0]

print("Mean:", mean_val)
print("Std Dev:", std_val)
print("Number of anomalies detected:", len(anomalies))

plt.figure(figsize=(14,6))
plt.plot(metric, label='Network Traffic', color='#1f77b4')
plt.scatter(anomalies, metric.iloc[anomalies],
            color='red', label='Anomalies', s=40)
plt.title("Anomaly Detection Using Z-Score")
plt.xlabel("Time (Hours)")
plt.ylabel("Bytes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/anomaly_detection.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. TRAIN / TEST SPLIT
train_size = int(len(df) * 0.8)
train = df['n_bytes'][:train_size]
test = df['n_bytes'][train_size:]

# 8. ARIMA MODEL
arima_model = ARIMA(train, order=(1,1,1))
arima_results = arima_model.fit()
arima_pred = arima_results.predict(start=test.index[0],
                                   end=test.index[-1])
arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))
print("ARIMA RMSE:", arima_rmse)

# 9. SARIMA MODEL
sarima_model = SARIMAX(train,
                       order=(1,1,1),
                       seasonal_order=(1,1,1,24))
sarima_results = sarima_model.fit(disp=False)
sarima_pred = sarima_results.predict(start=test.index[0],
                                     end=test.index[-1])
sarima_rmse = np.sqrt(mean_squared_error(test, sarima_pred))
print("SARIMA RMSE:", sarima_rmse)

# 10. MODEL COMPARISON
plt.figure(figsize=(12,5))
plt.plot(test, label="Test Data", color="green")
plt.plot(arima_pred, label="ARIMA Prediction", color="black")
plt.plot(sarima_pred, label="SARIMA Prediction", color="red")
plt.title("ARIMA vs SARIMA – Test Set Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# 11a. FINAL FORECAST (FULL SERIES)
final_model = SARIMAX(df['n_bytes'],
                      order=(1,1,1),
                      seasonal_order=(1,1,1,24))
final_results = final_model.fit(disp=False)
forecast = final_results.forecast(steps=24)

plt.figure(figsize=(12,5))
plt.plot(df['n_bytes'], label="Observed Data")
plt.plot(range(len(df), len(df)+24),
         forecast, label="24-Hour Forecast", color="red")
plt.title("Final Traffic Forecast (Next 24 Hours)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/24h_forecast_full.png", dpi=300, bbox_inches='tight')
plt.close()

# 11b. FINAL FORECAST (ZOOMED VIEW)
final_model = ARIMA(df['n_bytes'], order=(1,1,1))
final_results = final_model.fit()
forecast = final_results.forecast(steps=24)

zoom_last = 50
start_zoom = len(df) - zoom_last

plt.figure(figsize=(12,6))
plt.plot(range(start_zoom, len(df)),
         df['n_bytes'][start_zoom:],
         label="Observed (Last 50 Hours)", linewidth=2)
plt.plot(range(len(df), len(df)+24),
         forecast,
         label="24-Hour Forecast",
         linestyle="--", linewidth=3)
plt.title("Final Forecast: Last 50 Hours + 24-Hour Prediction")
plt.xlabel("Hour Index")
plt.ylabel("Bytes")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("figures/24h_forecast_zoom.png", dpi=300, bbox_inches='tight')
plt.close()
