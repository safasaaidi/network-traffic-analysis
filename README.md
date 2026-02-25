# Network Traffic Forecasting (ARIMA vs SARIMA)

This is a personal data science project focused on time series analysis
and network traffic forecasting using classical statistical models.

The goal of this project is to better understand network traffic behavior
and compare ARIMA and SARIMA models for short-term forecasting.

---

## Project description

In this project, I analyze real network traffic data collected on an hourly basis.
The work follows a complete data analysis pipeline, starting from data exploration
and preprocessing, to anomaly detection and forecasting.

Two models are implemented and compared:
- ARIMA as a baseline model
- SARIMA to include seasonal effects

The final objective is to forecast network traffic for the next 24 hours
and evaluate which model performs better on this dataset.

---

## Objectives
- Analyze network traffic patterns
- Check stationarity of the time series
- Detect anomalies in traffic data
- Build and compare ARIMA and SARIMA models
- Forecast traffic for the next 24 hours

---

## Models used

### ARIMA
- Stationarity verified using Augmented Dickey-Fuller test
- Parameters selected based on ACF and PACF analysis
- Used as baseline model

### SARIMA
- Seasonal extension of ARIMA
- Daily seasonality (24 hours) included
- Compared against ARIMA using RMSE

---

## Results

- ARIMA achieved better performance on the test set
- SARIMA tended to overestimate traffic values
- The simpler ARIMA model generalized better for this dataset

This result shows that more complex models are not always better,
especially when seasonality is weak or unstable.

---

## Dataset
- Hourly network traffic data
- 6,717 observations
- Main metrics: bytes, packets, flows
- Forecasting focused on the `n_bytes` variable

---

## Technologies
- Python
- pandas, numpy
- matplotlib, seaborn
- statsmodels
- scikit-learn

---

## How to run the project

### Run locally
```bash
pip install -r requirements.txt
python scripts/network_analysis.py
```
### Run with Docker
```bash
docker build -t network-traffic-analysis .
docker run network-traffic-analysis
```
