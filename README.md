#  Network Traffic Forecasting (ARIMA vs SARIMA)

Personal data science project focused on time series analysis and
network traffic forecasting using classical statistical models.

---

##  Objectives
- Analyze network traffic behavior
- Detect anomalies
- Compare ARIMA and SARIMA models
- Forecast traffic for the next 24 hours

---

##  Models Used
- ARIMA (baseline)
- SARIMA (seasonal extension)

---

##  Key Results
- ARIMA showed better generalization on the test set
- SARIMA tended to overestimate traffic values
- Simpler models proved more robust for this dataset

---

## Technologies
Python, pandas, numpy, matplotlib, seaborn, statsmodels, scikit-learn

---

##  How to Run
```bash
pip install -r requirements.txt
python main.py
