
# OPSD PowerDesk: Day-Ahead Load Forecasting

A day-ahead (24-step) electric load forecasting system using OPSD data for Germany (DE), France (FR), and Spain (ES). Includes classical and neural forecasting, anomaly detection, live simulation, and a simple dashboard.

---
## Installation

1.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Setup**:
    Place your Open Power System Data (OPSD) CSV file inside the `data/` directory.

-----

## Usage
### **Run like so**
```bash
streamlit run src/dashboard_app.py
```

-----

## Features

  * **Classical Forecasting**: Implementation of SARIMA/SARIMAX models.
  * **Neural Forecasting**: GRU network architecture (168-hour sliding window input → 24-hour output).
  * **Anomaly Detection**: Combination of statistical residual analysis and ML-based detection.
  * **Live Simulation**: Simulates data ingestion and performs online model adaptation.
  * **Visualization**: A clean Streamlit dashboard for monitoring forecasts and anomalies.

-----

