# 🚀 Advanced Stock Prediction & Forecasting

A Streamlit-powered ML application that compares **9 machine learning models** for stock price prediction and generates n-day future forecasts.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)

## Features

- **9 ML Models** trained and compared side-by-side:
  - Linear Regression, Ridge, Lasso
  - Decision Tree, Random Forest, Gradient Boosting, XGBoost
  - SVR, KNN Regressor
- **Train / Test Split Visualization** — color-coded chart showing training vs testing data
- **Performance Metrics Table** — MAE, MSE, RMSE, R², MAPE for every model (green = best, red = worst)
- **Actual vs Predicted Chart** — all model predictions overlaid on actual prices
- **R² Bar Chart** — instant visual accuracy comparison
- **N-Day Forecast** — recursive future price forecast from all models
- **Technical Indicators** — SMA, RSI, MACD, ATR, lagged features

## Project Structure

```
ml forecast/
├── app.py                  # Streamlit dashboard
├── data_loader.py          # Yahoo Finance data fetching
├── feature_engineering.py  # Technical indicators & feature creation
├── model_trainer.py        # Model training, evaluation & forecasting
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/krishjariwala11/Multimodal-ML-Stock-Forecasting-App.git
   cd Multimodal-ML-Stock-Forecasting-App
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Enter a ticker symbol (e.g. `AAPL`, `TSLA`, `RELIANCE.NS`), configure dates and forecast horizon, then click **Run Detailed Analysis**.

## Screenshots

After running the analysis you will see:

| Section | Description |
|---------|-------------|
| Historical Price Trend | Full price chart for the selected period |
| Train / Test Split | Blue (training) vs Red (test) with split marker |
| Performance Metrics | Table comparing all 9 models |
| Actual vs Predicted | Overlay chart of all model predictions |
| R² Bar Chart | Visual accuracy comparison |
| N-Day Forecast | Future price projections from each model |

## Tech Stack

- **Data**: [yfinance](https://github.com/ranaroussi/yfinance), [pandas_ta](https://github.com/twopirllc/pandas-ta)
- **ML**: scikit-learn, XGBoost
- **Visualization**: Plotly
- **UI**: Streamlit

## License

This project is open source and available under the [MIT License](LICENSE).
