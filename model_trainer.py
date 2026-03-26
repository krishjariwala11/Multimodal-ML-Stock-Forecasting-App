import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=1):
        super(PyTorchLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :]) # Take last time step
        out = self.fc(out)
        return out

def create_lstm_sequences(data_features, data_target, window=10):
    X, y = [], []
    for i in range(len(data_features) - window):
        X.append(data_features[i:i + window])
        y.append(data_target[i + window])
    return np.array(X), np.array(y)

def train_and_evaluate(df_feat, feature_cols):
    """
    Trains multiple models predicting NEXT DAY RETURNS to prevent data leakage.
    Back-calculates to absolute prices for visual evaluation.
    Also trains ARIMA and LSTM.
    """
    X = df_feat[feature_cols]
    y = df_feat['Target'] # Next day return
    prices = df_feat['Close'] 
    
    # Split data (time-series preserving)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Base prices to back-calculate predictions
    prices_train = prices.loc[X_train.index]
    prices_test = prices.loc[X_test.index]
    
    # True target prices for evaluation
    true_prices_test = prices_test * (1 + y_test)
    
    # Scale data for SVR / KNN / LSTM
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    models = {
        "Linear Regression": {"model": LinearRegression(), "scaled": False},
        "Ridge": {"model": Ridge(alpha=1.0), "scaled": False},
        "Lasso": {"model": Lasso(alpha=0.001, max_iter=10000), "scaled": False},
        "Decision Tree": {"model": DecisionTreeRegressor(max_depth=5, random_state=42), "scaled": False},
        "Random Forest": {"model": RandomForestRegressor(n_estimators=50, random_state=42), "scaled": False},
        "Gradient Boosting": {"model": GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, random_state=42), "scaled": False},
        "XGBoost": {"model": XGBRegressor(n_estimators=50, learning_rate=0.05, random_state=42), "scaled": False},
        "SVR": {"model": SVR(kernel='rbf', C=1, gamma='scale'), "scaled": True},
        "KNN": {"model": KNeighborsRegressor(n_neighbors=5), "scaled": True},
    }

    results = {}
    predictions = {}
    fitted_models = {}

    # 1. Traditional ML Models
    for name, cfg in models.items():
        model = cfg["model"]
        use_scaled = cfg["scaled"]

        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test

        model.fit(Xtr, y_train)
        pred_returns = model.predict(Xte)
        
        # Back-calculate predicted absolute prices: P_{t+1} = P_t * (1 + R_{t+1})
        pred_prices = prices_test.values * (1 + pred_returns)
        
        predictions[name] = pd.Series(pred_prices, index=y_test.index)
        fitted_models[name] = model

    # 2. ARIMA (Univariate)
    try:
        # ARIMA modeling uses absolute prices, not returns.
        arima = ARIMA(prices_train, order=(5,1,0))
        arima_fit = arima.fit()
        # Forecast test set length
        arima_forecast = arima_fit.forecast(steps=len(y_test))
        predictions["ARIMA"] = pd.Series(arima_forecast.values, index=y_test.index)
        fitted_models["ARIMA"] = arima_fit
    except Exception as e:
        print(f"ARIMA Failed: {e}")

    # 3. LSTM (Deep Learning - PyTorch)
    try:
        lstm_window = 10
        X_lstm = scaler.transform(X) # Full scaled X
        X_seq, y_seq = create_lstm_sequences(X_lstm, y.values, window=lstm_window)
        
        split_idx = int(len(X_seq) * 0.8)
        X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
        y_seq_train, y_seq_test = y_seq[:split_idx], y_seq[split_idx:]
        
        X_seq_train_t = torch.tensor(X_seq_train, dtype=torch.float32)
        y_seq_train_t = torch.tensor(y_seq_train, dtype=torch.float32).view(-1, 1)
        X_seq_test_t = torch.tensor(X_seq_test, dtype=torch.float32)
        
        torch.manual_seed(42)
        lstm = PyTorchLSTM(input_size=X.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(lstm.parameters(), lr=0.01)
        
        lstm.train()
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = lstm(X_seq_train_t)
            loss = criterion(outputs, y_seq_train_t)
            loss.backward()
            optimizer.step()
            
        lstm.eval()
        with torch.no_grad():
            lstm_pred_returns = lstm(X_seq_test_t).numpy().flatten()
        
        # Align prices
        lstm_base_prices = prices.values[split_idx + lstm_window : split_idx + lstm_window + len(lstm_pred_returns)]
        lstm_pred_prices = lstm_base_prices * (1 + lstm_pred_returns)
        
        # Pad with NaNs so length matches y_test, aligning to the end
        full_lstm_preds = np.full(len(y_test), np.nan)
        pad_len = min(len(y_test), len(lstm_pred_prices))
        full_lstm_preds[-pad_len:] = lstm_pred_prices[-pad_len:]
        
        predictions["LSTM"] = pd.Series(full_lstm_preds, index=y_test.index)
        fitted_models["LSTM"] = lstm
    except Exception as e:
        print(f"LSTM Failed: {e}")

    # Evaluate
    for name, pred in predictions.items():
        valid_idx = pred.dropna().index
        if len(valid_idx) == 0: continue
            
        true_p = true_prices_test.loc[valid_idx]
        pred_p = pred.loc[valid_idx]
        
        mae = mean_absolute_error(true_p, pred_p)
        mse = mean_squared_error(true_p, pred_p)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_p, pred_p)
        mape = np.mean(np.abs((true_p - pred_p) / true_p)) * 100

        results[name] = {
            "MAE": round(mae, 4), "MSE": round(mse, 4),
            "RMSE": round(rmse, 4), "R2": round(r2, 4), "MAPE (%)": round(mape, 2)
        }

    return results, predictions, true_prices_test, X_test, X_train, y_train, scaler, fitted_models

def forecast_n_days(df, results, fitted_models, feature_cols, n_days, engineer_func, scaler):
    """
    Generates an n-day recursive forecast.
    """
    scaled_models = {"SVR", "KNN"}
    forecasts = {}
    
    # ARIMA direct forecast
    if "ARIMA" in fitted_models:
        arima_fit = fitted_models["ARIMA"]
        forecasts["ARIMA"] = arima_fit.forecast(steps=n_days).tolist()

    tail_data = df.tail(100).copy()

    for name, res in results.items():
        if name == "ARIMA": continue # Handled above
        
        if name not in fitted_models: continue
        model = fitted_models[name]
        
        # Reset data for each model's recursive simulation
        current_data = tail_data.copy()
        model_forecasts = []
        
        # LSTM needs a sequence history state
        if name == "LSTM":
            for _ in range(n_days):
                df_temp = engineer_func(current_data)
                if df_temp.empty: break
                
                features_last_10 = df_temp[feature_cols].tail(10)
                if len(features_last_10) < 10: break
                
                features_scaled = scaler.transform(features_last_10)
                seq = np.array([features_scaled]) # shape (1, 10, features)
                seq_t = torch.tensor(seq, dtype=torch.float32)
                
                model.eval()
                with torch.no_grad():
                    pred_return = model(seq_t).item()
                    
                last_price = current_data['Close'].iloc[-1]
                next_price = last_price * (1 + pred_return)
                model_forecasts.append(next_price)
                
                next_date = current_data.index[-1] + pd.Timedelta(days=1)
                new_row = pd.DataFrame({'Open': [next_price], 'High': [next_price], 'Low': [next_price], 'Close': [next_price], 'Volume': [0]}, index=[next_date])
                current_data = pd.concat([current_data, new_row])
                
        else:
            # Traditional ML
            for _ in range(n_days):
                df_temp = engineer_func(current_data)
                if df_temp.empty: break
                    
                features = df_temp[feature_cols].tail(1)
                if name in scaled_models:
                    features = pd.DataFrame(scaler.transform(features), index=features.index, columns=features.columns)
                    
                pred_return = model.predict(features)[0]
                last_price = current_data['Close'].iloc[-1]
                next_price = last_price * (1 + pred_return)
                model_forecasts.append(next_price)
                
                next_date = current_data.index[-1] + pd.Timedelta(days=1)
                new_row = pd.DataFrame({'Open': [next_price], 'High': [next_price], 'Low': [next_price], 'Close': [next_price], 'Volume': [0]}, index=[next_date])
                current_data = pd.concat([current_data, new_row])
                
        forecasts[name] = model_forecasts
        
    return forecasts
