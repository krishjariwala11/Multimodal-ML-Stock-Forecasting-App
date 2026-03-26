import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def train_and_evaluate(X, y):
    """
    Trains multiple models and compares their performance.
    Returns results, predictions, y_test, X_test, X_train, y_train plus the split index.
    """
    # Split data (time-series preserving)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Scale data for SVR / KNN
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    models = {
        "Linear Regression": {"model": LinearRegression(), "scaled": False},
        "Ridge Regression": {"model": Ridge(alpha=1.0), "scaled": False},
        "Lasso Regression": {"model": Lasso(alpha=0.1, max_iter=10000), "scaled": False},
        "Decision Tree": {"model": DecisionTreeRegressor(max_depth=10, random_state=42), "scaled": False},
        "Random Forest": {"model": RandomForestRegressor(n_estimators=100, random_state=42), "scaled": False},
        "Gradient Boosting": {"model": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42), "scaled": False},
        "XGBoost": {"model": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42), "scaled": False},
        "SVR": {"model": SVR(kernel='rbf', C=100, gamma='scale'), "scaled": True},
        "KNN Regressor": {"model": KNeighborsRegressor(n_neighbors=5), "scaled": True},
    }

    results = {}
    predictions = {}

    for name, cfg in models.items():
        model = cfg["model"]
        use_scaled = cfg["scaled"]

        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test

        # Train
        model.fit(Xtr, y_train)

        # Predict
        y_pred = model.predict(Xte)
        predictions[name] = y_pred

        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        results[name] = {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "MAPE (%)": round(mape, 2),
            "model": model,
        }

    return results, predictions, y_test, X_test, X_train, y_train, scaler

def forecast_n_days(df, models_results, feature_cols, n_days, engineer_func, scaler=None):
    """
    Generates a recursive forecast for the next n days for each model.
    """
    # Identify which models need scaled input
    scaled_models = {"SVR", "KNN Regressor"}

    forecasts = {}
    tail_data = df.tail(100).copy()

    for name, res in models_results.items():
        model = res['model']
        current_data = tail_data.copy()
        model_forecasts = []

        for _ in range(n_days):
            df_temp = engineer_func(current_data)
            if df_temp.empty:
                break

            features = df_temp[feature_cols].tail(1)

            if name in scaled_models and scaler is not None:
                features = pd.DataFrame(scaler.transform(features), index=features.index, columns=features.columns)

            pred = model.predict(features)[0]
            model_forecasts.append(pred)

            last_date = current_data.index[-1]
            next_date = last_date + pd.Timedelta(days=1)

            new_row = pd.DataFrame({
                'Open': [pred], 'High': [pred], 'Low': [pred], 'Close': [pred], 'Volume': [0]
            }, index=[next_date])

            current_data = pd.concat([current_data, new_row])

        forecasts[name] = model_forecasts

    return forecasts
