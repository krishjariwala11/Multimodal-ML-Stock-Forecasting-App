import pandas as pd
import ta
import numpy as np

def engineer_features(df):
    """
    Engineers technical indicators and lag features for ML models.
    """
    # Ensure dataframe is a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Technical Indicators using ta library
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    
    # Volatility (ATR)
    df['ATR_14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    
    # Lagged features
    df['Close_Lag_1'] = df['Close'].shift(1)
    df['Close_Lag_2'] = df['Close'].shift(2)
    df['Close_Lag_3'] = df['Close'].shift(3)
    
    # Returns
    df['Returns'] = df['Close'].pct_change()
    
    # Target Variable: Next Day's price
    df['Target'] = df['Close'].shift(-1)
    
    # Drop rows with NaNs created by indicators/lags/target
    df.dropna(inplace=True)
    
    return df

def prepare_data_for_ml(df, feature_cols):
    """
    Splits data into features (X) and target (y).
    """
    X = df[feature_cols]
    y = df['Target']
    return X, y
