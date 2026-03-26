import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from data_loader import fetch_stock_data, get_ticker_info
from feature_engineering import engineer_features, prepare_data_for_ml
from model_trainer import train_and_evaluate, forecast_n_days

# Page Config
st.set_page_config(page_title="Stock Price Prediction ML App", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .stTable {
        background-color: #1e2130;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Advanced Stock Prediction & Forecasting by Krish Jariwala")
st.markdown("Comparing **9 ML models** and forecasting future prices.")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=365*2))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
forecast_days = st.sidebar.slider("Forecast Days", min_value=1, max_value=30, value=7)

# Color palette for 9 models
MODEL_COLORS = {
    'Linear Regression': '#00d4ff',
    'Ridge Regression': '#8B5CF6',
    'Lasso Regression': '#EC4899',
    'Decision Tree': '#F97316',
    'Random Forest': '#ffaa00',
    'Gradient Boosting': '#14B8A6',
    'XGBoost': '#00ff00',
    'SVR': '#EF4444',
    'KNN Regressor': '#A78BFA',
}

if st.sidebar.button("Run Detailed Analysis"):
    with st.spinner(f"Fetching data for {ticker}..."):
        df = fetch_stock_data(ticker, start_date, end_date)
        
    if df is not None:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        info = get_ticker_info(ticker)
        st.subheader(f"Analyzing {info.get('longName', ticker)}")
        
        # ── Historical Price Chart ──
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='#00d4ff')))
        fig_hist.update_layout(title="Historical Price Trend", template="plotly_dark", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Feature Engineering
        with st.spinner("Engineering features..."):
            df_feat = engineer_features(df)
            
        feature_cols = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_Signal', 'ATR_14', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Returns']
        X, y = prepare_data_for_ml(df_feat, feature_cols)
        
        # ── Model Training ──
        with st.spinner("Training 9 models and evaluating..."):
            results, predictions, y_test, X_test, X_train, y_train, scaler = train_and_evaluate(X, y)

        # ── Train / Test Split Visualization ──
        st.subheader("📊 Train / Test Split Visualization")
        st.markdown("The model is trained on **80%** of the data and tested on the remaining **20%**.")

        fig_split = go.Figure()

        # Training region (Close prices for training dates)
        train_close = df_feat.loc[y_train.index, 'Close']
        test_close = df_feat.loc[y_test.index, 'Close']

        fig_split.add_trace(go.Scatter(
            x=train_close.index, y=train_close.values,
            name=f'Training Data ({len(y_train)} samples)',
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy', fillcolor='rgba(0,212,255,0.08)'
        ))
        fig_split.add_trace(go.Scatter(
            x=test_close.index, y=test_close.values,
            name=f'Test Data ({len(y_test)} samples)',
            line=dict(color='#EF4444', width=2),
            fill='tozeroy', fillcolor='rgba(239,68,68,0.08)'
        ))

        # Vertical split line using add_shape (avoids Plotly _mean bug with Timestamps)
        split_date = y_test.index[0]
        fig_split.add_shape(
            type="line", x0=split_date, x1=split_date,
            y0=0, y1=1, yref="paper",
            line=dict(color="yellow", width=2, dash="dash")
        )
        fig_split.add_annotation(
            x=split_date, y=1, yref="paper",
            text="Train ← | → Test", showarrow=False,
            font=dict(color="yellow", size=12),
            xanchor="left", yanchor="bottom"
        )

        fig_split.update_layout(
            title="Data Used for Training vs Testing",
            template="plotly_dark",
            xaxis_title="Date", yaxis_title="Close Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_split, use_container_width=True)

        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Total Samples", len(X))
        with col_info2:
            st.metric("Training Samples", len(X_train))
        with col_info3:
            st.metric("Test Samples", len(X_test))
        
        # ── Performance Metrics Table ──
        st.subheader("📋 Performance Metrics Comparison")
        metrics_df = pd.DataFrame({
            name: {k: v for k, v in res.items() if k != 'model'} for name, res in results.items()
        }).T
        metrics_df.index.name = "Model"

        st.dataframe(
            metrics_df.style
                .highlight_min(axis=0, subset=['MAE', 'RMSE', 'MAPE (%)'], color='#166534')
                .highlight_max(axis=0, subset=['R2'], color='#166534')
                .highlight_max(axis=0, subset=['MAE', 'RMSE', 'MAPE (%)'], color='#991B1B')
                .highlight_min(axis=0, subset=['R2'], color='#991B1B')
                .format(precision=4),
            use_container_width=True,
            height=380
        )
        
        # ── Multi-Model Comparison: Actual vs Predicted ──
        st.subheader("📈 Actual vs Predicted — All Models")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(
            x=y_test.index, y=y_test.values,
            name='Actual Price',
            line=dict(color='white', width=3)
        ))
        for name, pred in predictions.items():
            fig_comp.add_trace(go.Scatter(
                x=y_test.index, y=pred,
                name=f'{name}',
                line=dict(color=MODEL_COLORS.get(name, 'gray'), dash='dot', width=1.5)
            ))
            
        fig_comp.update_layout(
            title="All Models — Predicted vs Actual on Test Set",
            template="plotly_dark",
            xaxis_title="Date", yaxis_title="Price",
            legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # ── Per-Model Error Bar Chart ──
        st.subheader("📊 Model Accuracy Comparison")
        bar_models = list(metrics_df.index)
        bar_colors = [MODEL_COLORS.get(m, 'gray') for m in bar_models]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=bar_models, y=metrics_df['R2'].values,
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in metrics_df['R2'].values],
            textposition='outside'
        ))
        fig_bar.update_layout(
            title="R² Score by Model (Higher is Better)",
            template="plotly_dark",
            xaxis_title="Model", yaxis_title="R² Score",
            yaxis_range=[min(0, metrics_df['R2'].min() - 0.05), 1.05]
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── N-Day Forecast ──
        st.subheader(f"🔮 {forecast_days}-Day Future Forecast")
        with st.spinner(f"Generating future forecast from all models..."):
             future_forecasts = forecast_n_days(df, results, feature_cols, forecast_days, engineer_features, scaler)
        
        future_dates = [df.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
        
        fig_fore = go.Figure()
        hist_context = df.tail(30)
        fig_fore.add_trace(go.Scatter(x=hist_context.index, y=hist_context['Close'], name='Recent Price', line=dict(color='white', width=2)))
        
        for name, forecast_vals in future_forecasts.items():
            full_dates = [df.index[-1]] + future_dates
            full_vals = [df['Close'].iloc[-1]] + forecast_vals
            fig_fore.add_trace(go.Scatter(
                x=full_dates, y=full_vals,
                name=f'Forecast: {name}',
                line=dict(color=MODEL_COLORS.get(name, 'gray'), width=2)
            ))
            
        fig_fore.update_layout(
            title=f"Next {forecast_days} Days Forecast — All Models",
            template="plotly_dark",
            xaxis_title="Date", yaxis_title="Price",
            legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_fore, use_container_width=True)

        # ── Summary Metrics ──
        st.subheader("🏆 Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
             st.metric("Current Price", f"{df['Close'].iloc[-1]:.2f}")
        with col2:
             best_model = metrics_df['R2'].idxmax()
             st.metric("Best Model (R²)", best_model)
        with col3:
             final_pred = future_forecasts[best_model][-1]
             st.metric(f"Forecasted Price (T+{forecast_days})", f"{final_pred:.2f}", delta=f"{final_pred - df['Close'].iloc[-1]:.2f}")

    else:
        st.error("Could not fetch data. Please check ticker symbol or try again after sometime.")
else:
    st.info("👈 Enter ticker and config in the sidebar, then click 'Run Detailed Analysis'.")
