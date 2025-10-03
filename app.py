# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")
st.title("📈 Stock Analysis and Prediction App (Educational)")

with st.sidebar:
    st.header("User Input Parameters")
    ticker = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, INFY.NS)", "AAPL")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    fetch = st.button("Fetch Data")

@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        return df
    except Exception:
        # return empty df if download fails
        return pd.DataFrame()

def safe_line_chart(df, cols, title=None):
    # Utility to avoid KeyError: show warning if cols missing
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.warning(f"Missing columns for chart: {missing}. Chart skipped.")
        return
    plot_df = df[cols].dropna()
    if plot_df.empty:
        st.warning("Not enough non-NaN rows to plot chart.")
    else:
        st.line_chart(plot_df)

if fetch:
    if start_date >= end_date:
        st.error("Start date must be before End date.")
    else:
        df = load_data(ticker, start_date, end_date)

        if df is None or df.empty:
            st.error("No data found for that ticker / date range. Try another symbol or range.")
        else:
            st.subheader(f"Raw data (last 5 rows) — {ticker.upper()}")
            st.dataframe(df.tail())

            # --- compute moving averages BEFORE plotting ---
            df['MA50'] = df['Close'].rolling(window=50).mean()
            df['MA200'] = df['Close'].rolling(window=200).mean()

            st.subheader(f"Stock Closing Price for {ticker.upper()}")
            safe_line_chart(df, ['Close'])

            st.subheader("Stock Price with Moving Averages (50 & 200-day)")
            safe_line_chart(df, ['Close', 'MA50', 'MA200'])

            # Feature engineering (simple lag features)
            data = df[['Close']].copy()
            data['Lag1'] = data['Close'].shift(1)   # close at t-1
            data['Lag2'] = data['Close'].shift(2)   # close at t-2
            data = data.dropna()

            if data.shape[0] < 10:
                st.warning("Very few rows available after creating features; model may be poor.")

            # Prepare X/y
            X = data[['Lag1', 'Lag2']].values
            y = data['Close'].values

            # Train/test split (time-respecting)
            try:
                model = LinearRegression()
                model.fit(X, y)
                df['Prediction'] = model.predict(X)

                st.subheader("Prediction Results")
                st.line_chart(df[['Close', 'Prediction']])

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")
