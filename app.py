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
    # download using yfinance
    df = yf.download(ticker, start=start, end=end)
    return df

if fetch:
    if start_date >= end_date:
        st.error("Start date must be before End date.")
    else:
        df = load_data(ticker, start_date, end_date)

        # check we got data
        if df is None or df.empty:
            st.error("No data found for that ticker or date range. Try different inputs.")
        else:
            st.subheader(f"Raw data (last 5 rows) — {ticker}")
            st.dataframe(df.tail())

            # ---------- IMPORTANT FIX: compute MA50 BEFORE plotting ----------
            df['MA50'] = df['Close'].rolling(window=50).mean()
            # ------------------------------------------------------------------

            st.subheader(f"Stock Closing Price for {ticker}")
            st.line_chart(df['Close'])

            st.subheader("Stock Price with 50-Day Moving Average")
            # the MA50 column now exists; safe to plot
            st.line_chart(df[['Close', 'MA50']])

            # --- Feature engineering for basic next-day prediction ---
            data = df[['Close']].copy()
            data['Lag1'] = data['Close'].shift(1)  # close at t-1 for predicting t
            data['Lag2'] = data['Close'].shift(2)  # close at t-2
            data = data.dropna()

            if len(data) < 10:
                st.warning("Only a small number of rows available after feature creation. Model quality may be poor.")

            X = data[['Lag1', 'Lag2']].values
            y = data['Close'].values

            # time-series split (no shuffle)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.subheader("Model Performance (test set)")
            st.write(f"MSE: {mse:.4f}")
            st.write(f"R² Score: {r2:.4f}")

            # plot actual vs predicted
            st.subheader("Actual vs Predicted (test set)")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(y_test, label='Actual')
            ax.plot(y_pred, label='Predicted')
            ax.legend()
            st.pyplot(fig)

            # --- Next-day prediction (use Close_t and Close_{t-1} as features) ---
            # For predicting t+1 we need Lag1 = Close_t and Lag2 = Close_{t-1}
            if len(df) >= 3:
                last_close = df['Close'].iloc[-1]        # Close_t
                prev_close = df['Close'].iloc[-2]        # Close_{t-1}
                next_pred = model.predict(np.array([[last_close, prev_close]]))[0]
                st.success(f"Predicted next-day close for {ticker}: {next_pred:.2f}")
            else:
                st.warning("Not enough rows to produce next-day prediction (need at least 3 rows).")
