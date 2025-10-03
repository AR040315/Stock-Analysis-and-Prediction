# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")
st.title("📈 Stock Analysis and Prediction App (Educational)")

# ---------------- Sidebar Inputs ----------------
with st.sidebar:
    st.header("User Input Parameters")
    ticker = st.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, INFY.NS)", "AAPL")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    future_days = st.slider("Days to Forecast", 1, 30, 7)   # forecast horizon
    fetch = st.button("Fetch & Predict")

# ---------------- Load Data ----------------
@st.cache_data(ttl=3600)
def load_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        return df
    except Exception:
        return pd.DataFrame()

# ---------------- Utility: Safe Line Chart ----------------
def safe_line_chart(df, cols, title=None):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.warning(f"⚠️ Missing columns for chart: {missing}. Chart skipped.")
        return
    plot_df = df[cols].dropna()
    if plot_df.empty:
        st.warning("⚠️ Not enough valid rows to plot chart.")
    else:
        if title:
            st.subheader(title)
        st.line_chart(plot_df)

# ---------------- Main Logic ----------------
if fetch:
    if start_date >= end_date:
        st.error("❌ Start date must be before End date.")
    else:
        df = load_data(ticker, start_date, end_date)

        if df.empty:
            st.error("❌ No data found for that ticker / date range.")
        else:
            st.subheader(f"Raw data (last 5 rows) — {ticker.upper()}")
            st.dataframe(df.tail())

            # Ensure we have a usable Close/Adj Close
            if "Close" in df.columns:
                df["Target"] = df["Close"]
            elif "Adj Close" in df.columns:
                df["Target"] = df["Adj Close"]
            else:
                st.error(f"No usable 'Close' column found. Available: {list(df.columns)}")
                st.stop()

            # Add Moving Averages
            df["MA50"] = df["Target"].rolling(window=50).mean()
            df["MA200"] = df["Target"].rolling(window=200).mean()

            # Charts
            safe_line_chart(df, ["Target"], f"Stock Closing Price for {ticker.upper()}")
            safe_line_chart(df, ["Target", "MA50", "MA200"], "Stock Price with Moving Averages")

            # ---------------- ML Prediction ----------------
            # Feature Engineering
            data = df[["Target"]].copy()
            data["Lag1"] = data["Target"].shift(1)
            data["Lag2"] = data["Target"].shift(2)
            data = data.dropna()

            if data.shape[0] < 10:
                st.warning("⚠️ Not enough rows for reliable prediction.")
            else:
                X = data[["Lag1", "Lag2"]].values
                y = data["Target"].values

                try:
                    model = LinearRegression()
                    model.fit(X, y)
                    data["Prediction"] = model.predict(X)

                    st.subheader("Prediction Results (on historical data)")
                    st.line_chart(data[["Target", "Prediction"]])

                    # Metrics
                    r2 = model.score(X, y)
                    st.write(f"🔹 R² Score: {r2:.4f}")

                    # ---------------- Future Forecast ----------------
                    st.subheader(f"🔮 Forecast for Next {future_days} Days")

                    # Start from last two known values
                    last_lag1 = data["Target"].iloc[-1]
                    last_lag2 = data["Target"].iloc[-2]

                    future_preds = []
                    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days, freq="B")

                    for d in range(future_days):
                        X_future = np.array([[last_lag1, last_lag2]])
                        y_future = model.predict(X_future)[0]
                        future_preds.append(y_future)

                        # Shift lags for next prediction
                        last_lag2 = last_lag1
                        last_lag1 = y_future

                    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_preds})
                    forecast_df = forecast_df.set_index("Date")

                    st.line_chart(forecast_df)

                    st.write("📅 Forecasted Prices:")
                    st.dataframe(forecast_df)

                except Exception as e:
                    st.error(f"⚠️ Error in prediction: {e}")
