import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("📈 Stock Analysis and Prediction App")

# Sidebar Inputs
st.sidebar.header("User Input Parameters")
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, TSLA, INFY.NS)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))

if st.sidebar.button("Fetch Data"):
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("⚠️ No data found. Try another stock symbol or date range.")
    else:
        st.subheader(f"Stock Closing Price for {ticker}")
        st.line_chart(df['Close'])

        # Moving Average
        df['MA50'] = df['Close'].rolling(50).mean()
        st.subheader("Stock Price with 50-Day Moving Average")
        st.line_chart(df[['Close','MA50']])

        # Feature Engineering
        df['Lag1'] = df['Close'].shift(1)
        df['Lag2'] = df['Close'].shift(2)
        df = df.dropna()

        # Train-Test Split
        X = df[['Lag1','Lag2']]
        y = df['Close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        st.subheader("Model Performance")
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("R² Score:", r2_score(y_test, y_pred))

        # Plot prediction vs actual
        st.subheader("Actual vs Predicted Stock Prices")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(y_test.values, label="Actual")
        ax.plot(y_pred, label="Predicted")
        ax.legend()
        st.pyplot(fig)

        # Next Day Prediction
        latest = df[['Lag1','Lag2']].iloc[-1].values.reshape(1,-1)
        predicted_price = model.predict(latest)
        st.success(f"Predicted Next Day Price: {predicted_price[0]:.2f} USD")
