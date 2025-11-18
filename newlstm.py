import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.title("📈 Marico Stock Price Prediction using LSTM")
st.write("Upload your stock dataset (Date, Open, High, Low, Close, Volume).")

# ---------------------------
# 1. Upload file
# ---------------------------
file = st.file_uploader("Upload Excel or CSV File", type=["csv", "xlsx"])

if file is not None:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.subheader("🔍 Preview of Uploaded Data")
    st.dataframe(df.head())

    # ---------------------------
    # 2. Use only Close price
    # ---------------------------
    data = df["Close (₹)"].values.reshape(-1, 1)

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # ---------------------------
    # 3. Create training sequences
    # ---------------------------
    sequence_length = 60  # past 60 days input

    X_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # reshape for LSTM [samples, time_steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # ---------------------------
    # 4. Build LSTM Model
    # ---------------------------
    st.subheader("🧠 Training LSTM Model...")

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X_train, y_train, batch_size=32, epochs=20)

    st.success("Model Training Completed Successfully!")

    # ---------------------------
    # 5. Predict future prices
    # ---------------------------
    st.subheader("📅 Predict Future Stock Price")

    future_days = st.slider("Select days to forecast:", 1, 60, 15)

    last_sequence = scaled_data[-sequence_length:]
    future_predictions = []

    current_seq = last_sequence.copy()

    for _ in range(future_days):
        pred = model.predict(current_seq.reshape(1, sequence_length, 1))[0][0]
        future_predictions.append(pred)

        # update the sequence
        current_seq = np.append(current_seq[1:], pred)
        current_seq = current_seq.reshape(sequence_length, 1)

    future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # ---------------------------
    # 6. Plot prediction graph
    # ---------------------------
    st.subheader("📈 Forecasted Price Chart")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(data)), data, label="Historical Price")
    ax.plot(range(len(data), len(data) + future_days), future_prices, label="Predicted Price")
    ax.set_title("Marico Stock Price Prediction")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price (₹)")
    ax.legend()

    st.pyplot(fig)

    # Show predicted values
    st.subheader("📌 Predicted Future Prices")
    future_df = pd.DataFrame({
        "Day": np.arange(1, future_days + 1),
        "Predicted Price (₹)": future_prices.flatten()
    })
    st.dataframe(future_df)
