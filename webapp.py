#ce code retoure une webapp qui montre les prevision de notre modéle 

#pour excecuter ce code il faut utiliser l'instruction (streamlit run webapp.py)dans le terminal
#il faut avoir streamlit au prealablement installer

import streamlit as st
from datetime import date, timedelta
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.metrics import MeanSquaredError

# Constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# App Title
st.title('Apple Stock Forecast App')

# Model Loading
@st.cache(allow_output_mutation=True)
def load_lstm_model():
    """Loads the saved LSTM model."""
    model = load_model('lstm_apple_stock_model.h5', custom_objects={'mse': MeanSquaredError()})
    return model

model = load_lstm_model()

# Function to load data from Yahoo Finance
@st.cache
def load_data(ticker):
    """Loads stock data for the given ticker."""
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

# Load Apple stock data
data_load_state = st.text('Loading data...')
data = load_data('AAPL')
data_load_state.text('Loading data... done!')

# Display raw data
st.subheader('Raw Data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    """Plots the raw stock data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Time Series Data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Data preparation for LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

sequence_length = 60  # Input sequence length for the LSTM model
last_sequence = scaled_data[-sequence_length:]

# Forecast future prices
def forecast_prices(model, last_sequence, n_days):
    """Forecasts the next n_days using the LSTM model."""
    predictions = []
    current_sequence = last_sequence.reshape(1, sequence_length, 1)  # Ensure the correct shape
    for _ in range(n_days):
        predicted_price = model.predict(current_sequence)[0, 0]  # Extract the scalar prediction
        predictions.append(predicted_price)
        # Prepare the next sequence
        predicted_price_reshaped = np.array(predicted_price).reshape(1, 1, 1)  # Shape: (1, 1, 1)
        current_sequence = np.append(current_sequence[:, 1:, :], predicted_price_reshaped, axis=1)
    return predictions

# Forecast the stock prices
n_days = st.slider('Days of Prediction:', 1, 30, value=10)
forecasted_prices = forecast_prices(model, last_sequence, n_days)
forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))

# Generate forecasted dates
last_date = data['Date'].iloc[-1]
forecasted_dates = [last_date + timedelta(days=i + 1) for i in range(n_days)]

# Display forecasted data
st.subheader('Forecast Data')
forecast_df = pd.DataFrame({'Date': forecasted_dates, 'Forecasted Price': forecasted_prices.flatten()})
st.write(forecast_df)

# Plot forecasted data
def plot_forecast_data():
    """Plots the forecasted stock data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Historical Prices"))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted Price'], name="Forecasted Prices", line=dict(dash='dot')))
    fig.layout.update(title_text='Apple Stock Price Forecast', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_forecast_data()
 