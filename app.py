import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Download data for a specific stock (e.g., Apple - AAPL)
stock = 'AAPL'
df = yf.download(stock, start="2015-01-01", end="2024-01-01")

# Display the first few rows
print(df.head())

df = df.fillna(method='ffill')  # Forward fill missing values

df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price', color='blue')
plt.title(f"{stock} Stock Price Trend")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.legend()
plt.show()


