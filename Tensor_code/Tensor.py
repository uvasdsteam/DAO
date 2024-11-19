from math import tanh
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Input
import matplotlib as plt
import datetime
import seaborn as sns 

def forecast_stock(ticker, start_date, end_date):
  data = yf.download(ticker, start=start_date, end=end_date)
  # use yf api to get data on the stock

  scaler = MinMaxScaler(feature_range=(0, 1))
  #create first scalar (find min max)
  scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
  #transform the info -- gets the close data for processing

  train_size = int(len(scaled_data) * 0.8)
  test_size = len(scaled_data) - train_size
  train_data, test_data = scaled_data[0:train_size,:], scaled_data[train_size:len(scaled_data),:]

  def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    #create empty arrays to append
    for i in range(len(dataset)-time_step-1):
      a = dataset[i:(i+time_step), 0]
      dataX.append(a)
      dataY.append(dataset[i + time_step, 0])
      #works on the interval
    return np.array(dataX), np.array(dataY)
  # returns the correct dataset for our model


  time_step = 60
  X_train, y_train = create_dataset(train_data, time_step)
  X_test, y_test = create_dataset(test_data, time_step)
  
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  #helps reshape so that we can do input and sequence work on the model

  model = Sequential()
  model.add(LSTM(50, return_sequences=True, activation=tanh))
  #working with LSTM part of tensorflow 
  model.add(LSTM(50))
  model.add(Dense(1))
  model.compile(loss='mean_squared_error', optimizer='adam')
  model.fit(X_train, y_train, epochs=35, batch_size=32)


  predictions = model.predict(X_test)
  predictions = scaler.inverse_transform(predictions)


stock_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
#current tickers that will iterated through the NN function

def get_latest_yfinance_date(ticker):
  df = yf.download(ticker, start="2010-01-01")
  latest_date = df.index[-1].strftime('%Y-%m-%d')
  return latest_date
#gets the most recent date from y-finance

# Get today's date
today = datetime.date.today().strftime('%Y-%m-%d')

for ticker in stock_tickers:
  latest_api_date = get_latest_yfinance_date(ticker)
  end_date = min(today, latest_api_date)
  result = forecast_stock(ticker, '2010-01-01', end_date) 
  #runs each dat through the model from 2010-today
  if result is not None:
    model, predictions = result
      # Proceed with using model and predictions
  else:
    print(f"Warning: forecast_stock returned None for ticker {ticker}")
      # Handle the case where forecast_stock fails, if necessary

