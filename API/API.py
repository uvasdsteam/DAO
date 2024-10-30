
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf

# format for setting up any stock: aapl = yf.Ticker("AAPL")
aapl = yf.Ticker("AAPL")
aapl_history = aapl.history(period ="max")
aapl_history.reset_index(inplace=True)
df = aapl_history[['Date','Close']]
