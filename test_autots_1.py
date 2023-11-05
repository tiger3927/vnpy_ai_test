import numpy as np
import pandas as pd
import datetime
from autots import AutoTS
from datetime import date, timedelta
# import yfinance as yf   # we can use this library to download some finance data
# d1 = date.today()
# end_date = d1.strftime("%Y-%m-%d")
# d2 = date.today()-timedelta(days=1000)
# start_date = d2.strftime("%Y-%m-%d")
#
# data = yf.download("BTC-USD"
#                   , start = start_date
#                   , end = end_date,
#                   progress = False)
#
# print(data.head())
# data["Date"] = data.index
# data.reset_index(drop=True, inplace= True)
# print(data.isnull().sum())

data_path = "./data/"
data = pd.read_csv(data_path+'ETHUSDT_5m_2020-01-01 00-00-00---2023-01-01 00-00-00.csv')
print(data.head())
# import plotly.graph_objects as go
# figure = go.Figure(go.Candlestick(x = data["Date"]
#                                  ,open = data["Open"]
#                                  ,close = data["Close"]
#                                  ,high = data["High"]
#                                  ,low = data["Low"]))
# figure.update_layout(title = " Bitcoin Price Analysis",
#                     xaxis_rangeslider_visible = False)
# figure.show()

# import seaborn as sns
#
# correlation = data.corr()
# correlation["Close"].sort_values(ascending = False)
#
# sns.heatmap(correlation, cmap= "coolwarm", annot = True)
# print(correlation)

model = AutoTS(forecast_length = 30, ensemble = "simple", frequency = "infer")
model = model.fit(data, date_col = "Date", value_col = "Close", id_col = None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)
