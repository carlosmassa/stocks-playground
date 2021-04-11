from datetime import datetime
from typing import List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as pdr
import requests
import urllib

yf.pdr_override()  # <== that's all it takes :-)

"""
# Tickers Tracking

by Dror Atariah ([LinkedIn](https://www.linkedin.com/in/atariah/) / [GitHub](https://github.com/drorata/stocks-playground))
"""


metrics = ["Open", "Close", "High", "Low"]


@st.cache()
def load_data(start_date: datetime, end_date: datetime, tickers: List[str]):
    return {
        ticker: pdr.get_data_yahoo(ticker, start=start_date, end=end_date)[metrics]
        for ticker in tickers
    }


start_date = st.sidebar.date_input(
    "Start date:", (datetime.now() - relativedelta(years=2)),
)
end_date = st.sidebar.date_input("End date:", datetime.now())
tickers = st.sidebar.text_input("Ticker(s), separated by commas:", "AAPL, AMZN, GOOGL")
if len(tickers) == 0:
    st.error("You must specify at least one ticker")
    raise st.ScriptRunner.StopException
tickers = [x.strip() for x in tickers.split(",")]

raw_tickers_data = load_data(start_date, end_date, tickers)
for ticker in tickers:
    if raw_tickers_data[ticker].empty:
        st.error(f"No data found for the ticker '{ticker}'")
        raise st.ScriptRunner.StopException

roll_types = ["Normal", "Exponential"]
roll_type = st.sidebar.radio("Averaging type", roll_types)

roll_avg_map = {"1 day": "1d", "7 days": "7d", "30 days": "30d", "120 days": "120d"}
roll_avg = st.sidebar.radio("Average rolling window of", list(roll_avg_map.keys()))

if roll_type not in roll_types:
    st.error("Something went wrong with the rolling type. Exiting")
    raise st.ScriptRunner.StopException

if roll_type == "Normal":
    tickers_data = {
        ticker: raw_tickers_data[ticker].rolling(roll_avg_map[roll_avg]).mean()
        for ticker in tickers
    }
if roll_type == "Exponential":
    tickers_data = {
        ticker: raw_tickers_data[ticker]
        .ewm(span=int(roll_avg_map[roll_avg][:-1]))
        .mean()
        for ticker in tickers
    }

changes = {
    ticker: 100
    * (tickers_data[ticker] - (tickers_data[ticker]).iloc[0])
    / (tickers_data[ticker]).iloc[0]
    for ticker in tickers_data.keys()
}


st.write(
    """## Change in percentage for all tickers

(Based on daily OPEN value)
"""
)
res = []
for key in tickers_data.keys():
    data = changes[key]["Open"]
    data.name = key
    res.append(data)
df = pd.DataFrame(res).transpose().reset_index()

st.plotly_chart(
    px.line(df, x="Date", y=tickers).update_layout(
        yaxis_title="% change", legend_title_text="Ticker"
    )
)

if st.checkbox("Show data"):
    st.write(df.set_index("Date"))

st.write("## Ticker behavior")
ticker = st.selectbox("Select ticker", tickers)
df = tickers_data[ticker].reset_index()

fig = go.Figure(
    data=[
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
        )
    ]
)
st.plotly_chart(fig, use_container_width=True)

value_type = st.selectbox("Select value", ["Open", "High", "Low", "Close"])
df = pd.concat(
    [tickers_data[ticker][value_type], raw_tickers_data[ticker][value_type]], axis=1
)
df.columns = ["Averaged", "Raw"]
df.reset_index(inplace=True)
fig = px.line(df, x="Date", y=["Averaged", "Raw"]).update_layout(
    yaxis_title=f"'{value_type}' value"
)
st.plotly_chart(fig, use_container_width=True)


##############################
# TEST
##############################
# Load Bitcoin Prices into a dataframe
# Ticker is customizable
ticker = "BTC"
# Cryptocompare URL and fiels
# Cryptocompare API KEY: 979ade059ddaa46d916f9b80f8a07d3448f53bad60a073a08d573580a7bffde6
# API Key not needed here!
base_url = 'https://min-api.cryptocompare.com/data/histoday'
ticker_field = 'fsym'
field_dict = {'tsym': 'USD','allData': 'true'}
# Convert the field dict into a url encoded string
url_args = "&" + urllib.parse.urlencode(field_dict)
ticker = ticker.upper()
globalURL = (base_url + "?" + ticker_field + "=" + ticker + url_args)
st.write("Downloading price Data from:", globalURL)

# Request the data
resp = requests.get(url=globalURL)
data = resp.json()
st.write(data["Response"])

# Parse the JSON into a Pandas DataFrame
try:
    df = pd.DataFrame.from_dict(data['Data'])
    df = df.rename(columns={'time': 'date'})
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df.set_index('date', inplace=True)
    df_save = df[['close', 'open', 'high', 'low']]
except Exception as e:
    self.errors.append(e)
    df_save = None

# Include percentage change and other columns
df = df_save
df['change'] = df['close'].pct_change()

# Show Log chart of data

fig = px.line(df, x="Date", y=["close"]).update_layout(
    yaxis_title='Historical Chart ('+ticker.upper()+') - Log Y axis'
)
st.plotly_chart(fig, use_container_width=True)

