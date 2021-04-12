from datetime import timedelta, date, datetime, timezone
from typing import List
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as pdr
import requests
import urllib
import numpy as np
import coinmetrics
from itertools import consecutive_groups

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
	            content:'© 2021 Carlos Massa'; 
	            visibility: visible;
	            display: block;
	            position: relative;
	            #background-color: red;
	            padding: 5px;
	            top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

#_max_width_()
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

today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
yesterday = datetime.strftime(datetime.now(timezone.utc) - timedelta(1), '%Y-%m-%d')
tomorrow = datetime.strftime(datetime.now(timezone.utc) + timedelta(1), '%Y-%m-%d')
one_month_from_now = datetime.strftime(datetime.now(timezone.utc) + timedelta(30), '%Y-%m-%d')
one_year_from_now = datetime.strftime(datetime.now(timezone.utc) + timedelta(365), '%Y-%m-%d')
four_years_from_now = datetime.strftime(datetime.now(timezone.utc) + timedelta(365 * 4), '%Y-%m-%d')
begin_timestamp = "2009-01-03"  # The `datetime` type is also accepted
end_timestamp = yesterday  # The `datetime` type is also accepted

##############################
# TEST - Cryptocompare
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
st.write("Capturing Data from:", globalURL)

# Request the data
resp = requests.get(url=globalURL)
data = resp.json()
st.write(data["Response"])

# Parse the JSON into a Pandas DataFrame
try:
    df = pd.DataFrame.from_dict(data['Data'])
    df = df.rename(columns={'time': 'date'})
    df['date'] = pd.to_datetime(df['date'], unit='s')
    #df.set_index('date', inplace=True)
    df['Id'] = df.reset_index().index
    df.set_index('Id', inplace=True)
    df_save = df[['date', 'close', 'open', 'high', 'low']]
except Exception as e:
    self.errors.append(e)
    df_save = None

# Include percentage change and other columns
df = df_save
df['change'] = df['close'].pct_change()
df['close_log'] = np.log10(df['close'])

st.write(df.head(5))

# Show Log chart of data

fig = px.line(df, x="date", y=["close"]).update_layout(
    yaxis_title='Historical Chart ('+ticker.upper()+') - Log Y axis'
)
st.plotly_chart(fig, use_container_width=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
	x=df['date'],
	y=df['close'],
	fill=None,
	fillcolor=None,
	yaxis='y',
	mode='lines',
	#line_color='rgba(0,0,128,1.0)', #Navy
	name='BTC Price'))

fig.update_layout(template='plotly_white')
fig.update_layout(title_text='BTC Price (source: Cryptocompare)', title_x=0.5)


fig.update_layout(
    shapes=[
        dict(                                      # Color palette: https://www.color-hex.com/color-palette/44237
            fillcolor="rgba(107,127,140, 0.2)",
            line={"width": 0},
            type="rect",
            x0="2009-01-03",
            x1="2012-11-28",
            xref="x",
            y0=0,
            y1=1,
            yref="paper"
        ),
        dict(
            fillcolor="rgba(142,154,162, 0.2)",
            line={"width": 0},
            type="rect",
            x0="2012-11-28",
            x1="2016-07-09",
            xref="x",
            y0=0,
            y1=1,
            yref="paper"
        ),
        dict(
            fillcolor="rgba(189,195,199, 0.2)",
            line={"width": 0},
            type="rect",
            x0="2016-07-09",
            x1="2020-05-12",
            xref="x",
            y0=0,
            y1=1,
            yref="paper"
        ),
        dict(
            fillcolor="rgba(233,234,235, 0.2)",
            line={"width": 0},
            type="rect",
            x0="2020-05-12",
            x1="2024-05-12",
            xref="x",
            y0=0,
            y1=1,
            yref="paper"
        ),
    ],
    separators=".,",
    showlegend=True,
    legend=dict(
        x=0.98,
        y=0.10,
        bgcolor='#ffffff',
        bordercolor='#000000',
        borderwidth=1,
        font=dict(color="black", size=13),
        traceorder='normal',
        xanchor='auto',
        yanchor='auto'
    ),
    hoverlabel=dict(namelength=-1),
    hovermode="x",
    yaxis=dict(
            hoverformat=",.2f",
            type="log",
            title=dict(text="Price (USD)", font=dict(color="black", size=14)),
            tickformat="$0,.2f",
            #tickprefix="$",
            tickmode="auto",
	    nticks=8,
            tickfont=dict(color="#000000", size=14),
            gridcolor="#e4f2fc",
            #domain=[0.33, 1]
        ),
    xaxis=dict(
        hoverformat="%Y-%m-%d",
        showgrid=True,
        type="date",
        title=dict(text="Date", font=dict(color="black", size=13)),
        tickmode="auto",
        tickfont=dict(color="#000000", size=13),
        gridcolor="#e4f2fc",
        range=['2009-12-20', one_year_from_now],
        zeroline=True,
    ),
    margin=dict(
        l=120,
        r=60,
        b=35,
        t=35,
        pad=4
    )
)
st.plotly_chart(fig, use_container_width=True)

##############################
# TEST - Coinmetrics
##############################

st.write("Capturing data from Coinmetrics...")
cm = coinmetrics.Community()

# Usage Examples ############################################################

# List the assets Coin Metrics has data for.
# supported_assets = cm.get_supported_assets()
# print("supported assets:\n", supported_assets)

# List all available metrics for BTC.
asset = "btc"
available_data_types = cm.get_available_data_types_for_asset(asset)
# print("available data types:\n", available_data_types)

# Fetch All Metrics data for BTC from 2009-01-01 to 2019-12-17.
asset = "btc"
metric = "AdrActCnt,BlkCnt,BlkSizeByte,BlkSizeMeanByte,CapMVRVCur,CapMrktCurUSD,CapRealUSD,DiffMean,FeeMeanNtv," \
         "FeeMeanUSD,FeeMedNtv,FeeMedUSD,FeeTotNtv,FeeTotUSD,IssContNtv,IssContPctAnn,IssContUSD,IssTotNtv,IssTotUSD," \
         "NVTAdj,NVTAdj90,PriceBTC,PriceUSD,ROI1yr,ROI30d,SplyCur,TxCnt,TxTfrCnt,TxTfrValAdjNtv,TxTfrValAdjUSD," \
         "TxTfrValMeanNtv,TxTfrValMeanUSD,TxTfrValMedNtv,TxTfrValMedUSD,TxTfrValNtv,TxTfrValUSD,VtyDayRet180d," \
         "VtyDayRet30d,VtyDayRet60d"  # If only Price and Roi is needed, then use: metric = "PriceUSD,ROI30d"

# metric = available_data_types
# metric = "PriceUSD"

asset_data = cm.get_asset_data_for_time_range(asset, metric, begin_timestamp, end_timestamp)
# print("data given timerange:\n", asset_data)

# Convert the data object we received to a Pandas DataFrame for further processing.
# We are reusing the `asset_data` from the previous step.
df = coinmetrics.cm_to_pandas(asset_data)
#st.write(df.head(5))


# Add Date in proper format to dataframe df
data = []

def date_range(date1, date2):
	for n in range(int((date2 - date1).days) + 1):
            yield date1 + timedelta(n)

start_dt = date(2009, 1, 3)
yesterday_obj = datetime.strptime(yesterday, '%Y-%m-%d')
end_dt = yesterday_obj.date()

for dt in date_range(start_dt, end_dt):
	data.append(dt.strftime("%Y-%m-%d"))

df.insert(0, 'date', data, True)

# remove duplicate columns
df = df.drop_duplicates().copy()

df['Id'] = df.reset_index().index
df.set_index('Id', inplace=True)

#df_save = df[['date', 'PriceUSD']]
#df = df_save


# Calculate variables
def RSI(series, period):
    # Get the difference in price from previous step
    delta = series.diff()
    # Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences
    delta = delta[1:]
    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    # Calculate the EWMA
    roll_up1 = up.ewm(span=period).mean()
    roll_down1 = down.abs().ewm(span=period).mean()
    # Calculate the RSI based on EWMA
    RS = roll_up1 / roll_down1
    return 100.0 - (100.0 / (1.0 + RS))

df['MA_10'] = df['PriceUSD'].rolling(window=10).mean()  # calculate MA10
df['MA_20'] = df['PriceUSD'].rolling(window=20).mean()  # calculate MA20
df['MA_50'] = df['PriceUSD'].rolling(window=50).mean()  # calculate MA50
df['MA_100'] = df['PriceUSD'].rolling(window=100).mean()  # calculate MA100
df['MA_140'] = df['PriceUSD'].rolling(window=140).mean()  # calculate MA100
df['MA_200'] = df['PriceUSD'].rolling(window=200).mean()  # calculate MA200
df['MA_730'] = df['PriceUSD'].rolling(window=730).mean()  # calculate MA730
df['MA_730x5'] = df['MA_730'] * 5  # calculate 5xMA730
df['MA_1458'] = df['PriceUSD'].rolling(window=1458).mean()  # calculate MA1458
df['MA_1458_diff'] = df['MA_1458'].diff()  # calculate 200WMA strength
df['MA_1458_growth'] = df['MA_1458'].pct_change().mul(100).round(3)  # calculate 200WMA %growth
df['MA_1458_const'] = df['PriceUSD'] / df['MA_1458_diff']
df['RSI_5'] = RSI(df['PriceUSD'], 5)  # calculate RSI_5
df['RSI_14'] = RSI(df['PriceUSD'], 14) # calculate RSI_14
df['MinPriceUSD50days'] = df['PriceUSD'].rolling(window=50).min()
df['MinPriceUSD100days'] = df['PriceUSD'].rolling(window=100).min()
df['MinPriceUSD200days'] = df['PriceUSD'].rolling(window=200).min()
df['MinPriceUSD365days'] = df['PriceUSD'].rolling(window=365).min()
df['MinPriceUSD2years'] = df['PriceUSD'].rolling(window=730).min()
df['MinPriceUSD4years'] = df['PriceUSD'].rolling(window=1460).min()
df['AllMinPriceEqual'] = np.where((df['MinPriceUSD50days'] == df['MinPriceUSD100days']), df['date'], 0)

# Calculation of lowest price forward
# reverse df
# expanding().min() to calculate min value of all previous values (which are the future ones because df is reversed)
# reverse df again
df2=df.iloc[::-1].copy()
df2['LowestPriceForward'] = df2['PriceUSD'].expanding().min()
df=df2.iloc[::-1].copy()



fig = go.Figure()
fig.add_trace(go.Scatter(
	x=df['date'],
	y=df['PriceUSD'],
	fill=None,
	fillcolor=None,
	mode='lines',
	#line_color='rgba(0,0,128,1.0)', #Navy
	name='BTC Price'))

fig.add_trace(go.Scatter(
	x=df['date'],
	y=df['MinPriceUSD50days'],
	fill=None,
	fillcolor=None,
	mode='lines',
	#line_color='rgba(0,0,128,1.0)', #Navy
	name='Highest Price for +50 consecutive days'))

fig.add_trace(go.Scatter(
	x=df['date'],
	y=df['MinPriceUSD100days'],
	fill=None,
	fillcolor=None,
	mode='lines',
	#line_color='rgba(0,0,128,1.0)', #Navy
	name='Highest Price for +100 consecutive days'))

fig.add_trace(go.Scatter(
	x=df['date'],
	y=df['MinPriceUSD200days'],
	fill=None,
	fillcolor=None,
	mode='lines',
	#line_color='rgba(0,0,128,1.0)', #Navy
	name='Highest Price for +200 consecutive days'))

fig.add_trace(go.Scatter(
	x=df['date'],
	y=df['MinPriceUSD365days'],
	fill=None,
	fillcolor=None,
	mode='lines',
	#line_color='rgba(0,0,128,1.0)', #Navy
	name='Highest Price for +365 consecutive days'))

fig.add_trace(go.Scatter(
	x=df['date'],
	y=df['MinPriceUSD2years'],
	fill=None,
	fillcolor=None,
	mode='lines',
	#line_color='rgba(0,0,128,1.0)', #Navy
	name='Highest Price for +2 years'))

fig.add_trace(go.Scatter(
	x=df['date'],
	y=df['LowestPriceForward'],
	fill=None,
	fillcolor=None,
	mode='lines',
	#line_color='rgba(0,0,128,1.0)', #Navy
	name='Lowest Price forward'))

# lines to add, specified by x-position
lines = {'a':"2018-09-24",'b':"2020-09-24"}

bottoms = df.loc[df['AllMinPriceEqual']!= 0,'date'].tolist()

st.write(bottoms)

for g in consecutive_groups(bottoms, lambda x: datetime.strptime(x, '%Y-%m-%d').toordinal()):
    st.write(list(bottoms))

# add lines using absolute references
for k in range(len(bottoms)):
    #print(k)
    fig.add_shape(type='line',
                yref="y",
                xref="x",
                x0=bottoms[k],
                y0=df['PriceUSD'].min()*1.2,
                x1=bottoms[k],
                y1=df['PriceUSD'].max()*1.2,
                line=dict(color='black', width=3))
    fig.add_annotation(
                x=bottoms[k],
                y=df['PriceUSD'].max()*1.25,
                yref='paper',
                showarrow=False,
                text=k)

fig.update_layout(template='plotly_white')
fig.update_layout(title_text='BTC Price (source: Coinmetrics)', title_x=0.5)


fig.update_layout(
    shapes=[
        dict(                                      # Color palette: https://www.color-hex.com/color-palette/44237
            fillcolor="rgba(107,127,140, 0.2)",
            line={"width": 0},
            type="rect",
            x0="2009-01-03",
            x1="2012-11-28",
            xref="x",
            y0=0,
            y1=1,
            yref="paper"
        ),
        dict(
            fillcolor="rgba(142,154,162, 0.2)",
            line={"width": 0},
            type="rect",
            x0="2012-11-28",
            x1="2016-07-09",
            xref="x",
            y0=0,
            y1=1,
            yref="paper"
        ),
        dict(
            fillcolor="rgba(189,195,199, 0.2)",
            line={"width": 0},
            type="rect",
            x0="2016-07-09",
            x1="2020-05-12",
            xref="x",
            y0=0,
            y1=1,
            yref="paper"
        ),
        dict(
            fillcolor="rgba(233,234,235, 0.2)",
            line={"width": 0},
            type="rect",
            x0="2020-05-12",
            x1="2024-05-12",
            xref="x",
            y0=0,
            y1=1,
            yref="paper"
        ),
    ],
    separators=".,",
    #showlegend=False,
    legend=dict(
        x=0.98,
        y=0.10,
        bgcolor='#ffffff',
        bordercolor='#000000',
        borderwidth=1,
        font=dict(color="black", size=13),
        traceorder='normal',
        xanchor='auto',
        yanchor='auto'
    ),
    hoverlabel=dict(namelength=-1),
    hovermode="x",
    yaxis=dict(
            hoverformat=",.2f",
            type="log",
            title=dict(text="Price (USD)", font=dict(color="black", size=14)),
            tickformat="$0,.2f",
            #tickprefix="$",
            tickmode="auto",
	    nticks=6,
            tickfont=dict(color="#000000", size=14),
            gridcolor="#e4f2fc",
            #domain=[0.33, 1]
        ),
    xaxis=dict(
        hoverformat="%Y-%m-%d",
        showgrid=True,
        type="date",
        title=dict(text="Date", font=dict(color="black", size=13)),
        tickmode="auto",
        tickfont=dict(color="#000000", size=13),
        gridcolor="#e4f2fc",
        range=['2009-12-20', one_year_from_now],
        zeroline=True,
    ),
    margin=dict(
        l=120,
        r=60,
        b=35,
        t=35,
        pad=4
    ),
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(
		label="≡",
		method="update",
		args=[{"showlegend": [False]}],
        	args2=[{"showlegend": [True]}]
	)]
    )]
)
st.plotly_chart(fig, use_container_width=True)
