import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from io import StringIO


my_param = st.slider("Total Stocks in Selection", min_value=5, max_value=500, value=50, step=5)


# Data Collection ##############################################################

# Get S&P 500 details from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

response = requests.get(url, headers=headers)
df_sp500 = pd.read_html(StringIO(response.text), attrs={'id': 'constituents'})[0]

df_sp500 = df_sp500[~df_sp500['Symbol'].isin(['GOOGL','FOXA','NWSA'])]
df_sp500['Symbol'] = df_sp500['Symbol'].str.replace('.', '-', regex=False)
df_sp500['Security'] = df_sp500['Security'].str.replace(r'\s*\(.*?\)', '', regex=True).str.strip()

# Selected Tickers
url = 'https://github.com/ayamnori0/data/raw/refs/heads/main/SP500_MarketCap.parquet'
df_sp500_marketcap = pd.read_parquet(url)

tickers = df_sp500_marketcap['Ticker'].iloc[:my_param].tolist()

# Yahoo data prices
df_data = yf.download(tickers, period='2d', interval='1d', group_by='ticker', threads=True)
 
df_data = df_data.xs('Close', level='Price', axis=1)
df_data = df_data.T.reset_index()
df_data.columns = [*df_data.columns[:-2], 'T_0', 'T_1']

df_data['T_1'] = df_data['T_1'].astype(object)
mask = df_data['T_1'].isna()
df_data.loc[mask, 'T_1'] = df_data.loc[mask, 'Ticker'].apply(lambda x: yf.Ticker(x).fast_info.get('lastPrice'))

# Yahoo data details
df_data = pd.merge(df_data,
                   df_sp500_marketcap[['Ticker', 'MarketCap']],
                   on='Ticker')

# DataFrame ####################################################################

# df
df = df_data[['Ticker', 'MarketCap', 'T_0', 'T_1']].copy()

df = df.rename(columns={
    'T_0': 'Prev_Price',
    'T_1': 'Price'
})

df['Price'] = df['Price'].astype('float64')

df['Pct_Change'] = (df.iloc[:, [-2, -1]]
                    .dropna()
                    .astype('double')
                    .pct_change(axis=1)
                    .iloc[:, -1]).mul(100).round(2)

df = pd.merge(df, df_sp500[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']],
              left_on='Ticker', right_on='Symbol')


# Chart ########################################################################

# Parameters
p_max = max(np.ceil(df['Pct_Change'].abs().max() / 5) * 5, 5)

custom_scale = [[0, '#DE1010'],
                [(p_max-4.0)/(p_max*2), '#DE1010'],

                [(p_max-0.5)/(p_max*2), '#662121'],
                [(p_max-0.5)/(p_max*2), '#8C8C87'],
                [(p_max+0.5)/(p_max*2), '#8C8C87'],
                [(p_max+0.5)/(p_max*2), '#19451A'],

                [(p_max+4.0)/(p_max*2), '#09ED12'],
                [1, '#09ED12']]

# Figure
fig = px.treemap(
    df,
    path=[px.Constant(f'S&P Top {my_param}'), 'GICS Sector', 'Ticker'],
    values='MarketCap',
    color='Pct_Change',
    custom_data=['Pct_Change', 'Security', 'Price'],

    color_continuous_scale=custom_scale,
    range_color=[-p_max , p_max],
    color_continuous_midpoint=0,
    hover_data={'MarketCap': False, 'Security': True, 'Pct_Change': True},
    title=f'S&P Top {my_param} - Daily Performance'
)

fig.update_traces(
    texttemplate="<b>%{label}</b><br>"
                 "%{customdata[0]:.2f}%",
    textposition="middle center",

    hovertemplate="<b>%{customdata[1]}</b><br>"
                   "<b>%{label}  </b>"
                   "%{customdata[2]:.2f},  "
                   "%{color:.2f}%"
)

fig.update_layout(margin=dict(t=50, l=25, r=25, b=25),
                  coloraxis_showscale=False)
#fig.show()
st.plotly_chart(fig, use_container_width=True)
