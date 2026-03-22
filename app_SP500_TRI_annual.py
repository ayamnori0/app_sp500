import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

import plotly.io as plt_io
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

# Data from GitHub
url = 'https://github.com/ayamnori0/data/raw/refs/heads/main/SP500_TRI_ANNUAL.parquet'
df = pd.read_parquet(url)

# Data Manipulation

t_neg = df[df['AnnualGrowth_TRI'] < 0]
t_pos = df[~(df['AnnualGrowth_TRI'] < 0)]

# color for growth between 0 - 0.5
c = pd.DataFrame({'EP': np.add(np.arange(0, 0.5, 0.001), 0.001)})
c = pd.concat([pd.DataFrame({'SP': c['EP'].values}, index=c.index+1), c], axis=1) \
      .sort_index().fillna({'SP': 0, 'EP': np.inf}) \
      .reset_index(names='C')

# pos
t_pos = pd.merge(t_pos.assign(key=1), c.assign(key=1), on='key').drop('key', axis=1)
t_pos['I'] = np.where(t_pos['AnnualGrowth_TRI'] >= t_pos['EP'], t_pos['EP'] - t_pos['SP'],
                      np.where(t_pos['AnnualGrowth_TRI'] > t_pos['SP'], t_pos['AnnualGrowth_TRI'] - t_pos['SP'], 0))
t_pos = t_pos[t_pos['I'] > 0]

# neg
c[['SP', 'EP']] *= -1
t_neg = pd.merge(t_neg.assign(key=1), c.assign(key=1), on='key').drop('key', axis=1)
t_neg['I'] = np.where(t_neg['AnnualGrowth_TRI'] <= t_neg['EP'], t_neg['EP'] - t_neg['SP'],
                      np.where(t_neg['AnnualGrowth_TRI'] < t_neg['SP'], t_neg['AnnualGrowth_TRI'] - t_neg['SP'], 0))
t_neg = t_neg[t_neg['I'] < 0]

del c

def f_AVG_AnnualGrowth(df, SY, EY, col):
  # from end of SY to end of EY
  # f_AVG_AnnualGrowth(t, 1928, 1930, 'TotalReturnIndex') is growth for years 1929 and 1930
  pom = (df.loc[EY, col] / df.loc[SY, col] - 1)
  return (pom + 1) ** (1 / (EY - SY)) - 1

# Parameters
plt_io.templates['MyTemplate'] = plt_io.templates['plotly_dark']

max_y = df['Year'].max()

# Figure
fig = make_subplots(rows=1, cols=1)

fig.add_traces(go.Bar(x=t_pos['Year'], y=t_pos['I'],
                      marker=dict(color = t_pos['C'],
                                  colorscale=[(0, '#0d3802'), (0.2, '#176b04'), (0.4, '#32b514'), (1, '#8df674')]),
                      showlegend=False, name='',
                      customdata=t_pos['AnnualGrowth_TRI'],
                      hovertemplate='<b>%{customdata:.2%}</b>',
                      ), rows=1, cols=1)

fig.add_traces(go.Bar(x=t_neg['Year'], y=t_neg['I'],
                      marker=dict(color = t_neg['C'],
                                  colorscale=[(0, '#700000'), (0.2, '#990202'), (0.4, '#cc2525'), (1, '#ff3c3c')]),
                      showlegend=False, name='',
                      customdata=t_neg['AnnualGrowth_TRI'],
                      hovertemplate='<b>%{customdata:.2%}</b>',
                      ), rows=1, cols=1)

fig.add_trace(go.Scatter(x=df['Year'], y=df['AnnualGrowth_PI'],
                         mode='markers', name='',
                         marker=dict(size=5, color='orange'),
                         hovertemplate='%{y:.2%}<extra></extra>',
                         ), row=1, col=1)

fig.add_hline(y=0, line_width=1, line_color='white', row=1, col=1)

# Axis *************************************************************************
fig.update_xaxes(type='linear', range=[df.iloc[-1]['Year']-1, df.iloc[0]['Year']+1], showticklabels=True,
                 tickfont_size=14, tickfont_family='Arial Black')

fig.update_yaxes(ticks='outside', tickwidth=0, tickcolor='rgba(0, 0, 0, 0.0)', tickfont_size=14,
                 ticklen=10, tickformat='.0%', tickfont_family='Arial Black',
                 row=1, col=1)

fig.update_traces(marker_line_width=0)

fig.update_layout(template='MyTemplate',
                  height=500, width=1000,
                  title='<span style="font-size: 22px; font-weight: bold">S&P 500 Total Return Index</span> <br>'
                        '<span style="font-size: 16px; ">Historical Annual Growth</span>',
                  showlegend=True,
                  legend=dict(y=0.11, x=0.17, tracegroupgap=100, entrywidth=100, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'),
                  hovermode='x',
                  barmode='stack',
                  plot_bgcolor='rgba(0,0,0,0)',
                  margin=dict(l=50, r=50, t=65, b=160)
                 )

# Shape ************************************************************************
fig.add_shape(type='rect', x0=0, x1=1, y0=0, y1=1, xref='x domain', yref='y domain',
              fillcolor='#17181c', layer='below', line_width=0, row=1, col=1)

# Text *************************************************************************
fig.add_annotation(x=-0.021, y=-0.2, xref='paper', yref='paper',
    text='<span style="font-size: 16px; font-weight: bold">Average Annual Total Return </span>',
    showarrow=False, align='left')

fig.add_annotation(x=-0.021, y=-0.52, xref='paper', yref='paper',
    text='<span style="font-size: 14px;">Over nearly a century (1929-2025), <b>Average Annual Total Return</b> was around</span>'
         f'<span style="font-size: 16px; color: #8df674;"> <b>+10%</b></span>. <br>'
         '<span style="font-size: 14px;">At its inception, the Standard and Poor\'s index covered 233 companies. </span> <br>'
         '<span style="font-size: 14px;">In 1957, it was expanded to include 500 companies and became the S&P 500 index as it is known today,</span> '
         '<span style="font-size: 14px;">(1957-2025) +10.7%.</span>',
    showarrow=False, align='left')

p1 = str(round(f_AVG_AnnualGrowth(df=df.set_index('Year'), SY=1928, EY=1940, col='TotalReturnIndex') * 100, 1))
p2 = str(round(f_AVG_AnnualGrowth(df=df.set_index('Year'), SY=1940, EY=1960, col='TotalReturnIndex') * 100, 1))
p3 = str(round(f_AVG_AnnualGrowth(df=df.set_index('Year'), SY=1960, EY=1980, col='TotalReturnIndex') * 100, 1))
p4 = str(round(f_AVG_AnnualGrowth(df=df.set_index('Year'), SY=1980, EY=2000, col='TotalReturnIndex') * 100, 1))
p5 = str(round(f_AVG_AnnualGrowth(df=df.set_index('Year'), SY=2000, EY=2025, col='TotalReturnIndex') * 100, 1))
fig.add_annotation(x=-0.021, y=-0.31, xref='paper', yref='paper', showarrow=False, align='left', bgcolor='#2d365e',
                   text= '<span style="font-size: 14px; color: #a5a6b0;"> 1929-1940 </span> <span style="font-size: 16px;"><b>'+ p1 +'% </b></span>'
                         '<span style="font-size: 14px; color: #a5a6b0;">    1941-1960 </span> <span style="font-size: 16px;"><b>'+ p2 +'% </b></span>'
                         '<span style="font-size: 14px; color: #a5a6b0;">    1961-1980 </span> <span style="font-size: 16px;"><b>'+ p3 +'% </b></span>'
                         '<span style="font-size: 14px; color: #a5a6b0;">    1981-2000 </span> <span style="font-size: 16px;"><b>'+ p4 +'% </b></span>'
                         '<span style="font-size: 14px; color: #a5a6b0;">    2001-2025 </span> <span style="font-size: 16px;"><b>'+ p5 +'% </b></span>'
                   )

fig.add_annotation(x=1.02, y=1.13, xref='paper', yref='paper',
                   bgcolor='#2d365e', borderpad=3,
    text=f'&nbsp;<span style="font-size: 14px; ">Closing Values at Year-End {str(max_y)} </span> <br>'
         f'&nbsp;<span style="font-size: 14px; ">S&P 500 Total Return Index:</span> <span style="font-size: 14px; "><b>{df.loc[df['Year'] == max_y, 'TotalReturnIndex'].round(1).values[0]:.0f}</b></span> <br>'
         f'&nbsp;<span style="font-size: 14px; ">S&P 500 Index:</span> <span style="font-size: 14px; "><b>{df.loc[df['Year'] == max_y, 'PriceIndex'].round(1).values[0]:.0f}</b></span>', showarrow=False, align='left')

# Legend text
fig.add_annotation(x=0.205, y=0.123, xref='paper', yref='paper',
    text='<span style="font-size: 14px">S&P 500 Price Index</span>', showarrow=False, align='left')

#fig.show()

st.set_page_config(layout="wide")

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.plotly_chart(fig, theme=None, use_container_width=True) 
