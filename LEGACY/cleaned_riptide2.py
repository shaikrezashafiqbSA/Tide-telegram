#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:24:48 2021

@author: shaikrezashafiq
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 17:47:32 2021

@author: Shaik Reza Shafiq
"""

import ccxt
import pandas as pd
import numpy as np
from numpy import ndarray
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
pio.renderers.default='browser'
from datetime import datetime
import time
import requests
from functools import reduce
import pandas_ta as ta
from numba import jit
# from numba import jitclass, types, typed
#%%
# GET TOP MARKET CAP coins
# top_pairs
top = requests.get("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false")
#%%
# GET KLINE DATA FROM DB

from binance.client import Client  # need the standard client order requests ???
import pandas as pd
import sqlite3
import os


def load_db(pair="BTCUSDT",intervals=['4h','6h'],from_date="1 Jan, 2020",db='binance_kline.db'):

    available_intervals = ['1m','3m','5m','15m','30m','1h','2h','4h','6h','8h','12h','1d','3d','1w','1M']
    if not set(intervals).issubset(available_intervals):
        print(f">>>>{intervals} not in \n{available_intervals}")
    client = Client()
    conn = sqlite3.connect(db)
    for interval in intervals:
        # for futures= futures_historical_klines
        print(f">>>>interval: {interval}")
        kline = client.get_historical_klines(pair, interval, from_date)  # "1 year ago UTC"
    
        df = pd.DataFrame(kline, dtype=float,
                          columns=['openTime', 'open', 'high', 'low', 'close', 'vol', 'closeTime', 'quote_vol', 'nTrades',
                                   'takerBuy_baseAssetVol', 'takerBuy_quoteAssetVol', '_ignore'])
        
        # df = df.iloc[:-1].copy() # drop the last frame due to incomplete bar
        
        table_name = pair + '_' + interval
        print(f">>>>{table_name} loaded to db")
        df.to_sql(table_name, conn, if_exists='append', index=False)


def query_db(pair,interval,db='binance_kline.db'):
    conn = sqlite3.connect(db)
    # Check if table exists
    table_exists=f""" select count(*) from sqlite_master where type='table' and name='{pair}_{interval}' """
    test=pd.read_sql_query(table_exists, conn)
    # if test.iloc[0,0]==0:
    #     load_db(pair,intervals=[interval])
    query = f""" SELECT * from {pair}_{interval}
                ORDER BY closeTime
            """ # LIMIT 10000
    df = pd.read_sql_query(query, conn)
    df["closeTime"]+=1
    df.drop_duplicates(subset=['closeTime'], inplace=True,keep="last")
    df['date_time'] = pd.to_datetime(df['closeTime'], unit='ms')
    df.set_index(keys=['date_time'], inplace=True, drop=False)
    return df


#%%

# from tvDatafeed import TvDatafeed,Interval

# username = 'YourTradingViewUsername'
# password = 'YourTradingViewPassword'



# tv=TvDatafeed(auto_login=False, chromedriver_path='/usr/bin/chromedriver')

#%%
# =============================================================================
# TIDE1
# =============================================================================


def drawdowns(df):
  wealth_index = (df+1).cumprod()
  prev_peak = wealth_index.cummax()
  dd = (wealth_index-prev_peak)/prev_peak 
  return dd

def rolling_sum(a, n=4):
    ret = np.cumsum(a, axis=1, dtype=float)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1:]

def calc_exponential_height(OHLCVT_array,window):   ## CHECK!! 
    heights = OHLCVT_array[:,1]-OHLCVT_array[:,2]
    rolling_sum_H_L = rolling_sum([heights],n=window)[0]
    # rolling_sum_H_L = np.full(len(heights),np.nan) 
    # for idx in range(window-1,len(heights)):
    #     # print(f"summing {start_idx} to {idx}")
    #     rolling_sum_H_L[idx] = np.sum(heights[idx-window+1:idx])
    # mpbw=(heights.rolling(window=w).sum())
    return (rolling_sum_H_L[-1] - heights[-window]+heights[-1])/window #(mpbw.iloc[-1]-heights[-w]+heights[-1])/w

# @jitclass
class indicators_tide:
    # datetime agnostic
    def __init__(self):
        self.tide=list()
        self.ebb=list()
        self.flow=list()
        self.closeTime=list()
        
    def calc_tide(self,OHLCVT_array,sensitivity=50, thresholds=10,ticksize=0.01,window_scale=1,lookback_windows=[5,20,67]):
        windows=[w*window_scale for w in lookback_windows]
        # Checks first is OHLCVT_array contains enough data to start calculating tides
        max_lookback=max(windows)
        if max_lookback>len(OHLCVT_array):
            # print("not enough data")
            self.tide.append(np.nan)
            self.ebb.append(np.nan)
            self.flow.append(np.nan)
            self.closeTime.append(OHLCVT_array[-1,-1])
            return {'tide':np.nan,'ebb':np.nan,'flow':np.nan}
        
        new_open = OHLCVT_array[-1,0]
        new_high = OHLCVT_array[-1,1]
        new_low  = OHLCVT_array[-1,2]
        
        if (len(self.tide) + len(self.ebb) + len(self.flow) == 0):
            previous_tide=1
            previous_ebb=1
        elif np.isnan(self.tide[-1]):
            previous_tide=1
            previous_ebb=1
        else:
            # print("using prev tide indicators")
            previous_tide = self.tide[-1]
            previous_ebb  = self.ebb[-1]
            
            
        # ====================================================================
        # START SIGNAL CALCULATION
        # ====================================================================
         # undertow = [[1 if new_open > previous_ebb else 0] if previous_tide == 0 else [1 if new_open < previous_ebb else 0]][0][0]
        " undertow "
        if previous_tide:
            if new_open<previous_ebb:
                undertow=1
            else:
                undertow=0
        else:
            if new_open > previous_ebb:
                undertow=1
            else:
                undertow=0
        
        # surftow = [[1 if new_high > previous_ebb else 0] if previous_tide == 0 else [1 if new_low < previous_ebb else 0]][0][0]
        " surftow "
        if previous_tide:
            if new_low < previous_ebb:
                surftow=1
            else:
                surftow=0
        else:
            if new_high > previous_ebb:
                surftow=1
            else:
                surftow=0
            
        
        " Calculate change in tide: flow"
        
        # heights = df["high"][-67:]-df["low"][-67:]
        heights = OHLCVT_array[:,1]-OHLCVT_array[:,2]
        heights = heights[-67:]
    
        max_exp_height=max([calc_exponential_height(OHLCVT_array,w) for w in windows])#calc_exponential_height(prices,lengths[0]),calc_exponential_height(prices,lengths[1]),calc_exponential_height(prices,lengths[2]))    #THIS CAN BE CHANGED TO separate rolling functions#
    
    
        # sensitivity=sensitivity/100
        max_exp_height_ranges = list(np.quantile(heights,np.linspace(0,1,thresholds)))
        max_exp_height_ranges=[0]+ max_exp_height_ranges + [np.inf]
        additives_range=np.linspace(0,np.quantile(heights,sensitivity/100),len(max_exp_height_ranges)+1)
        max_exp_height_ranges=list(zip(max_exp_height_ranges[0:-1],max_exp_height_ranges[1:]))
    
        i=0
        for maxmpb_range_i in max_exp_height_ranges:
            if maxmpb_range_i[0] <= max_exp_height <= maxmpb_range_i[1]:
                additive=additives_range[i]
                break
            else:
                i+=1
                
        " flow "
        # flow = [previous_ebb+additive if previous_tide == 1 else previous_ebb-additive][0] 
        if previous_tide:
            flow=previous_ebb + additive
        else: 
            flow=previous_ebb - additive
        
    
        
        " interim tides "
        # tide_1 = [1 if new_open >= flow else 0][0] 
        if new_open >= flow:
            tide_1=1
        else:
            tide_1=0
     
        # tide_2 = [[1 if new_low < previous_ebb else 0] if tide_1 == 1 else [1 if new_high > previous_ebb else 0]][0][0]
        if tide_1:
            if new_low < previous_ebb:
                tide_2=1
            else:
                tide_2=0
        else:
            if new_high > previous_ebb:
                tide_2=1
            else:
                tide_2=0
    
        # tide_3 =[[1 if surftow == 1 else 0] if tide_2 ==1 else [[ 0 if undertow == 1 else 1] if surftow == 0 else 0]][0][0]
        if tide_2:
            if surftow:
                tide_3=1
            else:
                tide_3=0
        else:
            if surftow:
                if undertow:
                    tide_3=0
                else:
                    tide_3=1
            else:
                tide_3=0
          
        # tide_4 = [[1 if new_low>=flow else 0] if tide_1 == 1 else [1 if new_high > flow else 0]][0][0]
        if tide_1:
            if new_low >= flow:
                tide_4=1
            else:
                tide_4=0
        else:
            if new_high > flow:
                tide_4=1
            else:
                tide_4=0
        
        
        " tide formulation "
        if tide_1 == 1 and tide_4 == 1 :
            new_tide = 1
        elif tide_1 == 0 and tide_4 == 0 :
            new_tide = 0
        else:
            new_tide = previous_tide
            
        " ebb Formulation "
        if tide_3 == 1: 
            interim_ebb = previous_ebb
        else:
            if tide_1 == 1:
                interim_ebb = new_low - additive
            elif tide_1 == 0:
                interim_ebb = new_high + additive
        
        " new ebb "
        if tide_1 == 1:
            if interim_ebb <= previous_ebb:
                new_ebb = previous_ebb
            else: 
                new_ebb = interim_ebb
        elif tide_1 == 0:
            if interim_ebb < previous_ebb:
                new_ebb = interim_ebb
            else:
                new_ebb = previous_ebb
                
        # ====================================================================
        # END SIGNAL CALCULATION
        # ====================================================================     

        self.tide.append(new_tide)
        self.ebb.append(new_ebb)
        self.flow.append(flow)
        self.closeTime.append(OHLCVT_array[-1,-1])
        
        return {'tide':new_tide,'ebb':new_ebb,'flow':flow}



class order_management:
    def __init__(self):
        self.long = []
        self.short= []
        self.pnl= []
    def golong(self,price,closeTime):
        entry={'closeTime':closeTime,'price':price}
        self.long.append(entry)
    def goshort(self,price,closeTime):
        entry={'closeTime':closeTime,'price':price}
        self.short.append(entry)
    

        # need current position, mark to market
#%%
# test tide0 speed
pair="BTCUSDT"
freq="1m"
klines = query_db(pair,freq)

df0 = pd.DataFrame(klines)
cols_to_keep = ["open","high","low","close","vol","closeTime"]
df0=df0[cols_to_keep]


#%%
df=df0.copy().tail(20000)
ar=df.to_numpy()







#%%
# =============================================================================
# GENERATION TEST (in streaming backtester format)
# =============================================================================
# print("============================================================================\nTIDE1 TIME\n============================================================================")

# send in updates as rows of pandas and np.array
# drip OHLCV... from stream


# @jit()
def backtest_slow_vs_fast(ar,freq=5,sensitivity=50, thresholds=10,lookback_windows=[5,20,67],ticksize=0.01,window_scale=1):
    tides_fast = indicators_tide()
    tides_slow = indicators_tide()
    orders_1 = order_management()
    orders_2 = order_management()
    ar1_collected = ndarray((1,6),np.float64) # should i collect entry prices here
    ar2_collected = ndarray((1,6),np.float64)
    i1=0
    i2=0
    for row in ar[0:]:
        
        # =============================================================================
        # Organise collected OHLCV 
        # =============================================================================
        ar1_collected = np.append(ar1_collected,[row],axis=0)
        # Drop first row of rubbish
        if i1==0:
            ar1_collected=np.delete(ar1_collected,0,0)
        i1+=1
        
        # =============================================================================
        #  Slow Signal
        # =============================================================================
        if (not row[5]/1000%(60*freq)) and (i1>=5):
            # Should collect OHLCV for past {freq} bars
            open = ar1_collected[-freq,0]
            high = np.max(ar1_collected[-freq:,1])
            low  = np.min(ar1_collected[-freq:,2])
            close= ar1_collected[-1,3]
            vol = np.sum(ar1_collected[-freq:,4]) 
            closeTime = ar1_collected[-1,5]
                
            ar2_collected = np.append(ar2_collected,[[open,high,low,close,vol,closeTime]],axis=0)
            if i2==0:
                ar2_collected = np.delete(ar2_collected,0,0)
            i2+=1
                
            # _________________
            # SIGNAL GENERATION
            # _________________
            tide_slow=tides_slow.calc_tide(ar2_collected,sensitivity=sensitivity, thresholds = thresholds, ticksize=ticksize,lookback_windows=lookback_windows)
            
            # ______________
            # ORDER DECISION
            # ______________
            if len(tides_slow.tide)>max(lookback_windows):
                if tides_slow.tide[-2] > tides_slow.tide[-1]:
                    # NEW + TIDE! 
                    orders_2.goshort(price=close,closeTime=closeTime)
                elif tides_slow.tide[-2] < tides_slow.tide[-1]:
                    orders_2.golong(price=close,closeTime=closeTime) 
                    
        # =============================================================================
        #  Fast Signal
        # =============================================================================   
        
        # _________________
        # SIGNAL GENERATION
        # _________________
        tide_fast=tides_fast.calc_tide(ar1_collected,sensitivity=sensitivity, thresholds = thresholds, ticksize=ticksize,lookback_windows=lookback_windows)
        
        # ______________
        # ORDER DECISION
        # ______________
        if len(tides_fast.tide)>max(lookback_windows):
            if tides_fast.tide[-2] > tides_fast.tide[-1]:
                # NEW + TIDE! 
                orders_1.goshort(price=row[3],closeTime=row[5])
            elif tides_fast.tide[-2] < tides_fast.tide[-1]:
                orders_1.golong(price=row[3],closeTime=row[5])       
    # return ar1_collected,tides_fast,ar2_collected,tides_slow   
    return ar1_collected, tides_fast, tides_slow, orders_1, orders_2
        

        
        
        
        
#%%

t0=time.time()


ar1,tides_fast,tides_slow,orders_fast, orders_slow= backtest_slow_vs_fast(ar,freq=3,sensitivity=90, thresholds=10,ticksize=0.01,window_scale=1,lookback_windows=[60,120,600])


print(f"time taken tide1: {time.time()-t0}")


#%%
# performance




#%%
# plotly studies
freqs = ["fast","slow"]
idf_dict={}
order_list=[]
for freq in freqs: 
    # Order df
    for orderside in ["long","short"]:
        order_df = eval(f"pd.DataFrame(orders_{freq}.{orderside})")
        order_df['date_time'] = pd.to_datetime(order_df['closeTime'], unit='ms')
        order_df.drop(columns=["closeTime"],inplace=True)
        order_df.set_index(keys=['date_time'], inplace=True, drop=True)
        order_df.rename(columns={"price":f"{orderside}_{freq}"},inplace=True)
        # order_dict[f"{freq}_{orderside}"] = order_df
        order_list.append(order_df)
    # indicators
    idf = eval(f"pd.DataFrame(list(zip(tides_{freq}.closeTime, tides_{freq}.tide,tides_{freq}.ebb)),columns=['closeTime','tide_{freq}','ebb_{freq}'])")
    idf.dropna(inplace=True)
    idf['date_time'] = pd.to_datetime(idf['closeTime'], unit='ms')
    idf.set_index(keys=['date_time'], inplace=True, drop=True)
    idf.drop(columns=["closeTime"],inplace=True)
    idf=idf.resample("1T").last()
    idf.fillna(method="ffill",inplace=True)
    
    idf_dict[freq]=idf
    # ar_freq = np.append(ar1,indicators,axis=1)
    # indicators_1m = np.array(list(zip(tides_1m.tide,tides_1m.ebb)))
    # indicators_5m = np.array(list(zip(tides_5m.tide,tides_5m.ebb)))
idf = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True), list(idf_dict.values()))   

# for freq in freqs: construct dfs
df2 = pd.DataFrame(ar1)
df2.rename(columns=dict(zip(range(0,df2.shape[1]),cols_to_keep)),inplace=True)
df2['date_time'] = pd.to_datetime(df2['closeTime'], unit='ms')
df2.drop(columns=["closeTime"],inplace=True)
df2.set_index(keys=['date_time'], inplace=True, drop=True) 
df2.dropna(inplace=True)
df3=pd.merge(df2,idf, how='inner', left_index=True, right_index=True)

# Add order infos to df3
for order_df  in order_list:
    df3=pd.merge(df3,order_df, how='outer', left_index=True, right_index=True)
df3=df3[df3["open"].notna()]

df4=df3.copy().tail(1000)
kline = go.Candlestick(x=df4.index,
                open=df4[f"open"],
                high=df4[f"high"],
                low=df4[f"low"],
                close=df4[f"close"],name="kline")
plot_list = [kline] 
colors = [['rgb(115, 211, 255)','rgb(246, 255, 0)'],['rgb(34, 11, 153)','rgb(189, 168, 15)']] #lightblue/lightyellow  , blue/yellow
for freq,color in zip(freqs,colors):
    # print(color) 
    tide_positive = eval(f"go.Scatter(x=df4.index, y=df4['ebb_{freq}'].where(df4['tide_{freq}']>0), name ='ebb{freq}+',mode='lines', marker= dict(color='{color[0]}'))") # blue
    plot_list.append(tide_positive)
    tide_negative = eval(f"go.Scatter(x=df4.index, y=df4['ebb_{freq}'].where(df4['tide_{freq}']==0), name ='ebb{freq}-',mode='lines', marker= dict(color='{color[1]}'))") # yellow
    plot_list.append(tide_negative)
    # plot orders
    go_long = eval(f"go.Scatter(x=df4.index, y=df4['long_{freq}'], name ='tide_{freq} long',mode='markers',marker_symbol= 'arrow-up-open', marker= dict(color='{color[0]}'))") 
    plot_list.append(go_long)
    go_short = eval(f"go.Scatter(x=df4.index, y=df4['short_{freq}'], name ='tide_{freq} short',mode='markers',marker_symbol= 'arrow-down-open', marker= dict(color='{color[1]}'))") 
    plot_list.append(go_short)
    
# =============================================================================
# # simple returns
# =============================================================================
df4["ret"]=df4["close"].pct_change().shift(-1)
df4["cumret"]=df4["ret"].cumsum()
buyhold = go.Scatter(x=df4.index, y=df4["cumret"],name=f"buyhold {pair}",marker_color="black")

# =============================================================================
# fast vs slow tide
# =============================================================================
plot_list_tideret=[]
for freq in freqs:
    df4[f"pos_{freq}"]=df4[f"tide_{freq}"].apply(lambda x: 1 if x>0 else -1)
    df4[f"ret_{freq}"] = df4[f"pos_{freq}"]*df4["ret"]
    df4[f"cumret_{freq}"]=df4[f"ret_{freq}"].cumsum()
    plottideret = go.Scatter(x=df4.index, y=df4[f"cumret_{freq}"],name=f"tide_{freq}")
    # plottideret = go.Bar(x=df4.index, y=df4[f"cumret_{freq}"],name=f"tide_{freq}")
    plot_list_tideret.append(plottideret)
    
# =============================================================================
# # tide studies
# =============================================================================
# histogram 
df4["diff"]=df4[f'ebb_{freqs[1]}']-df4[f'ebb_{freqs[0]}']
df4["diff_change"]=df4["diff"]

df4["tide_z"] = df4["diff"]#.diff()
study = go.Bar(x=df4.index, y=df4["tide_z"],name="tide_z",marker_color="darkred")
plot_list.append(study)

# add experimental signal to lot
df4["tide_z_pos"]=df4["tide_fast"]+df4["tide_slow"]
df4["ret_z"] = df4["tide_z_pos"]*df4["ret"]
df4["cumret_z"]=df4["ret_z"].cumsum()
z = go.Scatter(x=df4.index,y=df4["cumret_z"],name="test")
plot_list.append(z)

# =============================================================================
# collate plot list
# =============================================================================
plot_list.append(buyhold)
plot_list.append(plot_list_tideret[0])    
plot_list.append(plot_list_tideret[1])    


# =============================================================================
# PLOT
# =============================================================================
fig = make_subplots(rows=3, cols=1,
                    subplot_titles=(f'{pair}', f'ebb diff'),
                    shared_xaxes=True,row_heights=[0.7,0.2,0.1],vertical_spacing=0.05)
fig.add_traces(plot_list, rows=[1,1,1,1,1,1,1,1,1,2,3,3,3,3],cols=[1,1,1,1,1,1,1,1,1,1,1,1,1,1])  
fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(hovermode="x unified")
fig.show()

# freqs = ["1m","5m"]
# idf_dict={}
# for freq in freqs: 
#     idf = eval(f"pd.DataFrame(list(zip(tides_{freq}.closeTime, tides_{freq}.tide,tides_{freq}.ebb)),columns=['closeTime','tide_{freq}','ebb_{freq}'])")
#     idf.dropna(inplace=True)
#     idf['date_time'] = pd.to_datetime(idf['closeTime'], unit='ms')
#     idf.set_index(keys=['date_time'], inplace=True, drop=True)
#     idf.drop(columns=["closeTime"],inplace=True)
#     idf=idf.resample("1T").last()
#     idf.fillna(method="ffill",inplace=True)
    
#     idf_dict[freq]=idf
#     # ar_freq = np.append(ar1,indicators,axis=1)
#     # indicators_1m = np.array(list(zip(tides_1m.tide,tides_1m.ebb)))
#     # indicators_5m = np.array(list(zip(tides_5m.tide,tides_5m.ebb)))
# idf = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True), list(idf_dict.values()))   

# # for freq in freqs: construct dfs
# df2 = pd.DataFrame(ar1)
# df2.rename(columns=dict(zip(range(0,df2.shape[1]),cols_to_keep)),inplace=True)
# df2['date_time'] = pd.to_datetime(df2['closeTime'], unit='ms')
# df2.drop(columns=["closeTime"],inplace=True)
# df2.set_index(keys=['date_time'], inplace=True, drop=True) 
# df2.dropna(inplace=True)
# df3=pd.merge(df2,idf, how='inner', left_index=True, right_index=True)


# df4=df3.copy().tail(1000)
# kline = go.Candlestick(x=df4.index,
#                 open=df4[f"open"],
#                 high=df4[f"high"],
#                 low=df4[f"low"],
#                 close=df4[f"close"],name="kline")
# plot_list = [kline] 
# colors = [['rgb(115, 211, 255)','rgb(246, 255, 0)'],['rgb(34, 11, 153)','rgb(189, 168, 15)']] #lightblue/lightyellow  , blue/yellow
# for freq,color in zip(freqs,colors):
#     print(color) 
#     tide_positive = eval(f"go.Scatter(x=df4.index, y=df4['ebb_{freq}'].where(df4['tide_{freq}']>0), name ='ebb{freq}+',mode='lines', marker= dict(color='{color[0]}'))") # blue
#     plot_list.append(tide_positive)
#     tide_negative = eval(f"go.Scatter(x=df4.index, y=df4['ebb_{freq}'].where(df4['tide_{freq}']==0), name ='ebb{freq}-',mode='lines', marker= dict(color='{color[1]}'))") # yellow
#     plot_list.append(tide_negative)

# # tide studies
# df4["diff"]=df4[f'ebb_{freqs[0]}']-df4[f'ebb_{freqs[1]}']
# df4["diff_change"]=df4["diff"]

# df4["tide_z"] = df4["diff"]#.diff()
# study = go.Bar(x=df4.index, y=df4["tide_z"],name="tide_z",marker_color="darkred")

# plot_list.append(study)


# # simple returns
# df4["ret"]=df4["close"].pct_change().shift(-1)
# df4["cumret"]=df4["ret"].cumsum()
# buyhold = go.Scatter(x=df4.index, y=df4["cumret"],name=f"buyhold {pair}")
# plot_list.append(buyhold)
# for freq in freqs:
#     df4[f"pos_{freq}"]=df4[f"tide_{freq}"].apply(lambda x: 1 if x>0 else -1)
#     df4[f"ret_{freq}"] = df4[f"pos_{freq}"]*df4["ret"]
#     df4[f"cumret_{freq}"]=df4[f"ret_{freq}"].cumsum()
#     plottideret = go.Scatter(x=df4.index, y=df4[f"cumret_{freq}"],name=f"tide_{freq}")
#     plot_list.append(plottideret)
    
    
# fig = make_subplots(rows=3, cols=1,
#                     subplot_titles=(f'{pair}', f'ebb diff'),
#                     shared_xaxes=True,row_heights=[0.7,0.2,0.1],vertical_spacing=0.05)
# fig.add_traces(plot_list, rows=[1,1,1,1,1,2,3,3,3],cols=[1,1,1,1,1,1,1,1,1])  
# fig.update_layout(xaxis_rangeslider_visible=False)
# fig.update_layout(hovermode="x unified")
# fig.show()


#%%

# =============================================================================
# Same frequency tide studies
# =============================================================================

def event_driven_backtest(ar,freq=5,sensitivity1=50, thresholds1=10,lookback_windows1=[5,20,67],sensitivity2=50, thresholds2=10,lookback_windows2=[5,20,67],ticksize=0.01,window_scale=1):
    tides_1 = indicators_tide()
    tides_2 = indicators_tide()
    orders_1 = order_management()
    orders_2 = order_management()
    # tides_5m = indicators_tide()
    ar1_collected = ndarray((1,6),np.float64) # should i collect entry prices here
    # ar2_collected = ndarray((1,6),np.float64)
    i1=0
    i2=0
    for row in ar[0:]:
        
        # =============================================================================
        # Organise collected OHLCV 
        # =============================================================================
        ar1_collected = np.append(ar1_collected,[row],axis=0)
        # Drop first row of rubbish
        if i1==0:
            ar1_collected=np.delete(ar1_collected,0,0)
        i1+=1
        
        
        tide1_1m=tides_1.calc_tide(ar1_collected,sensitivity=sensitivity1, thresholds = thresholds1, lookback_windows=lookback_windows1, ticksize=ticksize)
        tide2_1m=tides_2.calc_tide(ar1_collected,sensitivity=sensitivity2, thresholds = thresholds2, lookback_windows=lookback_windows2, ticksize=ticksize)
        # print(tide1_i)
        
        # =============================================================================
        # TRADE DECISION (after warming up tide)
        # =============================================================================
        if len(tides_1.tide)>2:
            if tides_1.tide[-2] > tides_1.tide[-1]:
                # NEW + TIDE! 
                orders_1.goshort(price=row[3],closeTime=row[5])
            elif tides_1.tide[-2] < tides_1.tide[-1]:
                orders_1.golong(price=row[3],closeTime=row[5])     
        if len(tides_2.tide)>2:
            if tides_2.tide[-2] > tides_2.tide[-1]:
                # NEW + TIDE! 
                orders_2.goshort(price=row[3],closeTime=row[5])
            elif tides_2.tide[-2] < tides_2.tide[-1]:
                orders_2.golong(price=row[3],closeTime=row[5])   
    # return ar1_collected,tides_1m,ar2_collected,tides_5m   
    return ar1_collected, tides_1, tides_2, orders_1, orders_2
        
        
        
        
#%%

t0=time.time()
# ar1,tides_1m,ar2,tides_5m = event_driven_backtest(ar,2,sensitivity=90, thresholds=10,ticksize=0.0001,window_scale=1,lookback_windows=[60,120,600])
ar1,tides_1,tides_2,orders_1, orders_2 = event_driven_backtest(ar,2,sensitivity1=50, thresholds1=10,lookback_windows1=[60,120,360],
                                                                    sensitivity2=50, thresholds2=10,lookback_windows2=[600,1200,3600],
                                                                    ticksize=0.0001,window_scale=1)

print(f"time taken tide1: {time.time()-t0}")

#%%
freqs = ["1","2"]
idf_dict={}
order_list=[]
for freq in freqs: 
    # Order df
    for orderside in ["long","short"]:
        order_df = eval(f"pd.DataFrame(orders_{freq}.{orderside})")
        order_df['date_time'] = pd.to_datetime(order_df['closeTime'], unit='ms')
        order_df.drop(columns=["closeTime"],inplace=True)
        order_df.set_index(keys=['date_time'], inplace=True, drop=True)
        order_df.rename(columns={"price":f"{orderside}_{freq}"},inplace=True)
        # order_dict[f"{freq}_{orderside}"] = order_df
        order_list.append(order_df)
    # indicators
    idf = eval(f"pd.DataFrame(list(zip(tides_{freq}.closeTime, tides_{freq}.tide,tides_{freq}.ebb)),columns=['closeTime','tide_{freq}','ebb_{freq}'])")
    idf.dropna(inplace=True)
    idf['date_time'] = pd.to_datetime(idf['closeTime'], unit='ms')
    idf.set_index(keys=['date_time'], inplace=True, drop=True)
    idf.drop(columns=["closeTime"],inplace=True)
    idf=idf.resample("1T").last()
    idf.fillna(method="ffill",inplace=True)
    
    idf_dict[freq]=idf
    # ar_freq = np.append(ar1,indicators,axis=1)
    # indicators_1m = np.array(list(zip(tides_1m.tide,tides_1m.ebb)))
    # indicators_5m = np.array(list(zip(tides_5m.tide,tides_5m.ebb)))
idf = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True), list(idf_dict.values()))   

# for freq in freqs: construct dfs
df2 = pd.DataFrame(ar1)
df2.rename(columns=dict(zip(range(0,df2.shape[1]),cols_to_keep)),inplace=True)
df2['date_time'] = pd.to_datetime(df2['closeTime'], unit='ms')
df2.drop(columns=["closeTime"],inplace=True)
df2.set_index(keys=['date_time'], inplace=True, drop=True) 
df2.dropna(inplace=True)
df3=pd.merge(df2,idf, how='inner', left_index=True, right_index=True)

# Add order infos to df3
for order_df  in order_list:
    df3=pd.merge(df3,order_df, how='outer', left_index=True, right_index=True)


df4=df3.copy().tail(1000)
kline = go.Candlestick(x=df4.index,
                open=df4[f"open"],
                high=df4[f"high"],
                low=df4[f"low"],
                close=df4[f"close"],name="kline")
plot_list = [kline] 
colors = [['rgb(115, 211, 255)','rgb(246, 255, 0)'],['rgb(34, 11, 153)','rgb(189, 168, 15)']] #lightblue/lightyellow  , blue/yellow
for freq,color in zip(freqs,colors):
    print(color) 
    tide_positive = eval(f"go.Scatter(x=df4.index, y=df4['ebb_{freq}'].where(df4['tide_{freq}']>0), name ='ebb{freq}+',mode='lines', marker= dict(color='{color[0]}'))") # blue
    plot_list.append(tide_positive)
    tide_negative = eval(f"go.Scatter(x=df4.index, y=df4['ebb_{freq}'].where(df4['tide_{freq}']==0), name ='ebb{freq}-',mode='lines', marker= dict(color='{color[1]}'))") # yellow
    plot_list.append(tide_negative)
    # plot orders
    go_long = eval(f"go.Scatter(x=df4.index, y=df4['long_{freq}'], name ='tide_{freq} long',mode='markers',marker_symbol= 'arrow-up-open', marker= dict(color='{color[0]}'))") 
    plot_list.append(go_long)
    go_short = eval(f"go.Scatter(x=df4.index, y=df4['short_{freq}'], name ='tide_{freq} short',mode='markers',marker_symbol= 'arrow-down-open', marker= dict(color='{color[1]}'))") 
    plot_list.append(go_short)
# tide studies
df4["diff"]=df4[f'ebb_{freqs[0]}']-df4[f'ebb_{freqs[1]}']
df4["diff_change"]=df4["diff"]

df4["tide_z"] = df4["diff"]#.diff()
study = go.Bar(x=df4.index, y=df4["tide_z"],name="tide_z",marker_color="darkred")

plot_list.append(study)


# simple returns
df4["ret"]=df4["close"].pct_change().shift(-1)
df4["cumret"]=df4["ret"].cumsum()
buyhold = go.Scatter(x=df4.index, y=df4["cumret"],name=f"buyhold {pair}")
plot_list.append(buyhold)
for freq in freqs:
    df4[f"pos_{freq}"]=df4[f"tide_{freq}"].apply(lambda x: 1 if x>0 else -1)
    df4[f"ret_{freq}"] = df4[f"pos_{freq}"]*df4["ret"]
    df4[f"cumret_{freq}"]=df4[f"ret_{freq}"].cumsum()
    plottideret = go.Scatter(x=df4.index, y=df4[f"cumret_{freq}"],name=f"tide_{freq}")
    plot_list.append(plottideret)
    
    
fig = make_subplots(rows=3, cols=1,
                    subplot_titles=(f'{pair}', f'ebb diff'),
                    shared_xaxes=True,row_heights=[0.7,0.2,0.1],vertical_spacing=0.05)
fig.add_traces(plot_list, rows=[1,1,1,1,1,1,1,1,1,2,3,3,3],cols=[1,1,1,1,1,1,1,1,1,1,1,1,1])  
fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_layout(hovermode="x unified")
fig.show()



# fig1 = go.FigureWidget(data=plot_list,layout=layout)
# fig.update_layout(layout=layout)
#######
# fig.update_layout(xaxis_rangeslider_visible=False)
# fig.update_layout(hovermode="x unified")
# fig.show()
# #%%
# layout = dict(
#     title=f'{pair} {freq}',
#     xaxis=dict(
#         rangeselector=dict(
#             buttons=list([
#                 dict(count=1,
#                      label='1m',
#                      step='month',
#                      stepmode='backward'),
#                 dict(count=6,
#                      label='6m',
#                      step='month',
#                      stepmode='backward'),
#                 dict(count=1,
#                     label='YTD',
#                     step='year',
#                     stepmode='todate'),
#                 dict(count=1,
#                     label='1y',
#                     step='year',
#                     stepmode='backward'),
#                 dict(step='all')
#             ])
#         ),
#         rangeslider=dict(
#             visible = True
#         ),
#         type='date'
#     )
# )


# fig1 = go.FigureWidget(data=plot_list,layout=layout)
# fig1.update_layout(xaxis_rangeslider_visible=False)
# fig1.update_layout(hovermode="x")
# fig1.show()