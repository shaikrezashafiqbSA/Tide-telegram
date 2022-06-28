# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 17:47:32 2021

@author: Shaik Reza Shafiq
"""

import ccxt
import pandas as pd
import numpy as np
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
#%%
client  = ccxt.kucoin()
markets = client.load_markets()
ratelimit=0
#%%
# GET TOP MARKET CAP coins
# top_pairs
top = requests.get("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=100&page=1&sparkline=false")
#%%
# TIDE
def drawdowns(df):
  wealth_index = (df+1).cumprod()
  prev_peak = wealth_index.cummax()
  dd = (wealth_index-prev_peak)/prev_peak 
  return dd

def mpbcalc(prices,w):
    global mpb # Made global for histo for ranges
    mpb= prices['high']-prices['low']
    mpbw=(mpb.rolling(window=w).sum())
    return (mpbw.iloc[-1]-mpb.iloc[-w]+mpb.iloc[-1])/w
    
def tide0(prices,pmini=1,omx=1,sensitivity=50, thresholds=10,ticksize=0.01, mpbint=8,lengths=[5,20,67]): #,mpbm1=67,mpbm2=20,mpbm3=5
    # lengths dont affect thresholds of tide
    curr_open = prices["open"].iloc[-1]
    curr_high = prices["high"].iloc[-1]
    curr_low  = prices["low"].iloc[-1]
    
    # Arg 1 & 2
    mini_1 = [[1 if curr_open > omx else 0] if pmini == 0 else [1 if curr_open < omx else 0]][0][0]
    mini_2 = [[1 if curr_high > omx else 0] if pmini == 0 else [1 if curr_low < omx else 0]][0][0]
    mpb = prices["high"]-prices["low"]
    # Arg 3
    maxmpb=max([mpbcalc(prices,l) for l in lengths])#mpbcalc(prices,lengths[0]),mpbcalc(prices,lengths[1]),mpbcalc(prices,lengths[2]))    #THIS CAN BE CHANGED TO separate rolling functions#
    
    " Default mpbs or adaptive "
    # if m1 < 0:
    #     maxmpb_range = [2.6, 3.4, 4.5, 5.5, 6.5, 7.4, 8.2, 9.9] # HOW TO ABSTRACT THIS AWAY 
    #     additives_range = np.linspace(0.2,1,9)
    #     T_a = list(range(0,len(additives_range)))
    # else:      
    # thresholds: 1 ->most sensitive? split maxmpb range by many more quantiles
    #  if ticksize 0.01, then 100 steps to 1
    sensitivity=sensitivity/100
    maxmpb_range = list(np.quantile(mpb,np.linspace(0,1,thresholds)))
    maxmpb_range=[0]+ maxmpb_range + [np.inf]
    additives_range=np.linspace(0,np.quantile(mpb,sensitivity),len(maxmpb_range)+1)
    maxmpb_range=list(zip(maxmpb_range[0:-1],maxmpb_range[1:]))

    # m1=thresholds
    # m2=1-thresholds
    # maxmpb_range = np.linspace(np.percentile(mpb,m1),np.percentile(mpb,m2),mpbint)
    # additives_range=np.linspace(ticksize*4,ticksize*20,mpbint+1)
    i=0
    for maxmpb_range_i in maxmpb_range:
        if maxmpb_range_i[0] <= maxmpb <= maxmpb_range_i[1]:
            additive=additives_range[i]
            break
        else:
            i+=1
    # test = pd.DataFrame(list(zip(maxmpb_range, additives_range)))      
    # T_a=list(range(0,thresholds)) # mpbint
    
    # for i in T_a:
    #     if i<T_a[-1]:
    #         if maxmpb < maxmpb_range[i]:
    #             additive = additives_range[i]
    #             break
    #     else:
    #         additive=additives_range[-1]
    #         break                
 
    # Prepared additive for comparative calculation
    comparative = [omx+additive if pmini == 1 else omx-additive][0] 
    #print('Comparative: ', comparative, 'Additive: ', additive)
    
    " Compute mini_trendday "
    mini_trendday = [1 if curr_open >= comparative else 0][0] # Mini 1.0: comparative instead of omx
 
    " Arg 4: mini_3 "
    mini_3 = [[1 if curr_low < omx else 0] if mini_trendday == 1 else [1 if curr_high > omx else 0]][0][0]
    
    " Arg 5: PXMX "
    pxmx =[[1 if mini_2 == 1 else 0] if mini_3 ==1 else [[ 0 if mini_1 == 1 else 1] if mini_2 == 0 else 0]][0][0]
      
    
    " Mini Formulation "
    mini_trendday2 = [[1 if curr_low>=comparative else 0] if mini_trendday == 1 else [1 if curr_high > comparative else 0]][0][0]
    
    # New Mini for day end
    if mini_trendday == 1 and mini_trendday2 == 1 :
        nmini = 1
    elif mini_trendday == 0 and mini_trendday2 == 0 :
        nmini = 0
    else:
        nmini = pmini
        
    " MX Formulation "
    if pxmx == 1: 
        mxc = omx
    else:
        if mini_trendday == 1:
            mxc = curr_low - additive
        elif mini_trendday == 0:
            mxc = curr_high + additive
    
    # New mchange
    if mini_trendday == 1:
        if mxc <= omx:
            nmx = omx
        else: 
            nmx = mxc
    elif mini_trendday == 0:
        if mxc < omx:
            nmx = mxc
        else:
            nmx = omx
 
    #print('New Mini: ', nmini, ', New MX: ', nmx, ', New Comparative: ', comparative)
    return {'nmini':nmini,'nmx':nmx,'comparative':comparative}

def tidegenerator_sensitivity(ohlcv,ticksize,thresholds = 10, sensitivities=[10,40,90]):
    prices=ohlcv.copy()
    tidewindow = len(ohlcv)-67
  # Initialise starting params (will converge to true params after ~6 iterations)
    for sensitivity in sensitivities:
        mx=[]
        mini=[]
        maxmpbs=[]
        comparatives=[]
        additives=[]
        pmini=1
        omx = prices['high'][0]-prices['low'][0]
        
        for d in prices.index[-tidewindow:]:
          prices_d = prices.loc[:d]
          int_signals=tide0(prices_d,pmini=pmini,omx=omx,sensitivity=sensitivity,thresholds=thresholds,ticksize=ticksize)
          nmini=int_signals.get('nmini')
          nmx=int_signals.get('nmx')
          signals= tide0(prices_d,pmini=nmini,omx=nmx,sensitivity=sensitivity,thresholds=thresholds,ticksize=ticksize)
           
          # update pmini to nmini, omx to nmx
          pmini = nmini
          omx = nmx
           
          # Append lists for stats
          mx.append(omx)
          mini.append(pmini)
          maxmpbs.append(int_signals.get('maxmpb'))
          comparatives.append(signals.get('comparative')) ## THIS from int_signals to signals?
          additives.append(int_signals.get('additive'))
         
         
        # extend price df for mx,mini,comparative,additive,maxmpbs
        prices[f'tide{sensitivity}']=pd.DataFrame(mini,columns=['tide'], index=prices.index[-tidewindow :])
        prices[f'mx{sensitivity}']=pd.DataFrame(mx,columns=['mx'], index=prices.index[-tidewindow:])
        #prices['mx']=np.round(np.round((prices.mx)/ticksize)*ticksize,-int(math.floor(math.log10(ticksize)))) #round mx to the nearest ticksize---! rounds off for ticksize of dp 2 also to dp 1
        prices[f'comparative{sensitivity}']=pd.DataFrame(comparatives,columns=['comparative'], index=prices.index[-tidewindow:])
 
 
    return prices

def tidegenerator_threshold(ohlcv,ticksize, sensitivity = 0.5, thresholds_list=[10,100,1000]):
    prices=ohlcv.copy()
    tidewindow = len(ohlcv)-67
  # Initialise starting params (will converge to true params after ~6 iterations)
    for thresholds in thresholds_list:
        mx=[]
        mini=[]
        maxmpbs=[]
        comparatives=[]
        additives=[]
        pmini=1
        omx = prices['high'][0]-prices['low'][0]
        
        for d in prices.index[-tidewindow:]:
          prices_d = prices.loc[:d]
          int_signals=tide0(prices_d,pmini=pmini,omx=omx,sensitivity=sensitivity, thresholds=thresholds,ticksize=ticksize)
          nmini=int_signals.get('nmini')
          nmx=int_signals.get('nmx')
          signals= tide0(prices_d,pmini=nmini,omx=nmx,sensitivity=sensitivity,thresholds=thresholds,ticksize=ticksize)
           
          # update pmini to nmini, omx to nmx
          pmini = nmini
          omx = nmx
           
          # Append lists for stats
          mx.append(omx)
          mini.append(pmini)
          maxmpbs.append(int_signals.get('maxmpb'))
          comparatives.append(signals.get('comparative')) ## THIS from int_signals to signals?
          additives.append(int_signals.get('additive'))
         
         
        # extend price df for mx,mini,comparative,additive,maxmpbs
        prices[f'tide{thresholds}']=pd.DataFrame(mini,columns=['tide'], index=prices.index[-tidewindow :])
        prices[f'mx{thresholds}']=pd.DataFrame(mx,columns=['mx'], index=prices.index[-tidewindow:])
        #prices['mx']=np.round(np.round((prices.mx)/ticksize)*ticksize,-int(math.floor(math.log10(ticksize)))) #round mx to the nearest ticksize---! rounds off for ticksize of dp 2 also to dp 1
        prices[f'comparative{thresholds}']=pd.DataFrame(comparatives,columns=['comparative'], index=prices.index[-tidewindow:])
 
 
    return prices    
#%% 
# Sensitivity analysis
df=ohlcv.copy()
test=tidegenerator_sensitivity(df,ticksize=0.01,thresholds = 10, sensitivities=[1,90])
ax2 = test.filter(regex='mx').plot()

#%%

df=test.copy()
df["EMA20"] = ta.ema(df["close"], length=4)
df["EMA50"] = ta.ema(df["close"], length=4*7*2)
df["EMA_diff"]=df["EMA20"]-df["EMA50"]
df["EMA_pos"]=df["EMA_diff"].apply(lambda x: 1 if x>0 else -1)
df["ret"]=df["close"].pct_change().shift(-1)
df["ret_cumpnl"]=df["ret"].cumsum()
# df["EMA_cumpnl"]=df["EMA_pnl"].cumsum()
metrics={}
for tide in df.filter(regex="tide").columns:
    df[tide]=df[tide].apply(lambda x: 1 if x>0 else -1)
    # Pure tide
    exec(f"df['{tide}_pnl']=df['ret']*df['{tide}']")
    exec(f"df['{tide}_cumpnl']=df['{tide}_pnl'].cumsum()")
    # Tide + ema crossover
    exec(f"df['{tide}_cross_pnl']=df['ret']*(df['{tide}']+df['EMA_pos'])")
    exec(f"df['{tide}_cross_cumpnl']=df['{tide}_cross_pnl'].cumsum()")


tidepnl = df.filter(regex='cumpnl').plot()
#%%
df.dropna(inplace=True)
kline = go.Candlestick(x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],name="kline")

mxs = test.filter(regex='mx').columns
tides = test.filter(regex="tide").columns
plot_list=[]
plot_list.append(kline)

colors=[['rgb(82, 247, 129)','rgb(255, 114, 110)'],['rgb(30, 105, 51)','rgb(135, 25, 22)']]
for mx,tide,color_pair in zip(mxs,tides,colors):
    print(tide,mx)
    exec(f"{mx}_pos = go.Scatter(x=df.index, y=df['{mx}'].where(df['{tide}']>0), name ='{mx}+',line = dict(color='{color_pair[0]}'))")
    exec(f"plot_list.append({mx}_pos)")
    exec(f"{mx}_neg = go.Scatter(x=df.index, y=df['{mx}'].where(df['{tide}']==0), name ='{mx}-',line = dict(color='{color_pair[1]}'))")
    exec(f"plot_list.append({mx}_neg)")
# mx12h = go.Scatter(x=df.index, y=df["mx_12h"], name ='mx_12h',line = dict(color='blue'))
# mx1d = go.Scatter(x=df.index, y=df["mx_1d"], name ='mx_1d',line = dict(color='darkblue'))
# mx1d = go.Scatter(x=df.index, y=df["mx_1d"], name ='mx_1d',line = dict(color='navy'))
fig1 = go.Figure(plot_list)
fig1.show()

#%%
def gentide(pair="ETH/USDT",freqs=["4h,12h"],sensitivities=[1]): # freqs given must be in ascending order
    print(f"================ {pair} ================")
    if pair not in markets.keys():
        print(f"{pair} not in exchange catalogue")
        return None
    pair_details = markets[pair]
    min_cost = pair_details["limits"]["price"]["min"]
    df_dict = {}
    if client.has['fetchOHLCV']:
        for freq in freqs:
            print(f"... getting {freq} kline data from exchange REST API ...")
            klines = client.fetch_ohlcv(pair, freq)
            global ratelimit
            ratelimit+=1
            df = pd.DataFrame(klines)
            df.rename(columns={0:"timestamp",1:"open",2:"high",3:"low",4:"close",5:"vol"},inplace=True)
            if freq=="1d":
                df["timestamp"]=df["timestamp"]+1
            df["datetime"]=pd.to_datetime(df["timestamp"],unit="ms")
            df.set_index("datetime",inplace=True)
            df=df[["open","high","low","close","vol"]]
            print(f"... generating {freq} tides ...")
            prices=tidegenerator_sensitivity(df,ticksize=min_cost,thresholds = 10, sensitivities=sensitivities)
            prices = prices.rename(columns={col: col+f'_{freq}' 
                            for col in prices.columns if col not in ['datetime']})
            prices.dropna(inplace=True)
            prices[f"ret_{freq}"]=prices[f"close_{freq}"].pct_change().shift(-1)
            for s in sensitivities:
                prices[f"tide{s}_{freq}"]=prices[f"tide{s}_{freq}"].apply(lambda x: 1 if x>0 else -1)
                prices[f"pos{s}_{freq}"]=prices[f"tide{s}_{freq}"].shift(1)
                prices[f"pnl{s}_{freq}"]=prices[f"ret_{freq}"]*prices[f"pos{s}_{freq}"]
            prices=prices.resample(freqs[0]).last()
            prices.fillna(method="ffill",inplace=True)
            df_dict[freq]=prices

        df = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True), list(df_dict.values()))   
        # for freq in freqs:
        #     df[f"cumpnl_{freq}"]=df[f"pnl_{freq}"].cumsum()
    return df

freqs=["4h","6h"]                    
df = gentide(pair="BTC/USDT",freqs=freqs)    
df_0 = df.copy()

#%%
# Signal Analysis
df=df_0.copy()
df.dropna(inplace=True)
freq=freqs[0]
kline = go.Candlestick(x=df.index,
                open=df[f"open_{freq}"],
                high=df[f"high_{freq}"],
                low=df[f"low_{freq}"],
                close=df[f"close_{freq}"],name="kline")

mxs = df.filter(regex='mx1').columns
tides = df.filter(regex="tide1").columns
plot_list=[]
plot_list.append(kline)

colors=[['rgb(82, 247, 129)','rgb(255, 114, 110)'],['rgb(30, 105, 51)','rgb(135, 25, 22)'],['rgb(82, 247, 129)','rgb(255, 114, 110)'],['rgb(30, 105, 51)','rgb(135, 25, 22)']]
for mx,tide,color_pair in zip(mxs,tides,colors):
    print(tide,mx)
    exec(f"{mx}_pos = go.Scatter(x=df.index, y=df['{mx}'].where(df['{tide}']>0), name ='{mx}+')")#",line = dict(color='{color_pair[0]}'))")
    exec(f"plot_list.append({mx}_pos)")
    exec(f"{mx}_neg = go.Scatter(x=df.index, y=df['{mx}'].where(df['{tide}']==0), name ='{mx}-')")#",line = dict(color='{color_pair[1]}'))")
    exec(f"plot_list.append({mx}_neg)")
# mx12h = go.Scatter(x=df.index, y=df["mx_12h"], name ='mx_12h',line = dict(color='blue'))
# mx1d = go.Scatter(x=df.index, y=df["mx_1d"], name ='mx_1d',line = dict(color='darkblue'))
# mx1d = go.Scatter(x=df.index, y=df["mx_1d"], name ='mx_1d',line = dict(color='navy'))
fig1 = go.Figure(plot_list)
fig1.show()

#%%
min_freq=freqs[0]
mxs = df.filter(regex='mx').columns
df["mx_diff"]=df["mx1_6h"]-df["mx1_4h"] # mx crossover implies overbought/oversold levels, so flip positions here
df["mx_cross"]=df["mx_diff"].apply(lambda x: 1 if x>0 else -1)
df["mx_cross_pnl"]=df["mx_cross"]*df[f"ret_{min_freq}"]
df["mx_cross_cumpnl"]=df["mx_cross_pnl"].cumsum()
df[f"ret_{min_freq}_cumpnl"]=df[f"ret_{min_freq}"].cumsum()

df[["mx_cross_cumpnl",f"ret_{min_freq}_cumpnl"]].plot()
#%%        
# ax1 = df.filter(regex='cumpnl_').plot()
# ax2 = df.filter(regex='mx_').pct_change().plot()
# df[["cumpnl_4h","cumpnl_6h","cumpnl_12h"]].plot()
#%%  

# #%%
# # test riptide again
# df["diff"]=df["mx_4h"]-df["mx_1d"]
# df["xmx"]=df["diff"].apply(lambda x: 1 if x>=0 else -1)
# df["pos_xmx"]=df["xmx"].shift(1)
# df[f"pnl_xmx"]=df[f"ret_4h"]*df[f"pos_xmx"]
# df[f"cumpnl_xmx"]=df["pnl_xmx"].cumsum()
#%%
# #%%
# def riptide(pair="ETH/USDT"):
#     print(f"================ {pair} ================")
#     if pair not in markets.keys():
#         print(f"{pair} not in exchange catalogue")
#         return None
#     df_dict={}
#     freqs = ["1d","4h","1h"]
#     if client.has['fetchOHLCV']:
#         for freq in freqs:
#             print(f"... getting {freq} kline data from exchange REST API ...")
#             klines = client.fetch_ohlcv(pair, freq)
#             global ratelimit
#             ratelimit+=1
#             df = pd.DataFrame(klines)
#             df.rename(columns={0:"timestamp",1:"open",2:"high",3:"low",4:"close",5:"vol"},inplace=True)
#             if freq=="1d":
#                 df["timestamp"]=df["timestamp"]+1
#             df["datetime"]=pd.to_datetime(df["timestamp"],unit="ms")
#             df.set_index("datetime",inplace=True)
#             df=df[["open","high","low","close","vol"]]
#             print(f"... generating {freq} tides ...")
#             prices=tidegenerator(df,m1=-1,m2=-1,ticksize=0.01)
#             prices = prices.rename(columns={col: col+f'_{freq}' 
#                             for col in prices.columns if col not in ['datetime']})
#             df_dict[freq]=prices.resample("1H").last()
    
#     print("... DATA LOADED AND TESTING ...")        
#     df1 = df_dict["1h"]
#     df2 = df_dict["4h"]
#     df3 = df_dict["1d"]

#     df12=df1.merge(df2, left_index=True,right_index=True)
#     df = df12.merge(df3, left_index=True, right_index=True)
#     df.fillna(method="ffill",inplace=True)  
#     df["ret_1h"]=df["close_1h"].diff()#pct_change()
#     df.dropna(inplace=True)

#     # select buys and sells, so if mx_1h > mx_4h buy else sell
#     fee=1-(0.2/100) 
    
#     df["diff"]=df["mx_1h"]-df["mx_4h"]
#     df["pos"]=0
#     df["pos"]=df["diff"].apply(lambda x: 1 if x>=0 else -1)
#     df["pos_scale_1h"]=df["tide_1h"].apply(lambda x: 1 if x>0 else -1)
#     df["pos_scale_4h"]=df["tide_4h"].apply(lambda x: 1 if x>0 else -1)
#     df["pos_scale_1d"]=df["tide_1d"].apply(lambda x: 1 if x>0 else -1)
#     df["pos_scale"]=df["pos_scale_1h"]+df["pos_scale_4h"]+df["pos_scale_1d"]
#     df["pnl"]=df["ret_1h"]*df["pos"]*fee
#     df["scaled_pnl"]=df["pnl"]*abs(df["pos_scale"])*fee
    
    
    
#     sharpe1 = df["scaled_pnl"].cumsum().mean()/df["scaled_pnl"].cumsum().std()
#     dd1 = drawdowns(df["scaled_pnl"].cumsum().pct_change())
#     mdd1 = dd1.min()
#     df["cumpnl"]=df["pnl"].cumsum()
#     df["cumpnl_scaled"]=df["scaled_pnl"].cumsum()
    
#     fig = df[["cumpnl","cumpnl_scaled"]].plot(title=f"{pair} | sharpe: {round(sharpe1,5)} | mdd: {round(mdd1,3)}") 
#     base=str.split(pair,"/")[0]
#     quote=str.split(pair,"/")[1]
#     fig.figure.savefig(f"{base+'_'+quote}.png")
#     return fig,df
    
    
# #%%
# pair = "BTC/USDT"
# fig,df= riptide(pair=pair)
#%%
# # pair = "SOL/USDT"
# t0 =time.time()
# for pair in markets.keys():
#     while ratelimit/(time.time()-t0) > 100:
#         time.sleep(10)
#     test= riptide(pair=pair)
    
#%%    
    # klines plot
    
# #    
#     kline = go.Candlestick(x=df.index,
#                     open=df["open_1h"],
#                     high=df["high_1h"],
#                     low=df["low_1h"],
#                     close=df["close_1h"],name="kline")
    
#     mx1h = go.Scatter(x=df.index, y=df["mx_1h"], name ='mx_1h',line = dict(color='lightskyblue'))
#     mx4h = go.Scatter(x=df.index, y=df["mx_4h"], name ='mx_4h',line = dict(color='blue'))
#     # mx1d = go.Scatter(x=df.index, y=df["mx_1d"], name ='mx_1d',line = dict(color='navy'))
    
#     fig = go.Figure([kline,mx1h,mx4h])
#     fig.show()
    
#%%





#%%
# def mpbcalc(mpb,w):
#     mpbw=(mpb.rolling(window=w).sum())
#     return (mpbw.iloc[-1]-mpb.iloc[-w]+mpb.iloc[-1])/w
