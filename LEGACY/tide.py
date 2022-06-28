
from tvDatafeed import TvDatafeed,Interval
tv=TvDatafeed(username='shaikannuar',password='Trading651651!')

#%%
#@title Import modules
 
from IPython.display import Javascript
 
import pandas as pd
import numpy as np
import math
from scipy.stats import norm, kurtosis, skew
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from collections import OrderedDict
 
from datetime import date
from datetime import datetime
 
import sys, os

"""# 0) Model Functions

## a) Tide
"""

#@title Tide functions { form-width: "10px" }
"""
 
Tide Model =====================================================================================================================================================================
 
"""
 
 
##  TIDE Formula ---------------------------------------------------------------
def tide0(prices,pmini=1,omx=1, m1=-1,m2=-1,ticksize=0.1,mpbm1=67,mpbm2=20,mpbm3=5, mpbint=8):
    curr_open = prices.Open.iloc[-1]
    curr_high = prices.High.iloc[-1]
    curr_low  = prices.Low.iloc[-1]
    
    # Arg 1 & 2
    mini_1 = [[1 if curr_open > omx else 0] if pmini == 0 else [1 if curr_open < omx else 0]][0][0]
    mini_2 = [[1 if curr_high > omx else 0] if pmini == 0 else [1 if curr_low < omx else 0]][0][0]
    # Arg 3
    def mpbcalc(prices,w):
        global mpb # Made global for histo for ranges
        mpb= prices['High']-prices['Low']
        mpbw=(mpb.rolling(window=w).sum())
        return (mpbw.iloc[-1]-mpb.iloc[-w]+mpb.iloc[-1])/w
    
    maxmpb=max(mpbcalc(prices,mpbm1),mpbcalc(prices,mpbm2),mpbcalc(prices,mpbm3))    #THIS CAN BE CHANGED TO separate rolling functions#
    
    " Default mpbs or adaptive "
    mpb1=mpb.iloc[-126:]
    if m1 < 0:
        maxmpb_range = [2.6, 3.4, 4.5, 5.5, 6.5, 7.4, 8.2, 9.9]
        additives_range = np.linspace(0.2,1,9)
        T_a = list(range(0,len(additives_range)))
    else:      
        maxmpb_range = np.linspace(np.percentile(mpb,m1),np.percentile(mpb,m2),mpbint)
        additives_range=np.linspace(ticksize*4,ticksize*20,mpbint+1)# *ticksize/0.1
        T_a=list(range(0,mpbint))
 
    for i in T_a:
        if i<T_a[-1]:
            if maxmpb < maxmpb_range[i]:
                additive = additives_range[i]
                break
        else:
            additive=additives_range[-1]
            break                
 
            
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
    return {'nmini':nmini,'nmx':nmx,'comparative':comparative,'maxmpb':maxmpb,'additive':additive,'mpb':mpb}
 
 
 
 
"""
 
Tide Strengths =====================================================================================================================================================================
 
"""
 
## Tide Strengths calculator ----------------------------------------------------
def tidestrength(prices,ticksize,windows=[3,6,9,12],formula='maxdrawup_open'):
  dft=prices.copy()
  dfs = [g for i,g in dft.groupby(dft['tide'].ne(dft['tide'].shift()).cumsum())]
  dft_pos = [df for df in dfs if df.tide.sum() != 0]
  dft_neg = [df for df in dfs if df.tide.sum() == 0]
 
 
  pos_durs=[]
  pos_strs=[]
  neg_durs=[]
  neg_strs=[]
  if formula == 'maxdrawup':
    for cycles in windows:
      pos_durs.append([df.shape[0] for df in dft_pos[-cycles:]])
      pos_strs.append([roundtotick(df.High.max() - df['Low'][0],ticksize) for df in dft_pos[-cycles:]])
      neg_durs.append([df.shape[0] for df in dft_neg[-cycles:]])
      neg_strs.append([roundtotick(df['High'][0] - df.Low.min(),ticksize) for df in dft_neg[-cycles:]])
  elif formula == 'maxdrawup_open':
    for cycles in windows:
      pos_durs.append([df.shape[0] for df in dft_pos[-cycles:]])
      pos_strs.append([roundtotick(df.High.max() - df['Open'][0],ticksize) for df in dft_pos[-cycles:]])
      neg_durs.append([df.shape[0] for df in dft_neg[-cycles:]])
      neg_strs.append([roundtotick(df['Open'][0] - df.Low.min(),ticksize) for df in dft_neg[-cycles:]])
  else:
    for cycles in windows:
      pos_durs.append([df.shape[0] for df in dft_pos[-cycles:]])
      pos_strs.append([roundtotick(df.High.max() - df.Low.min(),ticksize) for df in dft_pos[-cycles:]])
      neg_durs.append([df.shape[0] for df in dft_neg[-cycles:]])
      neg_strs.append([roundtotick(df.High.max() - df.Low.min(),ticksize) for df in dft_neg[-cycles:]])
    
  return {'+durations':pos_durs,'-durations':neg_durs,'+strengths':pos_strs,'-strengths':neg_strs}
 
 
 
""" 
 
TIDE GENERATOR =====================================================================================================================================================================
 
"""
 
## Tide generator --------------------------------------------------------------
def tidegenerator(prices,m1,m2,ticksize,length_dh=7,length_dl=7,tidestr=None,str_formula='maxdrawup'):
  tidewindow = len(prices)-67
  # Initialise starting params (will converge to true params after ~6 iterations)
  mx=[]
  mini=[]
  maxmpbs=[]
  comparatives=[]
  additives=[]
  pmini=1
  omx = prices['High'][0]-prices['Low'][0]
 
 
 
  # Generate tides for every data point
  for d in prices.index[-tidewindow:]:
    prices_d = prices.loc[:d]
    int_signals=tide0(prices_d,pmini=pmini,omx=omx,m1=m1,m2=m2,ticksize=ticksize)
    nmini=int_signals.get('nmini')
    nmx=int_signals.get('nmx')
    signals= tide0(prices_d,pmini=nmini,omx=nmx,m1=m1,m2=m2,ticksize=ticksize)
 
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
  prices['tide']=pd.DataFrame(mini,columns=['tide'], index=prices.index[-tidewindow :])
  prices['mx']=pd.DataFrame(mx,columns=['mx'], index=prices.index[-tidewindow:])
  #prices['mx']=np.round(np.round((prices.mx)/ticksize)*ticksize,-int(math.floor(math.log10(ticksize)))) #round mx to the nearest ticksize---! rounds off for ticksize of dp 2 also to dp 1
  prices['comparative']=pd.DataFrame(comparatives,columns=['comparative'], index=prices.index[-tidewindow:])
  prices['additive']=pd.DataFrame(additives,columns=['additive'], index=prices.index[-tidewindow:])
  # ATR
  test=prices.copy()
  test['Close']=test['Close'].shift(1)
  global atr_window
  atr=test.apply(lambda x: max(abs(x.High-x.Low),abs(x.High-x.Close),abs(x.Low-x.Close)), axis = 1).rolling(window=atr_window).mean()
  prices['atr']=atr
  prices['maxmpb']=pd.DataFrame(maxmpbs,columns=['maxmpb'], index=prices.index[-tidewindow:])
 
  prices['xdh']=prices.High.rolling(length_dh).max()
  prices['xdl']=prices.Low.rolling(length_dl).min()
 
  if tidestr != None :
    pos_durs=[]
    pos_strs=[]
    neg_durs=[]
    neg_strs=[]
    ## TIDE STRENGTHS GENERATOR
    negwindow=tidestr[0]['window']
    poswindow=tidestr[1]['window']
    negdur=tidestr[0]['dur']
    posdur=tidestr[1]['dur']
    negstrg=tidestr[0]['strg']
    posstrg=tidestr[1]['strg']
    if negstrg < 0:
      for d in prices.index[-tidewindow:]:
        prices_d = prices.loc[:d]
        strengths= tidestrength(prices_d,ticksize)
        pos_durs.append(np.mean(strengths.get('+durations')[poswindow]))
        pos_strs.append(np.mean(strengths.get('+strengths')[poswindow]))
        neg_durs.append(np.mean(strengths.get('-durations')[negwindow]))
        neg_strs.append(np.mean(strengths.get('-strengths')[negwindow]))
 
      prices['+str']=pd.DataFrame(pos_strs,columns=['+str'], index=prices.index[-tidewindow:])
      prices['+str']=roundtotick(prices['+str'],ticksize)
      prices['+dur']=pd.DataFrame(pos_durs,columns=['+dur'], index=prices.index[-tidewindow:])
      prices['-str']=pd.DataFrame(neg_strs,columns=['-str'], index=prices.index[-tidewindow:])
      prices['-str']=roundtotick(prices['-str'],ticksize)
      prices['-dur']=pd.DataFrame(neg_durs,columns=['-dur'], index=prices.index[-tidewindow:])
      
    else:  
      for d in prices.index[-tidewindow:]:
        prices_d = prices.loc[:d]
        strengths= tidestrength(prices_d,ticksize,formula=str_formula)
        pos_durs.append(np.percentile(strengths.get('+durations')[poswindow],posdur))
        pos_strs.append(np.percentile(strengths.get('+strengths')[poswindow],posstrg))
        neg_durs.append(np.percentile(strengths.get('-durations')[negwindow],negdur))
        neg_strs.append(np.percentile(strengths.get('-strengths')[negwindow],negstrg))
 
      prices['+str']=pd.DataFrame(pos_strs,columns=['+str'], index=prices.index[-tidewindow:])
      prices['+str']=roundtotick(prices['+str'],ticksize)
      prices['+dur']=pd.DataFrame(pos_durs,columns=['+dur'], index=prices.index[-tidewindow:])
      prices['-str']=pd.DataFrame(neg_strs,columns=['-str'], index=prices.index[-tidewindow:])
      prices['-str']=roundtotick(prices['-str'],ticksize)
      prices['-dur']=pd.DataFrame(neg_durs,columns=['-dur'], index=prices.index[-tidewindow:])
 
 
  return prices

"""## b) Performance Analytics"""

#@title PNL and Expectancy functions
"""
 
PNLs and Expectancy =====================================================================================================================================================================
 
"""
 
def pnls(df):
  # global pospnls, negpnls
  dff=df.copy()
  pospnls=[]
  for t,cell in dff.iterrows():
    if (cell['+L'] == 1) and (cell['+S'] == 1): # closed long so collect +pnl* information and append to list
      pospnls.append(cell['+pnl*'])
    elif (type(cell['+L'])== list) and (type(cell['+S'])== list):
      pospnls.append(cell['+pnl*'][0])
  pospnls=np.array(pospnls)
 
 
  negpnls=[]
  for t,cell in dff.iterrows():
    if (cell['-L'] == 1) and (cell['-S'] == 1): # closed long so collect +pnl* information and append to list
      negpnls.append(cell['-pnl*'])
    elif (type(cell['-L'])== list) and (type(cell['-S'])== list):
      negpnls.append(cell['-pnl*'][0])
  negpnls=np.array(negpnls)
  return {'+':pospnls,'-':negpnls}
 
 
def expectancy(pnls,pointvalue):
  total=np.shape(pnls)[0]
  winners= np.sum(pnls>0)
  losers= np.sum(pnls<=0)
  winrate = winners/total
  loserate = losers/total
  avewin=np.mean(pnls[pnls>0])
  aveloss=np.mean(pnls[pnls<0])
  expectancy = ( winrate * avewin) + (loserate * aveloss)
  return OrderedDict({'total trades':np.round(total), 'winners':np.round(winners),'losers':np.round(losers), 'win rate':np.round(winrate,2), 'lose rate':np.round(loserate,2), 'average win':np.round(avewin*pointvalue,2),'average loss':np.round(aveloss*pointvalue,2),'expectancy':np.round(expectancy*pointvalue,2)})
 
def expectancies(df,pointvalue):
  pospnls=pnls(df)['+']
  negpnls=pnls(df)['-']
  totalpnls=np.concatenate([negpnls,pospnls])
  print(pd.DataFrame({'+tides':expectancy(pospnls,pointvalue),'-tides':expectancy(negpnls,pointvalue), 'total':expectancy(totalpnls,pointvalue)}))

#@title Standard Performance Metrics functions
"""
 
Standard Performance Metrics =====================================================================================================================================================================
 
"""
 
def MDD(df):
  wealth_index = (df+1).cumprod()
  prev_peak = wealth_index.cummax()
  dd = (wealth_index-prev_peak)/prev_peak 
  return dd
 
 
def summary(df,strat_name): # this should take into account length of backtest window
  df1=df.copy()
  ax=(df1).plot.hist(bins=50, figsize=(5,5), title='Distribution of returns of '+ strat_name)
  ax.set_xlabel("Returns %")
  ax.set_ylabel("Days")
  plt.plot([(df1).mean(),(df1).mean()],[0,16])
  plt.legend(['Mean = '+ str(round(df1.mean(),3))])
  plt.show()
  ## Statistics
  # cumret
  cumret = (df1+1).cumprod()[-1]-1
  # annualized returns (formula based off investing.com)
  annret = (1+cumret)**(252/len(df1)) - 1
  # annualized vol
  annvol = df1.std()*np.sqrt(252) 
  # Sharpe ratio
  sharpe = df1.mean() / df1.std()
  # skew
  skw = skew(df1)
  # kurtosis
  kurt = kurtosis(df1)
  # maxdrawdown
  mdd = min(MDD(df1))
 
 
  stats = ['{:0.1%}'.format(annret),
           '{:0.1%}'.format(cumret),
           '{:0.1%}'.format(annvol),
           '{:0.3}'.format(sharpe),
           '{:0.3}'.format(skw),
           '{:0.3}'.format(kurt),
           '{:0.1%}'.format(mdd) ]
 
  d = {'':['Annualized Returns',
           'Cumulative Returns',
           'Annualised Volatility',
           'Sharpe Ratio',
           'Skew',
           'Kurtosis', 
           'Max Drawdown'], 'Summary':stats}
 
  summary = pd.DataFrame(d).set_index('')
  # fig = go.Figure(data=[go.Table(summary)])
 
  print(summary)

"""## c) Graphs"""

#@title Graphing functions { form-width: "10%" }
"""
 
Graphs =====================================================================================================================================================================
 
"""
 
 
# Candlestsick plot + xdh/l + mx + comparative/tide ----------------------------
def tideplot(prices,height=600,width=1000):
  df=prices.copy()
  Candle=go.Candlestick(x=df['Date'],open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],name='OHLC')
  colors = {1: 'green',0: 'red'}
  bar = go.Bar(x=df.Date,y=df.comparative,name='tide/comparative',opacity=0.25,marker={'color':[colors[dff] for dff in df.tide]})
  MX_plot = go.Scatter(x=df.Date, y=df.mx.shift(1), name ='MX',line = dict(color='black',dash='dot'))
  new_MX_plot = go.Scatter(x=df.Date, y=df.mx, name ='new MX',line = dict(color='black'))
  xdh_plot = go.Scatter(x=df.Date, y=df.xdh.shift(1), name ='xdh',line = dict(color='lightgreen',dash='dot'))
  xdl_plot = go.Scatter(x=df.Date, y=df.xdl.shift(1), name ='xdl',line = dict(color='firebrick',dash='dot'))
  fig = go.Figure([Candle,MX_plot,new_MX_plot,xdh_plot,xdl_plot,bar])
  fig.update_layout(title='',yaxis_title='price',height=height,width=width) 
  fig.update_layout(xaxis_rangeslider_visible=False)
  fig.update_layout(yaxis=dict(range=[min(df.Low)-5,max(df.High)+5]),margin=go.layout.Margin(t=20,b=20,r=230),showlegend=False) #r=195
  fig.show()
 
# PNL + MDD profiles + Candlestick plot ------------------------------------
def tideplots(pointvalue,strats_df, strats,dimr=[400,1000],dimc=[600,1000],dimmmd=[240,1000],showsummary=True):
  df0=strats_df[0]
  fig1 = make_subplots(specs=[[{"secondary_y": True}]])
 
  # Background tide signal  
  colors = {1: 'green',0: 'red'}
  bar0 = go.Bar(x=df0.Date,y=df0.tide.replace(to_replace=0,value=1),name='tide',opacity=0.25,marker={'color':[colors[dff] for dff in df0.tide]}, xaxis='x1', yaxis='y2') 
 
  # PNL plot
  traces = [go.Scatter(x=df.Date,y=(df.pnl).cumsum(), name = strat) for df,strat in zip(strats_df,strats)] + [go.Scatter(x=df0.Date,y=(df0.BH).cumsum(), name = 'Buy & hold',line = dict(color='black'))]
  for trace in traces:
    fig1.add_trace(trace,secondary_y=False) 
  fig1.add_trace(bar0,secondary_y=True)
 
  fig1['layout']['yaxis2']['showgrid'] = False
  fig1.update_layout(yaxis_title='pnl ($)',height=dimr[0],width=dimr[1])
  fig1.update_layout(yaxis=dict(range=[min([trace['y'].min() for trace in traces[:]])-0.05,max([trace['y'].max() for trace in traces[:] ])+0.05])) #
 
  fig1.update_layout({"yaxis2.visible": False },margin=go.layout.Margin(t=20,b=0),xaxis_showticklabels=True)
  fig1.show()
 
  # MDD PLOT 
  fig2 = make_subplots(specs=[[{"secondary_y": True}]])
  traces1=[go.Scatter(x=df.Date,y=MDD(df.pnl.cumsum().pct_change()), name = strat) for df,strat in zip(strats_df,strats)] + [go.Scatter(x=df0.Date,y=MDD(df0.BH.cumsum().pct_change()), name = 'Buy & hold',line = dict(color='black'))]
  for trace in traces1:
    fig2.add_trace(trace,secondary_y=False) 
  fig2.add_trace(bar0,secondary_y=True)
 
  fig2['layout']['yaxis2']['showgrid'] = False
  fig2.update_layout(yaxis_title='mdd (%)',height=dimmmd[0],width=dimmmd[1])
  fig2.update_layout(yaxis=dict(range=[-1,0])) #min([trace['y'].min() for trace in traces1[:]])-0.05,max([trace['y'].max() for trace in traces1[:] ])
 
 
  fig2.update_layout({"yaxis2.visible": False },margin=go.layout.Margin(t=0,b=0,r=140),showlegend=False,xaxis_visible=False, xaxis_showticklabels=True) #r=130
  fig2.show()
 
  
  # BAR PLOT -------------------------------------------------------------------
  tideplot(df0,height=dimc[0],width=dimc[1])
  if showsummary==True:
    # Summary Histographs --------------------------------------------------------
    # Check for -inf and inf
    summary(df0.BH.cumsum().pct_change().fillna(0), 'Buy and Hold')
    print('here')
    for df,strat in zip(strats_df,strats):
      summary(df.pnl.cumsum().pct_change().fillna(0),strat)
      expectancies(df,pointvalue)
 
 
 
# window
def summaryplots(dfs,equity,plotwindow,pointvalue,ticksize,showstrats,dimr=[200,1700],dimmmd=[80,1700],dimc=[600,1700],showsummary=True):
  df0,df1,df121,df122=dfs.values()
  # global df0l,df1l,df12l,df2l,df3l
 
  df0l=df0.copy()[plotwindow[0]:plotwindow[1]]
  df1l=df1.copy()[plotwindow[0]:plotwindow[1]]
  df121l=df121.copy()[plotwindow[0]:plotwindow[1]]
  df122l=df122.copy()[plotwindow[0]:plotwindow[1]]
  df3l=df121l.copy()
  df3l['pnl']=df121l.pnl+df122l.pnl # combined
  for df in [df0l,df1l,df121l,df122l,df3l]:
    ## Change PNL to $ value
    df['pnl']=df.pnl*pointvalue
    df['BH']=df.BH*pointvalue
    ## Add initial equity to start of window
    df.at[df.index[0],'pnl']=df.at[df.index[0],'pnl']+equity
    df.at[df.index[0],'BH']=df.at[df.index[0],'BH']+equity
    print("CAT")
    
 
 
  ## Plot
 
  if showstrats=='base,pred':
    tideplots(pointvalue,[df0l,df1l],['Base Tide','Pred Tide'],dimr=dimr,dimmmd=dimmmd,dimc=dimc,showsummary=showsummary)
  elif showstrats=='base,preds':
    tideplots(pointvalue,[df0l,df1l,df12l],['Base Tide','Pred Tide','Pred2 Tide'],dimr=dimr,dimmmd=dimmmd,dimc=dimc,showsummary=showsummary)
  elif showstrats=='base,preds,cons':
    tideplots(pointvalue,[df0l,df1l,df12l,df2l],['Base Tide','Pred Tide','Pred2 Tide','Cons Tide'],dimr=dimr,dimmmd=dimmmd,dimc=dimc,showsummary=showsummary)
  elif showstrats=='all':
    tideplots(pointvalue,[df0l,df1l,df12l,df2l,df3l],['Base Tide','Pred Tide','Pred2 Tide','Cons Tide','Comb Tide'],dimr=dimr,dimmmd=dimmmd,dimc=dimc,showsummary=showsummary)
  elif showstrats=='pred2':
    tideplots(pointvalue,[df12l],['Pred2 Tide'],dimr=dimr,dimmmd=dimmmd,dimc=dimc,showsummary=showsummary)
  elif showstrats=='pred2,cons':
    tideplots(pointvalue,[df12l,df2l],['Pred2 Tide','Cons Tide'],dimr=dimr,dimmmd=dimmmd,dimc=dimc,showsummary=showsummary)
  elif showstrats=='pred2,cons,comb':
    tideplots(pointvalue,[df12l,df2l,df3l],['Pred2 Tide','Cons Tide','Comb Tide'],dimr=dimr,dimmmd=dimmmd,dimc=dimc,showsummary=showsummary)
  elif showstrats=='comb':
    tideplots(pointvalue,[df3l],['Comb Tide'],dimr=dimr,dimmmd=dimmmd,dimc=dimc,showsummary=showsummary)
  elif showstrats=='base,pred,comb':
    tideplots(pointvalue,[df0l,df1l,df12l,df3l],['Base Tide','Pred Tide','Pred2 Tide','Comb Tide'],dimr=dimr,dimmmd=dimmmd,dimc=dimc,showsummary=showsummary)
  elif showstrats=='pred2s':
    tideplots(pointvalue,[df1l, df121l,df122l],['Pred','Pred21 Tide','Pred22 Tide'],dimr=dimr,dimmmd=dimmmd,dimc=dimc,showsummary=showsummary)
 
 
## CURRENT INDICATORS ===================================================================================================================================================================================================
 
 
 
  if showsummary == False:#if show summary then remove latest indicators
    ## Indicators
    df=df121.copy()
    tdst=tidestrength(df,ticksize,windows=[3,6,9,12])
 
    if df0.tide[-1] == 1:
      currenttide='+'
      tidemult = 1
    elif df0.tide[-1] == 0:
      currenttide='-'
      tidemult = -1
 
    ## Function to generate historical strengths/durations
    def pasttides(df,ticksize,tide,str_or_dur,currenttide,mode):
      if tide == currenttide: # then need to exclude current tide in str/dur max/mean/min calculation
        tdst=tidestrength(df,ticksize,windows=[3+1,6+1,9+1,12+1])
        exclusion=-1
      elif tide != currenttide:
        tdst=tidestrength(df,ticksize,windows=[3,6,9,12])
        exclusion=None
 
      
      if mode == 'max':
        return [np.round(np.max(strs[:exclusion]),2) for strs in tdst[tide+str_or_dur]]
      elif mode == 'mean':
        return [np.round(np.mean(strs[:exclusion]),2) for strs in tdst[tide+str_or_dur]]
      elif mode == 'min':
        return [np.round(np.min(strs[:exclusion]),2) for strs in tdst[tide+str_or_dur]]
 
    ## Take profit points
    STRS = np.sort(np.unique(pasttides(df,ticksize,currenttide,'strengths',currenttide,'max') + pasttides(df,ticksize,currenttide,'strengths',currenttide,'mean') + pasttides(df,ticksize,currenttide,'strengths',currenttide,'min')))
    tpweights=[33,66,99]
    TPs = np.percentile(STRS,tpweights) ## what are these percentile dependant on? should be risk appetite?
    ## find entry to calculate + TP
    dfs = [g for i,g in df.groupby(df['tide'].ne(df['tide'].shift()).cumsum())]
    entry = dfs[-1].Open[0]
 
    # other indicators to be added into indicators table
    DMA20 = np.round(np.mean(df.Close[-20:]),2)
 
    colors = n_colors('rgb(255,255,255)', 'rgb(255, 255,255)', 7, colortype='rgb') +  n_colors('rgb(55,181,74)', 'rgb(164,212,152)', 3, colortype='rgb') + n_colors('rgb(55,181,74)', 'rgb(164,212,152)', 3, colortype='rgb') +   n_colors('rgb(239,59,59)', 'rgb(244,129,134)', 3, colortype='rgb') + n_colors('rgb(239,59,59)', 'rgb(244,129,134)', 3, colortype='rgb')
 
    indicators = go.Figure(data=[go.Table(header=dict(values=['<b>14day ATR</b>','<b>20DMA</b>',
                                                              '<b>TP1    ('+ str(tpweights[0])+'th %tile' +')</b>',
                                                              '<b>TP2    ('+ str(tpweights[1])+'th %tile' +')</b>',
                                                              '<b>TP3    ('+ str(tpweights[2])+'th %tile' +')</b>',
                                                              '<b> </b>',
                                                              '',
                                                              '<b>+STR    max</b>',
                                                              '<b>            mean</b>',
                                                              '<b>            min</b>',
 
                                                              '<b>+DUR max</b>',
                                                              '<b>             mean</b>',
                                                              '<b>             min</b>',
 
                                                              '<b>-STR    max</b>',
                                                              '<b>              mean</b>',
                                                              '<b>              min</b>',
 
                                                              '<b>-DUR max</b>',
                                                              '<b>              mean</b>',
                                                              '<b>              min</b>'],
                                                      font=dict(color='black', size=12)),
                                          
                                          cells=dict(values=[np.round(df.atr[-1],2),DMA20,
                                                             roundtotick((tidemult*TPs[0])+entry,ticksize),
                                                             roundtotick((tidemult*TPs[1])+entry,ticksize),
                                                             roundtotick((tidemult*TPs[2])+entry,ticksize),
                                                             ' ',
 
                                                            ['Recent Tide','Past 3 Tides','Past 6 Tides','Past 9 Tides','Past 12 Tides'],
                                                            ['-'] + pasttides(df,ticksize,'+','strengths',currenttide,'max'),
                                                            [tdst['+strengths'][0][-1]] + pasttides(df,ticksize,'+','strengths',currenttide,'mean'),
                                                            ['-'] + pasttides(df,ticksize,'+','strengths',currenttide,'min'),
 
                                                            ['-'] + pasttides(df,ticksize,'+','durations',currenttide,'max'),
                                                            [tdst['+durations'][0][-1]] + pasttides(df,ticksize,'+','durations',currenttide,'mean'),
                                                            ['-'] + pasttides(df,ticksize,'+','durations',currenttide,'min'),
 
                                                            ['-'] + pasttides(df,ticksize,'-','strengths',currenttide,'max'),
                                                            [tdst['-strengths'][0][-1]] + pasttides(df,ticksize,'-','strengths',currenttide,'mean'),
                                                            ['-'] + pasttides(df,ticksize,'-','strengths',currenttide,'min'),
 
                                                            ['-'] + pasttides(df,ticksize,'-','durations',currenttide,'max'),
                                                            [tdst['-durations'][0][-1]] + pasttides(df,ticksize,'-','durations',currenttide,'mean'),
                                                            ['-'] + pasttides(df,ticksize,'-','durations',currenttide,'min')
                                                            ],
                                                     fill_color=colors)
                                          )
                                ]
                           )
    # indicators.update_layout(margin=go.layout.Margin(t=20,b=0,r=500)) 
    indicators.update_layout(width=1700, margin=go.layout.Margin(t=20,b=0,r=228) ) 
    indicators.show()

"""## d) Strategies"""

#@title Position functions { form-width: "200px" }
"""
 
Position functions =====================================================================================================================================================================
 
"""
 
 
def position_update(prices,t,col,newdata):
  if type(prices.at[t, col]) == list:
    prices.at[t, col] = prices.at[t, col]+[newdata]
  elif type(prices.at[t, col]) == np.float64:
    prices.at[t, col] = [prices.at[t, col],newdata]
  elif type(prices.at[t, col]) == float:
    prices.at[t, col] = [prices.at[t, col],newdata]  
  elif type(prices.at[t, col]) == int:
    prices.at[t, col] = [prices.at[t, col],newdata]    
  else:
    prices.at[t, col]=newdata
  return prices
 
 
def position_extractor(prices,t,col):
  if type(prices.at[t, col]) == list:
    return prices.at[t, col][-1]
  elif type(prices.at[t, col]) == np.float64:
    return prices.at[t, col]
  else:
    return prices.at[t, col]
 
def pnlrounding(entry,exit,ticksize):
  return np.round(np.round((exit-entry)/ticksize)*ticksize,-int(math.floor(math.log10(ticksize))))
 
def roundtotick(price,ticksize):
  return np.round(np.round((price)/ticksize)*ticksize,-int(math.floor(math.log10(ticksize))))

#@title Base Tide { form-width: "200px" }
 
 
""" 
 
Base Tide ==========================================================================================================================================================================================================================================================================================================================================
 
"""
 
 
def streaming_base_tide(prices,m1,m2,ticksize,etide=None):
  prices=tidegenerator(prices,m1,m2,ticksize)
  if type(etide) == pd.core.series.Series:
    prices['tide']=etide 
 
 
  # STREAMING BACKTESTER
  prices['BH']=(prices.Close - prices.Close.shift(1)).fillna(0)
 
  prices['+entry']=None
  prices['+exit']=None
  prices['+L']=None
  prices['+S']=None
  prices['+pnl']=None
  prices['+pnl*']=None
 
  prices['-entry']=None
  prices['-exit']=None
  prices['-L']=None
  prices['-S']=None
  prices['-pnl']=None
  prices['-pnl*']=None
 
  prices[['+entry','+exit','-entry','-exit','+L','-L','+S','-S','+pnl','-pnl','+pnl*','-pnl*']]=prices[['+entry','+exit','-entry','-exit','+L','-L','+S','-S','+pnl','-pnl','+pnl*','-pnl*']].astype(object)
 
  for t,price in prices.iterrows():   
    ## Tide changed from 0 to 1 so LONG closing ===========================================================================================
    if (price.tide != prices.tide.shift(1))[t] & (price.tide == 1): 
      prices = trade(prices,price,t,'+',ticksize)
 
    ## Tide changed from 1 to 0 so SHORT closing ===========================================================================================
    elif (price.tide != prices.tide.shift(1))[t] & (price.tide == 0): 
      prices = trade(prices,price,t,'-',ticksize)
 
  ## Sum returns
  global pnlplus, pnlneg
  pnlplus=[]
  pnlneg=[]
  for t,cell in prices.fillna(value=0).iterrows():
    pnlplus.append(np.sum(cell['+pnl']))
    pnlneg.append(np.sum(cell['-pnl']))
 
  pnlplus=pd.Series(pnlplus)
  pnlplus.index=prices.index
 
  pnlneg=pd.Series(pnlneg)
  pnlneg.index=prices.index
 
  prices['pnl']=pnlplus+pnlneg
 
  prices['+pnl*']=prices['+exit']-prices['+entry']
  prices['-pnl*']=prices['-entry']-prices['-exit']
 
  return prices
 
"""
 
Base Tide trade ==========================================================================================================================================================================================================================================================================================================================================
 
"""
def trade(prices,price, t, tradetype,ticksize):
  if tradetype == '-':
    entry_position='S'
    exit_position ='L'
    position_sign = -1
    signal=0
    opp_signal=1
  elif tradetype == '+':
    entry_position='L'
    exit_position ='S'
    position_sign = 1
    signal=1
    opp_signal=0
  
 
  # Position entry/exit update
  position_update(prices,t,tradetype+'entry',price.Close) 
  position_update(prices,t,tradetype+'exit',price.Close) 
  price_entry = position_extractor(prices,t,tradetype+'entry')
  # Position size update
  position_update(prices,t,tradetype+entry_position,1)
  position_update(prices,t,tradetype+exit_position,0)
  # PNL update
  position_update(prices,t,tradetype+'pnl',position_sign*pnlrounding(price_entry,position_extractor(prices,t,tradetype+'exit'),ticksize)) 
  #print(t, ': Base ' + tradetype + 'tide ----------------------------- entry: ', price_entry)
 
  # look for exit - in this case; when EOD tide changes back to 0 ================================================================================================================================================
  for t_exit,price_exit in prices.loc[t:][1:].iterrows():    
    # If EOD is still same signal --------------------------------------------------------------------------------------------------------------------------
    if (price_exit.tide == signal): 
      # Position entry/exit update
      position_update(prices,t_exit,tradetype+'entry',price_entry)  
      position_update(prices,t_exit,tradetype+'exit',price_exit.Close)
      # Position sizes update
      position_update(prices,t_exit,tradetype+entry_position,1)
      position_update(prices,t_exit,tradetype+exit_position,0)
      # PNL update
      settlement_price = prices.Close.shift(1)[t_exit]
      position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,position_extractor(prices,t_exit,tradetype+'exit'),ticksize)) 
 
 
    # if EOD is opp signal ---------------------------------------------------------------------------------------------------------------------------------
    elif (price_exit.tide == opp_signal) & (prices.tide.shift(1)[t_exit] == signal): 
      # Position entry/exit update
      position_update(prices,t_exit,tradetype+'entry',price_entry)  
      position_update(prices,t_exit,tradetype+'exit',price_exit.Close)
      # Position sizes update
      position_update(prices,t_exit,tradetype+entry_position,1)
      position_update(prices,t_exit,tradetype+exit_position,1)
      # PNL update
      settlement_price = prices.Close.shift(1)[t_exit]
      position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,position_extractor(prices,t_exit,tradetype+'exit'),ticksize)) 
      break
  return prices

#@title Pred Tide { form-width: "200px" }
 
"""
 
Pred Tide ==========================================================================================================================================================================================================================================================================================================================================
 
"""
 
def streaming_predtide(prices,m1,m2,ticksize):
  prices=tidegenerator(prices,m1,m2,ticksize)
 
  # STREAMING BACKTESTER
  prices['BH']=(prices.Close - prices.Close.shift(1)).fillna(0)
 
  prices['+entry']=None
  prices['+exit']=None
  prices['+L']=None
  prices['+S']=None
  prices['+pnl']=None
  prices['+pnl*']=None
 
  prices['-entry']=None
  prices['-exit']=None
  prices['-L']=None
  prices['-S']=None
  prices['-pnl']=None
  prices['-pnl*']=None
 
  prices[['+entry','+exit','-entry','-exit','+L','-L','+S','-S','+pnl','-pnl','+pnl*','-pnl*']]=prices[['+entry','+exit','-entry','-exit','+L','-L','+S','-S','+pnl','-pnl','+pnl*','-pnl*']].astype(object)
 
  for t,price in prices.iterrows():
    ## PREDICTIVE
    ## Previous day tide 0 but today's opening > mx so LONG open 
    if (prices.tide.shift(1)[t] == 0) & (price.Open > prices.mx.shift(1)[t]):
      prices = pred_trade(prices,price,t,'+',ticksize)
 
    ## Previous day tide 1 but today's opening < mx so SHORT open 
    elif (prices.tide.shift(1)[t] == 1) & (price.Open < prices.mx.shift(1)[t]):
      prices = pred_trade(prices,price,t,'-',ticksize)
 
    ## BASE
    ## Tide changed from 0 to 1 so LONG closing ===========================================================================================
    elif (price.tide != prices.tide.shift(1))[t] & (price.tide == 1): 
      prices = trade(prices,price,t,'+',ticksize)
 
    ## Tide changed from 1 to 0 so SHORT closing ===========================================================================================
    elif (price.tide != prices.tide.shift(1))[t] & (price.tide == 0): 
      prices = trade(prices,price,t,'-',ticksize)
 
 
 
 
  ## Sum running pnl
  global pnlpos, pnlneg
  pnlpos=[]
  pnlneg=[]
 
  for t,cell in prices.fillna(value=0).iterrows():
    pnlpos.append(np.sum(cell['+pnl']))
    pnlneg.append(np.sum(cell['-pnl']))
 
  pnlpos=pd.Series(pnlpos)
  pnlpos.index=prices.index
  pnlneg=pd.Series(pnlneg)
  pnlneg.index=prices.index
  prices['pnl']=pnlpos+pnlneg
 
 
  # sum pnl*
  pnlpos=[]
  pnlneg=[]
  for t,cell in prices.iterrows():
    if type(cell['+entry']) == list:
      pnlpos.append([pnlrounding(entry,exit,ticksize) for entry,exit in zip(cell['+entry'],cell['+exit'])])
    elif type(cell['+entry']) == float:# or type(cell['-entry']) == np.float64:
      pnlpos.append(cell['+exit']-cell['+entry'])
    else:
      pnlpos.append(None)
 
  for t,cell in prices.iterrows():
    if type(cell['-entry']) == list:
      pnlneg.append([pnlrounding(entry,exit,ticksize) for entry,exit in zip(cell['-entry'],cell['-exit'])])
    elif type(cell['-entry']) == float:# or type(cell['-entry']) == np.float64:
      pnlneg.append(cell['-entry']-cell['-exit'])
    else:
      pnlneg.append(None)
 
 
 
  prices['+pnl*']=pnlpos
  prices['-pnl*']=pnlneg
 
  return prices
 
"""
 
Pred Tide trade ==========================================================================================================================================================================================================================================================================================================================================
 
"""
 
 
def pred_trade(prices,price, t, tradetype,ticksize,entryprice=0):
  if tradetype == '-': # predicted tide
    actualtide = '+'
    entry_position='S'
    exit_position ='L'
    position_sign = -1
    signal=1 # actual tide previous day
    opp_signal=0
  elif tradetype == '+':  # predicted tide
    actualtide = '-'
    entry_position='L'
    exit_position ='S'
    position_sign = 1
    signal=0 # actual tide previous day 
    opp_signal=1
 
  
  ## DEFINE ENTRY PRICE /////////////////////////////////////////////////////////
  if entryprice == 0:
    position_update(prices,t,tradetype+'entry',price.Open)   
  else:
    position_update(prices,t,tradetype+'entry',entryprice) 
  price_entry = position_extractor(prices,t,tradetype+'entry') 
  ## DEFINE ENTRY PRICE /////////////////////////////////////////////////////////
  print(t,': Pred ' + tradetype + 'tide ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ entry: ', price_entry)
 
  # ENTRY DAY SL: if EOD tide == previous day tide
  if  price.tide == signal:
    # Position exit update
    position_update(prices,t,tradetype+'exit',price.mx)                                                                                                                        #!!!! used to be close BUT must check for lookaheadbias !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Position size update
    position_update(prices,t,tradetype+entry_position,1)
    position_update(prices,t,tradetype+exit_position,1)  
    # PNL update
    position_update(prices,t,tradetype+'pnl',position_sign*pnlrounding(price_entry,position_extractor(prices,t,tradetype+'exit'),ticksize))
    print(t, ': Entry day SL, Reinstate ' + exit_position +' at mx:', price.mx)                                                                                                   #!!!! used to be close BUT must check for lookaheadbias !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ## REINSTATE POSITIONS (but at closing as entry price)    
    pred_trade(prices,price,t,actualtide,ticksize,entryprice=price.mx)                                                                                                            #!!!! used to be close BUT must check for lookaheadbias !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
 
  else: # else hold (loop) till next day k
    # Position exit update
    position_update(prices,t,tradetype+'exit',price.Close) 
    # Position size update
    position_update(prices,t,tradetype+entry_position,1)
    position_update(prices,t,tradetype+exit_position,0)  
    # PNL update
    position_update(prices,t,tradetype+'pnl',position_sign*pnlrounding(price_entry,position_extractor(prices,t,tradetype+'exit'),ticksize))
 
 
    # LOOP: look for exit - in this case; when EOD tide changes back to 0 ======================================================================================================================================================
    for t_exit,price_exit in prices.loc[t:][1:].iterrows(): 
 
      ## PREDICTIVE EXIT CONDITION ++++++++++++++++++++++++++++++
      if signal == 0:
        predsignal = price_exit.Open < prices.mx.shift(1)[t_exit]
      elif signal == 1:
        predsignal = price_exit.Open > prices.mx.shift(1)[t_exit]
      ## PREDICTIVE EXIT CONDITION ++++++++++++++++++++++++++++++
 
 
      # Check for secondary stoploss (new day open < mx ) if so EXIT AT OPEN, ELSE REINSTATE-----------------------------------------------------------------------
      if predsignal: 
        # Position entry/exit update
        position_update(prices,t_exit,tradetype+'entry',price_entry)
        position_update(prices,t_exit,tradetype+'exit',price_exit.Open)
 
        # Position size update  
        position_update(prices,t_exit,tradetype+entry_position,1)
        position_update(prices,t_exit,tradetype+exit_position,1) 
        
        # PNL Update
        settlement_price = prices.Close.shift(1)[t_exit]
        position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,position_extractor(prices,t_exit,tradetype+'exit'),ticksize))                       #!!!! used to be close BUT must check for lookaheadbias !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(t_exit, ': Secondary SL (Pred '+actualtide+') so exit:',position_extractor(prices,t_exit,tradetype+'exit'), 'settlement price:', settlement_price)
 
        # CHECK EOD if 
        if price_exit.tide == opp_signal:
            " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "
            print('!!!!')
 
 
        break
 
      else: # actual tide == predicted tide
        print(t_exit, ': HOLD since '+tradetype+ ' tide')
        # Position entry/exit update
        position_update(prices,t_exit,tradetype+'entry',price_entry)
        position_update(prices,t_exit,tradetype+'exit',price_exit.Close)
 
        # Position size update  
        position_update(prices,t_exit,tradetype+entry_position,1)
        position_update(prices,t_exit,tradetype+exit_position,0) 
        
        # Position PNL Update
        settlement_price = prices.Close.shift(1)[t_exit]
        position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,price_exit.Close,ticksize))
 
 
  return prices

#@title Pred2 Tide { form-width: "200px" }
 
 
"""
 
Pred2 Tide ==========================================================================================================================================================================================================================================================================================================================================
 
"""
 
 
 
def streaming_pred2tide(prices,m1,m2,ticksize,tidestr):
  prices=tidegenerator(prices,m1,m2,ticksize,tidestr=tidestr)
 
  # STREAMING BACKTESTER
  prices['BH']=(prices.Close - prices.Close.shift(1)).fillna(0)
 
  prices['+entry']=None
  prices['+exit']=None
  prices['+L']=None
  prices['+S']=None
  prices['+pnl']=None
  prices['+pnl*']=None
 
  prices['-entry']=None
  prices['-exit']=None
  prices['-L']=None
  prices['-S']=None
  prices['-pnl']=None
  prices['-pnl*']=None
 
  prices[['+entry','+exit','-entry','-exit','+L','-L','+S','-S','+pnl','-pnl','+pnl*','-pnl*']]=prices[['+entry','+exit','-entry','-exit','+L','-L','+S','-S','+pnl','-pnl','+pnl*','-pnl*']].astype(object)
 
  for t,price in prices.iterrows():
    ## PREDICTIVE
    ## Previous day tide 0 but today's opening > mx so LONG open 
    if (prices.tide.shift(1)[t] == 0) & (price.Open > prices.mx.shift(1)[t]):
      print(t,"PRED +TIDE HIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      prices = pred2_trade(prices,price,t,'+',ticksize)
 
    ## Previous day tide 1 but today's opening < mx so SHORT open 
    elif (prices.tide.shift(1)[t] == 1) & (price.Open < prices.mx.shift(1)[t]):
      print(t,"PRED -TIDE HIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      prices = pred2_trade(prices,price,t,'-',ticksize)
 
    ## BASE
    ## Tide changed from 0 to 1 so LONG closing ===========================================================================================
    elif (price.tide != prices.tide.shift(1))[t] & (price.tide == 1): 
      print(t,"BASE TIDE HIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      prices = trade(prices,price,t,'+',ticksize)
 
    ## Tide changed from 1 to 0 so SHORT closing ===========================================================================================
    elif (price.tide != prices.tide.shift(1))[t] & (price.tide == 0): 
      print(t,"BASE TIDE HIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      prices = trade(prices,price,t,'-',ticksize)
 
 
 
 
  ## Sum running pnl
  global pnlpos, pnlneg
  pnlpos=[]
  pnlneg=[]
 
  for t,cell in prices.fillna(value=0).iterrows():
    pnlpos.append(np.sum(cell['+pnl']))
    pnlneg.append(np.sum(cell['-pnl']))
 
  pnlpos=pd.Series(pnlpos)
  pnlpos.index=prices.index
  pnlneg=pd.Series(pnlneg)
  pnlneg.index=prices.index
  prices['pnl']=pnlpos+pnlneg
 
 
  # sum pnl*
  pnlpos=[]
  pnlneg=[]
  for t,cell in prices.iterrows():
    if type(cell['+entry']) == list:
      pnlpos.append([pnlrounding(entry,exit,ticksize) for entry,exit in zip(cell['+entry'],cell['+exit'])])
    elif type(cell['+entry']) == float:# or type(cell['-entry']) == np.float64:
      pnlpos.append(cell['+exit']-cell['+entry'])
    else:
      pnlpos.append(None)
 
  for t,cell in prices.iterrows():
    if type(cell['-entry']) == list:
      pnlneg.append([pnlrounding(exit,entry,ticksize) for entry,exit in zip(cell['-entry'],cell['-exit'])]) #pnlrounding(entry,exit)
    elif type(cell['-entry']) == float: #or type(cell['-entry']) == np.float64:
      pnlneg.append(cell['-entry']-cell['-exit'])
    else:
      pnlneg.append(None)
 
 
 
  prices['+pnl*']=pnlpos
  prices['-pnl*']=pnlneg
 
  return prices
 
 
 
"""
 
Pred2 Tide trade ==========================================================================================================================================================================================================================================================================================================================================
 
 
"""
 
 
def pred2_trade(prices,price, t, tradetype,ticksize,entryprice=0):
 
  ## DEFINE ENTRY PRICE /////////////////////////////////////////////////////////
  if entryprice == 0:
    position_update(prices,t,tradetype+'entry',price.Open)   
  else:
    position_update(prices,t,tradetype+'entry',entryprice) 
  price_entry = position_extractor(prices,t,tradetype+'entry') 
  ## DEFINE ENTRY PRICE /////////////////////////////////////////////////////////
 
  
  if tradetype == '-': # predicted tide
    actualtide = '+'
    entry_position='S'
    exit_position ='L'
    position_sign = -1
    signal=1 # actual tide previous day
    opp_signal=0
    TP = prices['+str'][t]  # flipped for mean reverting hypothesis
    earlyexit= price.Low < price_entry - TP
  elif tradetype == '+':  # predicted tide
    actualtide = '-'
    entry_position='L'
    exit_position ='S'
    position_sign = 1
    signal=0 # actual tide previous day 
    opp_signal=1
    TP = prices['-str'][t]
    earlyexit= price.High > price_entry + TP
 
  print(t,': Pred ' + tradetype + 'tide ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ entry: ', price_entry, ',TP: ',TP)
  
 
  # pred SL: hit comparative
 
 
  # TP at ATR (prelim draft)
  if earlyexit:
    # Position exit update
    position_update(prices,t,tradetype+'exit',price_entry + position_sign*TP)                                                                                                                        
    # Position size update
    position_update(prices,t,tradetype+entry_position,1)
    position_update(prices,t,tradetype+exit_position,1)  
    # PNL update
    position_update(prices,t,tradetype+'pnl',position_sign*pnlrounding(price_entry,position_extractor(prices,t,tradetype+'exit'),ticksize))
    print(t, ': Entry day early exit ' + exit_position +' at:',position_extractor(prices,t,tradetype+'exit') )  
 
  # ENTRY DAY SL: if EOD tide == previous day tide
  elif  price.tide == signal: # this is equivalent to price touching comparative but can test using low or high breaching comparative
    # Position exit update
    position_update(prices,t,tradetype+'exit',price.comparative)                                                                                                                        #!!!! must be comparative BUT must check for lookaheadbias !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Position size update
    position_update(prices,t,tradetype+entry_position,1)
    position_update(prices,t,tradetype+exit_position,1)  
    # PNL update
    position_update(prices,t,tradetype+'pnl',position_sign*pnlrounding(price_entry,position_extractor(prices,t,tradetype+'exit'),ticksize))
    print(t, ': Entry day SL, Reinstate ' + exit_position +' at mx:', price.mx)                                                                                                   
    ## REINSTATE POSITIONS (but at closing as entry price)    
    pred2_trade(prices,price,t,actualtide,ticksize,entryprice=price.mx)                                                                                                           
    
 
  else: # else hold (loop) till next day k
    # Position exit update
    position_update(prices,t,tradetype+'exit',price.Close) 
    # Position size update
    position_update(prices,t,tradetype+entry_position,1)
    position_update(prices,t,tradetype+exit_position,0)  
    # PNL update
    position_update(prices,t,tradetype+'pnl',position_sign*pnlrounding(price_entry,position_extractor(prices,t,tradetype+'exit'),ticksize))
 
 
    # LOOP: look for exit - in this case; when EOD tide changes back to 0 ======================================================================================================================================================
    for t_exit,price_exit in prices.loc[t:][1:].iterrows(): 
 
      ## PREDICTIVE EXIT CONDITION ++++++++++++++++++++++++++++++
      if signal == 0:
        predsignal = price_exit.Open < prices.mx.shift(1)[t_exit]
      elif signal == 1:
        predsignal = price_exit.Open > prices.mx.shift(1)[t_exit]
      ## PREDICTIVE EXIT CONDITION ++++++++++++++++++++++++++++++
 
      ## PREDICTIVE EARLY EXIT CONDITION ++++++++++++++++++++++++++++++
      if tradetype == '+':
        TP = price_exit['-str']  ################################################################################ TRY FRACTIONS 
        earlyexit= price_exit.High > price_entry + TP  
      elif tradetype == '-':
        TP = price_exit['+str']
        earlyexit= price_exit.Low < price_entry - TP
      ## PREDICTIVE EARLY EXIT CONDITION ++++++++++++++++++++++++++++++
 
      # TP at ATR (prelim draft)
      if earlyexit:
        # Position entry/exit update
        position_update(prices,t_exit,tradetype+'entry',price_entry)
        position_update(prices,t_exit,tradetype+'exit',price_entry + position_sign*TP)                                                                                                                        
        # Position size update
        position_update(prices,t_exit,tradetype+entry_position,1)
        position_update(prices,t_exit,tradetype+exit_position,1)  
        # PNL update
        settlement_price = prices.Close.shift(1)[t_exit]
        position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price ,position_extractor(prices,t_exit,tradetype+'exit'),ticksize)) 
        # position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(price_entry ,position_extractor(prices,t_exit,tradetype+'exit'),ticksize)) ## HERE ERROR!!!! PRICE_ENTRY supposed to be settlement
 
        print(t_exit, ': Early exit ' + exit_position +' at:',position_extractor(prices,t_exit,tradetype+'exit') )  
        break
 
      # Check for secondary stoploss (new day open < mx ) if so EXIT AT OPEN, ELSE REINSTATE-----------------------------------------------------------------------
      elif predsignal: 
        # Position entry/exit update
        position_update(prices,t_exit,tradetype+'entry',price_entry)
        position_update(prices,t_exit,tradetype+'exit',price_exit.Open)
 
        # Position size update  
        position_update(prices,t_exit,tradetype+entry_position,1) 
        position_update(prices,t_exit,tradetype+exit_position,1) 
        
        # PNL Update
        settlement_price = prices.Close.shift(1)[t_exit]
        position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,position_extractor(prices,t_exit,tradetype+'exit'),ticksize))                       #!!!! used to be close BUT must check for lookaheadbias !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print(t_exit, ': Secondary SL (Pred '+actualtide+') so exit:',position_extractor(prices,t_exit,tradetype+'exit'))
 
        # CHECK EOD if 
        if price_exit.tide == opp_signal:
            " !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! "
            print('!!!!')
 
 
        break
 
      else: # actual tide == predicted tide
        print(t_exit, ': HOLD since '+tradetype+ ' tide', 'TP: ', TP)
        # Position entry/exit update
        position_update(prices,t_exit,tradetype+'entry',price_entry)
        position_update(prices,t_exit,tradetype+'exit',price_exit.Close)
 
        # Position size update  
        position_update(prices,t_exit,tradetype+entry_position,1)
        position_update(prices,t_exit,tradetype+exit_position,0) 
        
        # Position PNL Update
        settlement_price = prices.Close.shift(1)[t_exit]
        position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,price_exit.Close,ticksize))
 
 
  return prices

#@title Cons Tide { form-width: "200px" }
 
"""
 
Cons Tide ==========================================================================================================================================================================================================================================================================================================================================
 
"""
 
 
def contra_constide(prices,m1,m2,SL,ticksize):
  prices=tidegenerator(prices,m1,m2,ticksize)
 
  # STREAMING BACKTESTER
  prices['BH']=(prices.Close - prices.Close.shift(1)).fillna(0)
 
  prices['+entry']=None
  prices['+exit']=None
  prices['+L']=None
  prices['+S']=None
  prices['+pnl']=None
  prices['+pnl*']=None
 
  prices['-entry']=None
  prices['-exit']=None
  prices['-L']=None
  prices['-S']=None
  prices['-pnl']=None
  prices['-pnl*']=None
 
  prices[['+entry','+exit','-entry','-exit','+L','-L','+S','-S','+pnl','-pnl','+pnl*','-pnl*']]=prices[['+entry','+exit','-entry','-exit','+L','-L','+S','-S','+pnl','-pnl','+pnl*','-pnl*']].astype(object)
  
 
  for t,price in prices.iterrows():   
    # if +tide prev day AND high > prev 7dh, short at prev 7dl ===========================================================================================
    if ( (price.High > prices.xdh.shift(1)[t])):
      prices = cons_trade(prices,price,t,'+',SL,ticksize)
 
    # if -tide prev day AND low < prev 7dl, buy at prev 7dl ===========================================================================================
    elif ( (price.Low < prices.xdl.shift(1)[t])):
      prices = cons_trade(prices,price,t,'-',SL,ticksize)
 
  ## Sum running pnl
  global pnlpos, pnlneg
  pnlpos=[]
  pnlneg=[]
 
  for t,cell in prices.fillna(value=0).iterrows():
    pnlpos.append(np.sum(cell['+pnl']))
    pnlneg.append(np.sum(cell['-pnl']))
 
  pnlpos=pd.Series(pnlpos)
  pnlpos.index=prices.index
  pnlneg=pd.Series(pnlneg)
  pnlneg.index=prices.index
  prices['pnl']=pnlpos+pnlneg
 
 
  # sum pnl*
  pnlpos=[]
  pnlneg=[]
  for t,cell in prices.iterrows():
    if type(cell['+entry']) == list:
      pnlpos.append([pnlrounding(entry,exit,ticksize) for entry,exit in zip(cell['+entry'],cell['+exit'])])
    elif (type(cell['+entry']) == float) or (type(cell['+entry']) == np.float64):
      pnlpos.append(cell['+exit']-cell['+entry'])
    else:
      pnlpos.append(None)
 
  for t,cell in prices.iterrows():
    if type(cell['-entry']) == list:
      pnlneg.append([pnlrounding(entry,exit,ticksize) for entry,exit in zip(cell['-entry'],cell['-exit'])])
    elif (type(cell['-entry']) == float) or (type(cell['-entry']) == np.float64):
      pnlneg.append(cell['-entry']-cell['-exit'])
    else:
      pnlneg.append(None)
 
 
 
  prices['+pnl*']=pnlpos
  prices['-pnl*']=pnlneg
 
 
 
  return prices
 
 
"""
 
Cons Tide trade
 
"""
 
def cons_trade(prices,price, t, tradetype,SL,ticksize):
  if tradetype == '+':
    entry_position='S' # Since contrarian
    exit_position ='L'
    position_sign = -1
    signal=1
    opp_signal=0
    xd='xdh'
    gappedopen=price.Open > prices[xd].shift(1)[t]
  elif tradetype == '-':
    entry_position='L' # Since contrarian
    exit_position ='S'
    position_sign = 1
    signal=0
    opp_signal=1
    xd='xdl'
    gappedopen=price.Open < prices[xd].shift(1)[t]
  SL=prices.atr.shift()[t]/2
  ## DEFINE ENTRY PRICE ////////////////////////////////////////////////////////
  # Check if open higher/lower than previous day xdh/xdl (Gapped entry)
  if gappedopen: # if gapped down, enter at opening
    position_update(prices,t,tradetype+'entry',price.Open) 
    if tradetype=='+':
      slsignaltoday= price.High > position_extractor(prices,t,tradetype+'entry') + SL
    elif tradetype=='-':
      slsignaltoday= price.Low < position_extractor(prices,t,tradetype+'entry') - SL                      
  else: # if gapped up, enter at opening
    position_update(prices,t,tradetype+'entry',prices[xd].shift(1)[t])  
    if tradetype=='+':
      slsignaltoday= price.High > prices[xd].shift(1)[t] + SL
    elif tradetype=='-':
      slsignaltoday= price.Low < prices[xd].shift(1)[t] - SL
 
  price_entry=position_extractor(prices,t,tradetype+'entry')
  ## DEFINE ENTRY PRICE ////////////////////////////////////////////////////////
  print(t,': Cons ' + tradetype + 'tide ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ entry: ', price_entry)
 
  # ENTRY DAY SL: exit at 1 point below or above 7dl/7dh
  if slsignaltoday:
    # Position exit update
    position_update(prices,t,tradetype+'exit',price_entry - position_sign*(SL))
    # Position size update
    position_update(prices,t,tradetype+entry_position,1)
    position_update(prices,t,tradetype+exit_position,1)  
    # PNL Update                        
    position_update(prices,t,tradetype+'pnl',position_sign*pnlrounding(price_entry,position_extractor(prices,t,tradetype+'exit'),ticksize)) 
    print(t,': Entry day SL: ' + exit_position + ' at buffer:',position_extractor(prices,t,tradetype+'exit') )
 
  else: # else hold (loop) till next day
    # Position exit update
    position_update(prices,t,tradetype+'exit',price.Close)
    # Position size update
    position_update(prices,t,tradetype+entry_position,1)
    position_update(prices,t,tradetype+exit_position,0) 
    # PNL Update 
    position_update(prices,t,tradetype+'pnl',position_sign*pnlrounding(price_entry,position_extractor(prices,t,tradetype+'exit'),ticksize))
 
 
    # LOOP: look for exit - in this case; buy at mx when price retraces back down ============================================================================================================================================= 
    for t_exit,price_exit in prices.loc[t:][1:].iterrows():
      SL=prices.atr.shift(1)[t_exit]/2
 
      ## STOP LOSS CONDITIONS ++++++++++++++++++++++++++++
      prevdaymx = prices.mx.shift(1)[t_exit]
      if tradetype == '-':
        cutmxsignal = price_exit.High >= prevdaymx
        slsignal= price_entry-SL > price_exit.Low # low
      elif tradetype == '+':
        cutmxsignal = price_exit.Low <= prevdaymx
        slsignal= price_entry+SL < price_exit.High # high
      ## STOP LOSS CONDITIONS ++++++++++++++++++++++++++++
 
 
      # Check for stoploss first (1 point above entry) THEN EXIT THERE! ---------------------------------------------------------------
      if slsignal: 
        # Position entry/exit update
        position_update(prices,t_exit,tradetype+'entry',price_entry)
        position_update(prices,t_exit,tradetype+'exit',price_entry - position_sign*SL)
 
        # Position sizes update
        position_update(prices,t_exit,tradetype+entry_position,1)
        position_update(prices,t_exit,tradetype+exit_position,1)
 
        # PNL Update 
        settlement_price = prices.Close.shift(1)[t_exit]
        position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,position_extractor(prices,t_exit,tradetype+'exit'),ticksize))
        print(t_exit,': SL at ', position_extractor(prices,t_exit,tradetype+'exit')  )
        break
 
      # If price cut mx TAKE PROFIT AT MX! --------------------------------------------------------------------------------------------
      # i.e.; for +tide, if price touches mx from above, and for -ve tide, if price touches mx from below
      elif cutmxsignal: #& (price_exit.tide == signal):
        # Position entry/exit update  
        position_update(prices,t_exit,tradetype+'entry',price_entry)
        position_update(prices,t_exit,tradetype+'exit',price_exit.Close)  # Wait till EOD and exit at close
 
        # Position sizes update
        position_update(prices,t_exit,tradetype+entry_position,1)   
        position_update(prices,t_exit,tradetype+exit_position,1) 
 
        # PNL Update
        settlement_price = prices.Close.shift(1)[t_exit]
        position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,position_extractor(prices,t_exit,tradetype+'exit'),ticksize)) 
        print(t_exit,': MX HIT so exit at: ',position_extractor(prices,t_exit,tradetype+'exit'))
        break
 
      
      # HOLD --------------------------------------------------------------------------------------------------------------------------
      else: 
        print(t_exit,': HOLD')
 
        # Position entry/exit update
        position_update(prices,t_exit,tradetype+'entry',price_entry)
        position_update(prices,t_exit,tradetype+'exit',price_exit.Close)
 
        # Position size update  
        position_update(prices,t_exit,tradetype+entry_position,1)
        position_update(prices,t_exit,tradetype+exit_position,0) 
        
        # Position PNL Update
        settlement_price = prices.Close.shift(1)[t_exit]
        position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,price_exit.Close,ticksize))
 
 
 
  return prices

#@title Cons2 Tide "UNDER CONSTRUCTION"{ form-width: "200px" }
 
"""
 
Cons2 Tide ==========================================================================================================================================================================================================================================================================================================================================
 
"""
 
 
def cons2tide(prices,m1,m2,SL,ticksize):
  prices=tidegenerator(prices,m1,m2,ticksize)
 
  # STREAMING BACKTESTER
  prices['BH']=(prices.Close - prices.Close.shift(1)).fillna(0)
 
  prices['+entry']=None
  prices['+exit']=None
  prices['+L']=None
  prices['+S']=None
  prices['+pnl']=None
  prices['+pnl*']=None
 
  prices['-entry']=None
  prices['-exit']=None
  prices['-L']=None
  prices['-S']=None
  prices['-pnl']=None
  prices['-pnl*']=None
 
  prices[['+entry','+exit','-entry','-exit','+L','-L','+S','-S','+pnl','-pnl','+pnl*','-pnl*']]=prices[['+entry','+exit','-entry','-exit','+L','-L','+S','-S','+pnl','-pnl','+pnl*','-pnl*']].astype(object)
  
 
  for t,price in prices.iterrows():   
    # if -tide prev day and price touches previous MX ===========================================================================================
    if  (prices.tide.shift(1)[t] == 0) & (price.High < prices.mx.shift(1)[t]):
      prices = cons2_trade(prices,price,t,'+',SL,ticksize)
 
    # if +tide prev day and price touches previous MX ===========================================================================================
    elif (prices.tide.shift(1)[t] == 1) & (price.Low > prices.mx.shift(1)[t]):
      prices = cons2_trade(prices,price,t,'-',SL,ticksize)
 
  ## Sum running pnl
  global pnlpos, pnlneg
  pnlpos=[]
  pnlneg=[]
 
  for t,cell in prices.fillna(value=0).iterrows():
    pnlpos.append(np.sum(cell['+pnl']))
    pnlneg.append(np.sum(cell['-pnl']))
 
  pnlpos=pd.Series(pnlpos)
  pnlpos.index=prices.index
  pnlneg=pd.Series(pnlneg)
  pnlneg.index=prices.index
  prices['pnl']=pnlpos+pnlneg
 
 
  # sum pnl*
  pnlpos=[]
  pnlneg=[]
  for t,cell in prices.iterrows():
    if type(cell['+entry']) == list:
      pnlpos.append([pnlrounding(entry,exit,ticksize) for entry,exit in zip(cell['+entry'],cell['+exit'])])
    elif (type(cell['+entry']) == float) or (type(cell['+entry']) == np.float64):
      pnlpos.append(cell['+exit']-cell['+entry'])
    else:
      pnlpos.append(None)
 
  for t,cell in prices.iterrows():
    if type(cell['-entry']) == list:
      pnlneg.append([pnlrounding(entry,exit,ticksize) for entry,exit in zip(cell['-entry'],cell['-exit'])])
    elif (type(cell['-entry']) == float) or (type(cell['-entry']) == np.float64):
      pnlneg.append(cell['-entry']-cell['-exit'])
    else:
      pnlneg.append(None)
 
 
 
  prices['+pnl*']=pnlpos
  prices['-pnl*']=pnlneg
 
 
 
  return prices
 
 
"""
 
Cons Tide trade
 
"""
 
def cons2_trade(prices,price, t, tradetype,SL,ticksize):
  if tradetype == '+':
    entry_position='S' # Since contrarian
    exit_position ='L'
    position_sign = -1
    signal=1
    opp_signal=0
    xd='xdh'
    gappedopen=price.Open > prices[xd].shift(1)[t]
  elif tradetype == '-':
    entry_position='L' # Since contrarian
    exit_position ='S'
    position_sign = 1
    signal=0
    opp_signal=1
    xd='xdl'
    gappedopen=price.Open < prices[xd].shift(1)[t]
  SL=prices.atr.shift()[t]/2
  ## DEFINE ENTRY PRICE ////////////////////////////////////////////////////////
  # Check if open higher/lower than previous day xdh/xdl (Gapped entry)
  if gappedopen: # if gapped down, enter at opening
    position_update(prices,t,tradetype+'entry',price.Open) 
    if tradetype=='+':
      slsignaltoday= price.High > position_extractor(prices,t,tradetype+'entry') + SL
    elif tradetype=='-':
      slsignaltoday= price.Low < position_extractor(prices,t,tradetype+'entry') - SL                      
  else: # if gapped up, enter at opening
    position_update(prices,t,tradetype+'entry',prices[xd].shift(1)[t])  
    if tradetype=='+':
      slsignaltoday= price.High > prices[xd].shift(1)[t] + SL
    elif tradetype=='-':
      slsignaltoday= price.Low < prices[xd].shift(1)[t] - SL
 
  price_entry=position_extractor(prices,t,tradetype+'entry')
  ## DEFINE ENTRY PRICE ////////////////////////////////////////////////////////
  print(t,': Cons ' + tradetype + 'tide ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ entry: ', price_entry)
 
  # ENTRY DAY SL: exit at 1 point below or above 7dl/7dh
  if slsignaltoday:
    # Position exit update
    position_update(prices,t,tradetype+'exit',price_entry - position_sign*(SL))
    # Position size update
    position_update(prices,t,tradetype+entry_position,1)
    position_update(prices,t,tradetype+exit_position,1)  
    # PNL Update                        
    position_update(prices,t,tradetype+'pnl',position_sign*pnlrounding(price_entry,position_extractor(prices,t,tradetype+'exit'),ticksize)) 
    print(t,': Entry day SL: ' + exit_position + ' at buffer:',position_extractor(prices,t,tradetype+'exit') )
 
  else: # else hold (loop) till next day
    # Position exit update
    position_update(prices,t,tradetype+'exit',price.Close)
    # Position size update
    position_update(prices,t,tradetype+entry_position,1)
    position_update(prices,t,tradetype+exit_position,0) 
    # PNL Update 
    position_update(prices,t,tradetype+'pnl',position_sign*pnlrounding(price_entry,position_extractor(prices,t,tradetype+'exit'),ticksize))
 
 
    # LOOP: look for exit - in this case; buy at mx when price retraces back down ============================================================================================================================================= 
    for t_exit,price_exit in prices.loc[t:][1:].iterrows():
      SL=prices.atr.shift(1)[t_exit]/2
 
      ## STOP LOSS CONDITIONS ++++++++++++++++++++++++++++
      prevdaymx = prices.mx.shift(1)[t_exit]
      if tradetype == '-':
        cutmxsignal = price_exit.High >= prevdaymx
        slsignal= price_entry-SL > price_exit.Low # low
      elif tradetype == '+':
        cutmxsignal = price_exit.Low <= prevdaymx
        slsignal= price_entry+SL < price_exit.High # high
      ## STOP LOSS CONDITIONS ++++++++++++++++++++++++++++
 
 
      # Check for stoploss first (1 point above entry) THEN EXIT THERE! ---------------------------------------------------------------
      if slsignal: 
        # Position entry/exit update
        position_update(prices,t_exit,tradetype+'entry',price_entry)
        position_update(prices,t_exit,tradetype+'exit',price_entry - position_sign*SL)
 
        # Position sizes update
        position_update(prices,t_exit,tradetype+entry_position,1)
        position_update(prices,t_exit,tradetype+exit_position,1)
 
        # PNL Update 
        settlement_price = prices.Close.shift(1)[t_exit]
        position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,position_extractor(prices,t_exit,tradetype+'exit'),ticksize))
        print(t_exit,': SL at ', position_extractor(prices,t_exit,tradetype+'exit')  )
        break
 
      # If price cut mx TAKE PROFIT AT MX! --------------------------------------------------------------------------------------------
      # i.e.; for +tide, if price touches mx from above, and for -ve tide, if price touches mx from below
      elif cutmxsignal: #& (price_exit.tide == signal):
        # Position entry/exit update  
        position_update(prices,t_exit,tradetype+'entry',price_entry)
        position_update(prices,t_exit,tradetype+'exit',price_exit.Close)  # Wait till EOD and exit at close
 
        # Position sizes update
        position_update(prices,t_exit,tradetype+entry_position,1)   
        position_update(prices,t_exit,tradetype+exit_position,1) 
 
        # PNL Update
        settlement_price = prices.Close.shift(1)[t_exit]
        position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,position_extractor(prices,t_exit,tradetype+'exit'),ticksize)) 
        print(t_exit,': MX HIT so exit at: ',position_extractor(prices,t_exit,tradetype+'exit'))
        break
 
      
      # HOLD --------------------------------------------------------------------------------------------------------------------------
      else: 
        print(t_exit,': HOLD')
 
        # Position entry/exit update
        position_update(prices,t_exit,tradetype+'entry',price_entry)
        position_update(prices,t_exit,tradetype+'exit',price_exit.Close)
 
        # Position size update  
        position_update(prices,t_exit,tradetype+entry_position,1)
        position_update(prices,t_exit,tradetype+exit_position,0) 
        
        # Position PNL Update
        settlement_price = prices.Close.shift(1)[t_exit]
        position_update(prices,t_exit,tradetype+'pnl',position_sign*pnlrounding(settlement_price,price_exit.Close,ticksize))
 
 
 
  return prices

#@title Price update function
def updateprices(ticker):
  prices = pd.read_excel('drive/My Drive/Prices/'+ticker+'.xlsx')
  # Set Date as index
  prices.index = pd.to_datetime(prices.Date,dayfirst=False)
  prices.Date = prices.index
  # Convert from int to float 
  prices.Open=prices.Open.astype(float)
  prices.High=prices.High.astype(float)
  prices.Low=prices.Low.astype(float)
  prices.Close=prices.Close.astype(float)
 
  ## UPDATE PRICES if need be
  print('Update with latest prices?')
  yn=input()
  if yn=='y':
    print('OHLC: ')
    #OHLC = float(input());
    OHLC=[float(i) for i in input().split(",")]
 
    
    current_prices=[pd.to_datetime(pd.to_datetime('today').date()),OHLC[0],OHLC[1],OHLC[2],OHLC[3]] 
    prices = prices.append(pd.Series(current_prices, index=prices.columns), ignore_index = True)
    prices.index = prices.Date
 
  try:
    if OHLC != "":
      print('Update Google Drive with latest prices?')
      if input()=='y':
        prices.drop(columns=['Date']).to_excel('/content/drive/My Drive/Prices/' +ticker+'.xlsx')
  except:
    print("Not updated") 
 
  return prices

