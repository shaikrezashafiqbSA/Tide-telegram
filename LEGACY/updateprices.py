# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:29:07 2021

@author: Shaik Reza Shafiq
"""

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