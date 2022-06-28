import ccxt
from utilities import read_config
import json
from datetime import datetime as dt
import pandas as pd
import numpy as np

#%% init
keys = json.loads(read_config.get_config('ACCT','keys'))
# keys['enableRateLimit']=True
# keys['options']={'fetchMyTradesMethod':'private_get_hist_orders'}
ccxt_client = ccxt.kucoin(keys)
#%% fetch balance
def get_balance():
    balance = ccxt_client.fetch_balance()
    bal_df = pd.DataFrame(balance["info"]["balances"])

#%% fetch closed orders
def fetchOrders(symbol):
    since = ccxt_client.parse8601('2021-10-01T00:00:00Z')
    to = ccxt_client.parse8601(str(dt.utcnow()))
    if to == None:
        to = ccxt_client.milliseconds()
    all_orders = []
    day = 24 * 3600 * 1000
    week = 7 * day
    limit = 20
    
    while since < to:
        end = since + week
        if end > to:
            end = to
        params = {'endAt': end}
        orders = ccxt_client.fetch_closed_orders(symbol, since, limit, params)
        print(ccxt_client.iso8601(since), '-', ccxt_client.iso8601(end), len(orders), 'orders')
        if len(orders) == limit:
            since = orders[-1]['timestamp']
        else:
            since += week
        all_orders.extend(orders)
    
    all_orders_df = pd.DataFrame(all_orders)
    if len(all_orders_df)>0:
        if any(all_orders_df['side'].str.contains('sell')):
            first_buy_after_last_sell_idx = np.where(all_orders_df['side'].eq('sell'),all_orders_df.index,0).max()+1
            all_orders_df = all_orders_df.iloc[first_buy_after_last_sell_idx:].copy()

        total_cost = all_orders_df['cost'].sum()
        total_amount = all_orders_df['amount'].sum()
        AEP = total_cost/total_amount
        # print(f"total_cost: {total_cost}\ntotal_amount: {total_amount}\nAEP: {AEP}")
        
        # get last price
        ticker = ccxt_client.fetchTicker(symbol)
        lastPrice = ticker['last']
        upnl = (lastPrice/AEP - 1)*100
        return {'orders':all_orders_df,'total_cost':total_cost, 'total_amount':total_amount,'AEP':AEP,'lastPrice':lastPrice, 'pct_upnl':upnl,'quote_qty':total_amount*lastPrice}
    else:
        return all_orders_df

# kcsusdt = fetchOrders(symbol="KCS-USDT")
#%%
pairs = ['BTC-USDT','ETH-USDT','KCS-USDT','HYDRA-USDT','SWASH-USDT',
         'ORAI-USDT','SHILL-USDT','HBAR-USDT','VR-USDT','LUNA-USDT',
         'DYP-USDT','SOL-USDT','FTM-USDT','NTVRK-USDT','XRP-USDT',
         'AXS-USDT','KONO-USDT','ANKR-USDT','ADA-USDT','XNL-USDT']
orders_dict={}
for pair in pairs:
    print(pair)
    orders = fetchOrders(symbol=pair)
    orders_dict[pair]=orders

orders_df = pd.DataFrame(orders_dict).T
summary_df = orders_df.copy()[['total_cost', 'total_amount', 'AEP', 'lastPrice', 'pct_upnl','quote_qty']]
summary_df.sort_values(by='total_cost',ascending=False,inplace=True)
# first_buy_after_last_sell_idx = np.where(orders['side'].eq('sell'),orders.index,0).max()+1
# openOrders = orders.iloc[first_buy_after_last_sell_idx:].copy()

deposits=ccxt_client.fetchDeposits()
deposits_df = pd.DataFrame(deposits)
total_USDT_deposited = deposits_df[deposits_df['currency']=='USDT']['amount'].sum()
