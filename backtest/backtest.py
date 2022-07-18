import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from performance.metrics import calc_equity_drawdown

# pip install pandas==1.4.3
def pickle_klines(data=None):
    w = ['rb' if data is None else 'wb'][0]
    with open(f'../database/klines_indicators_dict.pickle', w) as handle:
        if data is None:
            data = pickle.load(handle)
            return data
        else: 
            pickle.dump(data, handle)
            
def get_signal(idx,np_closePx, np_signal1, np_signal2,position="long",side="buy")-> bool:
    price = np_closePx[idx]
    if position == "long":
        if side == "buy":
            signal = (np_signal1[idx,0]>0) #and all(np_signal2[idx,-1:]>price)
        elif side == "sell":
            signal = (np_signal1[idx,0]<0) #and all(np_signal2[idx,-2:]<price)
    
    elif position == "short":
        if side == "buy":
            signal = (np_signal1[idx,0]<0) #and all(np_signal2[idx,-1:]<price)
        elif side == "sell":
            signal = (np_signal1[idx,0]>0) #and all(np_signal2[idx,-2:]>price)
            
    if (type(signal) != bool) and (type(signal) !=np.bool_):
        raise Exception(f"signal -->{signal}<-- should be bool, instead its typeis: {type(signal)}")
    return signal

def backtest(df0,
             trd_fees=0.0007,
             min_holding_period=1):
    df=df0.copy()
    # backtest related
    
    np_long_positions = np.full(len(df), np.nan)
    np_short_positions = np.full(len(df), np.nan)
    long_id = 1
    short_id = 1
    np_long_id = np.full(len(df), np.nan)
    np_short_id = np.full(len(df), np.nan)
    
    
    np_ret = np.full(len(df), np.nan)
    np_long_ret = np.full(len(df), np.nan)
    np_long_unret = np.full(len(df), np.nan)
    np_short_ret = np.full(len(df), np.nan)
    np_short_unret = np.full(len(df), np.nan)

    
    # forceCloseOut = True
    
    # signals related
    np_signal1 = df.filter(regex="tide$").values
    np_signal2 = df.filter(regex="ebb$").values
    
    
    # last price
    np_closePx = df['1h_close'].values
    np_long_entry = np.full(len(df), np.nan)
    np_long_exit = np.full(len(df), np.nan)
    np_short_entry = np.full(len(df), np.nan)
    np_short_exit = np.full(len(df), np.nan)
    # initialize state flags
    in_long_position  = False
    in_short_position = False
    
    for idx_row in range(len(df)):
            

        # ---------- #
        # ENTER LONG
        # ---------- #
        signal = get_signal(idx_row,np_closePx, np_signal1, np_signal2,position="long",side="buy")
        if (not in_long_position):
            signal = get_signal(idx_row,np_closePx, np_signal1, np_signal2,position="long",side="buy")
            if signal:
                np_long_positions[idx_row] = 1
                np_long_id[idx_row] = long_id
                np_long_unret[idx_row] = 0.0 # minus fees?
                
                in_long_position = True
                long_openIdx = idx_row
                
                long_entry_Px =  np_closePx[idx_row] 
                np_long_entry[idx_row] = long_entry_Px    
        # ---------- #
        # ENTER SHORT
        # ---------- #
        if not in_short_position:
            signal = get_signal(idx_row,np_closePx, np_signal1, np_signal2,position="short",side="buy")
            if signal:
                np_short_positions[idx_row] = -1
                np_short_id[idx_row] = short_id
                np_short_unret[idx_row] = 0.0 # minus fees?
                
                in_short_position = True
                short_openIdx = idx_row
                
                short_entry_Px = np_closePx[idx_row] 
                np_short_entry[idx_row] = short_entry_Px
                
        # LOOK TO EXIT LONG 
        if in_long_position and (idx_row >= long_openIdx + min_holding_period): 
            # ---------- #
            # EXIT LONG
            # ---------- #
            signal = get_signal(idx_row,np_closePx, np_signal1, np_signal2,position="long",side="sell")
            if signal:# or (forceCloseOut and idx_row == len(np_signal)-1): # -symmZ 
                np_long_positions[idx_row] = 0
                in_long_position = False

                
                long_exit_Px = np_closePx[idx_row]
                np_long_exit[idx_row] = long_exit_Px
                discrete_Long_ret = (long_exit_Px / long_entry_Px) - (2 * trd_fees) ## 2x trd_fees for entry & exit

                np_ret[idx_row] = discrete_Long_ret - 1
                np_long_ret[idx_row] = discrete_Long_ret - 1
                np_long_unret[idx_row] = discrete_Long_ret - 1
                
                np_long_id[idx_row] = long_id
                long_id += 1
                
            # ---------- #
            # STAY LONG
            # ---------- #
            else:
                np_long_id[idx_row] = long_id
                np_long_positions[idx_row] = np_long_positions[idx_row-1]      
                np_long_unret[idx_row] = (np_closePx[idx_row] / long_entry_Px) - 1
                
        # LOOK TO EXIT SHORT     
        if in_short_position and (idx_row >= short_openIdx + min_holding_period):
            # ---------- #
            # EXIT SHORT
            # ---------- #
            signal = get_signal(idx_row,np_closePx, np_signal1, np_signal2,position="short",side="sell")
            if signal:# or (forceCloseOut and idx_row == len(np_signal)-1): # symmZ
                np_short_positions[idx_row] = 0
                in_short_position = False

                
                short_exit_Px = np_closePx[idx_row]
                np_short_exit[idx_row] = short_exit_Px
                
                discrete_Short_ret = (short_entry_Px / short_exit_Px) - (2 * trd_fees) ## 2x trd_fees for entry & exit
                np_ret[idx_row] = discrete_Short_ret - 1
                np_short_ret[idx_row] = discrete_Short_ret - 1
                np_short_unret[idx_row] = discrete_Short_ret - 1
                
                np_short_id[idx_row] = short_id
                short_id += 1
                
            # ---------- #
            # STAY SHORT
            # ---------- #
            else:
                np_short_id[idx_row] = short_id
                np_short_positions[idx_row] = np_short_positions[idx_row-1]
                np_short_unret[idx_row] =  (short_entry_Px/np_closePx[idx_row]) - 1
                    
                    
    # END BACKTEST, summarise
    
    initial_column_names = list(df.columns)
    

    df["long_id"] = np_long_id
    df["long_positions"] = np_long_positions
    df["long_entry_price"] = np_long_entry
    df["long_exit_price"] = np_long_exit
    
    df["short_id"] = np_short_id
    df["short_positions"] = np_short_positions
    df["short_entry_price"] = np_short_entry
    df["short_exit_price"] = np_short_exit
    
    # df['runningPnl'] = df['positions'].shift() * df['1h_close'].pct_change() 
    df["buyhold"] = df['1h_close'].pct_change() 
    

    
    #% Discrete PnL from each trade. cannot cumsum without replacing nan with 0...hence we create 'Pnl_plot'
    
    df['Pnl'] = np_ret
    df["longUnPnl"] = np_long_unret
    df["longUnPnl_change"] = df["longUnPnl"].add(1).pct_change()

    gl = df.groupby("long_id").apply(lambda x: (x.longUnPnl+1).pct_change()).reset_index().set_index("date_time").drop(columns=["long_id"])
    df["longUnPnl_change"] = gl
    # df["longUnPnl_change"] = np.where(df["long_id"].isna(), np.nan, df["longUnPnl_change"])
    
    df["shortUnPnl"] = np_short_unret
    gs = df.groupby("short_id").apply(lambda x: (x.shortUnPnl+1).pct_change()).reset_index().set_index("date_time").drop(columns=["short_id"])
    df["shortUnPnl_change"] = gs
    # df["shortUnPnl_change"] = np.where(df["short_id"].isna(), np.nan, df["shortUnPnl_change"])
    
    # df["UnPnl_change"] = df["shortUnPnl_change"].fillna(0) + df["longUnPnl_change"].fillna(0)
    df["UnPnl_change"] = df[["shortUnPnl_change","longUnPnl_change"]].sum(axis=1)
    
    df['longPnl'] = np_long_ret
    df['shortPnl'] = np_short_ret
    
    df['Pnl'] = df['Pnl']#.fillna(0)  # needed to fillna(0) for cumsum in next step
    df['longPnl'] = df['longPnl']#.fillna(0) 
    df['shortPnl'] = df['shortPnl']#.fillna(0) 
    
    
    
    # takes into account PnL from minute-to-minute price variations
    
    df[["cum_buyhold",
        "cum_Pnl",
        "cum_longPnl",
        "cum_longUnPnl",
        "cum_shortPnl",
        "cum_shortUnPnl",
        "cum_UnPnl"]]=df[["buyhold",
                             "Pnl",
                             "longPnl",
                             "longUnPnl_change",
                             "shortPnl",
                             "shortUnPnl_change",
                             "UnPnl_change"]].cumsum().apply(np.exp)
    
    df.drop(columns=["UnPnl_change","longUnPnl_change", "shortUnPnl_change"],inplace=True)
    # df['cumsum_Pnl_plot'].plot(title=f'{pair} Pnl cumsum')
    df["cum_buyhold"].plot(label=f'buyhold')
    df['cum_UnPnl'].plot(label=f'strategy')
    df['cum_longUnPnl'].plot(label=f'strategy_longs')
    df['cum_shortUnPnl'].plot(label=f'strategy_shorts')
    plt.legend()
    plt.show()
    
    trade_cols = ["1h_tide","1h_close",'long_id',
                'long_positions',
                'long_entry_price',
                'long_exit_price',
                'longUnPnl',
                'longPnl',
                'short_id',
                'short_positions',
                'short_entry_price',
                'short_exit_price',
                'shortUnPnl',
                'shortPnl',
                'buyhold',
                'Pnl',
                'cum_buyhold',
                'cum_Pnl',
                'cum_longPnl',
                'cum_longUnPnl',
                'cum_shortPnl',
                'cum_shortUnPnl',
                'cum_UnPnl']
    df1=df[initial_column_names+trade_cols]
    trades = df[trade_cols]
    
    print(f"total # of trades: {df[['long_id','short_id']].max().sum()}")
    
    return {"df1":df1,"trades":trades}

#%%
def vectorised_backtest(df,fee=0.0001, holding_period=10):
    fig, axs = plt.subplots(2,1)
    
    df1=df.copy()
    df1["returns"] = np.log(df1['1h_close'] / df1['1h_close'].shift(1))

    # long only
    df1["position"] = df1["label"]

    df1["strategy"] = df1["position"].shift(1) * df1["returns"] - abs(df1["position"].diff()).fillna(0)*fee
    df1[["strategy","returns"]]=df1[["strategy","returns"]].cumsum().apply(np.exp)
    df1[["strategy","returns"]].plot(ax=axs[0])
    drawdown,maxdrawdown = calc_equity_drawdown(df1["strategy"]).values()
    drawdown.plot(ax=axs[1])
    
    no_trades = abs(df1["position"].diff().dropna()).astype(int).sum()
    print(f"Number of trades: {no_trades}, MDD: {maxdrawdown}")
    return df1


#%%

if __name__ == "__main__":
    klines_indicators_dict = pickle_klines()
    df = klines_indicators_dict["SGX_TWN1!"].copy()
    t0=time.time()
    df1,trades = backtest(df,trd_fees=0.00).values()
    t1=time.time()
    print(f"Backtest finished in {t1-t0}s")
    cols=list(df1.columns)
    # cols_to_get = ['1h_tide','1h_close','positions',
    #                "long_id","long_positions",'longUnPnl','longPnl',
    #                "short_id","short_positions", 'shortUnPnl','shortPnl',
    #                'Pnl', 'runningPnl', 'cum_runningPnl', 'longUnPnl','cum_longUnPnl','shortUnPnl', 'cum_shortUnPnl','UnPnl']
    # check = df1[cols_to_get]
    
        