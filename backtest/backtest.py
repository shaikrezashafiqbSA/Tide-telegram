import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from performance.metrics import calc_equity_drawdown, backtest_summary
from performance.backtest_plot import backtest_plots, multi_backtest_plots

# pip install pandas==1.4.3
def pickle_klines(data=None):
    w = ['rb' if data is None else 'wb'][0]
    with open('../database/klines_indicators_dict.pickle', w) as handle:
        if data is None:
            data = pickle.load(handle)
            return data
        else: 
            pickle.dump(data, handle)
            
            
def get_signal(i,np_closePx, signals_dict,position="long",side="buy",entry_i = None)-> bool:
            
    if position == "long":        
        if side == "buy": 
            signal = signals_dict["Y"][i]>0
        elif side == "sell":
            signal = signals_dict["Y"][i]<=0
            
    elif position == "short":
        if side == "buy": 
            signal = signals_dict["Y"][i]<0
        elif side == "sell":
            signal = signals_dict["Y"][i]>=0

        
    return signal


def get_signal_meta(i,np_closePx, signals_dict,position="long",side="buy",entry_i = None)-> bool:
    L_uq=0.99
    L_lq = 0.5
    L_q_lookback = 36
    
    S_uq=0.99
    S_lq = 0.5
    S_q_lookback = 36
    if i < L_q_lookback or i < S_q_lookback:
        return False


    if position == "long":        
        if side == "buy": 
            new_tide = signals_dict["signal"][i]==1 #and signals_dict["signal"][i-1]==-1
            signal_stronk = signals_dict["p"][i] >= np.quantile(signals_dict["p"][i-L_q_lookback:i+1],L_uq)
            signal = new_tide and signal_stronk
        elif side == "sell":
            signal_degrade = signals_dict["signal"][i]==1 and signals_dict["p"][i] <= np.quantile(signals_dict["p"][i-L_q_lookback:i+1],L_lq)
            # signal_degrade = signals_dict["signal"][i]==1 and signals_dict["p"][i] <= np.quantile(signals_dict["p"][entry_i:i],L_lq)
            signal_flip = signals_dict["signal"][i]==-1 and signals_dict["p"][i] >= np.quantile(signals_dict["p"][i-S_q_lookback:i+1],S_uq)
            signal = signal_degrade or signal_flip
            
    elif position == "short":
        if side == "buy": 
            new_tide = signals_dict["signal"][i]==-1 #and signals_dict["signal"][i-1]==1 
            signal_stronk = signals_dict["p"][i] >= np.quantile(signals_dict["p"][i-S_q_lookback:i+1],S_uq) 
            signal = new_tide and signal_stronk
        elif side == "sell":
            signal_degrade = signals_dict["signal"][i]==-1 and signals_dict["p"][i] <= np.quantile(signals_dict["p"][i-S_q_lookback:i+1],S_lq)
            # signal_degrade = signals_dict["signal"][i]==-1 and signals_dict["p"][i] <= np.quantile(signals_dict["p"][entry_i:i],S_lq)
            signal_flip = signals_dict["signal"][i]==1 and signals_dict["p"][i] >= np.quantile(signals_dict["p"][i-L_q_lookback:i+1],L_uq)
            signal = signal_degrade or signal_flip

    
    return signal

def get_signal_meta_target(i,np_closePx, signals_dict,position="long",side="buy",entry_i = None)-> bool:
    if position == "long":        
        if side == "buy": 
            signal = signals_dict["signal"][i]==1 and signals_dict["p_target"][i]==1
        elif side == "sell":
            signal =  signals_dict["signal"][i]==1 and signals_dict["p_target"][i]<1 
            
    elif position == "short":
        if side == "buy": 
            signal = signals_dict["signal"][i]==-1 and signals_dict["p_target"][i]==1
        elif side == "sell":
            signal = signals_dict["signal"][i]==-1 and signals_dict["p_target"][i]<1 

    return signal
# =============================================================================
# Trend Following
# =============================================================================

def get_signal_TF(i,np_closePx, signals_dict,position="long",side="buy",entry_i = None)-> bool: 
    """
    Just follow Tide
    """
    tf = "2h"

    # if position == "long":
    #     if side == "buy": 
    #         signal = signals_dict[f"{tf}_ebb_fast"][i]> signals_dict[f"{tf}_ebb_slow"][i] 
    #     elif side == "sell":
    #         signal = signals_dict[f"{tf}_ebb_fast"][i] < signals_dict[f"{tf}_ebb_slow"][i]
        
    # elif position == "short":
    #     if side == "buy":
    #         signal = signals_dict[f"{tf}_ebb_fast"][i] < signals_dict[f"{tf}_ebb_slow"][i]
    #     elif side == "sell":
    #         signal = signals_dict[f"{tf}_ebb_fast"][i]> signals_dict[f"{tf}_ebb_slow"][i]
    
    tide = "fast"
    if position == "long":
        if side == "buy": 
            signal = signals_dict[f"{tf}_tide_{tide}"][i]> 0
        elif side == "sell":
            signal = signals_dict[f"{tf}_tide_{tide}"][i] <= 0
        
    elif position == "short":
        if side == "buy":
            signal = signals_dict[f"{tf}_tide_{tide}"][i] <=0
        elif side == "sell":
            signal = signals_dict[f"{tf}_tide_{tide}"][i] > 0 

    return signal

# =============================================================================
# Mean Reversion
# =============================================================================
def get_signal_MR(i,np_closePx, signals_dict,position="long",side="buy",entry_i = None)-> bool: 
    """
    Mean Reversion around 1h ebb 1.43 sharpe on longs but only when window dressed :( 
    
    Parameters:
    'tide_fast': {'window': [12,24,36], "sensitivity": [10], "thresholds": [20], 'price':['open','high','low']},
    'tide_slow': {'window': [12,24,36], "sensitivity": [90], "thresholds": [20],'price':['open','high','low']},
        
    """
            
    if position == "long":
        if side == "buy": 
            below_ebb =  np_closePx[i] < signals_dict["1h_ebb_fast"][i] 
            pos_tide = signals_dict["1h_tide_slow"][i] > 0 and signals_dict["1h_tide_slow"][i-1] < 0
            signal = pos_tide and below_ebb
        elif side == "sell":
            
            TP =  np_closePx[i] >=  signals_dict["1h_ebb_fast"][i] and np_closePx[i] >=  signals_dict["1h_ebb_slow"][i]
            # TP =  np_closePx[i] <= signals_dict["upper_L"][i-1] and np_closePx[i] >= signals_dict["upper_L"][i-2]
            SL =  np_closePx[i] >= signals_dict["1h_ebb_fast"][i]  and np_closePx[i-1] >= signals_dict["1h_ebb_fast"][i-1]
            # TODO: NEED A PROPER SL signal here
            signal = TP 
        

    elif position == "short":
        # if side == "buy":
        #     above_support = signals_dict["1h_ebb_slow"][i] < np_closePx[i]
        #     overbought = signals_dict["1h_ebb_high"][i] > signals_dict["upper_S"][i-1] and signals_dict["1h_ebb_high"][i-1] < signals_dict["lower_S"][i-2]
        #     signal = overbought  
        # elif side == "sell":
        #     TP =  signals_dict["1h_ebb_slow"][i] >= np_closePx[i] and signals_dict["1h_ebb_slow"][i-1] < np_closePx[i-1]
        #     SL =  signals_dict["1h_ebb_fast"][i] >= np_closePx[i] and signals_dict["1h_ebb_fast"][i-1] >= np_closePx[i-1]
        #     signal = TP or SL
        signal = False            
            
    return signal



def _backtest(df0,
             fee=0.0007,
             slippage = 0.0003, # 3bps for slippage
             long_notional=1000,
             short_notional=1000,
             signals = ["Y", "d", "s","u"], 
             signal_function = None,
             window=None,
             min_holding_period=1,
             max_positions = 5):
    if window is None:
        df=df0.copy()
    else:
        df=df0.copy()[window[0]:window[1]]
        
    # signals related
    signals_dict = {}
    for signal in signals:
        signals_dict[signal] = df[signal].values
        
    if signal_function is None:
        _get_signal = get_signal
    elif signal_function == "MR":
        _get_signal = get_signal_MR
    elif signal_function == "TF":
        _get_signal = get_signal_TF
    elif signal_function == "meta":
        _get_signal = get_signal_meta
    elif signal_function == "meta_target":
        _get_signal = get_signal_meta_target

        
    # Piggyback signals dict for strat type TF / MR
    signals_dict["L"] = None         
    signals_dict["S"] = None  
    
    # positions
    np_long_positions = np.full(len(df), np.nan)
    np_short_positions = np.full(len(df), np.nan)
    
    
    # px/qty/cost details
    np_long_id = np.full(len(df), np.nan)
    np_long_entry = np.full(len(df), np.nan)
    np_long_cost = np.full(len(df), np.nan)
    np_long_qty = np.full(len(df), np.nan)
    np_long_exit = np.full(len(df), np.nan)
    
    np_short_id = np.full(len(df), np.nan)
    np_short_entry = np.full(len(df), np.nan)
    np_short_cost = np.full(len(df), np.nan)
    np_short_qty = np.full(len(df), np.nan)
    np_short_exit = np.full(len(df), np.nan)
    
    
    # pnl details
    np_pnl = np.full(len(df), np.nan)
    
    np_long_rpnl = np.full(len(df), np.nan)
    np_long_pnl = np.full(len(df), np.nan)
    np_long_fees = np.full(len(df), np.nan)
    
    np_short_rpnl = np.full(len(df), np.nan)
    np_short_pnl = np.full(len(df), np.nan)
    np_short_fees = np.full(len(df), np.nan)
    
    # last price
    np_closePx = df['1h_close'].values

    
    # initialize state flags
    in_long_position  = False
    in_short_position = False
    long_id = 1
    short_id = 1
    


    # t0=time.time()
    for i in range(len(df)):
            
        # =============================================================================
        # ENTRIES
        # =============================================================================

        # ---------- #
        # ENTER LONG
        # ---------- #
        signal = _get_signal(i,np_closePx, signals_dict,position="long",side="buy")
        if (not in_long_position):
            signal = _get_signal(i,np_closePx, signals_dict,position="long",side="buy")
            if signal:
                # Trackers
                np_long_positions[i] = 1
                np_long_id[i] = long_id
                in_long_position = True
                long_openIdx = i
                
                # px/qty/cost details
                long_entry_Px =  np_closePx[i]*(1+slippage) 
                np_long_entry[i] = long_entry_Px
                np_long_cost[i] = long_notional
                np_long_qty[i] = long_notional/long_entry_Px
                signals_dict["L"] = long_entry_Px
                
        # ---------- #
        # ENTER SHORT
        # ---------- #
        if not in_short_position:
            signal = _get_signal(i,np_closePx, signals_dict,position="short",side="buy")
            if signal:
                # Trackers
                np_short_positions[i] = -1
                np_short_id[i] = short_id
                in_short_position = True
                short_openIdx = i
                
                # px/qty/cost details
                short_entry_Px = np_closePx[i]*(1-slippage) 
                np_short_entry[i] = short_entry_Px
                np_short_cost[i] = short_notional 
                np_short_qty[i] = short_notional/short_entry_Px
                signals_dict["S"] = short_entry_Px
                
        # =============================================================================
        # EXITS
        # =============================================================================     
        
        # ========== #
        # LONG
        # ========== #
        if in_long_position and (i > long_openIdx): 
            signal = _get_signal(i,np_closePx, signals_dict,position="long",side="sell",entry_i = long_openIdx)
            
            # ---------- #
            # EXIT LONG
            # ---------- #            
            if i >= (long_openIdx + min_holding_period) and signal:

                # Trackers
                np_long_positions[i] = 0
                in_long_position = False

                # px/qty/cost details
                long_exit_Px = np_closePx[i]*(1+slippage) 
                np_long_exit[i] = long_exit_Px
                np_long_cost[i] = np_long_cost[i-1]
                np_long_qty[i] = np_long_qty[i-1]
                
                # pnl details
                discrete_Long_pnl = (long_exit_Px-long_entry_Px)*np_long_qty[i]
                fees = np_long_qty[i]*long_exit_Px*fee + np_long_qty[i]*long_entry_Px*fee
                discrete_Long_pnl-=fees
                
                # Records
                np_long_fees[i] = fees
                np_pnl[i] = discrete_Long_pnl 
                np_long_rpnl[i] = discrete_Long_pnl
                np_long_id[i] = long_id
                long_id += 1
                signals_dict["L"] = None
                
            # ---------- #
            # STAY LONG
            # ---------- #
            else:
                np_long_id[i] = long_id
                np_long_positions[i] = np_long_positions[i-1]      
                np_long_cost[i] = np_long_cost[i-1]
                np_long_qty[i] = np_long_qty[i-1]
                np_long_pnl[i] = (np_closePx[i]-long_entry_Px)*np_long_qty[i]
                
                
        # ========== #
        # SHORT
        # ========== #   
        if in_short_position and (i > short_openIdx):
            signal = _get_signal(i,np_closePx, signals_dict,position="short",side="sell",entry_i = short_openIdx)
            
            # ---------- #
            # EXIT SHORT
            # ---------- #
            if i >= (short_openIdx + min_holding_period) and signal:
                
                # Trackers
                np_short_positions[i] = 0
                in_short_position = False

                # px/qty/cost details
                short_exit_Px = np_closePx[i]*(1-slippage)
                np_short_exit[i] = short_exit_Px
                np_short_cost[i] = np_short_cost[i-1]
                np_short_qty[i] = np_short_qty[i-1]
                
                # pnl details
                discrete_Short_pnl = (short_entry_Px-short_exit_Px)*np_short_qty[i]
                fees = np_short_qty[i]*short_exit_Px*fee + np_short_qty[i]*short_entry_Px*fee
                discrete_Short_pnl-=fees
                
                # Records
                np_short_fees[i] = fees
                np_pnl[i] = discrete_Short_pnl
                np_short_rpnl[i] = discrete_Short_pnl
                np_short_id[i] = short_id
                short_id += 1
                signals_dict["S"] = None
                
            # ---------- #
            # STAY SHORT
            # ---------- #
            else:
                np_short_id[i] = short_id
                np_short_positions[i] = np_short_positions[i-1]
                np_short_cost[i] = np_short_cost[i-1]
                np_short_qty[i] = np_short_qty[i-1]
                np_short_pnl[i] = (short_entry_Px-np_closePx[i])*np_short_qty[i]
                    
    # END BACKTEST, summarise
    
    
    df["L_id"] = np_long_id
    df["L_positions"] = np_long_positions
    df["L_entry_price"] = np_long_entry
    df["L_cost"] = np_long_cost
    df["L_qty"] = np_long_qty
    df["L_exit_price"] = np_long_exit
    df["L_fees"] = np_long_fees
    df["L_pnl"] = np_long_pnl
    df["L_rpnl"] = np_long_rpnl
    
    df["S_id"] = np_short_id
    df["S_positions"] = np_short_positions
    df["S_entry_price"] = np_short_entry
    df["S_cost"] = np_short_cost
    df["S_qty"] = np_short_qty
    df["S_exit_price"] = np_short_exit
    df["S_fees"] = np_short_fees
    df["S_pnl"] = np_short_pnl
    df["S_rpnl"] = np_short_rpnl
    
    df["A_rpnl"] = np_pnl
    df["B_pnl"] = df["1h_close"].pct_change()
    
    return df


def backtest(df0,
             fee=0.0007,
             slippage = 0.0003, # 1bps for slippage
             long_notional=1000,
             short_notional=1000,
             signals = ["Y", "d", "s","u"], 
             signal_function=None,
             min_holding_period=1,
             max_positions = 5,
             plots=True,
             produce_signal =False,
             signal_name = None,
             window=None,
             horizon_labels=None,
             show_B=True):
    
    # if produce_signal:
    #     _window = window
    # else:
    #     _window = window
        
    t0=time.time()
    df = _backtest(df0,
                    fee = fee,
                    slippage = slippage, # 1bps for slippage
                    long_notional = long_notional,
                    short_notional = short_notional,
                    signals = signals, 
                    signal_function=signal_function,
                    window = window,
                    min_holding_period = min_holding_period,
                    max_positions = max_positions)
    dur_backtest = np.round(time.time()-t0,3)
    
    if produce_signal is False:
        t0=time.time()
        df_backtested, df_trades, df_summary = backtest_summary(df,long_notional,short_notional)
        dur_metrics = np.round(time.time()-t0,3)
        
        t0=time.time()
        if plots:
            backtest_plots(df_backtested,horizon_labels=horizon_labels,show_B = show_B)
        dur_plots = np.round(time.time()-t0,3)
            
        print(df_summary)
        print(f"\nBacktesting {df_backtested.index[0]} to {df_backtested.index[-1]} ({len(df)} rows)\nRuntimes\nbacktesting: {dur_backtest}s\nmetrics calc: {dur_metrics}s\nplots calc: {dur_plots}s")
        return df_backtested,df_trades,df_summary
    else:
        print(f"Runtime:\nGenerating signals {len(df)} rows: {dur_backtest}s")
        # if signal_name is None:
        #     df0["L_signal"] = df["L_positions"]
        #     df0["L_signal"].fillna(0,inplace=True)
        # else:
        #     df0[signal_name] = df["L_positions"]
        #     df0[signal_name].fillna(0,inplace=True)
        if signal_name is None:
            # df["S_positions"] = -1 * df["S_positions"]
            df0["signal"] = df[["L_positions","S_positions"]].sum(axis=1,skipna=True)
            df0["signal"].fillna(0,inplace=True)
        else:
            # df["S_positions"] = -1 * df["S_positions"]
            df0[signal_name] = df[["L_positions","S_positions"]].sum(axis=1,skipna=True)
            df0[signal_name].fillna(0,inplace=True)
        return df0
#%%


def multi_backtest(klines_indicators_dict,
                     fee=0.0007,
                     slippage = 0.0003, # 1bps for slippage
                     long_notional=1000,
                     short_notional=1000,
                     signals = ["Y", "d", "s","u"], 
                     min_holding_period=1,
                     max_positions = 5,
                     plots=True,
                     window=["2021-01-01","2022-12-31"]):
    
    backtested_dict = {}
    trades_dict = {}
    summary_dict = {}
    for instrument,df0 in tqdm(klines_indicators_dict.items()):
        df = _backtest(df0[window[0]:window[1]],
                        fee = fee,
                        slippage = slippage, # 1bps for slippage
                        long_notional = long_notional,
                        short_notional = short_notional,
                        signals = signals, 
                        window = window,
                        min_holding_period = min_holding_period,
                        max_positions = max_positions)

    
        df_backtested, df_trades, df_summary = backtest_summary(df,long_notional,short_notional)
        
        backtested_dict[instrument] = df_backtested
        trades_dict[instrument] = df_trades
        summary_dict[instrument] = df_summary
        
        
    if plots:
        multi_backtest_plots(backtested_dict)
    print(pd.concat(summary_dict))
    return backtested_dict,trades_dict,summary_dict
        