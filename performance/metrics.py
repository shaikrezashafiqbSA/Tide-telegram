import pandas as pd
import numpy as np
import time

comments_dict = {1.0:"initial entry",
                 10.0:"DCA",
                 100.0:"DCA up",
                 -1.0:"max TP",
                 -10.0:"close < HMA",
                 -100.0: "cross < HMA",
                 -1000.0:"trailing_stop",
                 -10000.0:"timed SL",
                 -100000.0:"hard SL"}

def calc_position_drawdown(continuous_ret): # equivalent to returns.cumsum() #
    continuous_ret=1+continuous_ret
  # wealth_index = (df+1).cumprod()
    prev_peak = continuous_ret.cummax()
    dd = (continuous_ret-prev_peak)/prev_peak 
    return dd

def calc_equity_drawdown(equity):
   returns=equity.pct_change()
   cumulative_returns = returns.cumsum() + 1
   peaks = cumulative_returns.cummax()
   drawdown = ((cumulative_returns-peaks)/peaks) *100
   drawdown = drawdown.fillna(0)
   

   return drawdown


def backtest_summary(df0,long_notional,short_notional):
    df=df0.copy()
    summary_index = ["L","S","A","B"]
    # print(df.columns)
    initial_column_names = list(df.columns)
    
    # CUM PNL CALCULATIONS
    df["cum_L_rpnl"] = df['L_rpnl'].cumsum().fillna(method="ffill")
    df["cum_S_rpnl"] = df['S_rpnl'].cumsum().fillna(method="ffill")
    
    df["cum_L_pnl"]=df[["cum_L_rpnl","L_pnl"]].sum(axis=1)
    df["cum_S_pnl"]=df[["cum_S_rpnl","S_pnl"]].sum(axis=1)
    df["cum_A_pnl"]=df[["cum_L_pnl","cum_S_pnl"]].sum(axis=1)
    
    df["cum_B_pnl"] = (((df["B_pnl"]+1).cumprod()-1)*long_notional)
    
    # % Equity Calculations
    df["cum_B_pnl%"] = (long_notional+df["cum_B_pnl"]).pct_change().add(1).cumprod()*100
    df["cum_A_pnl%"] = (long_notional+df["cum_A_pnl"]).pct_change().add(1).cumprod()*100
    df["cum_L_pnl%"] = (long_notional+df["cum_L_pnl"]).pct_change().add(1).cumprod()*100
    df["cum_S_pnl%"] = (short_notional+df["cum_S_pnl"]).pct_change().add(1).cumprod()*100
    
    df["dd_B"]=calc_equity_drawdown(df["cum_B_pnl%"])
    df["dd_A"]=calc_equity_drawdown(df["cum_A_pnl%"])
    df["dd_L"]=calc_equity_drawdown(df["cum_L_pnl%"])
    df["dd_S"]=calc_equity_drawdown(df["cum_S_pnl%"])
    
    
    trade_cols = ['L_id',
                'L_positions',
                'L_entry_price',
                'L_cost',
                'L_qty',
                'L_exit_price',
                'L_pnl',
                'L_rpnl',
                'L_fees',
                'S_id',
                'S_positions',
                'S_entry_price',
                'S_cost',
                'S_qty',
                'S_exit_price',
                'S_pnl',
                'S_rpnl',
                'S_fees',
                'A_rpnl',
                'cum_B_pnl',
                'cum_A_pnl',
                'cum_L_pnl',
                'cum_L_rpnl',
                'cum_S_pnl',
                'cum_S_rpnl']

    # df1=df[initial_column_names+trade_cols]
    trades = df[trade_cols].dropna(subset=['L_entry_price','L_exit_price','S_entry_price','S_exit_price'],how="all")
    # print(df.columns)
    # =============================================================================
    # METRICS CALCULATIONS
    # =============================================================================
    
    # Number of trades
    number_of_trades = {}
    for c in summary_index:
        if c == "B":
            number_of_trades[c] = 1
        elif c == "A":
            number_of_trades[c] = df[['L_id','S_id']].max().sum()
        else:
            number_of_trades[c] = df[f'{c}_id'].max()
            
            
    # Time in trade
    time_in_trade_mean = {}
    time_in_trade_max = {}
    time_in_trade_min = {}
    total_hours = len(df)
    for c in summary_index:
        if c == "B":
            time_in_trade_max[c] = total_hours
            time_in_trade_min[c] = total_hours
            time_in_trade_mean[c] = total_hours
        elif c in ["S","L"]:
            time_in_trade_max[c] = df[f"{c}_id"].value_counts().max() #max(len(c_trades_grouped.groups))
            time_in_trade_min[c] = df[f"{c}_id"].value_counts().min() #min(len(c_trades_grouped.groups))
            time_in_trade_mean[c] = df[f"{c}_id"].value_counts().mean()
        elif c == "A":
            time_in_trade_max[c] = max(time_in_trade_max.values())
            time_in_trade_min[c] = min(time_in_trade_min.values())
            time_in_trade_mean[c] = total_hours/number_of_trades[c]
    

    # winrate
    winrate = {}
    total_trades = {}
    for c in summary_index:
        if c == "B":
            winrate[c] = 100
            total_trades[c] = 1
        else:#if c in ["S","L"]:
            wins =  trades[trades[f'{c}_rpnl']>0][f'{c}_rpnl'].count()
            lose =  trades[trades[f'{c}_rpnl']<0][f'{c}_rpnl'].count()
            total_trades[c] = wins+lose
            winrate[c] = wins/(wins+lose)*100
        # elif c == "A":
        #     time_in_trade_max[c] = max(time_in_trade_max.values())
        #     time_in_trade_min[c] = min(time_in_trade_min.values())
        #     time_in_trade_mean[c] = total_hours/number_of_trades[c]
            
    
    # profit factor/best/worst trades
    p2g = {}
    avg_wins = {}
    avg_loss = {}
    for c in summary_index:
        if c == "B":
            w_trades =  df[df[f'{c}_pnl']>0][f'{c}_pnl']
            l_trades = df[df[f'{c}_pnl']<0][f'{c}_pnl']
            
            gains =  w_trades.sum()
            pains =  l_trades.sum()
            p2g[c] =  gains/np.abs(pains)
            avg_wins[c] = w_trades.mean()
            avg_loss[c] = l_trades.mean()
        else:
            w_trades =  df[df[f'{c}_rpnl']>0][f'{c}_rpnl']
            l_trades = df[df[f'{c}_rpnl']<0][f'{c}_rpnl']
            
            gains =  w_trades.sum()
            pains =  l_trades.sum()
            p2g[c] =  gains/np.abs(pains)
            avg_wins[c] = w_trades.mean()
            avg_loss[c] = l_trades.mean()
    # Sharpe
    # for strat in ["buyhold"]

    # sharpe = trades['long_rpnl'].mean() / trades['long_rpnl'].std() * np.sqrt(number_of_trades) 
    sharpes = {}
    for c in summary_index:
        N = 365*24
        if c == "B":
            ret = df[f"cum_{c}_pnl"].add(long_notional).pct_change()
            sharpes[c] = (ret.mean() / ret.std()) * np.sqrt(N)
        else:
            # Since hourly  #total_trades[c] / ((df.index[-1] - df.index[0]).days) # where N is number of trades in a day
            # ret = df[f"cum_{c}_pnl%"]
            # ret = df[f'{c}_rpnl']
            if c == "A":
                notional = long_notional+short_notional
            elif c=="L":
                notional = long_notional
            elif c=="S":
                notional = short_notional
            ret = df[f"cum_{c}_pnl"].add(notional).pct_change()
            sharpes[c] = (ret.mean() / ret.std()) * np.sqrt(N)
    
    # def sharpe(returns,N):
    # # >1 is good
    #     return returns.mean()*N/returns.std()/ np.sqrt(N)
    # def sortino(returns,N):
    #     # 0 - 1.0 is suboptimal, 1> good, 3> very good
    #     std_neg = returns[returns<0].std()*np.sqrt(N)
    #     return returns.mean()*N/std_neg

    # def calmar(returns,N,mdd):
    #     # > 0.5 is good, 3.0 to 5.0 is very good
    #     return returns.mean()*N/abs(mdd/100)
    
    # Fees
    fees = {}
    for c in summary_index:
        if c == "B":
            fees[c] = 0
        elif c in ["L","S"]:
            fees[c] = trades[f"{c}_fees"].sum()
        else:
            fees[c] = fees["L"]+fees["S"]
    
    # returns
    ret = {}
    for c in summary_index:
        ret[c]=np.round(df[f"cum_{c}_pnl%"].iloc[-1]-100,2)
    
    # pnl and starting and final equity
    pnl = {} # cum_B_pnl
    eq_start = {}
    eq_end = {}
    for c in summary_index:
        if c == "B":
            pnl[c] = df["cum_B_pnl"].iloc[-1]
            eq_start[c] = long_notional
            eq_end[c] = eq_start[c] + pnl[c]
        elif c in ["L","S"]:    
            pnl[c]=df[f"cum_{c}_rpnl"].iloc[-1]
            if c =="L":
                eq_start[c] = long_notional
            else:
                eq_start[c] = short_notional
            eq_end[c] = eq_start[c] + pnl[c]
        else:
            pnl[c]=pnl["L"]+pnl["S"]
            eq_start[c] = eq_start["L"]+eq_start["S"]
            eq_end[c] = eq_start[c] + pnl[c]
    # maxdrawdown
    mdd = {}
    for c in summary_index:
        mdd[c] = np.round(df[f"dd_{c}"].min(),2)
    
    # time underwater
            
            
    summary = {"Sharpe": sharpes,
               "Total Return %":ret,
               "Equity Start $":eq_start,
               "Total Return $":pnl,
               "Fees $": fees,
               "Equity End $" :eq_end,
               "avg_wins":avg_wins,
               "avg_loss":avg_loss,
               "Profit Factor":p2g,
               "total_trades": total_trades,
               "MDD %":mdd,
               "Win Rate %":winrate,
               "Time in Trade Mean":time_in_trade_mean,
               "Time in Trade Max":time_in_trade_max,
               "Time in Trade Min":time_in_trade_min}
    
    # Tidy up trades table
    output_trade_cols = ['L_id',
                        'L_positions',
                        'L_entry_price',
                        'L_cost',
                        'L_qty',
                        'L_exit_price',
                        'L_rpnl',
                        'L_fees',
                        'S_id',
                        'S_positions',
                        'S_entry_price',
                        'S_cost',
                        'S_qty',
                        'S_exit_price',
                        'S_rpnl',
                        'S_fees',
                        'A_rpnl',
                        'cum_L_rpnl',
                        'cum_S_rpnl']
    
    #  Tidy up summary tables
    summary_df = pd.DataFrame(summary)
    for col in summary.keys():
        summary_df[col] = summary_df[[col]].round(decimals=pd.Series([2,2,2,2,2,2,3,3,3,1,1,1,1,1,1], index=summary_df.T.index))
    summary_df = summary_df.astype(object).T
    return df, trades[output_trade_cols], summary_df