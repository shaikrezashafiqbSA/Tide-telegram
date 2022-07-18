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
   
   max_drawdown = np.round(drawdown.min(),3)
   return {"drawdown":drawdown,"maxdrawdown":max_drawdown}


def tradeBatch(df,df_type = "metrics_df"):
    # if df_type == "trades_summary_df":
    #     df1=df.copy()
    #     longPositions = np.array(df1["#Positions"])
    #     tradeBatchId_vector = np.full(len(longPositions), np.nan) # generate array of np.nan
    #     tradeBatchId = 1 # first batch will be 1
    #     for idx,entryNum  in enumerate(longPositions):
    #         if entryNum > 0:
    #             tradeBatchId_vector[idx] = tradeBatchId
    #         elif entryNum == 0:
    #             tradeBatchId_vector[idx] = tradeBatchId
    #             tradeBatchId += 1 # increment to indicate next trade
    #     combine_array = np.stack([longPositions,tradeBatchId_vector], axis=0)
    #     combine_array = combine_array.T
    #     df1["tradeBatch"]=tradeBatchId_vector
    #     return df1
    # elif df_type == "metrics_df":
    #     df1=df.copy()
    #     longPositions = np.array(df1["long_positions"])
    #     AEPs = np.array(df1["long_average_price"])
    #     tradeBatchId_vector = np.full(len(longPositions), np.nan) # generate array of np.nan
    #     tradeBatchId = 1 # first batch will be 1
    #     for idx,(entryNum,AEP)  in enumerate(zip(longPositions,AEPs)):
    #         if entryNum > 0:
    #             tradeBatchId_vector[idx] = tradeBatchId
    #         elif entryNum > 0:
    #             tradeBatchId_vector[idx] = tradeBatchId
    #         elif entryNum == 0 and not np.isnan(AEP):
    #             tradeBatchId_vector[idx] = tradeBatchId
    #             tradeBatchId += 1 # increment to indicate next trade
    #         elif entryNum == 0 and np.isnan(AEP):
    #             tradeBatchId_vector[idx] = float(np.nan)
    #     combine_array = np.stack([longPositions,tradeBatchId_vector], axis=0)
    #     combine_array = combine_array.T
    #     df1["long_id"]=tradeBatchId_vector
        
    #     shortPositions = np.array(df1["short_positions"])
    #     AEPs = np.array(df1["short_average_price"])
    #     tradeBatchId_vector = np.full(len(shortPositions), np.nan) # generate array of np.nan
    #     tradeBatchId = 1 # first batch will be 1
    #     for idx,(entryNum,AEP)  in enumerate(zip(shortPositions,AEPs)):
    #         if entryNum > 0:
    #             tradeBatchId_vector[idx] = tradeBatchId
    #         elif entryNum > 0:
    #             tradeBatchId_vector[idx] = tradeBatchId
    #         elif entryNum == 0 and not np.isnan(AEP):
    #             tradeBatchId_vector[idx] = tradeBatchId
    #             tradeBatchId += 1 # increment to indicate next trade
    #         elif entryNum == 0 and np.isnan(AEP):
    #             tradeBatchId_vector[idx] = float(np.nan)
    #     combine_array = np.stack([shortPositions,tradeBatchId_vector], axis=0)
    #     combine_array = combine_array.T
    #     df1["short_id"]=tradeBatchId_vector
        
    df1=df.copy() 
    # Gotta be careful, here where longposition=0 will still have fee values, but shld be ok since na - fee = na
    df1["long_fee_cum"]=df1.groupby("long_id")["long_fee"].cumsum().fillna(method="ffill")
    # df1["long_fee_pct_cum"]=df1.groupby("long_id")["long_fee_pct"].cumsum().fillna(method="ffill")
    df1["long_fee_pct_cum"]=df1.groupby("long_id")["long_fee_pct"].fillna(method="ffill")
    
    df1["short_fee_cum"]=df1.groupby("short_id")["short_fee"].cumsum().fillna(method="ffill")
    # df1["short_fee_pct_cum"]=df1.groupby("short_id")["short_fee_pct"].cumsum().fillna(method="ffill")
    df1["short_fee_pct_cum"]=df1.groupby("short_id")["short_fee_pct"].fillna(method="ffill")
    
    df1["long_unrealised_pnl_w_fees"] = df1["long_unrealised_pnl"]-df1["long_fee_cum"]
    df1["long_unrealised_pnl_pct_w_fees"] = df1["long_unrealised_pnl_pct"]-df1["long_fee_pct_cum"]
    
    df1["short_unrealised_pnl_w_fees"] = df1["short_unrealised_pnl"]-df1["short_fee_cum"]
    df1["short_unrealised_pnl_pct_w_fees"] = df1["short_unrealised_pnl_pct"]-df1["short_fee_pct_cum"]
    
    
    df1.rename({"long_unrealised_pnl":"long_unrealised_pnl_wo_fees",
                "long_unrealised_pnl_pct":"long_unrealised_pnl_pct_wo_fees",
                "short_unrealised_pnl":"short_unrealised_pnl_wo_fees",
                "short_unrealised_pnl_pct":"short_unrealised_pnl_pct_wo_fees"},
               axis=1,
               inplace=True)
    
    for side in ["long", "short"]:
        df1[f"{side}_cum_entry"]=df1.groupby([f"{side}_id"])[f"{side}_entry_qty"].cumsum()
        df1[f"{side}_cum_entry"]=df1.groupby([f"{side}_id"])[f"{side}_cum_entry"].apply(lambda group: group.ffill())
        df1[f"{side}_cum_exit"]=df1.groupby([f"{side}_id"])[f"{side}_exit_qty"].cumsum()
        df1[f"{side}_cum_exit"]=df1.groupby([f"{side}_id"])[f"{side}_cum_exit"].apply(lambda group: group.ffill())
        df1[f"{side}_cum_exit"]=df1.groupby([f"{side}_id"])[f"{side}_cum_exit"].apply(lambda group: group.fillna(0))
        df1[f"{side}_allocation"]=df1[f"{side}_cum_entry"] - df1[f"{side}_cum_exit"]
        for fee in ["w_fees","wo_fees"]:
            df1[f"{side}_realised_pnl_pct_{fee}"] = df1[df1[f"{side}_positions"] == 0.0][f"{side}_unrealised_pnl_pct_{fee}"]
            df1[f"{side}_realised_pnl_{fee}"] = df1[df1[f"{side}_positions"] == 0.0][f"{side}_unrealised_pnl_{fee}"]


    return df1

def convert_timedelta(duration):
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    return f"{hours}:{minutes}:{seconds}"



#%%
def calc_position_metrics(Data_manager,Strategy_manager,Order_manager,config):
    """
    Collates 
        - realised_pnl
        - realised_pnl_pct
        
    from Data_manager, Strategy_manager and Order_manager to calculate
        - realised_pnl 
        - realised_pnl_pct
        - unrealised_pnl_pct
    for both sides and full strategy
    """
    #%%
    total_allocation = config['strategy']['allocation_total_notional']
    allocation_per_pair_pct = config['strategy']['allocation_per_pair_pct']
    allocation_per_pair_notional = total_allocation*(allocation_per_pair_pct/100)
    
    window = config["BACKTEST"]["window"]
    fee = config["BACKTEST"]["fee"]
    
    # =============================================================================
    #  Collect data from trade managers
    # =============================================================================
    
    trades_df = Strategy_manager.get_signals_df().copy()
    orders_df = Order_manager.get_orders_df().copy()
    trades_df.index = trades_df.index.round("1T")
    
    trades_df[['long_fee','short_fee']]=np.nan
    trades_df.update(orders_df[['long_fee','short_fee']])
    trades_df[['long_fee_pct','short_fee_pct']]=np.nan
    trades_df.update(orders_df[['long_fee_pct','short_fee_pct']])
    trades_df[['long_id','short_id']] = np.nan
    trades_df.update(orders_df[['long_id','short_id']])
    # trades_df[['long_fee','short_fee']]=orders_df[['long_fee','short_fee']]
    # trades_df[['long_fee_pct','short_fee_pct']] = orders_df[['long_fee_pct','short_fee_pct']]
    # trades_df[['long_id','short_id']] = orders_df[['long_id','short_id']]
    
    trades_df['long_id'].fillna(method="ffill",inplace=True)
    trades_df['short_id'].fillna(method="ffill",inplace=True)
    trades_df = tradeBatch(trades_df)
    
    
    # Populate signals and klines df with metrics (cum pct change, drawdown)
    klines_1m = Data_manager.klines_dict["1m"].copy()
    klines_1m = klines_1m.sort_index()[window[0]:window[1]]
    trades_df = trades_df.sort_index().loc[window[0]:window[1]]
    
    # =============================================================================
    # Calculate Metrics
    # =============================================================================
    for side in ["long","short"]:
        
        for wf in ["_wo_fees","_w_fees"]:

            # ============
            # Cumulative $
            # ============
                        
            # cumulative realised pnl            
            trades_df[f"{side}_cum_realised_pnl{wf}"] = trades_df[f"{side}_realised_pnl{wf}"].cumsum()
            trades_df[f"{side}_cum_realised_pnl{wf}"] = trades_df[f"{side}_cum_realised_pnl{wf}"].fillna(method="ffill")
            trades_df[f"{side}_cum_realised_pnl{wf}"].fillna(0,inplace=True)
            
            # cumulative (unrealised + realised) pnl
            trades_df[f"{side}_cum_pnl{wf}"]=trades_df[trades_df[f"{side}_positions"]>0][[f"{side}_cum_realised_pnl{wf}",f"{side}_unrealised_pnl{wf}"]].sum(axis=1)
            trades_df[f"{side}_cum_pnl{wf}"].fillna(trades_df[f"{side}_cum_realised_pnl{wf}"],inplace=True)
            trades_df[f"{side}_cum_pnl{wf}"].fillna(method="ffill",inplace=True)
            
            
            # ============
            # Cumulative %
            # ============
            
            # cumulative realised pnl %          
            trades_df[f"{side}_cum_realised_pnl_pct{wf}"] = trades_df[f"{side}_realised_pnl_pct{wf}"].cumsum()
            trades_df[f"{side}_cum_realised_pnl_pct{wf}"] = trades_df[f"{side}_cum_realised_pnl_pct{wf}"].fillna(method="ffill")
            trades_df[f"{side}_cum_realised_pnl_pct{wf}"].fillna(0,inplace=True)
            
            # cumulative (unrealised + realised) pnl %
            trades_df[f"{side}_cum_pnl_pct{wf}"]=trades_df[trades_df[f"{side}_positions"]>0][[f"{side}_cum_realised_pnl_pct{wf}",f"{side}_unrealised_pnl_pct{wf}"]].sum(axis=1)
            trades_df[f"{side}_cum_pnl_pct{wf}"].fillna(trades_df[f"{side}_cum_realised_pnl_pct{wf}"],inplace=True)
            trades_df[f"{side}_cum_pnl_pct{wf}"].fillna(method="ffill",inplace=True)
    
    # combined pair pnls
    for wf in ["_wo_fees","_w_fees"]:   
        trades_df[f"combined_cum_pnl{wf}"]=trades_df[[f"long_cum_pnl{wf}",f"short_cum_pnl{wf}"]].sum(axis=1)    
        trades_df[f"combined_cum_pnl_pct{wf}"]=trades_df[[f"long_cum_pnl_pct{wf}",f"short_cum_pnl_pct{wf}"]].sum(axis=1)  
        
        trades_df[f"combined_realised_pnl_pct{wf}"]=trades_df[[f"long_realised_pnl_pct{wf}",f"short_realised_pnl_pct{wf}"]].sum(axis=1).replace(0, np.nan)
        trades_df[f"combined_realised_pnl{wf}"]=trades_df[[f"long_realised_pnl{wf}",f"short_realised_pnl{wf}"]].sum(axis=1).replace(0, np.nan)
        
    # Merge klines to signals
    metrics_instrument_df = pd.merge(klines_1m, trades_df, right_index=True, left_index= True, how="left")
    
    # Calculate equity related pnls
    for wf in ["_wo_fees","_w_fees"]:
        metrics_instrument_df[[f"long_cum_pnl{wf}",f"short_cum_pnl{wf}",f"combined_cum_pnl{wf}"]] = metrics_instrument_df[[f"long_cum_pnl{wf}",f"short_cum_pnl{wf}",f"combined_cum_pnl{wf}"]].fillna(method="ffill")
        metrics_instrument_df[f"equity{wf}"]=metrics_instrument_df[f"combined_cum_pnl{wf}"]+allocation_per_pair_notional
        metrics_instrument_df[f"drawdown{wf}"]=calc_equity_drawdown(metrics_instrument_df[f"equity{wf}"])["drawdown"]
        metrics_instrument_df[f"equity_pct_change{wf}"] = (metrics_instrument_df[f"equity{wf}"].pct_change()+1).cumprod()
        metrics_instrument_df["buyhold_pct_change"] = ((metrics_instrument_df["close"].pct_change())+1).cumprod()

    #%%
    return metrics_instrument_df
    
#%%
def calc_portfolio_metrics(results,config):
    """
    DEV
    ----
        This takes roughly 7 minutes to run for 5 pairs.
        Optimisation required
    Parameters
    ----------
    results : ordered dict
        This is a nested dict of instrument.
        asdasda.
    config : TYPE
        DESCRIPTION.
    
    Returns
    -------
    metrics_portfolio_df : TYPE
        DESCRIPTION.

    """
    #%%
    from functools import reduce
    import numpy as np
    import pandas as pd
    import time 
    
    allocation_total_notional = config["strategy"]["allocation_total_notional"]
    allocation_per_pair_pct = config["strategy"]["allocation_per_pair_pct"]
       
    
    # Concatenating results 
    
    if "TOTAL" in results.keys():
        del results["TOTAL"]
    t0=time.time()
    temp = []
    number_of_pairs = len(results)
    for side in ["long","short"]:
        for metric in ["positions","allocation","entry_qty","exit_qty","fee", "fee_pct"]:
            if metric == "_id":
                df_list = [result["metrics_instrument_df"][[f"{side}{metric}"]] for result in results.values()]
                total_metric_df = pd.concat(df_list,axis=1).sum(axis=1)
                total_metric_df.name = f"{side}{metric}"
            else:
                df_list = [result["metrics_instrument_df"][[f"{side}_{metric}"]] for result in results.values()]
                total_metric_df = reduce(lambda x, y: x.add(y, fill_value=0.0), df_list) #pd.concat(df_list,axis=1).sum(axis=1)
                total_metric_df.name = f"{side}_{metric}" 
            temp.append(total_metric_df)

    buyhold_list = [result["metrics_instrument_df"][[f"buyhold_pct_change"]] for result in results.values()]
    # buyhold_agg = (pd.concat(buyhold_list,axis=1).sum(axis=1)/number_of_pairs).replace(0,np.nan)
    buyhold_agg = reduce(lambda x,y: x.add(y,fill_value=np.nan), buyhold_list).sum(axis=1)/number_of_pairs
    buyhold_agg.name = "buyhold_pct_change"
    
    temp.append(buyhold_agg)
    
    temp_df = pd.concat(temp,axis=1)
    # temp_df = reduce(lambda x,y: x.add(y), temp)
    t1=time.time()
    for wf in ["_wo_fees","_w_fees"]:  
        metrics_to_aggregate = ["short_realised_pnl_pct","long_realised_pnl_pct","combined_realised_pnl_pct",
                                "short_realised_pnl","long_realised_pnl","combined_realised_pnl",
                                "long_cum_pnl","short_cum_pnl","combined_cum_pnl",
                                "long_cum_pnl_pct","short_cum_pnl_pct","combined_cum_pnl_pct"]
        
        for metric in metrics_to_aggregate:
            df_list = [result["metrics_instrument_df"][[f"{metric}{wf}"]] for result in results.values()]
            # if "realised" in metric:
            #     # total_metric_df = pd.concat(df_list,axis=1).sum(axis=1).replace(0, np.nan)
            #     total_metric_df = reduce(lambda x,y: x.add(y,fill_value=np.nan), df_list)
            # else:
            total_metric_df = reduce(lambda x,y: x.add(y, fill_value=0.0), df_list)
            total_metric_df.name = f"{metric}{wf}"
            temp.append(total_metric_df)
    
    
    metrics_df = pd.concat(temp,axis=1)
    # print(f"time taken: {time.time() - t1}")

    for wf in ["_wo_fees","_w_fees"]:  
            # print("Portfolio equity ..")
        metrics_df[f"equity{wf}"]=metrics_df[f"combined_cum_pnl{wf}"]+allocation_total_notional
        
        # print("Portfolio drawdown ..")
        metrics_df[f"drawdown{wf}"] = calc_equity_drawdown(metrics_df[f"equity{wf}"])["drawdown"]
        
        # print("Portfolio equity pct change ..")
        metrics_df[f"equity_pct_change{wf}"]  = (metrics_df[f"equity{wf}"].pct_change()+1).cumprod()
             

    # metrics_portfolio_df = metrics_df.filter(regex="TOTAL")
    # metrics_portfolio_df.columns = metrics_portfolio_df.columns.str.lstrip("TOTAL_")
    # print(f" ----- Time taken for getting TOTAL metrics: {time.time() -t0}")
    #%%
    temp = {}
    temp["metrics_instrument_df"] = metrics_df
    
    
    results["TOTAL"] = temp
    
    
#%%    
def calc_standard_metrics(results,config):    
    """

    Parameters
    ----------
    results : TYPE
        DESCRIPTION.
    config : TYPE
        This is not used in this function

    Returns
    -------
    dict
        DESCRIPTION.

    """
    allocation_total_notional = config["strategy"]["allocation_total_notional"]
    allocation_per_pair_pct = config["strategy"]["allocation_per_pair_pct"]
    allocation_per_pair = allocation_per_pair_pct/100*allocation_total_notional
    metrics_dict = {}
    for wf in ["_w_fees","_wo_fees"]:
        returns_metrics_per_w_wo_fees={}
        
        for pair,result in results.items():
            # df = full_results["results"][pair]["metrics_instrument_df"].copy()
            df = result["metrics_instrument_df"]
            returns_metrics_per_side = {}
            
            for side in ["long","short","combined"]:
                
                # ===================================
                # Returns related metrics
                # ===================================
                Total_return_usdt = df[f"{side}_cum_pnl{wf}"].iloc[-1]
                if pair == "TOTAL":
                    Total_return_pct = Total_return_usdt/allocation_total_notional*100
                    allocation = allocation_total_notional
                    equity = (df[f"{side}_cum_pnl{wf}"]+allocation)
                    returns = equity.pct_change()
                    dd,mdd = calc_equity_drawdown(equity).values()
                else:
                    Total_return_pct = Total_return_usdt/allocation_per_pair*100
                    allocation = allocation_per_pair
                    equity = (df[f"{side}_cum_pnl{wf}"]+allocation)
                    returns=equity.pct_change()
                    dd,mdd = calc_equity_drawdown(equity).values()
                benchmark_return_pct = (df.filter(regex="buyhold_pct_change").iloc[-1,0]-1)*100
                
                # ===================================
                # gains/pains
                # ===================================
                realised_pnl = df[f"{side}_realised_pnl{wf}"]
                gains = float(realised_pnl[realised_pnl>0].sum())
                pains = float(realised_pnl[realised_pnl<0].sum())
                try:
                    gains_to_pain_ratio = np.round(gains/abs(pains),4)
                except Exception as e:
                    gains_to_pain_ratio = "-"
                    
                # ===================================
                # Trade sizes
                # ===================================
                entry_qtys_df = df.filter(regex="entry_qty")
                exit_qtys_df = df.filter(regex="exit_qty")
                    
                if side in ["long","short"]:
                    min_entry_size = entry_qtys_df[entry_qtys_df>0].min()[f"{side}_entry_qty"]
                    max_entry_size = entry_qtys_df[entry_qtys_df>0].max()[f"{side}_entry_qty"]
                    min_exit_size = exit_qtys_df[exit_qtys_df>0].min()[f"{side}_exit_qty"]
                    max_exit_size = exit_qtys_df[exit_qtys_df>0].max()[f"{side}_exit_qty"]
                    
                elif side == "combined":
                    min_entry_size = entry_qtys_df[entry_qtys_df>0].min().min()
                    max_entry_size = entry_qtys_df[entry_qtys_df>0].max().max()
                    min_exit_size = exit_qtys_df[exit_qtys_df>0].min().min()
                    max_exit_size = exit_qtys_df[exit_qtys_df>0].max().max()
                    
                    
                # ===================================
                # FEES TABULATION
                # ===================================
                if wf == "_w_fees":    
                    if side in ["long", "short"]:
                        fees = df.filter(regex=f"{side}_fee$").sum().sum()
                        
                    elif side == "combined":
                        fees = df.filter(regex=f"fee$").sum().sum()
                elif wf == "_wo_fees":
                    fees = 0
                    
                # ===================================
                # Win rate/ Sharpe/Sortino/Calmar
                # ===================================
                number_winning_trades = realised_pnl[realised_pnl>0].count()
                number_of_losing_trades = realised_pnl[realised_pnl<0].count()
                numberOfTrades = int(number_winning_trades+number_of_losing_trades)
                try:
                    win_rate = float(number_winning_trades/(number_winning_trades+number_of_losing_trades))
                except:
                    win_rate = "-"
                    
                    
                N = numberOfTrades * 365 / (df.index[-1] - df.index[0]).days * 24*60
                
                try:
                    sharpe_ratio = np.round(sharpe(returns,N),4)
                except:
                    sharpe_ratio = "-"         
                        
                try:
                    sortino_ratio = np.round(sortino(returns,N),5)
                except:
                    sortino_ratio = "-"
                
                try:
                    calmar_ratio = np.round(calmar(returns,N,mdd),5)
                except:
                    calmar_ratio = "-"
                # ===================================
                # Metrics table 
                # ===================================
                returns_metrics_per_side[side]={"Starting Allocation (USDT)":np.round(allocation,4),
                                                "Total Return (USDT)": np.round(Total_return_usdt,4),
                                                "Total Fees Paid (USDT)":np.round(fees,4),
                                                "Total Return [%]": np.round(Total_return_pct,2), 
                                                "Benchmark Return [%]":np.round(benchmark_return_pct,2),
                                                "sharpe":sharpe_ratio,
                                                "sortino": sortino_ratio,
                                                "calmar": calmar_ratio,
                                                "Win Rate [%]":np.round(win_rate*100,2),
                                                "gains (USDT)": np.round(gains,2), 
                                                "pains (USDT)": np.round(pains,2),
                                                "gains / pain": gains_to_pain_ratio,
                                                "total trades": np.round(numberOfTrades,4),
                                                "maxdrawdown (%)":mdd,
                                                "min entry size": min_entry_size,
                                                "max entry size": max_entry_size,
                                                "min exit size": min_exit_size,
                                                "max exit size": max_exit_size}
                
            returns_metrics_per_w_wo_fees[pair]=returns_metrics_per_side
 
        metrics_dict[wf]=returns_metrics_per_w_wo_fees
        
    return metrics_dict

#%%
def sharpe(returns,N):
    # >1 is good
    return returns.mean()*N/returns.std()/ np.sqrt(N)

def sortino(returns,N):
    # 0 - 1.0 is suboptimal, 1> good, 3> very good
    std_neg = returns[returns<0].std()*np.sqrt(N)
    return returns.mean()*N/std_neg

def calmar(returns,N,mdd):
    # > 0.5 is good, 3.0 to 5.0 is very good
    return returns.mean()*N/abs(mdd/100)
#%%
def trade_analysis(metrics_df,pair="BTCUSDT",side="long",fee_toggle="w_fees"):
    """

    Parameters
    ----------
    metrics_df : TYPE
        DESCRIPTION.
    pair : TYPE, optional
        DESCRIPTION. The default is "BTCUSDT".
    side : TYPE, optional
        DESCRIPTION. The default is "long".
    fee_toggle : TYPE, optional
        DESCRIPTION. The default is "w_fees".

    Returns
    -------
    trades : TYPE
        DESCRIPTION.

    """
    # TODO: call results instead of metrics df. unless metrics_instrument_df
    wf=fee_toggle
    trades=metrics_df[[f"{pair}_{side}_id",f"{pair}_{side}_positions",f"{pair}_{side}_unrealised_pnl_pct_{wf}",f"{pair}_{side}_realised_pnl_pct_{wf}"]].copy()
    trades=trades[trades[f"{pair}_{side}_id"]>=0]
    trades.loc[trades[f"{pair}_{side}_positions"]==0,f"{pair}_{side}_positions"]= np.nan
    trades[f"{pair}_{side}_positions"] = trades[f"{pair}_{side}_positions"].fillna(method="ffill")
    
    tradeDuration = trades.groupby(f"{pair}_{side}_id")[f"{pair}_{side}_unrealised_pnl_pct_{wf}"].count()
    
    winning_trades_ID = trades[trades[f"{pair}_{side}_realised_pnl_pct_{wf}"]>0][f"{pair}_{side}_id"].unique()
    winning_trades = trades.loc[trades[f"{pair}_{side}_id"].isin(winning_trades_ID)]
    winning_tradesDurations = winning_trades.groupby(f"{pair}_{side}_id")[f"{pair}_{side}_unrealised_pnl_pct_{wf}"].count()
    winning_maxdrawdownsTrades= winning_trades.groupby(f"{pair}_{side}_id")[f"{pair}_{side}_unrealised_pnl_pct_{wf}"].min()*100**2
    
    losing_trades_ID = trades[trades[f"{pair}_{side}_realised_pnl_pct_{wf}"]<0][f"{pair}_{side}_id"].unique()
    losing_trades = trades.loc[trades[f"{pair}_{side}_id"].isin(losing_trades_ID)]
    losing_tradesDurations = losing_trades.groupby(f"{pair}_{side}_id")[f"{pair}_{side}_unrealised_pnl_pct_{wf}"].count()
    losing_maxdrawdownsTrades= losing_trades.groupby(f"{pair}_{side}_id")[f"{pair}_{side}_unrealised_pnl_pct_{wf}"].min()*100**2
    
    trades = trades.dropna(subset=[f"{pair}_{side}_realised_pnl_pct_{wf}"])
    trades = trades.merge(winning_tradesDurations.rename("winning_trade_duration"),left_on=f"{pair}_{side}_id",right_index = True,how="outer")
    trades = trades.merge(winning_maxdrawdownsTrades.rename("winning_maxdrawdowns"),left_on=f"{pair}_{side}_id",right_index = True,how="outer")
    trades = trades.merge(losing_tradesDurations.rename("losing_trade_duration"),left_on=f"{pair}_{side}_id",right_index = True,how="outer")
    trades = trades.merge(losing_maxdrawdownsTrades.rename("losing_maxdrawdowns"),left_on=f"{pair}_{side}_id",right_index = True,how="outer")
    trades[f"{pair}_{side}_realised_pnl_pct_{wf}"] = trades[f"{pair}_{side}_realised_pnl_pct_{wf}"]*100_00
    trades.drop(columns=f"{pair}_{side}_unrealised_pnl_pct_{wf}",inplace=True)
    return trades

if __name__ == "__main__":
    #%%
    import numpy as np
    allocation_total_notional = config["strategy"]["allocation_total_notional"]
    allocation_per_pair_pct = config["strategy"]["allocation_per_pair_pct"]
    allocation_per_pair = allocation_per_pair_pct/100*allocation_total_notional
    df = full_results["results"]["BTCUSDT"]["metrics_instrument_df"].copy()
    for side in ["long"]:
        for wf in ["_w_fees", "_wo_fees"]:
            print(f"{'-'*20}\n{side} -> {wf}")
            Total_return_usdt = df[f"{side}_cum_pnl{wf}"].iloc[-1]
            Total_return_pct = Total_return_usdt/allocation_per_pair*100
            allocation = allocation_per_pair
            equity = (df[f"{side}_cum_pnl{wf}"]+allocation)
            returns=equity.pct_change()
            # returns.hist(bins=50)
            cum_pnl = df[f"{side}_cum_pnl{wf}"]
            print(f"sum equity: {equity.sum()}")
            print(f"0 returns: {returns[returns==0.0].count()}")
            print(f"sum returns: {returns.sum()}")
            
            
            
            realised_pnl = df[f"{side}_realised_pnl{wf}"]
            number_winning_trades = realised_pnl[realised_pnl>0].count()
            number_of_losing_trades = realised_pnl[realised_pnl<0].count()
            numberOfTrades = int(number_winning_trades+number_of_losing_trades)
            N = numberOfTrades * 365 / (df.index[-1] - df.index[0]).days * 24*60
            # print(f"no. trades: {N}")
            
            realised_pnl.hist(bins=100)
            
            sharpe_ratio = np.round(sharpe(returns,N),4)
            print(f"sharpe: {sharpe_ratio}")
            
    ''' CONCLUSION: cum_pnl with fees > cum_pnl without fees!! -><-'''
    
    check = df[["long_cum_pnl_w_fees","long_cum_pnl_wo_fees"]]
    check1 = check["2020-03-06":"2020-03-12"]
