import numpy as np
import pandas as pd
import numba as nb


@nb.njit(cache=True)
def rolling_sum(heights, w=4):
    ret = np.cumsum(heights)
    ret[w:] = ret[w:] - ret[:-w]
    return ret[w - 1:]


@nb.njit(cache=True)
def calc_exponential_height(heights, w):  ## CHECK!!
    # heights = OHLCVT_array[:,1]-OHLCVT_array[:,2]
    rolling_sum_H_L = rolling_sum(heights, w)
    # rolling_sum_H_L = np.full(len(heights),np.nan)
    # for idx in range(window-1,len(heights)):
    #     # print(f"summing {start_idx} to {idx}")
    #     rolling_sum_H_L[idx] = np.sum(heights[idx-window+1:idx])
    # mpbw=(heights.rolling(window=w).sum())
    exp_height = (rolling_sum_H_L[-1] - heights[-w] + heights[-1]) / w
    return exp_height  # (mpbw.iloc[-1]-heights[-w]+heights[-1])/w


@nb.njit(cache=True)
def calc_tide(open_i: np.array,
              high_i: np.array,
              low_i: np.array,
              close_i: np.array,
              previous_tide: float,
              previous_ebb: float,
              previous_flow: float,
              windows: np.array,
              thresholds: int,
              sensitivity: float) -> np.array:
    # Checks first is OHLCVT_array contains enough data to start calculating tides
    # if max_lookback>len(OHLCV):
    #     # print("not enough data")
    #     tide.append(np.nan)
    #     ebb.append(np.nan)
    #     flow.append(np.nan)
    #     closeTime.append(OHLCVT_array[-1,-1])
    #     return {'tide':np.nan,'ebb':np.nan,'flow':np.nan}

    new_open = open_i[-1]
    new_high = high_i[-1]
    new_low = low_i[-1]

    if np.isnan(previous_tide):
        previous_tide = 1
        previous_ebb = 1

    # ====================================================================
    # START SIGNAL CALCULATION
    # ====================================================================
    # undertow = [[1 if new_open > previous_ebb else 0] if previous_tide == 0 else [1 if new_open < previous_ebb else 0]][0][0]
    " undertow "
    if previous_tide:
        if new_open < previous_ebb:
            undertow = 1
        else:
            undertow = 0
    else:
        if new_open > previous_ebb:
            undertow = 1
        else:
            undertow = 0

    # surftow = [[1 if new_high > previous_ebb else 0] if previous_tide == 0 else [1 if new_low < previous_ebb else 0]][0][0]
    " surftow "
    if previous_tide:
        if new_low < previous_ebb:
            surftow = 1
        else:
            surftow = 0
    else:
        if new_high > previous_ebb:
            surftow = 1
        else:
            surftow = 0

    " Calculate change in tide: flow"

    # heights = df["high_i"][-67:]-df["low_i"][-67:]
    heights = high_i - low_i
    heights = heights[-max(windows):]

    w_0 = 0
    for w in windows:
        w_i = calc_exponential_height(heights, w)
        if w_i > w_0:
            max_exp_height = w_i
            w_0 = w_i
    # max_exp_height=max([calc_exponential_height(heights,w) for w in windows])#calc_exponential_height(prices,lengths[0]),calc_exponential_height(prices,lengths[1]),calc_exponential_height(prices,lengths[2]))    #THIS CAN BE CHANGED TO separate rolling functions#

    # sensitivity=sensitivity/100
    max_exp_height_ranges = list(np.quantile(heights, np.linspace(0, 1, thresholds)))
    max_exp_height_ranges = [0] + max_exp_height_ranges + [np.inf]
    additives_range = np.linspace(0, np.quantile(heights, sensitivity / 100), len(max_exp_height_ranges) + 1)
    max_exp_height_ranges = list(zip(max_exp_height_ranges[0:-1], max_exp_height_ranges[1:]))

    i = 0
    for maxmpb_range_i in max_exp_height_ranges:
        if maxmpb_range_i[0] <= max_exp_height <= maxmpb_range_i[1]:
            additive = additives_range[i]
            break
        else:
            i += 1

    " flow "
    # flow = [previous_ebb+additive if previous_tide == 1 else previous_ebb-additive][0]
    if previous_tide:
        flow = previous_ebb + additive
    else:
        flow = previous_ebb - additive

    " interim tides "
    # tide_1 = [1 if new_open >= flow else 0][0]
    if new_open >= flow:
        tide_1 = 1
    else:
        tide_1 = 0

    # tide_2 = [[1 if new_low < previous_ebb else 0] if tide_1 == 1 else [1 if new_high > previous_ebb else 0]][0][0]
    if tide_1:
        if new_low < previous_ebb:
            tide_2 = 1
        else:
            tide_2 = 0
    else:
        if new_high > previous_ebb:
            tide_2 = 1
        else:
            tide_2 = 0

    # tide_3 =[[1 if surftow == 1 else 0] if tide_2 ==1 else [[ 0 if undertow == 1 else 1] if surftow == 0 else 0]][0][0]
    if tide_2:
        if surftow:
            tide_3 = 1
        else:
            tide_3 = 0
    else:
        if surftow:
            if undertow:
                tide_3 = 0
            else:
                tide_3 = 1
        else:
            tide_3 = 0

    # tide_4 = [[1 if new_low>=flow else 0] if tide_1 == 1 else [1 if new_high > flow else 0]][0][0]
    if tide_1:
        if new_low >= flow:
            tide_4 = 1
        else:
            tide_4 = 0
    else:
        if new_high > flow:
            tide_4 = 1
        else:
            tide_4 = 0

    " tide formulation "
    if tide_1 == 1 and tide_4 == 1:
        new_tide = 1
    elif tide_1 == 0 and tide_4 == 0:
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

    return new_tide, new_ebb, flow


@nb.njit(cache=True)
def nb_tide(open_: np.array,
            high: np.array,
            low: np.array,
            close: np.array,
            windows: np.array,
            thresholds: int,
            sensitivity: float,
            fixed_window: bool):  # , tide: np.array, ebb: np.array, flow:np.array):
    # ,sensitivity=sensitivity, thresholds=10,ticksize=0.01,window_scale=1,lookback_windows=[5,20,67]):
    # indicators = [[tide_0,ebb_0,flow_0],
    #               [tide_0,ebb_0,flow_0],
    #               ...
    #               ...
    #               [tide_n,ebb_n,flow_n]]

    n = len(close)
    max_lookback = np.max(windows)

    tide = np.full(n, np.nan)
    ebb = np.full(n, np.nan)
    flow = np.full(n, np.nan)
    tide_n = np.full(n, np.nan)
    
    previous_tide = np.nan
    previous_ebb = np.nan
    previous_flow = np.nan
    counter = 0
    for i in range(max_lookback, n + 1):
        # i_s = [i for i in range(max_lookback,n)]
        if fixed_window:
            open_i = open_[i - max_lookback:i]
            high_i = high[i - max_lookback:i]
            low_i = low[i - max_lookback:i]
            close_i = close[i - max_lookback:i]
        else:
            # expanding window
            open_i = open_[:i]
            high_i = high[:i]
            low_i = low[:i]
            close_i = close[:i]
        tide_i, ebb_i, flow_i = calc_tide(open_i=open_i,
                                          high_i=high_i,
                                          low_i=low_i,
                                          close_i=close_i,
                                          previous_tide=previous_tide,
                                          previous_ebb=previous_ebb,
                                          previous_flow=previous_flow,
                                          windows=windows,
                                          thresholds=thresholds,
                                          sensitivity=sensitivity)
        
        tide_unused, ebb_unused, flow_i = calc_tide(open_i=open_i,
                                                      high_i=high_i,
                                                      low_i=low_i,
                                                      close_i=close_i,
                                                      previous_tide=tide_i,
                                                      previous_ebb=ebb_i,
                                                      previous_flow=flow_i,
                                                      windows=windows,
                                                      thresholds=thresholds,
                                                      sensitivity=sensitivity)
        
        if previous_tide != tide_i:
            counter+=1
            
        previous_tide = tide_i
        previous_ebb = ebb_i
        previous_flow = flow_i
        
        tide_n[i-1] = counter
        tide[i - 1] = tide_i
        ebb[i - 1] = ebb_i
        flow[i - 1] = flow_i
        
    return tide, ebb, flow, tide_n


def calc_tides(df, sensitivity:int=50, thresholds:int=10, windows:np.ndarray=np.array([5, 20, 67])):
    assert type(sensitivity) == int
    assert type(thresholds) == int
    assert type(windows) == np.ndarray
    
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    tides, ebb, flow,tide_n  = nb_tide(open_=open_,
                               high=high,
                               low=low,
                               close=close,
                               windows=windows,
                               thresholds=thresholds,
                               sensitivity=sensitivity,
                               fixed_window=True)
    
    df["tide"] = tides
    df["tide"]=df["tide"].replace(0,-1)
    df["tide_id"] = tide_n
    df["ebb"] = ebb
    df["flow"] = flow
    # Other metrics needed from id
    # len of tide
    # str of tide (close[-1] - close[0])
    
    return df
    
def calc_slopes(df0,
                slope_lengths:list=[7,10,14,20,28,40,56,80],
                scaling_factor:float = 1.0,
                lookback:int = 500,
                upper_quantile = 0.9,
                logRet_norm_window = 10,
                suffix=""):
    hour_List = np.array(slope_lengths) * scaling_factor
    min_List = [int(hour) for hour in hour_List]
    close = list(df0.filter(regex="close$").columns)[0]
    df = df0.copy()
    df_temp = pd.DataFrame()
    df_temp['logRet'] = np.log(1+df[close].pct_change())
    df_temp['logRet_norm'] = df_temp['logRet'] / df_temp['logRet'].rolling(logRet_norm_window).std()
    
    # we need to fillna(0) because some pockets have zero mobvement in price => o/0 = nan
    df_temp['logRet_norm'] = df_temp['logRet_norm'].fillna(0)
    df_temp['logLevel_norm'] = df_temp['logRet_norm'].rolling(lookback).sum()

    df_temp['logRet_norm'].isna().sum()
    df_temp['logLevel_norm'].isna().sum()

    slopeNames = []
    
    for minutes in min_List:
        slope_name = 'slope_' + str(minutes)
        slopeNames.append(slope_name)
        df_temp[slope_name] = (df_temp['logLevel_norm'] - df_temp['logLevel_norm'].shift(periods=minutes)) / minutes
        
    if suffix != "":
        suffix = "_"+suffix
    df[f'slope_avg{suffix}'] = df_temp[slopeNames].mean(axis=1, skipna=False)
    lower_quantile = round(1.0 - upper_quantile, 3)
    

    df[f'slope_u{suffix}'] = df[f'slope_avg{suffix}'].rolling(lookback).quantile(upper_quantile)
    df[f'slope_l{suffix}'] = df[f'slope_avg{suffix}'].rolling(lookback).quantile(lower_quantile)
    return df
    
@nb.njit(cache=True)
def continuous_resampling(closetime, open_, high, low, close, vol, x):
    x=x-1
    n = len(close)
    r_closetime = np.full(n,np.nan)
    r_open = np.full(n, np.nan)
    r_high = np.full(n, np.nan)
    r_low = np.full(n, np.nan)
    r_close = np.full(n, np.nan)
    r_vol = np.full(n, np.nan)

    for i in range(x,n):
        r_closetime[i] = closetime[i]
        r_open[i]=open_[i-x]
        r_high[i]=np.max(high[i-x:i])
        r_low[i] = np.min(low[i-x:i])
        r_close[i] = close[i]
        r_vol[i] = np.sum(vol[i-x:i])
    
    return r_closetime,r_open,r_high,r_low,r_close,r_vol

def calc_continuous_resample(df,window):
    closeTime = df["closeTime"].to_numpy()
    open_ = df["open"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    vol = df["volume"].to_numpy()
    
    r_closetime,r_open,r_high,r_low,r_close,r_vol = continuous_resampling(closeTime,open_, high, low, close, vol, window)
    
    
    test= pd.DataFrame({"open":r_open,
                        "high":r_high,
                        "low":r_low,
                        "close":r_close,
                        "volume":r_vol,
                        "closeTime": r_closetime})
    # test['date_time'] = pd.to_datetime(test['closeTime'], unit='s').round("1T")
    test.index = df.index

    return test

import math 

@nb.njit(cache=True)
def get_dollar_bars(time_bars, dollar_threshold): #function credit to Max Bodoia
    # initialize an empty list of dollar bars
    dollar_bars = list()

    # initialize the running dollar volume at zero
    running_volume = 0

    # initialize the running high and low with placeholder values
    running_high, running_low = 0, math.inf

    # for each time bar...
    for i in range(len(time_bars)):
        # get the timestamp, open, high, low, close, and volume of the next bar
        next_close, next_high, next_low, next_open, next_timestamp, next_volume = [time_bars[i][k] for k in ['close', 'high', 'low', 'open', 'closeTime', 'volume']]
        # get the midpoint price of the next bar (the average of the open and the close)
        midpoint_price = ((next_open) + (next_close))/2

        # get the approximate dollar volume of the bar using the volume and the midpoint price
        dollar_volume = next_volume * midpoint_price

        # update the running high and low
        running_high, running_low = max(running_high, next_high), min(running_low, next_low)

        # if the next bar's dollar volume would take us over the threshold...
        if dollar_volume + running_volume >= dollar_threshold:

            # set the timestamp for the dollar bar as the timestamp at which the bar closed (i.e. one minute after the timestamp of the last minutely bar included in the dollar bar)
            bar_timestamp = next_timestamp + 60*60
            
            # add a new dollar bar to the list of dollar bars with the timestamp, running high/low, and next close
            dollar_bars += [{'timestamp': bar_timestamp, 'open': next_open, 'high': running_high, 'low': running_low, 'close': next_close}]

            # reset the running volume to zero
            running_volume = 0

            # reset the running high and low to placeholder values
            running_high, running_low = 0, math.inf

        # otherwise, increment the running volume
        else:
            running_volume += dollar_volume

    # return the list of dollar bars
    return dollar_bars

def calc_dollar_bars(df, dollar_threshold):
    time_bars = df.to_dict('records') 
    dollar_bars = get_dollar_bars(time_bars,dollar_threshold)
    dollar_bars = pd.DataFrame(dollar_bars)
    dollar_bars['date_time'] = pd.to_datetime(dollar_bars['timestamp'], unit='s').round("1T")
    dollar_bars.set_index(keys=['date_time'], inplace=True, drop=False)
    return dollar_bars

def calc_tide_metrics(klines_indicators_dict):
    #%%
    test = {}
    for instrument, df in klines_indicators_dict.items():
        df1 = df.copy()
        tide_labels = list(df1.filter(regex="tide$").columns) 
        mx_labels = list(df1.filter(regex="ebb$").columns)
        cols_to_get = ["1h_open","1h_high","1h_low","1h_close"] + tide_labels + mx_labels
        df1 = df1[cols_to_get].tail(2).copy()
        
        
        
        
        
        # RENAME AND TIDY UP
        for mx_label in mx_labels:
            tf,relabel = mx_label.split("_")
            df1.rename(columns={mx_label:f"{tf}_mx"},inplace=True)
        
        df1.reset_index(inplace=True)
        df1["instrument"] = instrument
        
        df1.set_index("instrument")
        df1 = df1.reset_index(drop=True)
        test[instrument] = df1
        

    final_df = pd.concat(test, axis=0)
    final_df.reset_index(drop=True, inplace=True)
    
    # final_df = final_df.set_index(keys="instrument")
    # final_df.sort_index(inplace=True)
    # multi_index = [(ins,dt) for ins,dt in final_df[["instrument", "date_time"]].to_dict().items()]
    # final_df.set_index(["instrument", "date_time"],inplace=True)
    final_df = final_df.round(2)
    tides = list(final_df.filter(regex="tide").columns)
    final_df[tides] = final_df.filter(regex="tide").astype(int)
    final_df.set_index(keys=["instrument", "date_time"],inplace=True)
    final_df.sort_index(inplace=True)
    
    s = final_df.style
    for idx, group_df in final_df.groupby('instrument'):
        s.set_table_styles({group_df.index[0]: [{'selector': '', 'props': 'border-top: 3px solid black;'}]}, 
                           overwrite=False, axis=1)

    mx_labels = list(final_df.filter(regex="mx$").columns)
    final_df1 = s.apply(tide_colors, axis=0, subset=list(final_df.filter(regex="tide").columns))
    # final_df1 = final_df.groupby("instrument").rank().style.background_gradient(subset=["1h_open","1h_high","1h_low","1h_close"]+mx_labels)
#%%
    return final_df1
    
#%%
def tide_colors(series):

    g = 'background-color: green;'
    r = 'background-color: orange;'
    w = ''

    return [r if e < 0 else g if e >0 else w for e in series]  


if __name__ == "__main__":
    
    print("calculating dollar bars")
    dollar_bars = get_dollar_bars(df,dollar_threshold=10000)
