import numpy as np
import pandas as pd
import numba as nb
# from strategy.resampler import calc_ohlc_from_series

@nb.njit(cache=True)
def nb_ewma(close_px: np.array, window: int, alpha=None) -> np.array:
    """
    default ewma uses a=2/(n+1)

    RSI function uses Wilder's MA which req a=1/n   <<--- see nb_rsi function

    INPUTS:
            close_px: np.array(float)
            window: int
            alpha: float / None

        RETURNS:
            ema: np.array(float)
    """
    n = len(close_px)
    ewma = np.full(n, np.nan)

    if not alpha:
        alpha = 2.0 / float(window + 1)  # default alpha is 2/(n+1)
    w = 1.0

    ewma_old = close_px[0]
    if np.isnan(ewma_old):  # to account for possibility that arr_in[0] may be np.nan
        ewma_old = 0.0

    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1.0 - alpha) ** i
        ewma_old = ewma_old * (1 - alpha) + close_px[i]
        ewma[i] = ewma_old / w
    return ewma

@nb.njit(cache=True)
def nb_wma(close_px: np.array, window: int) -> np.array:
    """ Weighted moving average

        INPUTS:
            close_px: np.array(float)
            window: int

        RETURNS:
            wma: np.array(float)
    """
    n = len(close_px)
    wma = np.full(n, np.nan)
    weights = np.arange(window ) +1   # linear increasing weights [1,2,3,...,13,14]
    weights_sum = np.sum(weights)

    for idx in range(window, n):
        # explicitly declare the window. No other reference to closePrices_np
        price_window = close_px[idx - window + 1:idx + 1]  # up to but not including

        wma[idx] = np.sum(weights * price_window) / weights_sum

    return wma

@nb.njit(cache=True)
def nb_hma(close_px: np.array, window: int) -> np.array:
    """ computes the Hull Moving Average

        Uses the nb_wma function (compiled in Numba) to compute the Weighted Moving Avg .
        Roughly 100x speed-up vs pandas_ta

        INPUTS:
            close_px: np.array(float)
            window: int

        RETURNS:
            hma: np.array(float)
    """
    wma_half = nb_wma(close_px, int(window / 2))
    wma_full = nb_wma(close_px, int(window))

    # vector operation
    hma_input = ( 2 *wma_half) - wma_full
    hma = nb_wma(hma_input, window=int(np.sqrt(window)))

    return hma

def calc_emas(df,price="close", window=89, label=14):    
    # Ensure no nans
    df.dropna(inplace=True)
    price = df[price].to_numpy()


    ema = nb_ewma(price, window=window)
    
    df[f"EMA_{label}"]= ema
    
    return df

@nb.njit(cache=True)
def nb_rsi(close_px: np.array, window: int) -> np.array:
    """ This method has dependency on another Numba function: nb_ewma

    INPUTS:
            close_px: np.array(float)
            window: int
            alpha: float / None

    RETURNS:
            rsi: np.array(float)

    """
    n = len(close_px)
    close_diff = np.full(n, np.nan)
    close_diff[1:] = close_px[1:] - close_px[:-1]

    up = np.maximum(close_diff, 0.0)
    down = -1 * np.minimum(close_diff, 0.0)

    ma_up = nb_ewma(up, window=window, alpha=1/window)
    ma_down = nb_ewma(down, window=window, alpha=1/window)
    ma_down[ma_down == 0.0] = np.nan  # this step to prevent ZeroDivision error when eval ma_up / ma_down

    rsi = ma_up / ma_down
    rsi = 100.0 - (100.0 / (1.0 + rsi))

    return rsi

def calc_rsis(df,price="close", window=13, label=13):    
    # Ensure no nans
    df.dropna(inplace=True)
    price = df[price].to_numpy()


    rsi = nb_rsi(price, window=window)
    
    df[f"RSI_{label}"]= rsi
    
    return df

@nb.njit(cache=True)
def nb_mfi(high: np.array, low: np.array, close: np.array, volume: np.array, window: int) -> np.array:
    """
    nb_mfi is ~2x speed of pta.mfi. Not that much speed-up. Same values output as pta.mfi

     INPUTS:
        high: np.array(float)
        low: np.array(float)
        close: np.array(float)
        volume: np.array(float)
        window: int

    RETURNS:
            mfi: np.array(float)
    """
    n = len(close)
    typicalPrice = (high+low+close) / 3
    raw_money_flow = volume * typicalPrice

    # create an index of Bool type
    pos_mf_idx = np.full(n, False)
    pos_mf_idx[1:] = np.diff(typicalPrice) > 0

    # assign values of raw_money_flow to pos_mf where pos_mf_idx == True...Likewise for neg_mf
    pos_mf = np.full(n, np.nan)
    neg_mf = np.full(n, np.nan)
    pos_mf[pos_mf_idx] = raw_money_flow[pos_mf_idx]
    neg_mf[~pos_mf_idx] = raw_money_flow[~pos_mf_idx]

    psum = np.full(n, np.nan)
    nsum = np.full(n, np.nan)

    for i in range(window, n):
        psum[i] = np.nansum(pos_mf[i-window+1:i+1])
        nsum[i] = np.nansum(neg_mf[i-window+1:i+1])

    mfi = 100 * psum / (psum + nsum)

    return mfi

def calc_mfi(df, window, label=14):#, ohlc=None):
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    volume = df["volume"].to_numpy()
    
    mfi = nb_mfi(high,low, close, volume, window)
    df[f"MFI_{label}"]= mfi
    
    # if ohlc is not None:
    #     df[[f"MFI_{label}_open",f"MFI_{label}_high",f"MFI_{label}_low",f"MFI_{label}_close"]] = calc_ohlc_from_series(df,col_name=f"MFI_{label}", window=ohlc)

    return df


# ============================== TIDES ========================================
    
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
    if np.isnan(previous_ebb):
        previous_ebb = new_open

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
            windows: np.array,
            thresholds: int,
            sensitivity: float,
            fixed_window: bool):

    n = len(open_)
    max_lookback = np.max(windows)

    tide = np.full(n, np.nan)
    # tide2 = np.full(n, np.nan)
    # tidestr = np.full(n, np.nan)
    ebb = np.full(n, np.nan)
    flow = np.full(n, np.nan)
    # tide_ids = np.full(n, np.nan)
    
    previous_tide = np.nan
    previous_ebb = np.nan
    previous_flow = np.nan
    # pos_tide_id = 0
    # neg_tide_id = 0
    for i in range(max_lookback, n + 1):
        # i_s = [i for i in range(max_lookback,n)]
        if fixed_window:
            open_i = open_[i - max_lookback:i]
            high_i = high[i - max_lookback:i]
            low_i = low[i - max_lookback:i]
            # close_i = close[i - max_lookback:i]
        else:
            # expanding window
            open_i = open_[:i]
            high_i = high[:i]
            low_i = low[:i]
            # close_i = close[:i]
        tide_i, ebb_i, flow_i = calc_tide(open_i=open_i,
                                          high_i=high_i,
                                          low_i=low_i,
                                          # close_i=close_i,
                                          previous_tide=previous_tide,
                                          previous_ebb=previous_ebb,
                                          previous_flow=previous_flow,
                                          windows=windows,
                                          thresholds=thresholds,
                                          sensitivity=sensitivity)
        
        # tide_unused, ebb_unused, flow_i = calc_tide(open_i=open_i,
        #                                               high_i=high_i,
        #                                               low_i=low_i,
        #                                               close_i=close_i,
        #                                               previous_tide=tide_i,
        #                                               previous_ebb=ebb_i,
        #                                               previous_flow=flow_i,
        #                                               windows=windows,
        #                                               thresholds=thresholds,
        #                                               sensitivity=sensitivity)
        
        # if previous_tide != tide_i:
        #     if tide_i > 0: 
        #         pos_tide_id+=1
        #         tide_id = pos_tide_id
        #     elif tide_i < 0:
        #         neg_tide_id-=1
        #         tide_id = neg_tide_id
        # if ebb_i > previous_ebb:
        #     tide2[i - 1] = 1
        #     tidestr[i - 1] = ebb_i/previous_ebb -1
        # elif ebb_i < previous_ebb:
        #     tide2[i - 1] = 0
        #     tidestr[i - 1] = previous_ebb/ebb_i -1
        # else:
        #     tide2[i - 1] = tide2[i - 2]
        #     tidestr[i - 1] = 0
            
        previous_tide = tide_i
        previous_ebb = ebb_i
        previous_flow = flow_i
        
        # tide_ids[i-1] = tide_id
        tide[i - 1] = tide_i
        ebb[i - 1] = ebb_i
        flow[i - 1] = flow_i
        
    return tide, ebb, flow


def calc_tides(df, sensitivity:int=50,
               thresholds:int=10, 
               windows:np.ndarray=np.array([5, 20, 67]),
               price=["open","high","low"],
               fixed_window = True,
               suffix="_fast"):
    assert type(sensitivity) == int
    assert type(thresholds) == int
    assert type(windows) == np.ndarray
    
    open_ = df[price[0]].to_numpy()
    high = df[price[1]].to_numpy()
    low = df[price[2]].to_numpy()
    # close = df["close"].to_numpy()

    tides, ebb, flow = nb_tide(open_=open_,
                               high=high,
                               low=low,
                               # close=close,
                               windows=windows,
                               thresholds=thresholds,
                               sensitivity=sensitivity,
                               fixed_window=fixed_window)
    
    df[f"tide{suffix}"] = tides
    df[f"tide{suffix}"]=df[f"tide{suffix}"].replace(0,-1)
    # df["tide_id"] = tide_ids
    df[f"ebb{suffix}"] = ebb
    df[f"flow{suffix}"] = flow
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
    

#%% VOLUME PROFILE
def nb_vp():
    pass


#%% Tide metrics
def calc_tide_metrics(klines_indicators_dict):
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


# if __name__ == "__main__":
    
#     print("calculating dollar bars")
#     dollar_bars = get_dollar_bars(df,dollar_threshold=10000)
# #%%
#     if input("Proceed? ") == "y":
#         print("yes")