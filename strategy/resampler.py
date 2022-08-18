import numpy as np
import pandas as pd
import numba as nb


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
        r_high[i]=np.max(high[i-x:i+1])
        r_low[i] = np.min(low[i-x:i+1])
        r_close[i] = close[i]
        r_vol[i] = np.sum(vol[i-x:i+1])
    
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

# @nb.njit(cache=True)
# def series_to_ohlc(data, x):
#     x=x-1
#     n = len(data)
#     r_open = np.full(n, np.nan)
#     r_high = np.full(n, np.nan)
#     r_low = np.full(n, np.nan)
#     r_close = np.full(n, np.nan)

#     for i in range(x,n):
#         r_open[i]=data[i-x]
#         r_high[i]=np.max(data[i-x:i+1])
#         r_low[i] = np.min(data[i-x:i+1])
#         r_close[i] = data[i]
    
#     return r_open,r_high,r_low,r_close

# def calc_ohlc_from_series(df,col_name,window):
#     data = df[col_name].to_numpy()
    
#     r_open,r_high,r_low,r_close = series_to_ohlc(data, window)    
    
#     test= pd.DataFrame({f"{col_name}_open":r_open,
#                         f"{col_name}_high":r_high,
#                         f"{col_name}_low":r_low,
#                         f"{col_name}_close":r_close,})
#     # test['date_time'] = pd.to_datetime(test['closeTime'], unit='s').round("1T")
#     test.index = df.index

#     return test