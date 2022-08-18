import numpy as np
import pandas as pd
import numba as nb
from typing import Tuple, List

from tqdm import tqdm

def compute_triple_barrier_labels(
    price_series: pd.Series, 
    event_timestamps: pd.Series, 
    horizon_delta: int, 
    upper_delta: float=None, 
    lower_delta: float=None, 
    vol_span: int=20, 
    upper_z: float=None,
    lower_z: float=None,
    upper_label: int=1, 
    lower_label: int=-1,
    tf=4,
    labels = 2) -> Tuple[pd.Series, pd.Series]:
    assert labels in [2,3]
    """
    Calculate event labels according to the triple-barrier method. 
    
    price_series: 
    
    Return a series with both the original events and the labels. Labels 1, 0, 
    and -1 correspond to upper barrier breach, vertical barrier breach, and 
    lower barrier breach, respectively. 
    Also return series where the index is the start date of the label and the 
    values are the end dates of the label.
    """

    time_horizon_delta = horizon_delta*60*60*tf # to timestamp seconds if 4, then 60 *60
    series = pd.Series(np.log(price_series.values), index=event_timestamps)#price_series.index)

    
    n = len(price_series)
    np_labels = np.full(n, np.nan)
    np_label_dates = np.full(n, np.nan)
    
    if upper_z or lower_z:
        threshold = series.ewm(span=vol_span).std()
        threshold *= np.sqrt(horizon_delta / vol_span)

    for i,event_date in tqdm(enumerate(event_timestamps)):
        # print(i,event_date)
        date_barrier = event_date + time_horizon_delta

        start_price = series.loc[event_date]
        log_returns = series.loc[event_date:date_barrier] - start_price

        # First element of tuple is 1 or -1 indicating upper or lower barrier
        # Second element of tuple is first date when barrier was crossed
        candidates: List[Tuple[int, pd.Timestamp]] = list()

        # Add the first upper or lower date to candidates
        if upper_delta:
            _date = log_returns[log_returns > upper_delta].first_valid_index()
            if _date:
                candidates.append((upper_label, _date))
    
        if lower_delta:
            _date = log_returns[log_returns < lower_delta].first_valid_index()
            if _date:
                candidates.append((lower_label, _date))

        # Add the first upper_z and lower_z to candidates
        # if upper_z:
        #     upper_barrier = upper_z * threshold[event_date]
        #     _date = log_returns[log_returns > upper_barrier].first_valid_index()
        #     if _date:
        #         candidates.append((upper_label, _date))

        # if lower_z:
        #     lower_barrier = lower_z * threshold[event_date]
        #     _date = log_returns[log_returns < lower_barrier].first_valid_index()
        #     if _date:
        #         candidates.append((lower_label, _date))

        if candidates:
            # If any candidates, return label for first date
            label, label_date = min(candidates, key=lambda x: x[1])
        else:
            # If there were no candidates, time barrier was touched
            if labels == 3:
                label, label_date = 0, date_barrier
            elif labels == 2:
            # CHeck if return at this time_barrier is pos or neg then label -1/1 -> dont want 0
                end_returns = log_returns.iloc[-1]
                if end_returns >0:
                    label, label_date = upper_label, date_barrier
                else:
                    label, label_date = lower_label, date_barrier
        
        np_labels[i] = label
        np_label_dates[i] = label_date
        
    # Output

    event_spans_labels = pd.DataFrame({"t":event_timestamps, 't1':np_label_dates,'label': np_labels})
    return event_spans_labels



def calc_triple_barrier(df0,
                        col_series="1h_close",
                        col_timestamps = "1h_closeTime",
                        horizon_delta = 4,
                        vol_span = 10,
                        labels =2,
                        upper_z = 1.8,
                        upper_delta=None,
                        lower_z = -1.8,
                        lower_delta=None,
                        resample=None,
                        side = None,
                        fill_no_trades=True):
    if resample is not None:
        df=df0.resample(resample).last().copy()
        tf=int(resample[:-1])
    else:
        df=df0.copy()
        tf=1
    # timestamps =df.index
    
    event_spans_labels = compute_triple_barrier_labels(price_series = df[col_series],
                                                        event_timestamps = df[col_timestamps],
                                                        horizon_delta=horizon_delta,
                                                        vol_span = vol_span,
                                                        upper_delta=upper_delta,
                                                        lower_delta=lower_delta,
                                                        upper_z=upper_z,
                                                        lower_z=lower_z,
                                                        tf=tf,
                                                        labels=labels
                                                    )
    
    # Concatenate labels
    df = pd.merge(df0,event_spans_labels, right_index=True, left_index=True, how="left")
    if resample is not None:
        df["label"].fillna(method="bfill", inplace=True)
        
    if fill_no_trades and labels != 2:
        df["label"].replace(0, np.NaN, inplace=True)
        df["label"].fillna(method="bfill",inplace=True)
    
    if side is not None:
        try:
            # df.reset_index(inplace=True)
            # df.set_index("1h_closeTime", drop=False,inplace=True)
            # df["ret"] = np.log(df["1h_close"].shift(horizon_delta)/df["1h_close"])
            # df["ret"]=df["ret"].shift(-horizon_delta-1)
            # # np.log(df[col_series].loc[df['t1'].values].values) - np.log(df[col_series].loc[df['t1']])
            temp = df["label"] * df[side]  # meta-labeling
            # Label incorrect side as 0
            df["p_target"] = np.where(temp<= 0,0,1)
            # df.loc[df[f"meta_{side}"]<= 0, 'bin'] = 0
            # df.set_index("date_time",inplace=True)
        except Exception as e:
            # df.set_index("date_time",inplace=True)
            print(e)
            return df
        
        
    return df



    # except Exception as e:
    #     return event_spans_labels




# from tqdm import tqdm
# class tripleBarrier:
    
#     """
    
#     tB = tripleBarrier(df1,
#                    col_name="1h_close",
#                    vol_lookback = 10,
#                    vol_delta = pd.Timedelta(hours=4),
#                    horizon_delta = pd.Timedelta(hours=4),
#                    barrier_multiplier = [1,1])
#     df2 = tB.calcTripleBarrier()
    
#     """
#     def __init__(self,df: pd.DataFrame,
#                  col_name:str,
#                  vol_lookback: int = 100,
#                  vol_delta: pd.Timedelta = pd.Timedelta(hours=1),
#                  horizon_delta: pd.Timedelta = pd.Timedelta(minutes=15),
#                  barrier_multiplier: list = [1,1]
#                 ):
#         self.df = df
#         self.col_name = col_name
#         self.vol_lookback = vol_lookback
#         self.vol_delta = vol_delta
#         self.horizon_delta = horizon_delta
#         self.barrier_multiplier = barrier_multiplier
        
#     """
#     main method: assignTripleBarrier()
    
    
#     """
        
#     def get_vol(self,data_col, vol_lookback,vol_delta):
#         prices = data_col.copy()
#         df0 = prices.index.searchsorted(prices.index - vol_delta)
#         df0 = df0[df0 > 0]  # 1.2 align timestamps of p[t-1] to timestamps of p[t]
#         df0 = pd.Series(prices.index[df0-1],   
#                index=prices.index[prices.shape[0]-df0.shape[0] : ])  # 1.3 get values by timestamps, then compute returns
#         df0 = prices.loc[df0.index] / prices.loc[df0.values].values - 1 
#         # 2. estimate rolling standard deviation
#         df0 = df0.ewm(span=vol_lookback).std()
#         return df0
    
#     def get_horizons(self,prices, horizon_delta=pd.Timedelta(minutes=15)):
#         t1 = prices.index.searchsorted(prices.index + horizon_delta)
#         t1 = t1[t1 < prices.shape[0]]
#         t1 = prices.index[t1]
#         t1 = pd.Series(t1, index=prices.index[:t1.shape[0]])
#         t1.name = "t1"
#         return t1
#     def get_touches(self,prices, events, barrier_multiplier: list):
#         '''
#         events: pd dataframe with columns
#         t1: timestamp of the next horizon
#         threshold: unit height of top and bottom barriers
#         side: the side of each bet
#         barrier_multiplier: multipliers of the threshold to set the height of 
#                top/bottom barriers
#         '''
#         out = events[['t1']].copy(deep=True)
#         if barrier_multiplier[0] > 0: 
#             thresh_uppr = barrier_multiplier[0] * events['threshold']
#         else:
#             thresh_uppr = pd.Series(index=events.index) # no uppr thresh
#         if barrier_multiplier[1] > 0:
#             thresh_lwr = -barrier_multiplier[1] * events['threshold']
#         else:
#             thresh_lwr = pd.Series(index=events.index)  # no lwr thresh
#         for loc, t1 in tqdm(events['t1'].iteritems()):
#             df0=prices[loc:t1]                              # path prices
#             df0=(df0 / prices[loc] - 1) * events.side[loc]  # path returns
#             out.loc[loc, 'touch_top'] = df0[df0 < thresh_lwr[loc]].index.min()  # earliest touch_bot
#             out.loc[loc, 'touch_bot'] = df0[df0 > thresh_uppr[loc]].index.min() # earliest touch_top
#         return out
    
#     def get_labels(self,touches):
#         out = touches.copy(deep=True)
#         # pandas df.min() ignores NaN values
#         first_touch = touches[['touch_bot', 'touch_top']].min(axis=1)
#         for loc, t in tqdm(first_touch.iteritems()):
#             if pd.isnull(t):
#                 out.loc[loc, 'label'] = 0
#             elif t == touches.loc[loc, 'touch_bot']: 
#                 out.loc[loc, 'label'] = -1
#             else:
#                 out.loc[loc, 'label'] = 1
#         return out
    
#     def calcTripleBarrier(self):
#         data = self.df.copy()
#         threshold=self.get_vol(data[self.col_name],
#                                vol_lookback = self.vol_lookback,
#                                vol_delta = self.vol_delta)
#         data = data.assign(threshold=threshold.dropna())
#         t1 = self.get_horizons(data, horizon_delta = self.horizon_delta)
#         data = pd.merge(data, t1, left_index=True, right_index=True, how="left").dropna()
#         events_raw = data[['t1', 'threshold']] 
#         events = events_raw.assign(side=pd.Series(-1., events_raw.index)) # long only
#         touches = self.get_touches(data[self.col_name], events, self.barrier_multiplier)
#         touches = self.get_labels(touches)
#         data = data.assign(label=touches.label)
        
#         # for debugging
#         self.events_raw = events_raw
#         self.events = events
#         self.touches =touches
        
        
#         return data
#%%
if __name__ == "__main__":
    #%%
    import pandas as pd
    import numpy as np
    from data.pickler import pickle_klines
    df = pickle_klines()["SGX_TWN1!"]
    

    #%%
    from strategy.labels import calc_triple_barrier
    df_labelled = calc_triple_barrier(df,
                                  col_series="1h_close",
                                  col_timestamps = "1h_closeTime",
                                  horizon_delta = 4,
                                  vol_span = 10,
                                  upper_z = 1.8,
                                  lower_z = -1.8)
    df_labelled=df_labelled[["1h_open","1h_high","1h_low","1h_close","1h_closeTime","t","t1","label"]]


    #%%%
    # print("calculating dollar bars")
    # dollar_bars = calc_dollar_bars(df,dollar_threshold=1000000)