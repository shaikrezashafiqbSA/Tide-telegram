import pandas as pd
import numpy as np
import math 

# @nb.njit(cache=True)
def get_dollar_bars(time_bars, dollar_threshold): #function credit to Max Bodoia
    # initialize an empty list of dollar bars
    dollar_bars = []

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

    

def calc_dollar_bars(df0, dollar_threshold, timeframe="1h"):
    df=df0.copy()
    df=df.filter(regex=timeframe)
    #  Need close high low open, closeTime, volume
    df=df[[f"{timeframe}_close",f"{timeframe}_high",f"{timeframe}_low",f"{timeframe}_open",f"{timeframe}_closeTime",f"{timeframe}_volume"]]
    df.rename(columns={f"{timeframe}_close": "close",
                       f"{timeframe}_high": "high",
                       f"{timeframe}_low": "low",
                       f"{timeframe}_open": "open",
                       f"{timeframe}_closeTime":"closeTime",
                       f"{timeframe}_volume":"volume"},inplace=True)
    time_bars = df.to_dict('records') 
    dollar_bars = get_dollar_bars(time_bars,dollar_threshold)
    dollar_bars = pd.DataFrame(dollar_bars)
    dollar_bars['date_time'] = pd.to_datetime(dollar_bars['timestamp'], unit='s').round("1T")
    dollar_bars.set_index(keys=['date_time'], inplace=True, drop=False)
    return dollar_bars