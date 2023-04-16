import pandas as pd
import numpy as np
import time
import sqlite3
from tqdm import tqdm

from strategy.indicators_management import indicators_manager
from strategy.resampler import calc_continuous_resample
from data.klines_ccxt import get_klines as get_klines_ccxt
from data.klines_TV import get_klines as get_klines_TV

class DataManager(indicators_manager):
    def __init__(self,
                 instruments,
                 timeframes,
                 indicators,
                 postprocess_klines = None,
                 preprocess_klines = None,
                 resample = True,
                 update_db = False,
                 db_path = "D:/OneDrive/database/"
                 ):
        
        super().__init__(indicators,
                         postprocess_klines,
                         preprocess_klines)
        
        # This should be in indicators manager
        
        self.instruments = instruments
        self.timeframes = timeframes
        self.resample = resample
        self.db_path= db_path
        self.update_db = update_db
        self.klines_dict = {}
        self.klines_indicators_dict = {}
        
    def check_asset_class(self,instrument):
        *exchange,symbol = instrument.split("_")
        if exchange[0] in ["kucoin", "ftx","binance"]:
            asset_class = "crypto"
        else:
            asset_class = "equities"
        return asset_class
    
    def resample_map(self,df,freq):
        """
        converts string 2h,3h to 2,3 and 1d to 12
        """
        bars_in_day = df["close"].groupby(df.index.date).count().median()
        freq_number = int(freq[:-1])
        timeframe = freq[-1]
        if timeframe == "h":
            x= freq_number
        elif timeframe == "d":
            x= freq_number * bars_in_day
        elif timeframe == "w":
            x= freq_number * bars_in_day*5
        return int(x)
    

    def load_klines(self, verbose=False):
        # TODO: specify timeframe to resample from. In this case, 1h resamples into higher timeframe
        for instrument in tqdm(self.instruments):
            temp = {}
            for freq in self.timeframes:                
                if self.resample and (freq in self.timeframes[1:]):
                    df = temp[self.timeframes[0]].copy()
                    freq_resample = self.resample_map(df,freq)
                    klines = calc_continuous_resample(df,freq_resample)
                else:
                    # GET data
                    asset_class = self.check_asset_class(instrument)
                    if asset_class == "crypto":
                        klines = get_klines_ccxt(instrument=f"{instrument}_{freq}",
                                                 db_path = self.db_path,
                                                 update=self.update_db,
                                                 reload=False)
                    else:
                        klines = get_klines_TV(instrument=f"{instrument}_{freq}", db_path = self.db_path,
                                               update=self.update_db,
                                               reload=False)
                    
                temp[freq] = klines
            self.klines_dict[instrument] = temp
                    
    def calc_indicators(self, verbose=False):
        for instrument in tqdm(self.instruments):
            temp = {}
            for freq in self.timeframes:                
                if self.resample and (freq in self.timeframes[1:]):
                    df = self.klines_dict[instrument][self.timeframes[0]].copy()
                    freq_resample = self.resample_map(df,freq)
                    klines = calc_continuous_resample(df,freq_resample)
                else:
                    # GET data
                    klines = self.klines_dict[instrument][freq].copy()
                # print(f"Generating Tide {instrument}_{freq} ...")
                # Tide = tide(sensitivity=50,thresholds=10,lookback_windows=[5,20,67])
                # klines = calc_tides(klines,sensitivity=self.sensitivity, thresholds=self.thresholds, windows=self.lookback_windows)
                klines = self._preprocess_klines(klines,freq)
                if self.indicators is not None:
                    klines = self._calc_indicators(klines,freq)
                klines = self._postprocess_klines(klines,freq)
                
                klines=klines.add_prefix(f"{freq}_")
                klines.dropna(inplace=True)
                temp[freq] = klines
            
            # Merge all freq for each instrument
            df = temp[self.timeframes[0]]
            for tf in self.timeframes[1:]:
                df = pd.merge(df,temp[tf], left_index=True,right_index=True,how="outer")
                    
            df = df.ffill()
            df.fillna(0,inplace=True)
            # df.dropna(inplace=True)
            
            self.klines_indicators_dict[instrument] = df #klines_indicators_TF_dict
            
    def load_data(self):
        self.load_klines()
        self.calc_indicators()
        return self.klines_indicators_dict
#%%
              
if __name__ == "__main__":
    #%%
    from data.data_management import DataManager
    config = {"general":{"db_update": False,
                         "output": "telegram/"},
              "strategy": {"instruments":["ftx_ETH/USD",
                                          "ftx_BTC/USD",
                                          "kucoin_BTC/USDT",
                                          "kucoin_ETH/USDT",
                                          "SGX_CN1!",
                                          "SGX_TWN1!",
                                          "SGX_SGP1!",
                                          ],
                           "timeframes": ["1h","4h", "24h", "48h"],
                           "indicators": {'tide': {'window': [5,20,67], "sensitivity": [10], "thresholds": [5]},
                                          'mfi': {'length': [14], 'close': ['close']},
                                          'ema': {'length': [81], 'close': ['close']}
                                          },
                           "resample": True
                          },
              
              }

    
    
    data_manager = DataManager(instruments = config["strategy"]["instruments"],
                                    update_db = config["general"]["db_update"],
                                    timeframes = config["strategy"]["timeframes"],
                                    indicators = config["strategy"]["indicators"],
                                    resample = config["strategy"]["resample"],
                                    )
    
    klines_indicators_dict =  data_manager.load_data()
    klines_dict = data_manager.klines_dict
    