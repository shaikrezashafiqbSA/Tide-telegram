import pandas as pd
import numpy as np
import time
import sqlite3
from tqdm import tqdm
import pandas_ta as ta

from tvDatafeed import TvDatafeed, Interval
from strategy.indicators_management import indicators_manager
from strategy.indicators import calc_continuous_resample

class klines_manager(indicators_manager):
    def __init__(self,
                 timeframes,
                 indicators,
                 klines_db_location,
                 postprocess_klines = None,
                 preprocess_klines = None,
                 resample = True,
                 update_db = False
                 ):
        
        super().__init__(indicators,
                         postprocess_klines,
                         preprocess_klines)
        
        # This should be in indicators manager
        
        self.timeframes = timeframes
        self.klines_db_location = klines_db_location
        self.conn = sqlite3.connect(self.klines_db_location)
        self.resample = resample
        self.update_db = update_db
        if self.update_db:
            self.tv = TvDatafeed()
        self.instruments = {}
        self.klines_dict = {}
        self.klines_indicators_dict = {}
        self.klines_dict2 = {}
        
        
        self.klines_freq_dict = {"1m":Interval.in_1_minute,
                                 "3m":Interval.in_3_minute,
                                 "5m":Interval.in_5_minute,
                                 "15m":Interval.in_15_minute,
                                 "30m":Interval.in_30_minute,
                                 "45m":Interval.in_45_minute,
                                 "1h":Interval.in_1_hour,
                                 "2h":Interval.in_2_hour,
                                 "3h":Interval.in_3_hour,
                                 "4h":Interval.in_4_hour,
                                 "1d":Interval.in_daily,
                                 "1w":Interval.in_weekly,
                                 "1M":Interval.in_monthly}
        
        
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
    
    
    def load_equities(self,
                      instruments=["CME_BTC1!",
                                   "SGX_CN1!",
                                   "SGX_TWN1!",
                                   "SGX_SGP1!",
                                   "HKEX_HSI1!",
                                   "HKEX_TCH1!",
                                   "HKEX_ALB1!",
                                   "COMEX_MINI_MGC1!",
                                   "NASDAQ_TSLA",
                                   "NASDAQ_NFLX",
                                   "NYSE_SE",
                                   "CME_MINI_ES1!",
                                   "CME_MINI_NQ1!"
                                   ],
                            number_of_bars = 100000):
        
        self.instruments = instruments
        
        for instrument in tqdm(instruments):
            klines_TF_dict = {}
            if self.resample:
                timeframes_to_scrap = [self.timeframes[0]]
            for freq in timeframes_to_scrap:
                # if resample True then only scrap 1h
                klines = self.scrap_from_tradingview(instrument,freq,number_of_bars)
                    
                
                klines = self.load_equities_to_db(klines, instrument, freq)
               
                klines_TF_dict[freq]=klines
                    
            self.klines_dict[instrument] = klines_TF_dict    
          
            
    def scrap_from_tradingview(self, instrument, freq, number_of_bars):
        interval = self.klines_freq_dict[freq]
        *exchange,ticker = instrument.split("_")
        exchange = "_".join(exchange)
        
        # LOAD TVDATAFEED        
        tries = 0
        while tries < 10:
            try:
                klines = self.tv.get_hist(ticker,exchange,interval=interval,n_bars=number_of_bars)
            except Exception as e:
                print(e)
                print("RETRYING ... {tries}")
                time.sleep(10)
                tries+=1
            else:
                return klines
                break
            
            
    def load_equities_to_db(self, klines, instrument, freq):
        klines["closeTime"]=klines.index.map(pd.Timestamp.timestamp)
        klines.drop(columns=["symbol"],inplace=True)
        table_name = f"TV_{instrument}_{freq}"
        klines.to_sql(table_name,self.conn, if_exists="append", index=False)
        return klines
    
    
    def load_equities_from_db(self,
                              instruments= ["CME_BTC1!",
                                            "SGX_CN1!",
                                            "SGX_TWN1!",
                                            "SGX_SGP1!",
                                            "HKEX_HSI1!",
                                            "HKEX_TCH1!",
                                            "HKEX_ALB1!",
                                            "COMEX_MINI_MGC1!",
                                            "NASDAQ_TSLA",
                                            "NASDAQ_NFLX",
                                            "NYSE_SE",
                                            "CME_MINI_ES1!",
                                            "CME_MINI_NQ1!"
                                            ]):
        print("Loading klines")
        for instrument in tqdm(instruments):
            klines_TF_dict = {}
            for freq in self.timeframes:
                if self.resample and (freq in self.timeframes[1:]): 
                    df = None
                else:
                    table_name = f"TV_{instrument}_{freq}"      
                    query = f"""SELECT * from '{table_name}' order by closeTime"""
                    df0 = pd.read_sql_query(query, self.conn)
                    df=df0.copy()
                    # Set datetime index from closeTime 
                    df.drop_duplicates(subset=['closeTime'], inplace=True, keep="last")
                    if len(df) < len(df0):
                        df.to_sql(table_name,self.conn, if_exists="replace", index=False)
                    df['date_time'] = pd.to_datetime(df['closeTime'], unit='s').round("1T")
                    df.set_index(keys=['date_time'], inplace=True, drop=True)
                klines_TF_dict[freq]=df
            self.klines_dict[instrument] = klines_TF_dict  


    def calc_indicators(self,klines_dict=None,verbose=False):
        print("Calculating Indicators")
        if klines_dict is None:
            klines_dict = self.klines_dict
            
            
        for instrument in tqdm(klines_dict.keys()):
            klines_indicators_TF_dict = {}
            for freq in self.timeframes:                
                if self.resample and (freq in self.timeframes[1:]):
                   df = klines_dict[instrument][self.timeframes[0]].copy()
                   freq_resample = self.resample_map(df,freq)
                   klines = calc_continuous_resample(df,freq_resample )
                else:
                   klines = klines_dict[instrument][freq].copy()
                 
                # print(f"Generating Tide {instrument}_{freq} ...")
                # Tide = tide(sensitivity=50,thresholds=10,lookback_windows=[5,20,67])
                # klines = calc_tides(klines,sensitivity=self.sensitivity, thresholds=self.thresholds, windows=self.lookback_windows)
                if self.indicators is not None:
                    klines = self._calc_indicators(klines)
  
                klines=klines.add_prefix(f"{freq}_")
                klines.dropna(inplace=True)
                klines_indicators_TF_dict[freq] = klines
            self.klines_dict2[instrument] = klines_indicators_TF_dict
            
            # Merge all freq for each instrument
            df = klines_indicators_TF_dict[self.timeframes[0]]
            for tf in self.timeframes[1:]:
                df = pd.merge(df,klines_indicators_TF_dict[tf], left_index=True,right_index=True,how="outer")
                    
            df = df.ffill()
            df.dropna(inplace=True)
            
            self.klines_indicators_dict[instrument] = df #klines_indicators_TF_dict
                
