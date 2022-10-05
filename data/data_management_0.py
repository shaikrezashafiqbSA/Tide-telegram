import time
import numpy as np
from datetime import datetime
import pandas as pd
import binance
from binance import Client
import sqlite3

from logger.logger_file import logger


class Data_Manager:
    def __init__(self, timeframes: dict,
                 indicators: dict,
                 indicators_needed_by_datetime: datetime,
                 klines_db_location: str,
                 update_db: bool, 
                 klines_type: str,
                 Indicators_manager): 
        # initialise
        self.klines_type = klines_type
        if self.klines_type == "futures":
            self.klines_type_binance = binance.enums.HistoricalKlinesType.FUTURES
        elif self.klines_type == "spot":
            self.klines_type_binance = binance.enums.HistoricalKlinesType.SPOT
        else:
            logger.warning("klines_type not supported: either 'spot' or 'futures'")
            
        # Initialise klines containers
        self.klines_dict = {}
        

        self.timeframes = timeframes
        self.indicators = indicators
        self.indicators_needed_by_datetime = indicators_needed_by_datetime
        self.klines_db_location = klines_db_location
        self.update_db = update_db
        self.db_status = self.check_db(klines_db_location=self.klines_db_location)
        self.updatedToDB = {}
        self.Indicators_manager = Indicators_manager
        
        # payload_timestamp
        self.payload_timestamp: pd.core.indexes.datetimes.DatetimeIndex = None
        self.from_date = 0
        self.max_indicator_length=0
        # RESAMPLING related dicts
        self.time_units = {'m': 1, 'h': 60, 'd': 24 * 60, 'w': 7 * 24 * 60, 'M': 30 * 7 * 24 * 60}
        # Get db status
        self.calc_lookback_date()
        
        pd.options.mode.chained_assignment = None
# =============================================================================
#     Load historical klines for warming up indicators
# =============================================================================
    def load_db(self, pair="BTCUSDT", timeframe='1m', from_date="1 Jan, 2021", if_exists='append') -> None:
        logger.info(f">>>> loading {pair}_{timeframe} ...")
        # available_timeframe = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d',
        #                         '1w', '1M']
        # if not set(timeframe).issubset(available_timeframe):
        #     logger.info(f">>>>{timeframe} not in \n{available_timeframe}")
        client = Client()
        conn = sqlite3.connect(self.klines_db_location)

        # for futures= futures_historical_klines
        logger.info(f">>>> loading {pair}_{timeframe} from {from_date}")
        klines = client.get_historical_klines(pair, timeframe, from_date,klines_type=self.klines_type_binance)  # "1 year ago UTC"
        logger.info(f">>>> loaded {pair}_{timeframe}")
        df = pd.DataFrame(klines, dtype=float,
                          columns=['openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quote_vol',
                                   'nTrades',
                                   'takerBuy_baseAssetVol', 'takerBuy_quoteAssetVol', '_ignore'])
        df.drop(columns=['_ignore'],inplace=True)

 
        table = self.klines_type+ "_" + pair + "_" + timeframe 
        
        logger.info(f">>>> {table} loaded to klines_db")
        df.to_sql(table, conn, if_exists=if_exists, index=False)


# =============================================================================
#     Check all tables for start and end date
# =============================================================================
    def check_db(self,klines_db_location) -> dict:
        conn = sqlite3.connect(klines_db_location)
        # conn = sqlite3.connect(klines_db_location)
        cursor = conn.cursor()
        # CHECK TABLES AVAILABLE IN DB
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_tables = cursor.fetchall()
        # logger.info(f"available tables: {all_tables}")

        # FIND MAX DATES in tables
        tables_dict = {}
        for table in all_tables:
            # table = all_tables[1]
            table = table[0]
            sql = f""" select * from {table} ORDER BY closeTime DESC LIMIT 1"""
            sql1 = f""" select * from {table} ORDER BY closeTime ASC LIMIT 1"""
            first_row = pd.read_sql_query(sql1, conn)
            latest_row = pd.read_sql_query(sql, conn)
            earliest_closeTime = first_row["closeTime"]
            latest_closeTime = latest_row["closeTime"]
            earliest_datetime = pd.to_datetime(earliest_closeTime, unit='ms')
            latest_datetime = pd.to_datetime(latest_closeTime, unit='ms')
            if len(latest_row) == 0:
                tables_dict[table] = np.nan
            else:
                tables_dict[table] = (earliest_datetime[0],latest_datetime[0])
        return tables_dict


# =============================================================================
# Query table from db
# =============================================================================
    def query_db(self, pair="1INCHUSDT", timeframe='4h', from_date="1 Jan, 2021") -> pd.DataFrame:
        
        # Initialise connection to klines_db
        conn = sqlite3.connect(self.klines_db_location)
        
        # sqlite3 table name
        table = self.klines_type+ "_" + pair + "_" + timeframe 
        
        if (table not in self.db_status) or pd.isna(self.db_status[table]):
            logger.info(f"{table} does not exist in {self.klines_db_location} --->>> load_db")
            self.load_db(pair=pair.upper(),
                         timeframe=str(timeframe),
                         from_date=from_date)
        
        # if table does not start from from_date then reload entire db and replace    
        elif datetime.strptime(from_date,"%d %b, %Y") < self.db_status[table][0] and (self.update_db):
            logger.info(f"{table} only updated from {self.db_status[table][0]} in {self.klines_db_location} --->>> reload table (via load_db)")
            self.load_db(pair=pair.upper(),
                         timeframe=str(timeframe),
                         from_date=from_date,
                         if_exists = "replace")
            
        # if table is not updated to current time AND update_db TRUE then load_db from last available date
        elif (not self.db_status[table][1] == time.time()) and (self.update_db):
            logger.info(f"{table} only updated till {self.db_status[table][1]} in {self.klines_db_location} --->>> update table (via load_db)")
            self.load_db(pair=pair.upper(),
                         timeframe=str(timeframe),
                         from_date=self.db_status[table][1].strftime("%H:%M %d %b, %Y"))
    
        
        # if update_db FALSE then use table as is
        elif not self.update_db:
            logger.info(f"Not updating {table}")

        # Query table from sqlite3 database
        query = f""" 
                    SELECT * from {table}
                    ORDER BY closeTime
                """
        df = pd.read_sql_query(query, conn)

        # Set datetime index from closeTime 
        df.drop_duplicates(subset=['closeTime'], inplace=True, keep="last")
        df['date_time'] = pd.to_datetime(df['closeTime'], unit='ms').round("1T")
        df.set_index(keys=['date_time'], inplace=True, drop=False)
        
        # Select klines from selected date onwards
        df=df[from_date:]
        return df


            
# =============================================================================
# Query table from db and populate with indicators      
# =============================================================================
    def load_ohlcv(self, pair:str = "BTCUSDT", workers=0) -> dict:
        indicators_needed_by_datetime = self.indicators_needed_by_datetime  # datetime.datetime(2021, 1, 1)
        
        # Get longest indicator and lowest timeframe to calculate max 1m lookback required
        
        # Load each given timeframes 
        if self.indicators is None: 
            logger.info("No indicators provided: no need for warmup data- loading klines from {self.from_date} ...")
            from_date = indicators_needed_by_datetime.strftime("%d %b, %Y")
            for timeframe_label,timeframe in self.timeframes.items():
                logger.info(f"Loading {timeframe} klines from {self.from_date} ... ")
                klines_TF = self.query_db(pair=pair, timeframe=timeframe, from_date=self.from_date) 
                self.klines_dict[timeframe_label] = klines_TF
                
        elif self.timeframes is None:
            logger.error(f"Higher timeframes not defined- loading 1m klines from {self.from_date} ...")
            
            klines_1m = self.query_db(pair=pair, timeframe="1m", from_date=self.from_date) 
            self.klines_dict["1m"]=klines_1m
            
        elif self.timeframes is not None:
            logger.info(f"Loading 1m klines from {self.from_date} ... ")
            klines_1m = self.query_db(pair=pair, timeframe="1m", from_date=self.from_date) 
            self.klines_dict["1m"]=klines_1m
            
            # =====================
            # Get higher timeframes OHLCV if specified else just 1m data klines loaded
            # =====================
            for timeframe_label,timeframe in self.timeframes.items():
                logger.info(f"Loading {timeframe} klines from {self.from_date} ... ")
                klines_TF = self.query_db(pair=pair, timeframe=timeframe, from_date=self.from_date) 
                self.klines_dict[timeframe_label] = klines_TF
        else:
            raise Exception(f"Something else produced: {pair}/{workers}\n timeframes: {self.timeframes}\nIndicators: {self.indicators}")
        

    def calc_lookback_date(self):
        all_indicators_params_list = sum(sum([list(indicator.values()) for indicator in self.indicators.values()],[]),[])
        all_indicators_param_list_nonstr = [param for param in all_indicators_params_list if type(param) != str]
        max_indicator_length = max(all_indicators_param_list_nonstr)
        self.max_indicator_length = max_indicator_length
        # get lookback date required for indicators (so that can start with all available indicators on start date)
        max_lookback_required_in_mins = max([int(freq[:-1])*self.time_units[freq[-1]] for freq in list(self.timeframes.values())]) 
        max_lookback_required_in_mins = max_lookback_required_in_mins * max_indicator_length
        
        from_date = self.indicators_needed_by_datetime - pd.Timedelta(value= max_lookback_required_in_mins, unit="m") # 1d * 24* 60* 100
        from_date = from_date.strftime("%d %b, %Y")
        self.from_date = from_date

# =============================================================================
# Calculate indicators (vectorised)
# =============================================================================
    def calc_all_indicators(self):
        logger.info("Calculating indicators ...")
        t0 = time.time()
        for timeframe_label,klines in self.klines_dict.items():
            self.Indicators_manager.preprocess_klines(klines,timeframe_label) 
            
            self.Indicators_manager.calc_indicators(klines)
            
            self.Indicators_manager.postprocess_klines(klines,timeframe_label)
        logger.info(f"time taken to calculate indicators {time.time() - t0}") 
 
# =============================================================================
# Calculate indicators (event-based)
# =============================================================================
    def calc_indicators(self,payload):
        # Update klines_dict with new data
        
        # Then calculate indicators
        for timeframe_label,klines in payload["klines"].items():
            self.Indicators_manager.preprocess_klines(klines)
            
            self.Indicators_manager.calc_indicators(klines)
            
            self.Indicators_manager.postprocess_klines(klines)
            
        payload["type"]="trade"
        
        
    def build_payloads(self,window):
        self.payload_timestamps = self.klines_dict["1m"][window[0]:window[1]].index
        self.needAllIndicatorsAtStartFlag = True
        
    def get_payload(self, payload_timestamp):
        """
        Need to consider lookback of payload for ta indicators
        240m EMA181 requires ~ 200 lookback
        60m EMA181 requires ~ 800
        1m EMA181 requires ~ 2000
        
        2021-01-02 23:57:00    32034.772784
        2021-01-02 23:58:00    32036.980885
        2021-01-02 23:59:00    32038.844502
        
        """
        payload = {}
        klines= {}
        klinesClosed={}
        for timeframe_label,klines_df in self.klines_dict.items():
            # TODO: consider rolling window instead of expanding for performance?
            temp_df = klines_df.loc[:payload_timestamp].tail(self.max_indicator_length*10) 
            if self.needAllIndicatorsAtStartFlag:
                closed_flag = True
            elif temp_df.index[-1] == payload_timestamp:
                closed_flag = True
            else:
                closed_flag = False
            klinesClosed[f"{timeframe_label}"] = closed_flag
            klines[f"{timeframe_label}"] = temp_df
        
        payload["timestamp"]=payload_timestamp
        payload["type"] = "klines"
        payload["closeTime"]=klines["1m"]["closeTime"][-1]
        payload["closePrice"]=klines["1m"]["close"][-1]
        payload["klinesClosed"]=klinesClosed
        payload["klines"]=klines
        
        
        return payload


    def update_payload_to_klines_dict(self,payload):
        payload_timestamp = payload["timestamp"]
        for timeframe_label, klines_df in self.klines_dict.items():
            if payload["klinesClosed"][timeframe_label]:
                payload_TF = payload["klines"][timeframe_label]
                
                if self.needAllIndicatorsAtStartFlag:
                    #  THIS IS NEEDED COS at this point klines_dict klines_TF has only 12 columns and not 12 + ta cols
                    self.indicator_columns = payload_TF.columns.difference(klines_df.columns)
                    self.klines_dict[timeframe_label]=self.klines_dict[timeframe_label].merge(payload["klines"][timeframe_label][self.indicator_columns],left_index=True,right_index=True,how="left")
                else:
                    self.klines_dict[timeframe_label].loc[payload_timestamp,self.indicator_columns] = payload["klines"][timeframe_label].loc[payload_timestamp,self.indicator_columns]
                    # self.klines_dict[timeframe_label].loc[payload_timestamp,:] = payload["klines"][timeframe_label].loc[payload_timestamp,:]

        self.needAllIndicatorsAtStartFlag = False
        
        
if __name__ == "__main__":
    #%%
    # LOAD DB IN SEQUENCE
    import time
    import pandas as pd
    import numpy as np
    import pandas_ta as ta
    from datetime import datetime
    from data.data_management import Data_Manager
    from strategy.indicators_management import Indicators_Manager
    
    timeframes = {'5m':'5m',"15m":"15m","1h":"1h","4h":"4h"}
    
    indicators = {'slope':{'length': [2, 3, 5, 8, 13, 21, 34, 55, 89, 233]},
      'ema':{'length': [2, 3, 5, 8, 13, 21, 34, 55, 89, 233]}}
    klines_db_location = './database/binance_kline.db'
    update_db=False
    indicators_needed_by_datetime = datetime.strptime("2018-01-01", "%Y-%m-%d") 
    
    
    Indicators_manager = Indicators_Manager(indicators = indicators,
                                calc_derived_indicators= None,
                                calc_preprocessed_klines = None)
    
    
    
    Data_manager = Data_Manager(timeframes=timeframes,
                    indicators=indicators,
                    indicators_needed_by_datetime=indicators_needed_by_datetime,
                    klines_db_location=klines_db_location,
                    update_db=update_db,
                    klines_type='spot',
                    Indicators_manager=Indicators_manager)
    
    db_status_df0= pd.DataFrame(Data_manager.db_status)
    db_status_df = db_status_df0.filter(regex="spot").T
    
    # -----------------------------------
    # load klines
    # -----------------------------------
    #%% ['BTCUSDT', 'ETHUSDT', 'XTZUSDT', 'ATOMUSDT', "XRPUSDT","ZECUSDT", "ZILUSDT","IOTXUSDT","ZRXUSDT", "LUNAUSDT"]
    #   ['BTCUSDT', 'ETHUSDT', 'XTZUSDT', 'ATOMUSDT', 'XRPUSDT','ZECUSDT', 'ZILUSDT','ZRXUSDT', 'DOGEUSDT', 'XMRUSDT']
    for pair in ['DOGEUSDT', 'XMRUSDT']: 
        while True:
            try:
                print(f"{'='*50}")
                t0=time.time()
                print(f"LOADING: {pair} ... ")
                Data_manager.load_ohlcv(pair=pair)
                print(f"LOADED: {pair} - {time.time()-t0} ")
            except Exception as e:
                print(f"FAILED TO LOAD: {pair} because \n {e}")
                time.sleep(60*10)
                print(f"RETRYING: {pair}")
            else:
                break
        
        
        
        
        