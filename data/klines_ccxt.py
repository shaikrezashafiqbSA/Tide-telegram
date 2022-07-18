from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
import ccxt

def _fetch_ohlcv(client,
                 symbol,
                 timeframe,
                 since, 
                 limit,
                 max_retries):
    num_retries = 0
    while num_retries < max_retries:
        try:
            ohlcv = client.fetch_ohlcv(symbol, timeframe, since, limit)
            return ohlcv
        except Exception as e:
            print(e)
            print(f"RETRY - {num_retries}")
            num_retries += 1
            client.sleep(10000)
            
def fetch_ohlcv_from_exchange(exchange,
                              symbol,
                              timeframe, 
                              since='2017-11-01 00:00:00',
                              limit = 1500,
                              max_retries = 3
                              ):
    client = getattr(ccxt, exchange)({'enableRateLimit': True})
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = client.parse8601(since)
    # preload all markets from the exchange
    markets = client.load_markets()
    
    # timestamp now
    until_timestamp = client.milliseconds() 
    
    # timeframe (eg: 1h) to timestamp delta
    timeframe_duration_in_seconds = client.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    
    # timedelta to paginate by 
    timedelta = limit * timeframe_duration_in_ms
    
    all_ohlcv = []
    while True:
        # Fetch backwards from until_timestamp
        fetch_since = until_timestamp - timedelta
        client.sleep(3000)
        ohlcv = _fetch_ohlcv(client, symbol, timeframe, fetch_since, limit, max_retries)
        
        fetched_since = ohlcv[0][0]
        fetched_until = ohlcv[-1][0]
        print(f"{symbol} fetched: {client.iso8601(fetched_since)} to {client.iso8601(fetched_until)}")
        if fetched_since >= until_timestamp:# or fetched_since is None:
            break
        until_timestamp = fetched_since
        all_ohlcv = all_ohlcv + ohlcv
        total_fetched_since = all_ohlcv[0][0]
        total_fetched_until = all_ohlcv[-1][0]
        
        print(f"---> {len(all_ohlcv)} | {client.iso8601(total_fetched_until)} to {client.iso8601(total_fetched_since)}")
        if fetch_since < since:
            break
        
    df = pd.DataFrame(all_ohlcv)
    df.columns = ["closeTime","open", "high","low","close","volume"]
    df=df[["open", "high","low","close","volume","closeTime"]]
    df["closeTime"]=df["closeTime"]/1000
    df.index = pd.to_datetime(df["closeTime"],unit='s')
    return df


def save_to_db(data,
               table_name = "kucoin_BTC/USDT_1h",
               db_path = "D:/OneDrive/database/kucoin_klines.db",
               if_exists = "update"
               ):
    conn = sqlite3.connect(db_path)
    data.to_sql(table_name, conn, if_exists=if_exists, index=False)
    
def load_from_db(table_name = "kucoin_BTC/USDT_1h",
                 db_path = "D:/OneDrive/database/kucoin_klines.db"
                 ):
    conn = sqlite3.connect(db_path)
    query = f""" 
                SELECT * from '{table_name}'
                ORDER BY closeTime
            """
    df0 = pd.read_sql_query(query, conn)
    
    # Check for duplicates, if there are, update db
    df=df0.copy()
    # Set datetime index from closeTime 
    df.drop_duplicates(subset=['closeTime'], inplace=True, keep="last")
    if len(df) < len(df0):
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    df['date_time'] = pd.to_datetime(df['closeTime'], unit='s')
    df.set_index(keys=['date_time'], inplace=True, drop=True)
                
    return df


def check_db(db_path="D:/OneDrive/database/ftx_klines.db"):

    unit = "s"
        
    conn = sqlite3.connect(db_path)
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
        exchange,instrument,timeframe = table.split("_")
        sql = f""" select * from '{table}' ORDER BY closeTime DESC LIMIT 1"""
        sql1 = f""" select * from '{table}' ORDER BY closeTime ASC LIMIT 1"""
        sql2 = f"""SELECT COUNT(*) FROM '{table}' """
        
        first_row = pd.read_sql_query(sql1, conn)
        latest_row = pd.read_sql_query(sql, conn)
        len_row = pd.read_sql_query(sql2, conn).iloc[0,0]
        earliest_closeTime = first_row["closeTime"]
        latest_closeTime = latest_row["closeTime"]
        earliest_datetime = pd.to_datetime(earliest_closeTime, unit = unit)
        latest_datetime = pd.to_datetime(latest_closeTime, unit = unit)
        if len(latest_row) == 0:
            tables_dict[table] = (exchange, instrument, timeframe,np.nan, np.nan,np.nan)
        else:
            tables_dict[table] = (exchange, instrument, timeframe,len_row, earliest_datetime[0],latest_datetime[0])
    tables_df = pd.DataFrame(tables_dict).T
    # Check first if tables_df is empty
    if len(tables_df) != 0:
        tables_df.columns = ['exchange', 'instrument', 'timeframe','len', 'start_date', 'end_date']
        
    return tables_df


def get_klines(instrument="ftx_BTC/USD_1h",
               update=False,
               from_date = None,
               to_date=None,
               reload=False,
               limit = 1500,
               max_retries=3):
    exchange,symbol,timeframe = instrument.split("_")
    db_path = f"D:/OneDrive/database/{exchange}_klines.db"
    db_tables = check_db(db_path)
    if (instrument in db_tables.index) and not reload:
        if not update:
            # If not update, just take everything from db
            output = load_from_db(table_name = f"{exchange}_{symbol}_{timeframe}",
                                  db_path = db_path
                                  )
            return output
        elif update:
            print(f"{instrument} is outdated in {db_path}\nINTIATING PAGINATED REST API CALLS ... ")
            ohlcv = fetch_ohlcv_from_exchange(exchange=exchange,
                                              symbol=symbol,
                                              timeframe=timeframe,
                                              since=str(db_tables.loc[instrument,"end_date"]),
                                              limit=limit,
                                              max_retries=max_retries)
            
            print(f"save_to_db: {instrument} ... ")
            save_to_db(ohlcv,
                       table_name = f"{exchange}_{symbol}_{timeframe}",
                       db_path = db_path,
                       if_exists = "append"
                       )
            
            print(f"returning {instrument} ... ")
            output = load_from_db(table_name = f"{exchange}_{symbol}_{timeframe}",
                                  db_path = db_path
                                  )
            return output
    elif (instrument not in db_tables.index) or reload:
        print(f"{instrument} does not exist in {db_path}\nINTIATING PAGINATED REST API CALLS ... ")
        ohlcv = fetch_ohlcv_from_exchange(exchange=exchange,
                                          symbol=symbol,
                                          timeframe=timeframe,
                                          limit=limit,
                                          max_retries=max_retries)
        
        print(f"save_to_db: {instrument} ... ")
        save_to_db(ohlcv,
                   table_name = f"{exchange}_{symbol}_{timeframe}",
                   db_path = db_path,
                   if_exists = "replace"
                   )
        
        print(f"returning {instrument} ... ")
        output = load_from_db(table_name = f"{exchange}_{symbol}_{timeframe}",
                              db_path = db_path
                              )
        
        return output
        
    
        

    
    #%%
if __name__ == "__main__":

    
    db_tables = check_db()
    
    #%%
    import ccxt 
    import pandas as pd
    client = ccxt.ftx()
    markets = client.load_markets()
    markets_df = pd.DataFrame(markets).T
    fut = pd.DataFrame([m for m in markets.values() if m["swap"]])
    spot =  pd.DataFrame([m for m in markets.values() if m["spot"]])
    
    since = "2022-07-15"
    since = client.parse8601(since)
    ohlcv = client.fetch_ohlcv(symbol="ETH/USD", timeframe="1h", since=since)
    # mark = client.fetch_ohlcv(symbol="ETH/USD", timeframe="1h", since=since, params={"price":"index"})
    #%% funding rates
    from data.klines_ccxt import check_db
    db_tables = check_db()
    
    #%%
    from data.klines_ccxt import get_klines
    # test = get_klines(instrument="ftx_BTC-PERP_1h", update=False, reload=False)
    test = get_klines(instrument="ftx_BTC/USD_1h", update=False, reload=False)
    

