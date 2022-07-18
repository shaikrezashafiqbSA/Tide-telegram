from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
import time

from tvDatafeed import TvDatafeed, Interval

klines_freq_dict = {"1m":Interval.in_1_minute,
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
            
def fetch_ohlcv_from_exchange(exchange,
                              symbol,
                              timeframe, 
                              limit = 150000,
                              max_retries = 3
                              ):
    client =  TvDatafeed()
    
    interval = klines_freq_dict[timeframe]
    
    # LOAD TVDATAFEED        
    tries = 0
    while tries < 10:
        try:
            klines = client.get_hist(symbol=symbol,
                                      exchange=exchange,
                                      interval=interval,
                                      n_bars=limit)
        except Exception as e:
            print(e)
            print("RETRYING ... {tries}")
            time.sleep(10)
            tries+=1
        else:
            klines["closeTime"]=klines.index.map(pd.Timestamp.timestamp)
            # klines["closeTime"] = klines["closeTime"].astype(np.int64)
            klines.drop(columns=["symbol"],inplace=True)
            klines = klines[["open","high","low","close","volume","closeTime"]]
            return klines
            break
        
    return klines


def save_to_db(data,
               table_name = "CME_MINI_NQ1!_1h",
               db_path = "D:/OneDrive/database/TV_klines.db",
               if_exists = "append"
               ):
    conn = sqlite3.connect(db_path)
    data.to_sql(table_name, conn, if_exists=if_exists, index=False)
    
def load_from_db(table_name = "CME_MINI_NQ1!_1h",
                 db_path = "D:/OneDrive/database/TV_klines.db"
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


def check_db(db_path="D:/OneDrive/database/TV_klines.db"):
    if db_path.split("/")[-1].split("_")[0] == "TV":
        unit = "s"
    else:
        unit = "ms"
        
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
        *exchange,instrument,timeframe = table.split("_")
        exchange = "_".join(exchange)
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
        tables_df.columns = ['exchange', 'instrument', 'timeframe','len', 'start_date', 'end_date']
        
    return tables_df


def get_klines(instrument="CME_MINI_NQ1!_1h",
               update=False,
               from_date = None,
               to_date=None,
               reload=False,
               limit = 1500,
               max_retries=3):
    *exchange,symbol,timeframe = instrument.split("_")
    exchange = "_".join(exchange)
    
    db_path = f"D:/OneDrive/database/TV_klines.db"
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
        print(f"{instrument} does not exist in {db_path}\nINTIATING REST API CALLS ... ")
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
#%%
    from data.klines_TV import check_db
    db_tables = check_db()
    
    #%%

    #%% funding rates
    
    
    from data.klines_TV import get_klines
    test = get_klines(instrument="SGX_CN1!_1h", update=False, reload=False)
    #%%
    from data.klines_TV import fetch_ohlcv_from_exchange
    test1 = fetch_ohlcv_from_exchange(exchange="SGX",
                                  symbol="CN1!",
                                  timeframe="1h", 
                                  limit = 150000,
                                  max_retries = 3
                                  )

    #%% RENAME
    # import the sqlite3 module
    import sqlite3
    
    # Create a connection object
    connection  = sqlite3.connect("D:/OneDrive/database/TV_klines.db")

    # Get a cursor
    cursor      = connection.cursor()
    
    # Rename the SQLite Table
    for table in db_tables.index:
        name = "_".join(table.split("_")[1:])
        
        renameTable = f"ALTER TABLE '{table}' RENAME TO '{name}'"
        
        cursor.execute(renameTable)
        print(f"{table} to {name}")
    
    # close the database connection
    connection.close()

    #%% DROP
    # import the sqlite3 module
    import sqlite3
    
    # Create a connection object
    connection  = sqlite3.connect("D:/OneDrive/database/TV_klines.db")

    # Get a cursor
    cursor      = connection.cursor()
    
    to_drop = db_tables[db_tables["timeframe"]!="1h"].index
    # Rename the SQLite Table
    for table in to_drop:
        # print(table)
        dropTable = f"DROP TABLE '{table}'"
        
        cursor.execute(dropTable)
        print(f"dropped {table}")
    
    # close the database connection
    connection.close()