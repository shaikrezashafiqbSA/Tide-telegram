from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
import ccxt

def _fetch_funding_rates(client,
                         until="2022-01-01 00:00:00", 
                         symbol="BTC/USD:USD",
                         limit=500,
                         max_retries=3):
    
    if isinstance(until, str):
        until = client.parse8601(until)
        
    num_retries = 0
    while num_retries < max_retries:
        try:
            fr = client.fetchFundingRateHistory(symbol=symbol,params={"until":until}, limit=limit) 
            return fr
        except Exception as e:
            print(e)
            print(f"RETRY - {num_retries}")
            num_retries += 1
            client.sleep(10000)
            
            
def fetch_funding_rates_from_exchange(exchange,
                                      symbol,
                                      limit=500, max_retries=3):
    client = getattr(ccxt, exchange)({'enableRateLimit': True})
    # convert since from string to milliseconds integer if needed

    until_timestamp = str(datetime.utcnow().replace(microsecond=0, second=0, minute=0))
    until_timestamp = client.parse8601(until_timestamp)
    # preload all markets from the exchange
    markets = client.load_markets()
    # assert symbol in markets.keys()
    
    # timestamp to start
    since_timestamp = client.parse8601("2017-11-01 00:00:00")#client.milliseconds()
    
    # timeframe (eg: 1h) to timestamp delta 1h for ftx only
    timeframe_duration_in_seconds = client.parse_timeframe("1h")
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    
    # timedelta to paginate by 
    timedelta = limit * timeframe_duration_in_ms
    
    all_fr = []
    while True:
        fetch_till = since_timestamp + timedelta
        client.sleep(3000)
        fr = _fetch_funding_rates(client=client, symbol=symbol, until=fetch_till, limit=limit, max_retries=max_retries)
        if len(fr)==0:
            since_timestamp+=timedelta
            continue
        fetched_since = fr[-1]["timestamp"]
        fetched_until = fr[0]["timestamp"]
        print(f"{symbol} fetched: {client.iso8601(fetched_since)} to {client.iso8601(fetched_until)}")
        if fetched_since > until_timestamp:
            break
        since_timestamp = fetched_since
        all_fr = all_fr + fr
        total_fetched_since = all_fr[0]["timestamp"]
        total_fetched_until = all_fr[-1]["timestamp"]
        
        print(f"---> {len(all_fr)} | {client.iso8601(total_fetched_since)} to {client.iso8601(total_fetched_until)}")
        if fetch_till > until_timestamp:
            break
        
    df = pd.DataFrame(all_fr)
    df.columns = ["info", "symbol","fundingRate", "closeTime", "datetime"]
    df=df[["fundingRate", "closeTime"]]
    df.index = pd.to_datetime(df["closeTime"],unit='ms')
    df.sort_index(inplace=True)
    return df


# def get_alt_data(instrument="ftx_BTC/USD_1h",
#                  update=False,
#                  from_date = None,
#                  to_date=None,
#                  reload=False,
#                  limit = 1500,
#                  max_retries=3):
#     exchange,symbol,timeframe = instrument.split("_")
#     db_path = f"D:/OneDrive/database/{exchange}_fr.db"
#     db_tables = check_db(db_path)
#     if (instrument in db_tables.index) and not reload:
#         if not update:
#             # If not update, just take everything from db
#             output = load_from_db(table_name = f"{exchange}_{symbol}_{timeframe}",
#                                   db_path = db_path
#                                   )
#             return output
#         elif update:
#             print(f"{instrument} is outdated in {db_path}\nINTIATING PAGINATED REST API CALLS ... ")
#             ohlcv = fetch_ohlcv_from_exchange(exchange=exchange,
#                                               symbol=symbol,
#                                               timeframe=timeframe,
#                                               since=str(db_tables.loc[instrument,"end_date"]),
#                                               limit=limit,
#                                               max_retries=max_retries)
            
#             print(f"save_to_db: {instrument} ... ")
#             save_to_db(ohlcv,
#                        table_name = f"{exchange}_{symbol}_{timeframe}",
#                        db_path = db_path,
#                        if_exists = "update"
#                        )
            
#             print(f"returning {instrument} ... ")
#             output = load_from_db(table_name = f"{exchange}_{symbol}_{timeframe}",
#                                   db_path = db_path
#                                   )
#             return output
#     elif (instrument not in db_tables.index) or reload:
#         print(f"{instrument} does not exist in {db_path}\nINTIATING PAGINATED REST API CALLS ... ")
#         ohlcv = fetch_ohlcv_from_exchange(exchange=exchange,
#                                           symbol=symbol,
#                                           timeframe=timeframe,
#                                           limit=limit,
#                                           max_retries=max_retries)
        
#         print(f"save_to_db: {instrument} ... ")
#         save_to_db(ohlcv,
#                    table_name = f"{exchange}_{symbol}_{timeframe}",
#                    db_path = db_path,
#                    if_exists = "replace"
#                    )
        
#         print(f"returning {instrument} ... ")
#         output = load_from_db(table_name = f"{exchange}_{symbol}_{timeframe}",
#                               db_path = db_path
#                               )
        
#         return output

if __name__ == "__main__":
    #%%
    from data.fundingRates import fetch_funding_rates_from_exchange
    fr = fetch_funding_rates_from_exchange(exchange="kucoin",
                                          symbol="BTCUSDM")
    
    #%%
    from data.klines_ccxt import save_to_db, check_db
    symbol = "BTC-PERP-FR"
    exchange="ftx"
    timeframe = "1h"
    save_to_db(fr,
               table_name = f"{exchange}_{symbol}_{timeframe}",
               db_path = "D:/OneDrive/database/ftx_alt.db",
               if_exists = "replace"
               )
    
    #%%
    tables_df = check_db(db_path="D:/OneDrive/database/ftx_alt.db")
