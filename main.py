import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import time
import telegram
from apscheduler.schedulers.background import BackgroundScheduler

from data.klines_management import klines_manager

"""
TODO:
    - 1hour continuous resample for 4h, 1d, 1w to get stabler, continuouos tides

"""
def tides_update():
    send_to_telegram = True # True
    update_db = True # True
    resample = True
    if send_to_telegram:
        telegram_auth_token = "5403909034:AAF5TVshAENvwEvhNRQ_H7Jf519nPGrgnxA"
        chat_id = -1001604110225
        
        
        # SGT = pytz.timezone('Asia/Singapore')
        # raw_SGT= dt.now(SGT)
        # curr_date = raw_SGT.strftime("%d-%m-%Y")
        # curr_time = raw_SGT.strftime("%H:%M:%S")
        
        
        bot = telegram.Bot(telegram_auth_token)
        bot.send_message(text="job starting ...", chat_id=chat_id)
    
    config = {"general":{"klines_db_location":"/Users/Shaik Reza Shafiq/Desktop/Tide/database/TV_klines.db",
                         "output": "telegram/"},
              "strategy": {"timeframes": ["1h","4h", "24h", "48h"],
                          "indicators": {'tide': {'window': [5,20,67], "sensitivity": [10], "thresholds": [5]},
                                         'atr': {'length': [14]},
                                         'mfi': {'length': [14], 'close': ['close']},
                                         'ema': {'length': [39], 'close': ['close']}
                                         },
                          },
              
              }
    
    t1 = time.time()
    timeframes = config["strategy"]["timeframes"]
    indicators = config["strategy"]["indicators"]
    klines_db_location = config["general"]["klines_db_location"]
    
    
    Klines_Manager = klines_manager(timeframes,
                                    indicators,
                                    klines_db_location,
                                    resample = resample,
                                    update_db = update_db)
    
    
    #%%
    instruments =["CME_BTC1!",
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
                  ]
    if update_db:
        Klines_Manager.load_equities(instruments=instruments)
    
    Klines_Manager.load_equities_from_db(instruments)
    
    Klines_Manager.calc_indicators()
    df=Klines_Manager.klines_indicators_dict["SGX_SGP1!"].copy()
    
    #%%
    
    from performance.plots import get_plotly#,calc_performance
    date_now = dt.now().strftime("%Y-%m-%d")
    date_1_month_ago =(dt.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    get_plotly(Klines_Manager.klines_indicators_dict,
               window=[date_1_month_ago, date_now],
               cols_to_plot=["MFI_14"])
    t2 = np.round(time.time()-t1,2)
    #%%
    
    # Send html file
    if send_to_telegram:
        file = open("D:/Users/Shaik Reza Shafiq/Desktop/Tide/tides.html",'rb')
        bot.send_document(chat_id, file)
        print("Sent tides.html to telegram")
    #%%   
    from strategy.indicators import calc_tide_metrics
    import dataframe_image as dfi 
    
    pd.set_option('display.max_columns', None)
    klines_indicators_dict = Klines_Manager.klines_indicators_dict
    
    metrics_df = calc_tide_metrics(klines_indicators_dict)
    dfi.export(metrics_df, f"telegram/summary.png")
    if send_to_telegram:
        bot.send_photo(chat_id, photo=open(f'telegram/summary.png', 'rb'),caption="Market Snapshot")
        print(f"Sent summary snapshot to telegram")
    
    #%%
    t2 = np.round(time.time()-t1,2)
    if send_to_telegram:
        print("FINISHED")
        msg = f"job completed in {t2}s"
        bot.send_message(text=msg, chat_id=chat_id)
        
        print(msg)
#%%
scheduler = BackgroundScheduler(daemon=False,timezone="Singapore")
scheduler.add_job(tides_update, 'cron', minute='20')
scheduler.start()