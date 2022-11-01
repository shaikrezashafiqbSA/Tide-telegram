import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import time
import telegram
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler

from data.data_management import DataManager
from performance.plots import get_plotly
from strategy.indicators import calc_tide_metrics
import dataframe_image as dfi 

"""
TODO:
    - mx vs price warnings

"""
class TidesUpdate:  
    def __init__(self, 
                 instruments, 
                 db_update,
                 config = None,
                 send_to_telegram = True):
        self.instruments = instruments
        self.db_update = db_update
        self.asset_class = instruments["asset_class"]
        self.send_to_telegram = send_to_telegram
        if config is None:
            self.config = {"general":{"db_update": self.db_update,
                                      # "db_path": "C:/Users/shaik/OneDrive/database/",  #"D:/OneDrive/database/",
                                      "db_path": "D:/OneDrive/database/",
                                      "output": "telegram/"},
                           "strategy": {"timeframes": ["1h","4h", "24h", "48h"],
                                        "indicators": {'tide_fast': {'window': [12,24,36],
                                                                     "sensitivity": [10],
                                                                     "thresholds": [20],
                                                                     'price':['open','high','low']},
                                                       'mfi': {'length': [14], 'price': ['close']},
                                                       'ema': {'length': [81], 'price': ['close']}
                                                       },
                                        "resample": True
                                        },
                           }

    def load_data(self):  
        
        data_manager = DataManager(instruments = self.instruments["instruments"],
                                   db_path = self.config["general"]["db_path"],
                                   update_db = self.config["general"]["db_update"],
                                   timeframes = self.config["strategy"]["timeframes"],
                                   indicators = self.config["strategy"]["indicators"],
                                   resample = self.config["strategy"]["resample"],
                                   )
        
        klines_indicators_dict =  data_manager.load_data()
        self.klines_indicators_dict = klines_indicators_dict
    
    #%%
    def update(self):
        if self.send_to_telegram:
            telegram_auth_token = "5403909034:AAF5TVshAENvwEvhNRQ_H7Jf519nPGrgnxA"
            chat_id = -1001604110225

            bot = telegram.Bot(telegram_auth_token)
            bot.send_message(text=f"{self.asset_class} job starting ...", chat_id=chat_id)
            
        # =======================================
        #  Get klines    
        # =======================================
        t1 = time.time()
        self.load_data()
               
        # =======================================
        #  Plot
        # =======================================        
        date_now = dt.now().strftime("%Y-%m-%d")
        date_1_month_ago =(dt.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        
        get_plotly(filename = f"tides_{self.asset_class}",
                   df_dict = self.klines_indicators_dict,
                   window=[date_1_month_ago, date_now],
                   cols_to_plot=["MFI_14"])
        t2 = np.round(time.time()-t1,2)

        # =======================================
        #  Send plotly html to telegram
        # =======================================
        if self.send_to_telegram:
            file = open(f"tides_{self.asset_class}.html",'rb')
            bot.send_document(chat_id, file)

            # retry = 0
            # while retry < 1:
            #     try:
            #         bot.send_document(chat_id, file)
            #     except Exception as e:
            #         retry += 1
            #         bot.send_message(text=f"{self.asset_class} ERROR {e} ---> retry {retry} in 5 minutes", chat_id=chat_id)
            #         print(f"{self.asset_class} ERROR {e} ---> retry {retry} in 5 minutes")
            #         time.sleep(300)



            print(f"Sent tides_{self.asset_class} to telegram")

        pd.set_option('display.max_columns', None)
        
        metrics_df = calc_tide_metrics(self.klines_indicators_dict)
        dfi.export(metrics_df, f"telegram/summary_{self.asset_class}.png")
        if self.send_to_telegram:
            bot.send_photo(chat_id, photo=open(f'telegram/summary_{self.asset_class}.png', 'rb'),
                           caption=f"{self.asset_class} snapshot")
            # retry = 0
            # while retry < 1:
            #     try:
            #         bot.send_photo(chat_id, photo=open(f'telegram/summary_{self.asset_class}.png', 'rb'),caption=f"{self.asset_class} snapshot")
            #     except Exception as e:
            #         retry += 1
            #         bot.send_message(text=f"{self.asset_class} ERROR {e} ---> retry {retry} in 5 minutes", chat_id=chat_id)
            #         print(f"{self.asset_class} ERROR {e} ---> retry {retry} in 5 minutes")
            #         time.sleep(300)



            print(f"Sent summary_{self.asset_class}.png to telegram")
        

        t2 = np.round(time.time()-t1,2)
        if self.send_to_telegram:
            print("FINISHED")
            msg = f"{self.asset_class} job completed in {t2}s"
            bot.send_message(text=msg, chat_id=chat_id)
            
            print(msg)
            
#%%
if __name__ == "__main__":
    test= False
    if not test:
        instruments_equities = {"asset_class":"equities",
                                "instruments":["CME_MINI_ES1!",
                                               "CME_MINI_NQ1!",
                                               "SGX_CN1!",    
                                               "SGX_TWN1!",
                                               "TWSE_2330",
                                               "SGX_SGP1!",
                                               "HKEX_HSI1!",
                                               "HKEX_TCH1!",
                                               "HKEX_ALB1!",
                                               "COMEX_MINI_MGC1!",
                                               "NASDAQ_TSLA",
                                               "NASDAQ_NFLX",
                                               "NYSE_SE",
                                               ]
                                }
        
        instruments_crypto = {"asset_class":"crypto",
                              "instruments":["ftx_BTC/USD",
                                            "ftx_ETH/USD",
                                            "ftx_TSM/USD",
                                            "ftx_NVDA/USD",
                                            "ftx_AMD/USD",
                                            "ftx_TSLA/USD",
                                            ]
                              }
        
        tide_equities = TidesUpdate(instruments=instruments_equities,
                                    db_update = True,
                                    send_to_telegram = True)
        
        tide_crypto = TidesUpdate(instruments=instruments_crypto,
                                  db_update = True,
                                  send_to_telegram = True)
        tide_equities.update()
        tide_crypto.update()

        # scheduler = BackgroundScheduler(daemon=False,timezone="Singapore")
        scheduler = BlockingScheduler(timezone="Singapore")
        scheduler.add_job(func=tide_equities.update, 
                          trigger='cron',
                          minute="15")
        scheduler.add_job(func=tide_crypto.update, 
                          trigger='cron',
                          minute="1")
        
        scheduler.start()
    else:

        instruments_equities = {"asset_class": "equities",
                                "instruments": ["CME_MINI_ES1!",
                                                "CME_MINI_NQ1!",
                                                "SGX_CN1!",
                                                "SGX_TWN1!",
                                                "TWSE_2330",
                                                "SGX_SGP1!",
                                                "HKEX_HSI1!",
                                                "HKEX_TCH1!",
                                                "HKEX_ALB1!",
                                                "COMEX_MINI_MGC1!",
                                                "NASDAQ_TSLA",
                                                "NASDAQ_NFLX",
                                                "NYSE_SE",
                                                ]
                                }

        instruments_crypto = {"asset_class": "crypto",
                              "instruments": ["ftx_BTC/USD",
                                              "ftx_ETH/USD",
                                              "ftx_TSM/USD",
                                              "ftx_NVDA/USD",
                                              "ftx_AMD/USD",
                                              "ftx_TSLA/USD",
                                              ]
                              }

        tide_equities = TidesUpdate(instruments=instruments_equities,
                                    db_update=True,
                                    send_to_telegram=True)

        tide_crypto = TidesUpdate(instruments=instruments_crypto,
                                  db_update=True,
                                  send_to_telegram=True)
        

        
        tide_equities.update()
        tide_crypto.update()
        
        
    
