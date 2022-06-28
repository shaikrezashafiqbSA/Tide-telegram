import numpy as np
import pandas as pd
import pandas_ta
import pandas_ta as ta
import warnings

from strategy.indicators import calc_tides, calc_continuous_resample

class indicators_manager:
    """
    This class is responsible for calculating technical indicators (using pandas_ta)
    This class also inherits kline/indicators processing from strategy class object, but why? 
    
    
    """
    def __init__(self,
                 indicators: dict = {"hma":{'length':[20,40,80]}}, # "col_names" : ("ADX","DMP","DMN") for multiple outputs
                 postprocess_klines = None,
                 preprocess_klines = None
                 ):
        self.indicators = indicators
        self.indicators_list = pd.DataFrame().ta.indicators(as_list=True)
        if postprocess_klines is not None:
            self.postprocess_klines = postprocess_klines
        if preprocess_klines is not None:
            self.preprocess_klines = preprocess_klines
        # Build list according to what ta.Strategy accepts
        indicators_params = []
        for indicator,params in indicators.items():
            if indicator == "tide":
                continue
            kind = indicator
            if len(params) == 1:
                for param_name,param_value_list in params.items():
                    for param_value in param_value_list:
                        indicators_params.append({"kind":kind, 
                                                  param_name:param_value}
                                                 )
            else: 
                try:
                    df = pd.DataFrame(params)
                except:
                    max_param_len = max([len(param) for param in params.values() ])
                    for param_name,param in params.items():
                        if len(param) < max_param_len:
                            params[param_name] = param*max_param_len
                    df = pd.DataFrame(params)
                df["kind"]=kind
                for idx in range(len(df)):
                    ind_dict_i ={}
                    for indicator_label in df.columns:
                        ind_dict_i[indicator_label] = df[indicator_label].iloc[idx]
                    indicators_params.append(ind_dict_i)
       
        self.indicators_factory = ta.Strategy(name="general_bot",ta=indicators_params)

    def _calc_indicators(self,klines: pd.DataFrame,workers=0):
        if "tide" in self.indicators.keys():
            sensitivity = self.indicators["tide"]["sensitivity"]
            thresholds = self.indicators["tide"]["thresholds"]
            windows = self.indicators["tide"]["window"]
            # print(f"\n\n sensitivity: {sensitivity}, type: {type(sensitivity)}")
            if len(sensitivity) >1 or type(sensitivity) is not int:
                # warnings.warn(f"More than 1 sensitivity parameter not supported yet \n--> selecting 1st of {sensitivity}")
                sensitivity = int(sensitivity[0])
            if len(thresholds) >1 or type(thresholds) is not int:
                # warnings.warn(f"More than 1 threshold parameter not supported yet \n--> selecting 1st of {thresholds}")     
                thresholds = int(thresholds[0])
            if type(windows) is not np.ndarray:
                windows = np.array(windows)
                
            klines = calc_tides(klines,sensitivity=sensitivity, thresholds=thresholds, windows=windows)
            
        klines.ta.cores = workers
        klines.ta.strategy(self.indicators_factory,verbose=False) 
        
        return klines
        
    def _preprocess_klines(self,klines: pd.DataFrame):
        print("NO PREPROCESSING OF KLINES")
        return klines
   
    
    def _postprocess_klines(self,klines: pd.DataFrame):
        print("NO POSTPROCESSING OF KLINES")
        return klines
    # def onPayload(self,payload):
    #     # Calculate TA stuff here
    #     if payload["type"] == "klines":
    #         return self.klinesHandler(payload)

        
        
        
        
#%%
    # def calc_indicator(klines,
    #                    indicator="LINEARREG_SLOPE",
    #                    params={'timeperiod':[20,40,80]}
    #                    ,
    #                    ):
    #     # indicator_factory = vbt.pandas_ta(indicator) # pandas_ta too slow
    #     # catalogue_talib = talib.get_functions()
    #     # try:
    #     #     indicator_factory = vbt.talib(indicator)
    #     # except Exception as e:
    #     #     # print(f"{e} - trying pandas_ta")
    #     indicator_factory = vbt.talib(indicator)
            
    #      #Else if not in pandas or talib then use jeremy's nb indicators
    #     # if custom:
    #     #     indicator_factory = IndicatorFactory(
    #     #                                             class_name='HMA',
    #     #                                             module_name=__name__,
    #     #                                             short_name='hma',
    #     #                                             input_names=['close'],
    #     #                                             param_names=['winsize'],
    #     #                                             output_names=['hma']
    #     #                                         ).from_apply_func(
    #     #                                             hma_nb,
    #     #                                             kwargs_to_args=None,
    #     #                                             ewm=False,
    #     #                                             adjust=False
    #     #                                         )
            
            
    #     output_names = indicator_factory.output_names
    #     input_names = indicator_factory.input_names
    #     param_names = indicator_factory.param_names
        
    #     input_arguments = {}#{"close":klines["close"],"timeperiod":timeperiod}
    #     for input_name in input_names:
    #         input_arguments[input_name]=klines[input_name]
    #     for param_name in param_names:
    #         try:
    #             input_arguments[param_name]=params[param_name]   
    #         except:
    #             # print(f"{indicator} parameter: {param_name} not provided, using default values")
    #             pass
    #     if len(output_names) == 1:
    #         df = getattr(indicator_factory.run(**input_arguments),output_names[0]) # Gotta fix for multiple outputs?
    #     else: 
    #         dfs=[]
    #         for output_name_idx in range(len(output_names)+1):
    #             dfs.append(getattr(indicator_factory.run(**input_arguments),output_names[0]) )
    #         df = pd.concat(dfs, axis=1)
    
    #     return df
#%%    
# from datetime import datetime
# from config.parser import config_to_dict
# config = config_to_dict("config/config-dev.ini")    
# indicators =config["strategy"]["indicators"]
# test= Indicators_Manager(indicators=indicators)
# klines = klines_LTF.copy()


# t0 = time.time()
# df = test.calc_indicator(klines)
# print(f"{time.time()-t0}")


# test.calc_derived_indicator()

#%%
# self.klines_indicators_dict = self.calc_indicators_for_all_timeframes(klines_dict=self.klines_dict,continuous_resample=continuous_resample,workers=workers)
# self.klines_indicators_dict = self.calc_derived_indicators()
        
#     def calc_derived_indicators(self):
#         klines_indicators_dict = {}
#         for TF,klines_indicators in self.klines_indicators_dict.items():
#             klines_indicators_postprocessed = self._calc_indicators_postprocessed(klines_indicators,TF=TF)
#             klines_indicators_dict[TF] = klines_indicators_postprocessed
        
#         return klines_indicators_dict
# # =============================================================================
# # Calculate and insert indicators into klines dataframes
# # =============================================================================
#     def calc_indicators_for_all_timeframes(self, klines_dict, continuous_resample:bool,workers=0):
#         # klines_dict of pairs with 1m, LTF MTF, HTF
#         # BUILD TECHNICAL INDICATORS (pandas_ta uses multiprocessing across TAs)
#         klines_indicators_dict = {}
#         for TF,klines in klines_dict.items():
#             klines_indicators = klines.copy()
#             klines_indicators.ta.cores = workers
#             klines_indicators.ta.strategy(self.technical_indicators_suite,verbose=False)   
#             # klines_indicators.ta.strategy(technical_indicators_suite) 
            
#             for indicator_params in self.indicators_params:
#                 # To transition to pandas ta suite-like object for multiple TAs? 
#                 indicator = indicator_params["kind"]
#                 if "hma" == indicator:
#                     klines[f"{indicator}"] = hma_nb(klines['close'].values, winsize=indicator_params['length'])
            
#             if TF != "1m":
#                 klines_indicators_dict[TF] = klines_indicators
#             else:
#                 klines_indicators_dict['1m'] = klines_indicators
        
            
    
#         return klines_indicators_dict
    

# # =============================================================================
# # Calculate and insert indicators into klines dataframes
# # =============================================================================
#     def calc_indicators(self, klines, timeframe, workers=0):
#         # klines_dict of pairs with 1m, LTF MTF, HTF
#         # BUILD TECHNICAL INDICATORS (pandas_ta uses multiprocessing across TAs)
#         # klines_indicators_dict = {}
#         # for TF,klines in klines_dict.items():
#         klines_indicators = klines.copy()
#         klines_indicators.ta.cores = workers
#         klines_indicators.ta.strategy(self.technical_indicators_suite,verbose=False)   
#             # klines_indicators.ta.strategy(technical_indicators_suite) 
            
#         for indicator_params in self.indicators_params:
#             # To transition to pandas ta suite-like object for multiple TAs? 
#             indicator = indicator_params["kind"]
#             if "hma" == indicator:
#                 klines[f"{indicator}"] = hma_nb(klines['close'].values, winsize=indicator_params['length'])
        
#         return klines_indicators
#%%

# if __name__ == "__main__":
#     import pandas_ta
#     import numpy as np
#     import pandas as pd
#     import vectorbt as vbt
#     from vectorbt.indicators.factory import IndicatorFactory
#     HMA = IndicatorFactory(
#         class_name='HMA',
#         module_name=__name__,
#         short_name='hma',
#         input_names=['close'],
#         param_names=['winsize'],
#         output_names=['hma']
#     ).from_apply_func(
#         hma_nb,
#         kwargs_to_args=None,
#         ewm=False,
#         adjust=False
#     )
        
        
       
#     klines = klines_1m.copy().head(5000)
    
    
#     timeit slope_talib = vbt.talib("LINEARREG_SLOPE").run(klines_1m["close"],[20]).real
    
#     indicator="EMA"
#     timeperiod=[7,10,14,20,28,40,56,80]
    
    
    
#     # def indicator_generator(klines,indicator,params)
#     generator_ema = vbt.talib("EMA")
#     inputs_required = generator_ema.input_names
#     param_names = generator_ema.param_names
    
#     # dictionary for generator_ema input argument: {"close":klines["close"],"timeperiod":timeperiod}
#     input_arguments = {}#{"close":klines["close"],"timeperiod":timeperiod}
#     for input_required in inputs_required:
#         input_arguments[input_required]=klines[input_required]
#     for param_name in param_names:
#         input_arguments[param_name]=timeperiod
    
    
#     indicators_ema = generator_ema.run(**input_arguments)
#     output_names = generator_ema.output_names
#     get_indicators_ema = getattr(indicators_ema,output_names[0])
    
#     indicators = {"HMA":{'length': [7,10,14,20,28,40,56,80]},
#                   "LINEARREG_SLOPE":{'timeperiod':[7,10,14,20,28,40,56,80]}
#                   }
#     #%%
#     def calc_indicator(klines,
#                        indicator="HMA",
#                        params={'length':[20,40,80]}
#                        ,
#                        ):
#         # indicator_factory = vbt.pandas_ta(indicator) # pandas_ta too slow
#         # catalogue_talib = talib.get_functions()
#         # try:
#         #     indicator_factory = vbt.talib(indicator)
#         # except Exception as e:
#         #     # print(f"{e} - trying pandas_ta")
#         indicator_factory = vbt.pandas_ta(indicator)
            
#          #Else if not in pandas or talib then use jeremy's nb indicators
#         # if custom:
#         #     indicator_factory = IndicatorFactory(
#         #                                             class_name='HMA',
#         #                                             module_name=__name__,
#         #                                             short_name='hma',
#         #                                             input_names=['close'],
#         #                                             param_names=['winsize'],
#         #                                             output_names=['hma']
#         #                                         ).from_apply_func(
#         #                                             hma_nb,
#         #                                             kwargs_to_args=None,
#         #                                             ewm=False,
#         #                                             adjust=False
#         #                                         )
            
            
#         output_names = indicator_factory.output_names
#         input_names = indicator_factory.input_names
#         param_names = indicator_factory.param_names
        
#         input_arguments = {}#{"close":klines["close"],"timeperiod":timeperiod}
#         for input_name in input_names:
#             input_arguments[input_name]=klines[input_name]
#         for param_name in param_names:
#             try:
#                 input_arguments[param_name]=params[param_name]   
#             except:
#                 # print(f"{indicator} parameter: {param_name} not provided, using default values")
#                 pass
#         if len(output_names) == 1:
#             df = getattr(indicator_factory.run(**input_arguments),output_names[0]) # Gotta fix for multiple outputs?
#         else: 
#             dfs=[]
#             for output_name_idx in range(len(output_names)+1):
#                 dfs.append(getattr(indicator_factory.run(**input_arguments),output_names[0]) )
#             df = pd.concat(dfs, axis=1)
    
#         return df
    
#     klines = klines_LTF.copy()
    
#     # timeit calc_indicator(klines)
#     test = calc_indicator(klines)
#     #%%
#     def calc_indicator_pandas_ta(klines):
#         dfs = pd.DataFrame()
#         for param in [20,40,80]:
#             df=klines.ta.ema(param)
#             dfs[param]=df
#         return dfs

#     #%%
    
#     klines = klines_LTF.copy()
#     indicators_params = [{"kind":"hma","length":20}, {"kind":"hma","length":40}, {"kind":"hma","length":80}]
#     import pandas_ta as ta
#     technical_indicators_suite = ta.Strategy(name="ta_suite",ta=indicators_params)
#     klines.ta.cores = 0
#     # timeit klines.ta.strategy(technical_indicators_suite,verbose=False) 

    
    
    