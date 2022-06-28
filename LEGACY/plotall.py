# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 14:59:49 2021

@author: Shaik Reza Shafiq
"""

#@title # 0) Mount google drive and import modules { form-width: "50px" }
 
from IPython.display import Javascript
 
import pandas as pd
import numpy as np
import math
from scipy.stats import norm, kurtosis, skew
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly
from plotly.subplots import make_subplots
from plotly.colors import n_colors
from collections import OrderedDict
 
from datetime import date
from datetime import datetime
 
import sys, os
 
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
 
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
 
# from google.colab import drive
# drive.mount('/content/drive', force_remount=True)
 
# py_file_location = "/content/drive/My Drive/tide"
# sys.path.append(os.path.abspath(py_file_location))
import tide as tide
from tide import *
 
##### 
 
def plotall(ticker, ticksize, pointvalue, window=['2017','2018','2021'],m1=23, m2=97, graphwindow=['2020-10-21','2020']):
  ## ==================================================================================================================================================================================================
  #                                                                                          Set Parameters                                                                                           =
  ## ==================================================================================================================================================================================================
  ## Instrument 
  # ticker='QZ'                                                                                       # <------------------
  # ticksize=0.05                                                                                     # <------------------
  # pointvalue=100                                                                                    # <------------------
  initial_equity=10000                                                              
  prices=updateprices(ticker) #1255.5, 1261.25, 1249.75, 1255
  verbose = False
  
  ## Tide parameters ---------------------------------------------------------------------------------
  # Pred TP parameters
  tide.TP1=[{'window':-1,'dur':25, 'strg':33},{'window':-1,'dur':25,'strg':33}] ## changed strg from -1 (average) to 25 (25th percentile)
  tide.TP2=[{'window':-1,'dur':25, 'strg':66},{'window':-1,'dur':25,'strg':80}]  
  
  
  
  
  # Other signals' parameters ------------------------------------------------------------------------
  tide.atr_window=14
  tide.dh_window=7
  tide.dl_window=7
 
 
  ## ==================================================================================================================================================================================================
  #                                                                                              Backtest                                                                                             =
  ## ==================================================================================================================================================================================================
  if verbose == False:
    with HiddenPrints():
      ## BASE TIDE
      df0 = prices[window[0]:window[2]].copy()
      streaming_base_tide(df0,m1,m2,ticksize);
      ## PRED TIDE
      df1 = prices[window[0]:window[2]].copy()
      streaming_predtide(df1,m1,m2,ticksize);
  else: 
    ## BASE TIDE
    df0 = prices[window[0]:window[2]].copy()
    streaming_base_tide(df,m1,m2,ticksize);
    ## PRED TIDE
    df1 = prices[window[0]:window[2]].copy()
    streaming_predtide(df1,m1,m2,ticksize);
 
 
 
 
  ## ==================================================================================================================================================================================================
  #                                                                                        Performance Summary                                                                                        =
  ## ==================================================================================================================================================================================================
  df={'base':df0,'pred':df1,'pred21':df0,'pred22':df0}
  summaryplots(df,initial_equity,graphwindow,pointvalue=pointvalue,ticksize=ticksize, showstrats='base,pred',showsummary=False) # 1 point = $100, $10k - 100 points
  # Increase height of output
  # display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 5000})'''))