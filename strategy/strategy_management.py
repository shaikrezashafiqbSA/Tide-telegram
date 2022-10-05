import numpy as np
import pandas as pd
from config.parser import getStrategy
from strategy.strategy_state import STRATEGY_STATE

class Strategy_Manager:
    def __init__(self,
                 strategy: str,
                 config_strategy: dict,
                 Order_manager,
                 verbose=True):

        self.verbose = verbose

        # Initialise attributes
        self.strategy = None
        self.trailing_initial_stop_upnl_pct = -100
        self.trailing_engaged_upnl_pct = 100
        self.trailing_take_profit_upnl_pct = 100
        for key, value in config_strategy.items():
            setattr(self, key, value)
        # =============================+
        # GET STRATEGY 
        # ==============================
        self.strategy = strategy
        self.config_strategy = config_strategy
        self.strategy = getStrategy(strategy=self.strategy, config_strategy=self.config_strategy)

        self.Order_manager = Order_manager

        # ==============================
        # Initialise tradelogs
        # ==============================

        # Strategy time series containers
        self.TS_realised_pnl_long = {}
        self.TS_unrealised_pnl_long = {}
        self.TS_unrealised_pnl_pct_long = {}
        self.TS_AEP_long = {}

        self.TS_realised_pnl_short = {}
        self.TS_unrealised_pnl_short = {}
        self.TS_unrealised_pnl_pct_short = {}
        self.TS_AEP_short = {}

        self.TS_long_comments = {}
        self.TS_short_comments = {}
        self.TS_long_positions = {}
        self.TS_short_positions = {}

        self.TS_long_entry_qty = {}
        self.TS_long_entry_base_qty = {}
        self.TS_long_entry_price = {}

        self.TS_long_exit_qty = {}
        self.TS_long_exit_base_qty = {}
        self.TS_long_exit_price = {}
        self.TS_long_trailing_stop = {}
        
        self.TS_short_entry_qty = {}
        self.TS_short_entry_base_qty = {}
        self.TS_short_entry_price = {}

        self.TS_short_exit_qty = {}
        self.TS_short_exit_base_qty = {}
        self.TS_short_exit_price = {}
        self.TS_short_trailing_stop = {}

        # ==============================
        # initialise Strategy state
        # ==============================
        self.long_positions = 0
        self.short_positions = 0

        self.current_timestamp = 0
        self.previous_Timestamp = 0

        self.long_average_price = np.nan
        self.long_unpnl = np.nan

        self.short_average_price = np.nan
        self.short_unpnl = np.nan

        self.long_trailing_stop = np.nan
        self.long_trailing_stop_engaged = None
        
        self.short_trailing_stop = np.nan
        self.short_trailing_stop_engaged = None

    # ==========================================================================================================================================================
    # START
    # CALCULATE SIGNAL (consumes 4 dataframes of height = lookback)
    # ==========================================================================================================================================================
    # TODO: typing
    def onPayload(self, payload):
        if payload["type"] == "klines":
            self.strategy.consume_payload(payload)
            return

            # This method consumes 1m, LTF, MTF, HTF (optional) dataframes to generate signals
        self.current_timestamp = payload["closeTime"]
        self.close_price = payload["closePrice"]

        self.update_strategy_state()
        payload["strategy_state"] = self.get_strategy_state()

        trail_adjust_value = self.strategy.calc_trail_adjust(payload)
        self.update_trailing_stop(trail_adjust_value=trail_adjust_value)

        orders_list = self.strategy.consume_payload(payload)

        """
               {"position": "long",
                "side": "buy",
                "type": "market",
                "price": None,
                "quantity_notional": 100,
                "comments": "signal"}

        """

        # =============================================================================
        # NO SIDE   
        # =============================================================================         
        if len(orders_list) == 0:
            pass
        
        for order in orders_list:
            
            if order is None:
                continue
            
            # =============================================================================
            # LONG SIDE        
            # =============================================================================
            elif order["position"] == "long":
                if order["side"] == "buy":
                    order["current_timestamp"] = self.current_timestamp
    
                    if self.verbose: print(f"\n{order}")
    
                    # TODO: AWAIT ORDER FILL only then update trade_logs
                    order_response = self.Order_manager.long_entry(long_entry_order=order)
                    self.updateTradeLogs(position="long", side="entry", order_response=order_response)
    
                elif order["side"] == "sell":
                    order["current_timestamp"] = self.current_timestamp
    
                    if self.verbose: print(f"\n{order}")
    
                    # TODO: AWAIT ORDER FILL only then update trade_logs
                    order_response = self.Order_manager.long_exit(long_exit_order=order)
                    self.updateTradeLogs(position="long", side="exit", order_response=order_response)
    
                elif order["side"] == None:
                    self.updateTradeLogs(position="long", side=None, order_response=None)
    
            # =============================================================================
            # SHORT SIDE        
            # =============================================================================
            elif order["position"] == "short":
                if order["side"] == "buy":
                    order["current_timestamp"] = self.current_timestamp
    
                    if self.verbose: print(f"\n{order}")
    
                    # TODO: AWAIT ORDER FILL only then update trade_logs
                    order_response = self.Order_manager.short_entry(short_entry_order=order)
                    self.updateTradeLogs(position="short", side="entry", order_response=order_response)
    
                elif order["side"] == "sell":
                    order["current_timestamp"] = self.current_timestamp
    
                    if self.verbose: print(f"\n{order}")
    
                    # TODO: AWAIT ORDER FILL only then update trade_logs
                    order_response = self.Order_manager.short_exit(short_exit_order=order)
                    self.updateTradeLogs(position="short", side="exit", order_response=order_response)
    
                elif order["side"] == None:
                    self.updateTradeLogs(position="short", side=None, order_response=None)


            # CATCH OTHERS
    
            else:
                raise Exception(f"Something else produced: {order}")
        # =============================================================================
        # END
        # =============================================================================

        self.previous_timestamp = self.current_timestamp

    # ==========================================================================================================================================================
    # CALCULATE SIGNAL
    # END
    # ==========================================================================================================================================================

    def onOrderUpdate(self, order_response):
        """

        Parameters
        ----------
        order_response : TYPE
            DESCRIPTION.
            
        Returns
        -------
        updates strategyState

        """
        return NotImplemented

    # =============================================================================
    # Trail Stop
    # =============================================================================
    def update_trailing_stop(self, trail_adjust_value):
        if trail_adjust_value == None:
            pass
        else:
            # Initial position so set initial stoploss
            if self.long_positions == 0:
                pass
    
            elif (self.long_trailing_stop_engaged == None) and self.long_positions > 0:
                self.long_trailing_stop = self.trailing_initial_stop_upnl_pct / 100
                self.long_trailing_stop_engaged = False
    
            # upnl has hit trailing threshold so trail up the stoploss
            elif self.long_unpnl >= (self.trailing_engaged_upnl_pct / 100) and (
                    self.long_trailing_stop_engaged == False) and self.long_positions > 0:
                self.long_trailing_stop = self.trailing_take_profit_upnl_pct / 100
                self.long_trailing_stop_engaged = True
    
            # Update trailing stop if close > previous trailing stop
            elif self.long_unpnl >= self.long_trailing_stop and (
                    self.long_trailing_stop_engaged == True) and self.long_positions > 0:
                updated_trail = self.long_trailing_stop + trail_adjust_value
                if self.long_unpnl >= updated_trail:
                    self.long_trailing_stop = updated_trail
    
            # Initial position so set initial stoploss
            if self.short_positions == 0:
                pass
    
            elif (self.short_trailing_stop_engaged == None) and self.short_positions > 0:
                self.short_trailing_stop = self.trailing_initial_stop_upnl_pct / 100
                self.short_trailing_stop_engaged = False
    
            # upnl has hit trailing threshold so trail up the stoploss
            elif self.short_unpnl >= (self.trailing_engaged_upnl_pct / 100) and (
                    self.short_trailing_stop_engaged == False) and self.short_positions > 0:
                self.short_trailing_stop = self.trailing_take_profit_upnl_pct / 100
                self.short_trailing_stop_engaged = True
    
            # Update trailing stop if close > previous trailing stop
            elif self.short_unpnl >= self.short_trailing_stop and (
                    self.short_trailing_stop_engaged == True) and self.short_positions > 0:
                updated_trail = self.short_trailing_stop + trail_adjust_value
                if self.short_unpnl >= updated_trail:
                    self.short_trailing_stop = updated_trail

    # =============================================================================
    # Strategy States
    # =============================================================================
    def update_strategy_state(self):
        if self.long_positions > 0:
            self.long_unpnl = self.close_price / self.long_average_price - 1
            self.in_long_position = True
        elif self.long_positions == 0:
            self.long_unpnl = np.nan
            self.in_long_position = False
        else:
            raise Exception(
                f"Error in determining strategy state- long_positions: {self.long_position}, short_positions: {self.short_positions}")
                
        if self.short_positions > 0:
            self.short_unpnl = 1 - self.close_price / self.short_average_price
            self.in_short_position = True
        elif self.short_positions == 0:
            self.short_unpnl = np.nan
            self.in_short_position = False
        else:
            raise Exception(
                f"Error in determining strategy state- long_positions: {self.long_positions}, short_positions: {self.short_positions}")

    def get_strategy_state(self):
        data = {"long":{"in_position":self.in_long_position,
                          "unpnl":self.long_unpnl,
                          "pnl_notional":self.TS_realised_pnl_long,
                          "unpnl_notional":self.TS_unrealised_pnl_long,
                          "open_positions_df":self.Order_manager.long_open_positions_df,
                          "trailing_stop_engaged": self.long_trailing_stop_engaged,
                          "trailing_stop": self.long_trailing_stop
                          },
                "short":{"in_position":self.in_short_position,
                         "unpnl":self.short_unpnl,
                         "pnl_notional":self.TS_realised_pnl_short,
                         "unpnl_notional":self.TS_unrealised_pnl_short,
                         "open_positions_df":self.Order_manager.short_open_positions_df,
                         "trailing_stop_engaged": self.short_trailing_stop_engaged,
                         "trailing_stop": self.short_trailing_stop
                         }} 
        
        strategy_state = STRATEGY_STATE(self.current_timestamp, data)
        return strategy_state

    # =============================================================================
    # Trade logs
    # ============================================================================= 
    def updateTradeLogs(self, side, position, order_response):
        self.update_strategy_state()
        self.TS_long_trailing_stop[self.current_timestamp] = self.long_trailing_stop
        self.TS_short_trailing_stop[self.current_timestamp]= self.short_trailing_stop
        # ====================================
        # LONG UPDATE
        # ====================================

        # Position updates
        if position == "long":
            if side == None and order_response == None:
                self.TS_long_positions[self.current_timestamp] = self.long_positions
                self.TS_AEP_long[self.current_timestamp] = self.long_average_price
                self.TS_unrealised_pnl_pct_long[self.current_timestamp] = self.long_unpnl
                self.TS_unrealised_pnl_long[self.current_timestamp] = (self.close_price - self.long_average_price) * self.Order_manager.long_open_positions_df["cum_amt"].iloc[-1]

            # Entry updates
            elif side == "entry" and order_response is not None:
                self.long_average_price = order_response["long_average_price"]
                self.long_positions = order_response["long_positions"]

                long_entry_timestamp = order_response["long_entry_timestamp"]

                self.TS_long_positions[long_entry_timestamp] = self.long_positions
                self.TS_long_entry_price[long_entry_timestamp] = order_response["long_entry_price"]
                self.TS_long_entry_qty[long_entry_timestamp] = order_response["long_entry_qty"]
                self.TS_long_entry_base_qty[long_entry_timestamp] = order_response["long_entry_amt"]
                self.TS_AEP_long[long_entry_timestamp] = self.long_average_price
                self.TS_long_comments[long_entry_timestamp] = order_response["comments"]

                self.TS_unrealised_pnl_pct_long[self.current_timestamp] = (self.close_price / self.long_average_price) - 1
                self.TS_unrealised_pnl_long[self.current_timestamp] = (self.close_price - self.long_average_price) * self.Order_manager.long_open_positions_df["cum_amt"].iloc[-1]

            # Exit updates
            elif side == "exit" and order_response is not None:
                self.long_positions = order_response["long_positions"]
                long_exit_timestamp = order_response["long_exit_timestamp"]

                self.TS_long_positions[long_exit_timestamp] = self.long_positions
                self.TS_long_exit_price[long_exit_timestamp] = order_response["long_exit_price"]
                self.TS_long_exit_qty[long_exit_timestamp] = order_response["long_exit_qty"]
                self.TS_long_exit_base_qty[long_exit_timestamp] = order_response["long_exit_amt"]
                self.TS_AEP_long[long_exit_timestamp] = self.long_average_price
                self.TS_long_comments[long_exit_timestamp] = order_response["comments"]
                self.TS_unrealised_pnl_pct_long[self.current_timestamp] = (order_response["long_exit_price"] / self.long_average_price) - 1
                self.TS_unrealised_pnl_long[self.current_timestamp] = order_response["long_exit_amt"] * (order_response["long_exit_price"] - self.long_average_price)
                self.TS_realised_pnl_long[self.current_timestamp] = order_response["long_exit_amt"] * (order_response["long_exit_price"] - self.long_average_price)

                # REDUCE/RESET 
                # self.number_of_available_longs = int(self.max_longs_per_pair)
                if order_response["long_open_positions_df"] is None:
                    self.long_average_price = np.nan
                    self.long_unpnl = 0.0
                    self.long_positions = 0
                    self.long_trailing_stop_engaged = None
                    self.long_trailing_stop = np.nan



        # ====================================
        # SHORT UPDATE
        # ====================================

        # Position updates
        elif position == "short":
            if side == None and order_response == None:
                self.TS_short_positions[self.current_timestamp] = self.short_positions
                self.TS_AEP_short[self.current_timestamp] = self.short_average_price
                self.TS_unrealised_pnl_pct_short[self.current_timestamp] = self.short_unpnl
                self.TS_unrealised_pnl_short[self.current_timestamp] = (self.short_average_price - self.close_price) * self.Order_manager.short_open_positions_df["cum_amt"].iloc[-1]

            # Entry updates
            elif side == "entry" and order_response is not None:
                self.short_average_price = order_response["short_average_price"]
                self.short_positions = order_response["short_positions"]
                # self.number_of_available_shorts -= 1

                short_exit_timestamp = order_response["short_entry_timestamp"]
                self.TS_short_positions[short_exit_timestamp] = self.short_positions
                self.TS_short_entry_price[short_exit_timestamp] = order_response["short_entry_price"]
                self.TS_short_entry_qty[short_exit_timestamp] = order_response["short_entry_qty"]
                self.TS_short_entry_base_qty[short_exit_timestamp] = order_response["short_entry_amt"]
                self.TS_AEP_short[short_exit_timestamp] = self.short_average_price
                self.TS_short_comments[short_exit_timestamp] = order_response["comments"]

                self.TS_unrealised_pnl_pct_short[self.current_timestamp] = 1 - (self.close_price / self.short_average_price)
                self.TS_unrealised_pnl_short[self.current_timestamp] = (self.short_average_price - self.close_price) * self.Order_manager.short_open_positions_df["cum_amt"].iloc[-1]

            # Exit updates
            elif side == "exit" and order_response is not None:
                self.short_positions = order_response["short_positions"]
                short_exit_timestamp = order_response["short_exit_timestamp"]

                self.TS_short_positions[short_exit_timestamp] = self.short_positions
                self.TS_short_exit_price[short_exit_timestamp] = order_response["short_exit_price"]
                self.TS_short_exit_qty[short_exit_timestamp] = order_response["short_exit_qty"]
                self.TS_short_exit_base_qty[short_exit_timestamp] = order_response["short_exit_amt"]
                self.TS_AEP_short[short_exit_timestamp] = self.short_average_price
                self.TS_short_comments[short_exit_timestamp] = order_response["comments"]
                self.TS_unrealised_pnl_pct_short[self.current_timestamp] = 1 - (
                        order_response["short_exit_price"] / self.short_average_price)
                self.TS_unrealised_pnl_short[self.current_timestamp] = order_response["short_exit_amt"] * (self.short_average_price - order_response["short_exit_price"])
                self.TS_realised_pnl_short[self.current_timestamp] = order_response["short_exit_amt"] * (self.short_average_price - order_response["short_exit_price"])

                # REDUCE/RESET 
                if order_response["short_open_positions_df"] is None:
                    self.short_average_price = np.nan
                    self.short_unpnl = 0.0
                    self.short_positions = 0
                    self.short_trailing_stop_engaged = None
                    self.short_trailing_stop = np.nan

    # ==========================================================================================================================================================
    # ==========================================================================================================================================================
    # Other methods
    # ==========================================================================================================================================================
    # ==========================================================================================================================================================

    def get_signals_df(self):
        # aggregate signals to df (like in backtest)

        signals_TS = {}
        signals_TS["long_positions"] = self.TS_long_positions
        signals_TS["long_entry_qty"] = self.TS_long_entry_qty
        signals_TS["long_entry_amt"] = self.TS_long_entry_base_qty
        signals_TS["long_entry_price"] = self.TS_long_entry_price
        signals_TS["long_average_price"] = self.TS_AEP_long
        signals_TS["long_exit_qty"] = self.TS_long_exit_qty
        signals_TS["long_exit_amt"] = self.TS_long_exit_base_qty
        signals_TS["long_exit_price"] = self.TS_long_exit_price
        signals_TS["long_trailing_stop"] = self.TS_long_trailing_stop
        signals_TS["long_comments"] = self.TS_long_comments
        signals_TS["long_realised_pnl"] = self.TS_realised_pnl_long
        signals_TS["long_unrealised_pnl"] = self.TS_unrealised_pnl_long
        signals_TS["long_unrealised_pnl_pct"] = self.TS_unrealised_pnl_pct_long


        signals_TS["short_positions"] = self.TS_short_positions
        signals_TS["short_entry_qty"] = self.TS_short_entry_qty
        signals_TS["short_entry_amt"] = self.TS_short_entry_base_qty
        signals_TS["short_entry_price"] = self.TS_short_entry_price
        signals_TS["short_average_price"] = self.TS_AEP_short
        signals_TS["short_exit_qty"] = self.TS_short_exit_qty
        signals_TS["short_exit_amt"] = self.TS_short_exit_base_qty
        signals_TS["short_exit_price"] = self.TS_short_exit_price
        signals_TS["short_trailing_stop"] = self.TS_short_trailing_stop
        signals_TS["short_comments"] = self.TS_short_comments
        signals_TS["short_realised_pnl"] = self.TS_realised_pnl_short
        signals_TS["short_unrealised_pnl"] = self.TS_unrealised_pnl_short
        signals_TS["short_unrealised_pnl_pct"] = self.TS_unrealised_pnl_pct_short

        # TS_long_positions has most ALL INDEX use this as merge on left
        signals_df = pd.DataFrame.from_dict(signals_TS)
        signals_df.index.name = 'date_time'
        signals_df.reset_index(inplace=True)
        signals_df["date_time"] = pd.to_datetime(signals_df['date_time'], unit='ms')
        signals_df.set_index(keys=['date_time'], inplace=True, drop=False)

        self.signals_df = signals_df
        return signals_df

