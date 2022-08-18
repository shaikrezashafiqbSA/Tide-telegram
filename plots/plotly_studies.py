from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly_resampler import FigureWidgetResampler

class build:
    def __init__(self,
                 row_heights: list=[1,2],
                 cols_to_plot = ["index","label"],
                 height: int = 500,
                 width: int = 500
                ):
        self.height = height
        self.width = width
        self.number_subplots = len(cols_to_plot)
        self.row_heights = row_heights
        self.cols_to_plot = cols_to_plot

        self.fig = FigureWidgetResampler(make_subplots(rows=self.number_subplots,
                                                       cols=1, 
                                                       shared_xaxes = True, 
                                                       vertical_spacing = 0.05,
                                                       row_heights = self.row_heights, 
                                                       specs =[[{"type":"scatter"}]]*self.number_subplots,
                                                       #subplot_titles = self.cols_to_plot
                                                      ))
        
        
    def plot(self,df):
        row=1    
        for col in self.cols_to_plot:
            print(f"plotting: {col}")
            if type(col)==list:
                print(f"plotting: {col} --> LIST DETECTED")
                # Plot these cols in 1 plot
                for col_i in col:
                    if "OHLC" in col_i:
                        col_name,tf = col_i.split("_")
                        klines_ax = go.Candlestick(x=df.index,
                                                   open=df[f"{tf}_open"],
                                                   high=df[f"{tf}_high"],
                                                   low=df[f"{tf}_low"],
                                                   close=df[f"{tf}_close"],
                                                   name=f"{col_i}",
                                                   increasing_line_color='rgb(14,203,129)',
                                                   decreasing_line_color='rgb(233,67,89)')
                        self.fig.append_trace(klines_ax,row=row,col=1)
                    elif "L_entry_price" in col_i:
                        entries_ax=go.Scattergl(x=df.index, y=df[f"L_entry_price"],mode='markers',marker_symbol= 'arrow-up',name="long entry",marker = dict(color='blue',size=13)) #name="longEntry_price"
                        self.fig.append_trace(entries_ax,row=row,col=1)
                    elif "L_exit_price" in col_i:
                        exits_ax= go.Scattergl(x=df.index, y=df[f"L_exit_price"],mode='markers',marker_symbol= 'arrow-down-open',name="long exit",marker = dict(color='blue',size=13)) #name="longExit_price"
                        self.fig.append_trace(exits_ax,row=row,col=1)
                    elif "S_entry_price" in col_i:  
                        entries_ax=go.Scattergl(x=df.index, y=df[f"S_entry_price"],mode='markers',marker_symbol= 'arrow-down',name="short entry",marker = dict(color='black',size=13)) #name="longEntry_price"
                        self.fig.append_trace(entries_ax,row=row,col=1)
                    elif "S_exit_price" in col_i:
                        exits_ax= go.Scattergl(x=df.index, y=df[f"S_exit_price"],mode='markers',marker_symbol= 'arrow-up-open',name="short exit",marker = dict(color='black',size=13)) #name="longExit_price"
                        self.fig.append_trace(exits_ax,row=row,col=1)
                        
                    elif ("sig" in col_i) or (len(df[col_i].unique()) < 4):
                        print(f"{col} -> cat data detected")
                        ax = go.Scattergl(x=df.index, y=df[col_i],name=col_i, mode='markers', marker = dict(size=4))
                        self.fig.append_trace(ax,row=row,col=1)
                        
                    else:
                        ax = go.Scattergl(x=df.index, y=df[col_i],name=col_i)
                        self.fig.append_trace(ax,row=row,col=1)
                row+=1
            elif ("sig" in col) or (len(df[col].unique()) < 4):
                print(f"{col} -> cat data detected")
                ax = go.Scattergl(x=df.index, y=df[col],name=col, mode='markers', marker = dict(size=4))
                self.fig.append_trace(ax,row=row,col=1)
                row+=1
            else:
                # Note: plotly_resampler only supports scattergl so other go objected will not be resampled
                ax = go.Scattergl(x=df.index, y=df[col],name=col)
                self.fig.append_trace(ax,row=row,col=1)
                row+=1

        self.fig.update_layout(autosize=False,width=self.width,height=self.height)
        self.fig.update_layout(xaxis_rangeslider_visible=False)
        self.fig.update_xaxes(rangeslider_visible=False)
        self.fig.update_layout(hovermode="x")
    
        return self.fig