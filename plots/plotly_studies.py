# pip install plotly
# pip install plotly-resampler 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly_resampler import FigureWidgetResampler

class plotly_studies:
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
    def build(self,df):
        row=1    
        for col in self.cols_to_plot:
            print(f"plotting: {col}")
            if type(col)==list:
                # Plot these cols in 1 plot
                for col_i in col:
                    ax = go.Scattergl(x=df.index, y=df[col_i],name=col_i)
                    self.fig.append_trace(ax,row=row,col=1)
                row+=1
            elif ("sig" in col) or (len(df[col].unique()) < 5):
                print(f"{col} -> cat data detected")
                ax = go.Scattergl(x=df.index, y=df[col],name=col, mode='markers', marker = dict(size=4))
                self.fig.append_trace(ax,row=row,col=1)
                row+=1
            else:
                # Note: plotly_resampler only supports scattergl so other go objected will not be resampled
                ax = go.Scattergl(x=df.index, y=df[col],name=col)
                self.fig.append_trace(ax,row=row,col=1)
                row+=1

        self.fig.update_l