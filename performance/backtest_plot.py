import matplotlib.pyplot as plt

def backtest_plots(df,horizon_labels=None,show_B=True):
    fig, axs = plt.subplots(3,2,gridspec_kw={'height_ratios': [4,2,1], 'width_ratios':[2,1]},figsize=(13, 5))
    # print(df.columns)
    # $ pnl
    # df["cum_buyhold"].plot(label=f'B', color="black",ax=axs[0])
    # df['cum_pnl'].plot(label=f'A', color="blue",ax=axs[0])
    # df['cum_long_pnl'].plot(label=f'L', color="green",ax=axs[0])
    # df['cum_short_pnl'].plot(label=f'S', color="orange",ax=axs[0])
    # axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # % pnl
    if show_B:
        df["cum_B_pnl%"].plot(label=f'B%', color="black",ax=axs[0,0])
    if horizon_labels is None:
        df['cum_A_pnl%'].plot(label=f'A%', color="blue",ax=axs[0,0])
        df['cum_L_pnl%'].plot(label=f'L%', color="green",ax=axs[0,0])
    elif len(horizon_labels)==1:
        df['cum_A_pnl%'].plot(label=f'A%', color="blue",ax=axs[0,0]).axvline(x=horizon_labels[0], color='red', ls="--")
        df['cum_L_pnl%'].plot(label=f'L%', color="green",ax=axs[0,0])
    else:
        df['cum_A_pnl%'].plot(label=f'A%', color="blue",ax=axs[0,0]).axvline(x=horizon_labels[0], color='red', ls="--")
        df['cum_L_pnl%'].plot(label=f'L%', color="green",ax=axs[0,0]).axvline(x=horizon_labels[1], color='green',ls="--")
        
    df['cum_S_pnl%'].plot(label=f'S%', color="orange",ax=axs[0,0]).axhline(100,color="red")
    axs[0,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))    
    
    # realised
    df.reset_index().dropna(subset="L_rpnl").plot.scatter(x='date_time', y='L_rpnl',label=f'L', color="green",ax=axs[1,0]).axhline(0,color="red")
    df.reset_index().dropna(subset="S_rpnl").plot.scatter(x='date_time', y ='S_rpnl',label=f'S', color="orange",ax=axs[1,0])
    axs[1,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # drawdown
    # df["dd_B"].plot(label=f'B%', color="black",ax=axs[2,0])
    # df['dd_A'].plot(label=f'A%', color="blue",ax=axs[2,0])
    df['dd_L'].plot(label=f'L%', color="green",ax=axs[2,0])
    df['dd_S'].plot(label=f'S%', color="orange",ax=axs[2,0]).axhline(100,color="red")
    axs[2,0].legend(loc='center left', bbox_to_anchor=(2, 0.5))
    axs[2,0].set_ylim([-15, 0])
    
    # returns distribution
    df['L_rpnl'].hist(bins=100,ax=axs[0,1],color="green")
    df['S_rpnl'].hist(bins=100,ax=axs[1,1],color="orange")
    

 
def multi_backtest_plots(backtested_dict):
    fig, axs = plt.subplots(nrows=4,ncols=1,sharex='col',gridspec_kw={'height_ratios': [1,1,1,1], 'width_ratios':[1]},figsize=(13, 5))
    # print(df.columns)
    # $ pnl
    # df["cum_buyhold"].plot(label=f'B', color="black",ax=axs[0])
    # df['cum_pnl'].plot(label=f'A', color="blue",ax=axs[0])
    # df['cum_long_pnl'].plot(label=f'L', color="green",ax=axs[0])
    # df['cum_short_pnl'].plot(label=f'S', color="orange",ax=axs[0])
    # axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for instrument,df in backtested_dict.items():
        
        # % pnl
        df["cum_B_pnl%"].plot(label=f'B_{instrument}',ax=axs[0])#.axhline(100,color="red")
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))  
        
        df['cum_A_pnl%'].plot(label=f'A_{instrument}',ax=axs[1])
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))   
        
        # df["cum_B_pnl%"].plot(label=f'B%', color="black",ax=axs[1])
        df['cum_L_pnl%'].plot(label=f'L_{instrument}',ax=axs[2])#.axhline(100,color="red")
        axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))   
        
        # df["cum_B_pnl%"].plot(label=f'B%', color="black",ax=axs[2]).axhline(100,color="red")
        df['cum_S_pnl%'].plot(label=f'S_{instrument}',ax=axs[3])#.axhline(100,color="red")
        axs[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))   
        
        
        # df["dd_B"].plot(label=f'B%', color="black",ax=axs[1,0])
        # df['dd_A'].plot(label=f'A%', color="blue",ax=axs[1,0])
        # df['dd_L'].plot(label=f'L%', color="green",ax=axs[1,0])
        # df['dd_S'].plot(label=f'S%', color="orange",ax=axs[1,0]).axhline(100,color="red")
        # axs[1,0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # axs[1,0].set_ylim([-100, 0])
        
        # # returns distribution
        # df['L_rpnl'].hist(bins=100,ax=axs[0,1],color="green")
        # df['S_rpnl'].hist(bins=100,ax=axs[1,1],color="orange")
        # # axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
 