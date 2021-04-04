import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ta
from datetime import datetime
from matplotlib.dates import DateFormatter

# Plotly, only used in HODL
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# For getting the data
from BinanceData import fetchData
from tensortrade.data.cdd import CryptoDataDownload

# Get the data for the benchmarks
# The data needs to consist of the Open, High, Low, Close and Volume (OHLCV)

# Method one, use CryptoDataDownload
#cdd = CryptoDataDownload()
#ohlcv = cdd.fetch("Bitfinex", "USD", "BTC", "d")[-220:]    
#ohlcv.set_index('date', inplace = True)

# Method two, use Binance data 
# Found here: https://github.com/StephanAkkerman/TensorTrade
coin = "BAT"
ohlcv = fetchData(symbol=(coin + "USDT"), amount=2, timeframe='4h')
ohlcv.set_index('date', inplace = True)

# Matplotlib method for plotting candles in axs
def pltCandles(axs):
    
    width=0.4
    width2=0.1
    up=ohlcv[ohlcv.close>=ohlcv.open]
    down=ohlcv[ohlcv.close<ohlcv.open]

    axs.bar(up.index,up.close-up.open,width,bottom=up.open,color='g')
    axs.bar(up.index,up.high-up.close,width2,bottom=up.close,color='g')
    axs.bar(up.index,up.low-up.open,width2,bottom=up.open,color='g')
    axs.bar(down.index,down.close-down.open,width,bottom=down.open,color='r')
    axs.bar(down.index,down.high-down.open,width2,bottom=down.open,color='r')
    axs.bar(down.index,down.low-down.close,width2, bottom=down.close,color='r')

# Uses plotly
def HODL():      
    
    # Init 2 plots
    fig = make_subplots(rows=2, cols=1)

    # First plot is for the ohlcv data
    fig.add_trace(go.Candlestick(x=ohlcv.index,
                                open=ohlcv['open'],
                                high=ohlcv['high'],
                                low=ohlcv['low'],
                                close=ohlcv['close'], 
                                name = 'Price'), 
                                row=1, 
                                col=1)

    fig.update_layout(xaxis_rangeslider_visible=False)

    # Second plot is for the net worth
    fig.add_trace(go.Scatter(x=ohlcv.index, 
                             y=ohlcv['close'], 
                             mode='lines', 
                             name='Net Worth', 
                             marker={'color': 'DarkGreen'}), 
                             row=2, 
                             col=1)
    fig.show()    

# Uses matplotlib
def HODL2():

    fig, axs = plt.subplots(2, figsize=(20,8), gridspec_kw={'height_ratios': [2, 1]})

    pltCandles(axs[0])
    
    plt.grid()

    portfolio = 100000
    btcvalue = ohlcv['close'][0]
    btcinport = portfolio / btcvalue
    portfoliodf = btcinport * ohlcv['close']
    axs[1].plot(portfoliodf)

    axs[0].title.set_text('BTC Daily Candles')
    axs[1].title.set_text('Net Worth')   
    plt.tight_layout() 

# Based on  https://www.jasonlee.mobi/projects/2018/5/18/building-a-momentum-trading-strategy-using-python
def SMACrossover():

    # VARIABLES
    short_window = 40
    long_window = 100

    signals = pd.DataFrame(index = ohlcv.index)
    signals['signal'] = 0.0

    #SMA of short window
    signals['short_mavg'] = ohlcv['close'].rolling(window=short_window, min_periods=1, center=False).mean()
       
    #SMA of long window
    signals['long_mavg'] = ohlcv['close'].rolling(window=long_window, min_periods=1, center=False).mean()

    #create signals for cross over
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)

    # Generate Trading orders
    signals['positions'] = signals['signal'].diff()
    
    # initialize plot figure
    fig, axs = plt.subplots(2, figsize=(20,8), gridspec_kw={'height_ratios': [2, 1]})

    # Plots the price candles in axs[0]
    pltCandles(axs[0])
    plt.grid()

    # plot short and long moving averages
    axs[0].plot(signals[['short_mavg', 'long_mavg']], lw=2.)

    # plot buy signals
    axs[0].plot(signals.loc[signals.positions == 1.0].index,
                 signals.short_mavg[signals.positions == 1.0],
                 '^', markersize=20, color='g')

    # plot sell signals
    axs[0].plot(signals.loc[signals.positions == -1.0].index,
                signals.short_mavg[signals.positions == -1.0],
                 'v', markersize=20, color='r')

    # === Backtesting === 
    initial_capital = float(100000)

    # Find the amount to buy
    start = (signals.signal.values != 0).argmax()
    initial_price = ohlcv.iloc[start, 3]
    initial_hold = initial_capital / initial_price

    # Positions keeps track of the amount owned
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions['Position in USD'] = signals['signal'] * initial_hold
    pos_diff = positions.diff()

    # Make a new dataframe to keep track of holdings and cash
    portfolio = pd.DataFrame()
    portfolio['holdings'] = (positions.multiply(ohlcv['close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(ohlcv['close'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']

    axs[0].title.set_text('BTC Daily Candles')

    # plot equity curve in dollars 
    axs[1].plot(portfolio['total'])
    axs[1].title.set_text('Net Worth')
    axs[1].ticklabel_format(useOffset=False, axis='y')
    axs[1].margins(x=0)

    plt.tight_layout() 

# Based on https://radiant-brushlands-42789.herokuapp.com/kaabar-sofien.medium.com/back-testing-the-rsi-divergence-strategy-on-fx-in-python-c3680e7e2960
def RSIDiv():
    
    # Add RSI to dataset
    ohlcv['RSI'] = ta.momentum.RSIIndicator(close = ohlcv['close']).rsi()
    
    # Drop volume and add RSI
    ohlc = ohlcv[['open', 'high', 'low', 'close', 'RSI']]
    
    # 0 in these columns mean hold, 1 buy, -1 sell 
    # Set all values in positions to default: hold
    ohlc['positions'] = 0

    # === PARAMETERS ===
    # These are the bounds for the RSI, 30 & 70 are the default values on TradingView for the RSI
    lower_barrier = 40
    upper_barrier = 60

    # Increasing this can increase performance, but will make it slower to calculate
    width = 20

    # Bullish Divergence
    # Start at 14, because RSI window = 14, so below that values wil be NaN
    for i in range(14, len(ohlc)):
        try:
            #Scouting for when the RSI goes under the 30 level.
            if ohlc.iat[i, 4] < lower_barrier:

                #Scout for when it resurfaces above it.
                for a in range(i+1, i + width):
                    if ohlc.iat[a, 4] > lower_barrier:

                        #Scout for whenever the RSI dips again under the 30 while not going lower than the first dip.
                        #Simultaneously, the prices should be lower now than they were around the first dip. 
                        for r in range(a + 1, a + width):
                            if ohlc.iat[r, 4] < lower_barrier and ohlc.iat[r, 4] > ohlc.iat[i, 4] and ohlc.iat[r, 3] < ohlc.iat[i, 3]:
                               #Scout for whenever the RSI resurfaces and completes the divergence pattern.
                               for s in range(r + 1, r + width): 
                                    if ohlc.iat[s, 4] > lower_barrier:
                                        # Change value in positions column to 1 (meaning buy)
                                        ohlc['positions'][s + 1] = 1
                                        break
                                    else:
                                        continue
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
        except IndexError:
            pass

    # Bearish divergence
    for i in range(14, len(ohlc)):
        try:
            if ohlc.iat[i, 4] > upper_barrier:

                for a in range(i+1, i + width):
                    if ohlc.iat[a, 4] < upper_barrier:

                        for r in range(a + 1, a + width):
                            if ohlc.iat[r, 4] > upper_barrier and ohlc.iat[r, 4] < ohlc.iat[i, 4] and ohlc.iat[r, 3] > ohlc.iat[i, 3]:
                                for s in range(r + 1, r + width): 
                                    if ohlc.iat[s, 4] < upper_barrier:
                                        # Change value in Bear_div column to -1 (sell)
                                        ohlc['positions'][s + 1] = -1
                                        break
                                    else:
                                        continue
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
        except IndexError:
            pass

    # To fit the BSH Strategy
    # Keep the first buy signal after -1 (and in the beginning)
    # Keep first sell signal
    buy = False
    sell = True

    for i in range(len(ohlc)): 
        check = ohlc.iat[i, 5]

        if check == 1:
            sell = False

            if buy == True:
                ohlc['positions'][i] = 0
            if buy == False:
                buy = True
        
        if check == -1:
            buy = False

            if sell == True:
                ohlc['positions'][i] = 0
            if sell == False:
                sell = True

    # Make ohlc['signal'] for backtesting
    ohlc['signal'] = 0
    bought = False
    for i in range(len(ohlc)): 
        check = ohlc.iat[i, 5]

        if check == 1:
            bought = True

        if check == -1:
            bought = False

        if bought == True:
            ohlc['signal'][i] = 1   

    # === Backtesting === 
    initial_capital = float(100000)

    # Find the amount to buy
    start = (ohlc.signal.values != 0).argmax()
    initial_price = ohlc.iloc[start, 3]
    initial_hold = initial_capital / initial_price

    # Positions keeps track of the amount owned
    positions = pd.DataFrame(index=ohlc.index).fillna(0.0)
    positions['Position in USD'] = ohlc['signal'] * initial_hold
    pos_diff = positions.diff()

    # Make a new dataframe to keep track of holdings and cash
    portfolio = pd.DataFrame()
    portfolio['holdings'] = (positions.multiply(ohlc['close'], axis=0)).sum(axis=1)
    portfolio['cash'] = initial_capital - (pos_diff.multiply(ohlc['close'], axis=0)).sum(axis=1).cumsum()
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']

    # === PLOTTING === 
    # initialize plot figure
    # axs[0] for price and signals
    # axs[1] for RSI
    # axs[2] for portfolio value
    fig, axs = plt.subplots(3, figsize=(20,8), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Plots the price candles in axs[0]
    pltCandles(axs[0])
    plt.grid()

    # Plot buy signals
    axs[0].plot(ohlc.loc[ohlc.positions == 1.0].index,
                 ohlc.close[ohlc.positions == 1.0],
                 '^', markersize=20, color='g')

    # plot sell signals
    axs[0].plot(ohlc.loc[ohlc.positions == -1.0].index,
                 ohlc.close[ohlc.positions == -1.0],
                 'v', markersize=20, color='r')

    # plot rsi
    axs[1].plot(ohlc["RSI"], color='purple', lw=2.)

    # plot portfolio value
    axs[2].plot(portfolio['total'], lw=2.)

    #plt.show()
    axs[0].title.set_text('BTC Daily Candles')
    axs[1].title.set_text('RSI')
    axs[2].title.set_text('Net Worth')

    plt.tight_layout()

if __name__ == '__main__':     
    #HODL()
    #HODL2()
    #SMACrossover()
    RSIDiv()

    plt.show()
    