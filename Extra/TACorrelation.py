# For calculating
import numpy as np
import pandas as pd
import ta

# For showing the heatmap
import matplotlib.pyplot as plt
import seaborn as sns

# For fetching the data
from tensortrade.data.cdd import CryptoDataDownload
from BinanceData import fetchData

def makePlot(type):

    # Fetch the data
    # Option 1: Use CryptoDataDownload
    #cdd = CryptoDataDownload()
    #df = cdd.fetch("Bitfinex", "USD", "BTC", "d")
        
    # Option 2: Use Binance data
    # Found here: https://github.com/StephanAkkerman/TensorTrade
    coin = "BAT"
    df = fetchData(symbol=(coin + "USDT"), amount=2, timeframe='4h')

    # Drop unix and set 'date' as index
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df = df.set_index("date")

    # Calculate absolute correlation and clean up names
    if (type == 'momentum'):
        df = ta.add_momentum_ta(df, high = 'high', low = 'low', close = 'close', volume = 'volume').corr().abs()
        df.columns = df.columns.str.replace('momentum_', '')
        df.index = df.index.str.replace('momentum_', '')

    if (type == 'volume'):
        df = ta.add_volume_ta(df, high = 'high', low = 'low', close = 'close', volume = 'volume').corr().abs()
        df.columns = df.columns.str.replace('volume_', '')
        df.index = df.index.str.replace('volume_', '')

    if (type == 'trend'):
        df = ta.add_trend_ta(df, high = 'high', low = 'low', close = 'close').corr().abs()
        df.columns = df.columns.str.replace('trend_', '')
        df.index = df.index.str.replace('trend_', '')

    if (type == 'volatility'):
        df = ta.add_volatility_ta(df, high = 'high', low = 'low', close = 'close').corr().abs()
        df.columns = df.columns.str.replace('volatility_', '')
        df.index = df.index.str.replace('volatility_', '')

    if (type == 'others'):
        df = ta.add_others_ta(df, close = 'close').corr().abs()
        df.columns = df.columns.str.replace('volumeothers_', '')
        df.index = df.index.str.replace('volumeothers_', '')

    if (type == 'all'):
        df = ta.add_all_ta_features(df, open = 'open', high = 'high', low = 'low', close = 'close', volume = 'volume').corr().abs()

    # Remove this from heatmap
    df = df.drop(index=['open', 'high', 'low', 'close', 'volume'], columns=['open', 'high', 'low', 'close', 'volume'])
        
    # Plot the heatmap
    plt.subplots(figsize=(10,10))
    # https://seaborn.pydata.org/tutorial/color_palettes.html for possible colors
    sns.heatmap(df, annot=False, linewidth=0, center=0.5, cmap=sns.color_palette("viridis", as_cmap=True))
    plt.tight_layout()
    plt.yticks(rotation=0)
    plt.show()

def correlation():
    coin = "BAT"
    df = fetchData(symbol=(coin + "USDT"), amount=2, timeframe='4h')

    # Drop unix and set 'date' as index
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    df = df.set_index("date")

    df = ta.add_all_ta_features(df, open = 'open', high = 'high', low = 'low', close = 'close', volume = 'volume').corr().abs()
    df = df.drop(index=['open', 'high', 'low', 'close', 'volume'], columns=['open', 'high', 'low', 'close', 'volume'])

    # Show only the necessary indicators in the console
    df = df.mean(axis=0)
    print(df[df <= 0.3])

def getLength():
    coin = "BAT"
    df = fetchData(symbol=(coin + "USDT"), amount=2, timeframe='4h')

    # Drop unix and set 'date' as index
    df = df[['date', 'open', 'high', 'low', 'close', 'volume']] #5 indicators (date not included in count)
    df = df.set_index("date")
    test = df.copy()
    fresh1 = df.copy()
    fresh2 = df.copy()
    fresh3 = df.copy()
    fresh4 = df.copy()

    momentum = ta.add_momentum_ta(fresh1, high = 'high', low = 'low', close = 'close', volume = 'volume')
    volume = ta.add_volume_ta(fresh2, high = 'high', low = 'low', close = 'close', volume = 'volume')
    trend = ta.add_trend_ta(fresh3, high = 'high', low = 'low', close = 'close')
    volatility = ta.add_volatility_ta(fresh4, high = 'high', low = 'low', close = 'close')

    print("Amount of momentum indicators: " + str(len(momentum.columns) - 5))
    print("Amount of volume indicators: " + str(len(volume.columns) - 5))
    print("Amount of trend indicators: " + str(len(trend.columns) - 5))
    print("Amount of volatility indicators: " + str(len(volatility.columns) - 5))

if __name__ == '__main__':
    correlation()