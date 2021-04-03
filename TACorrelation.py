# For calculating
import numpy as np
import pandas as pd
import ta

# For showing the heatmap
import matplotlib.pyplot as plt
import seaborn as sns

# For fetching the data
from tensortrade.data.cdd import CryptoDataDownload
from BinanceData import fetch4Hour

def makePlot(type):

        # Fetch the data
        # Option 1: Use CryptoDataDownload
        #cdd = CryptoDataDownload()
        #df = cdd.fetch("Bitfinex", "USD", "BTC", "d")
        
        # Option 2: Use Binance data
        # Found here: https://github.com/StephanAkkerman/BinanceExtras/blob/master/BinanceData.py
        coin = "BAT"
        df = fetch4Hour(symbol=(coin + "USDT"), days=200)

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

        # Remove this from heatmap
        df = df.drop(index=['open', 'high', 'low', 'close', 'volume'], columns=['open', 'high', 'low', 'close', 'volume'])
        
        # Plot the heatmap
        plt.subplots(figsize=(10,10))
        # https://seaborn.pydata.org/tutorial/color_palettes.html for possible colors
        sns.heatmap(df, annot=False, linewidth=0, center=0.5, cmap=sns.color_palette("viridis", as_cmap=True))
        plt.tight_layout()
        plt.yticks(rotation=0)
        plt.show()

        # Show only the necessary indicators in the console
        df = df.mean(axis=0)
        print(df[df <= 0.4])

if __name__ == '__main__':
    makePlot('trend')