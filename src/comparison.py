import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ta
from datetime import datetime
from matplotlib.dates import DateFormatter


def benchmark(comparison_list, data_used, coin):
    """
    comparison_list: a chronological list of the net worth produced by the RL agent
    data_used: a pandas DataFrame constisting of the OHLCV data, with columns named 'open', 'high', 'low', 'close', 'volume'
    coin: a string of the name of the coin used
    """

    # BUY AND HOLD
    btcvalue = data_used["close"][0]
    btcinport = 100000 / btcvalue
    portfoliodf = btcinport * data_used["close"]
    portfoliodf.plot(figsize=(20, 8), label="Buy And Hold")

    # RSI
    # Add RSI to dataset
    data_used["RSI"] = ta.momentum.RSIIndicator(close=data_used["close"]).rsi()
    ohlc = data_used[["open", "high", "low", "close", "RSI"]]
    ohlc["positions"] = 0
    lower_barrier = 40
    upper_barrier = 60
    width = 20

    # Bullish Divergence
    # Start at 14, because RSI window = 14, so below that values wil be NaN
    for i in range(14, len(ohlc)):
        try:
            # Scouting for when the RSI goes under the 30 level.
            if ohlc.iat[i, 4] < lower_barrier:

                # Scout for when it resurfaces above it.
                for a in range(i + 1, i + width):
                    if ohlc.iat[a, 4] > lower_barrier:

                        # Scout for whenever the RSI dips again under the 30 while not going lower than the first dip.
                        # Simultaneously, the prices should be lower now than they were around the first dip.
                        for r in range(a + 1, a + width):
                            if (
                                ohlc.iat[r, 4] < lower_barrier
                                and ohlc.iat[r, 4] > ohlc.iat[i, 4]
                                and ohlc.iat[r, 3] < ohlc.iat[i, 3]
                            ):
                                # Scout for whenever the RSI resurfaces and completes the divergence pattern.
                                for s in range(r + 1, r + width):
                                    if ohlc.iat[s, 4] > lower_barrier:
                                        # Change value in positions column to 1 (meaning buy)
                                        ohlc["positions"][s + 1] = 1
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

                for a in range(i + 1, i + width):
                    if ohlc.iat[a, 4] < upper_barrier:

                        for r in range(a + 1, a + width):
                            if (
                                ohlc.iat[r, 4] > upper_barrier
                                and ohlc.iat[r, 4] < ohlc.iat[i, 4]
                                and ohlc.iat[r, 3] > ohlc.iat[i, 3]
                            ):
                                for s in range(r + 1, r + width):
                                    if ohlc.iat[s, 4] < upper_barrier:
                                        # Change value in Bear_div column to -1 (sell)
                                        ohlc["positions"][s + 1] = -1
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
                ohlc["positions"][i] = 0
            if buy == False:
                buy = True

        if check == -1:
            buy = False

            if sell == True:
                ohlc["positions"][i] = 0
            if sell == False:
                sell = True

    # Make ohlc['signal'] for backtesting
    ohlc["signal"] = 0
    bought = False
    for i in range(len(ohlc)):
        check = ohlc.iat[i, 5]

        if check == 1:
            bought = True

        if check == -1:
            bought = False

        if bought == True:
            ohlc["signal"][i] = 1

    # === Backtesting ===
    initial_capital = float(100000)

    # Find the amount to buy
    start = (ohlc.signal.values != 0).argmax()
    initial_price = ohlc.iloc[start, 3]
    initial_hold = initial_capital / initial_price

    # Positions keeps track of the amount owned
    positions = pd.DataFrame(index=ohlc.index).fillna(0.0)
    positions["Position in USD"] = ohlc["signal"] * initial_hold
    pos_diff = positions.diff()

    # Make a new dataframe to keep track of holdings and cash
    portfolio = pd.DataFrame()
    portfolio["holdings"] = (positions.multiply(ohlc["close"], axis=0)).sum(axis=1)
    portfolio["cash"] = (
        initial_capital
        - (pos_diff.multiply(ohlc["close"], axis=0)).sum(axis=1).cumsum()
    )
    portfolio["total"] = portfolio["cash"] + portfolio["holdings"]

    portfolio["total"].plot(label="RSI Divergence")

    # SMA
    short_window = 40
    long_window = 100

    signals = pd.DataFrame(index=data_used.index)
    signals["signal"] = 0.0

    # SMA of short window
    signals["short_mavg"] = (
        data_used["close"]
        .rolling(window=short_window, min_periods=1, center=False)
        .mean()
    )

    # SMA of long window
    signals["long_mavg"] = (
        data_used["close"]
        .rolling(window=long_window, min_periods=1, center=False)
        .mean()
    )

    # create signals for cross over
    signals["signal"][short_window:] = np.where(
        signals["short_mavg"][short_window:] > signals["long_mavg"][short_window:],
        1.0,
        0.0,
    )

    # Generate Trading orders
    signals["positions"] = signals["signal"].diff()

    # === Backtesting ===
    initial_capital = float(100000)

    # Find the amount to buy
    start = (signals.signal.values != 0).argmax()
    initial_price = data_used.iloc[start, 3]
    initial_hold = initial_capital / initial_price

    # Positions keeps track of the amount owned
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    positions["Position in USD"] = signals["signal"] * initial_hold
    pos_diff = positions.diff()

    # Make a new dataframe to keep track of holdings and cash
    portfolio = pd.DataFrame()
    portfolio["holdings"] = (positions.multiply(data_used["close"], axis=0)).sum(axis=1)
    portfolio["cash"] = (
        initial_capital
        - (pos_diff.multiply(data_used["close"], axis=0)).sum(axis=1).cumsum()
    )
    portfolio["total"] = portfolio["cash"] + portfolio["holdings"]

    portfolio["total"].plot(label="SMA Crossover")

    portfolio["RL Agent"] = comparison_list

    portfolio["RL Agent"].plot(label="RL Agent")

    plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    plt.legend()
    plt.title("Net worth trading " + coin)
    plt.tight_layout()
    plt.show()
