# Benchmarking 
This is a simple code that displays 3 simple trading strategies and their return. It can be used as a simple benchmark for your trading strategy.
These strategies are:
- Buy and hold
- SMA crossover
- RSI divergence

# BinanceData
To make the BinanceData.py function:
Change the 'publicKey' and 'privateKey' in Client on line 10 to your Binance API keys.

How to use:

Add the BinanceData.py to same directory you're working in.
Write: from BinanceData import fetchData
To get the latest 500 daily data points of OHLCV on the BTCUSDT pair, write: fetchData(symbol="BTCUSDT", amount=1, timeframe='1d')
Symbol can be any pair availble on Binance. Amount is the amount of rows you would like to have returned times 500, so amount=2 will return 1000 rows. Supported time frames are: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'.

Note: The volume is converted to USDT in this example, volume will always be converted to the second coin in a pair.

# TACorrelation
Displays a heatmap of absolute correlation of technical analysis indicators in the same group. 
