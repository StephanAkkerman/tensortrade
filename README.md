# TensorTrade
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MIT License](https://img.shields.io/github/license/StephanAkkerman/TensorTrade.svg?color=brightgreen)](https://opensource.org/licenses/MIT)

---

This is the reinforcement learning code I used for my thesis about how to trade low market capitulization cryptocurrencies.

## Features
- Fetches up to date historical data from Binance, using a custom script.
- Plots a comparison of the reinforcement learning agent and simple trading strategies (see section Images for more info).
- Some extras, such as an analysis of all TA indicators available for the TA library.

## Dependencies
The required packages to run this code can be found in the `requirements.txt` file. To run this file, execute the following code block:
```
$ pip install -r requirements.txt 
```
Alternatively, you can install the required packages manually like this:
```
$ pip install <package>
```

## How to run
- Clone the repository
- Run `$ python src/main.py`
- See result

# Images
## Comparison
After testing the RL agent a graph is plotted, showing the net worth of the agent compared to the benchmarks.
![Image of benchmark](https://github.com/StephanAkkerman/TensorTrade/blob/main/img/Picture1.png)

## TACorrelation
Displays a heatmap of absolute correlation of technical analysis indicators in the same group.
This is how the heatmap of trend indicators looks like.

![Image of heatmap](https://github.com/StephanAkkerman/TensorTrade/blob/main/img/Trend_heatmap.png)
