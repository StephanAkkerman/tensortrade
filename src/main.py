import numpy as np
import pandas as pd
import ray
import matplotlib.pyplot as plt
import tensortrade.env.default as default
import ta
import tensorflow as tf

import ray.rllib.agents.impala as impala
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ppo.appo as appo
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.a3c.a3c as a3c
import ray.rllib.agents.a3c.a2c as a2c

from ray import tune
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import ExperimentPlateauStopper
from ray.rllib.utils.exploration.epsilon_greedy import EpsilonGreedy

from tensortrade.env.default.actions import (
    TensorTradeActionScheme,
    SimpleOrders,
    ManagedRiskOrders,
    BSH,
)
from tensortrade.env.default.rewards import (
    TensorTradeRewardScheme,
    SimpleProfit,
    RiskAdjustedReturns,
    PBR,
)
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger
from tensortrade.env.generic import ActionScheme, TradingEnv, Renderer
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.core import Clock
from tensortrade.oms.instruments import ExchangePair, Instrument
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.orders import Order, proportion_order, TradeSide, TradeType
from data import fetchData
from comparison import benchmark

# === Define custom instruments ===
# Precision of BAT on Binance is 4 (because 4 digits follow the comma)
BAT = Instrument("BAT", 4, "Basic Attention Token")
NANO = Instrument("NANO", 4, "Nano")

# Used for the benchmark comparison
trainData = pd.DataFrame()
testData = pd.DataFrame()


def start():

    # === Coin used in this run ===
    coin = "BTC"
    coinInstrument = BTC

    # amount=1 -> 500 rows of data
    # Max 4500 rows in total for BAT
    candles = fetchData(symbol=(coin + "USDT"), amount=9, timeframe="4h")

    # Add prefix in case of multiple assets
    data = candles.add_prefix(coin + ":")

    # Divide the data in test (last 20%) and training (first 80%)
    dataEnd = (int)(len(data) * 0.2)

    trainLength = len(data) - dataEnd

    # Print the amount of rows that are used for training and testing
    print("Training on " + (str)(trainLength) + " rows")
    print("Testing on " + (str)(dataEnd) + " rows")

    # Used for benchmark
    trainData = candles[50:-dataEnd]
    trainData.set_index("date", inplace=True)
    testData = candles[-dataEnd:]
    testData.set_index("date", inplace=True)

    def create_env(config):

        # Use config param to decide which data set to use
        # Reserve 50 rows of data to fill in NaN values
        if config["train"] == True:
            df = data[50:-dataEnd]
            envData = candles[50:-dataEnd]
            taData = data[:-dataEnd]
        else:
            df = data[-dataEnd:]
            envData = candles[-dataEnd:]
            taData = data[-dataEnd - 50 :]

        # === OBSERVER ===
        p = Stream.source(df[(coin + ":close")].tolist(), dtype="float").rename(
            ("USD-" + coin)
        )

        # === EXCHANGE ===
        # Commission on Binance is 0.075% on the lowest level, using BNB (https://www.binance.com/en/fee/schedule)
        binance_options = ExchangeOptions(commission=0.0075, min_trade_price=10.0)
        binance = Exchange("binance", service=execute_order, options=binance_options)(p)

        # === ORDER MANAGEMENT SYSTEM ===
        # Start with 100.000 usd and 0 assets
        cash = Wallet(binance, 100000 * USD)
        asset = Wallet(binance, 0 * coinInstrument)

        portfolio = Portfolio(USD, [cash, asset])

        # === OBSERVER ===
        dataset = pd.DataFrame()

        # Use log-returns instead of raw OHLCV. This is a refined version of naive standarization
        # log(current_price / previous_price) = log(current_price) - log(previous_price)
        # If log value below 0 current_price > previous_price
        # Above 0 means current_price < previous_price
        dataset["log_open"] = np.log(taData[(coin + ":open")]) - np.log(
            taData[(coin + ":open")].shift(1)
        )
        dataset["log_low"] = np.log(taData[(coin + ":low")]) - np.log(
            taData[(coin + ":low")].shift(1)
        )
        dataset["log_high"] = np.log(taData[(coin + ":high")]) - np.log(
            taData[(coin + ":high")].shift(1)
        )
        dataset["log_close"] = np.log(taData[(coin + ":close")]) - np.log(
            taData[(coin + ":close")].shift(1)
        )
        dataset["log_vol"] = np.log(taData[(coin + ":volume")]) - np.log(
            taData[(coin + ":volume")].shift(1)
        )

        # === TECHNICAL ANALYSIS ===
        # Extra features not described in research, therefore not used.
        # BB_low = ta.volatility.BollingerBands(close = taData[(coin + ':close')], window = 20).bollinger_lband()
        # BB_mid = ta.volatility.BollingerBands(close = taData[(coin + ':close')], window = 20).bollinger_mavg()
        # BB_high = ta.volatility.BollingerBands(close = taData[(coin + ':close')], window = 20).bollinger_hband()

        # Difference between close price and bollinger band
        # dataset['BB_low'] =  np.log(BB_low) - np.log(taData[(coin + ':close')])
        # dataset['BB_mid'] =  np.log(BB_mid) - np.log(taData[(coin + ':close')])
        # dataset['BB_high'] =  np.log(BB_high) - np.log(taData[(coin + ':close')])

        # Take log-returns to standardize
        # Log-returns can not be used if value is 0 or smaller.
        # Use pct_change() instead
        # IDEA: Maybe use volume or close to standardize

        # This line is necessary otherwise read only errors shows up
        taData = taData.copy()

        # For some reasons there are erros when using pct_change() for these indicators
        adi = ta.volume.AccDistIndexIndicator(
            high=taData[(coin + ":high")],
            low=taData[(coin + ":low")],
            close=taData[(coin + ":close")],
            volume=taData[(coin + ":volume")],
        ).acc_dist_index()
        dataset["adi"] = adi.pct_change()

        fi = ta.volume.ForceIndexIndicator(
            close=taData[(coin + ":close")], volume=taData[(coin + ":volume")]
        ).force_index()
        dataset["fi"] = fi.pct_change()

        macd_diff = ta.trend.MACD(close=taData[(coin + ":close")]).macd_diff()
        dataset["macd_diff"] = macd_diff.pct_change()

        dpo = ta.trend.DPOIndicator(close=taData[(coin + ":close")]).dpo()
        dataset["dpo"] = dpo.pct_change()

        # Too many outliers in the dataset
        # vpt = ta.volume.VolumePriceTrendIndicator(close=taData[(coin + ':close')], volume=taData[(coin + ':volume')]).volume_price_trend()
        # dataset['vpt'] = vpt.pct_change()
        # em = ta.volume.EaseOfMovementIndicator(high=taData[(coin + ':high')], low=taData[(coin + ':low')], volume=taData[(coin + ':volume')]).ease_of_movement()
        # dataset['em'] = em.pct_change()

        kst_sig = ta.trend.KSTIndicator(close=taData[(coin + ":close")]).kst_sig()
        dataset["kst_sig"] = kst_sig.pct_change()

        kst_diff = ta.trend.KSTIndicator(close=taData[(coin + ":close")]).kst_diff()
        dataset["kst_diff"] = kst_diff.pct_change()

        nvi = ta.volume.NegativeVolumeIndexIndicator(
            close=taData[(coin + ":close")], volume=taData[(coin + ":volume")]
        ).negative_volume_index()
        dataset["nvi"] = np.log(nvi) - np.log(nvi.shift(1))

        bbw = ta.volatility.BollingerBands(
            close=taData[(coin + ":close")]
        ).bollinger_wband()
        dataset["bbw"] = np.log(bbw) - np.log(bbw.shift(1))

        kcw = ta.volatility.KeltnerChannel(
            high=taData[(coin + ":high")],
            low=taData[(coin + ":low")],
            close=taData[(coin + ":close")],
        ).keltner_channel_wband()
        dataset["kcw"] = np.log(kcw) - np.log(kcw.shift(1))

        dcw = ta.volatility.DonchianChannel(
            high=taData[(coin + ":high")],
            low=taData[(coin + ":low")],
            close=taData[(coin + ":close")],
        ).donchian_channel_wband()
        dataset["dcw"] = np.log(dcw) - np.log(dcw.shift(1))

        psar_up = ta.trend.PSARIndicator(
            high=taData[(coin + ":high")],
            low=taData[(coin + ":low")],
            close=taData[(coin + ":close")],
        ).psar_up()
        dataset["psar_up"] = np.log(psar_up) - np.log(psar_up.shift(1))

        # These indicators have a mean independent of the OHLCV data
        # IDEA: Use log-returns on these as an extra indicator
        # Has a mean of 0
        dataset["cmf"] = ta.volume.ChaikinMoneyFlowIndicator(
            high=taData[(coin + ":high")],
            low=taData[(coin + ":low")],
            close=taData[(coin + ":close")],
            volume=taData[(coin + ":volume")],
        ).chaikin_money_flow()
        dataset["ppo"] = ta.momentum.PercentagePriceOscillator(
            close=taData[(coin + ":close")]
        ).ppo()
        dataset["ppo_signal"] = ta.momentum.PercentagePriceOscillator(
            close=taData[(coin + ":close")]
        ).ppo_signal()
        dataset["ppo_hist"] = ta.momentum.PercentagePriceOscillator(
            close=taData[(coin + ":close")]
        ).ppo_hist()
        dataset["ui"] = ta.volatility.UlcerIndex(
            close=taData[(coin + ":close")]
        ).ulcer_index()
        dataset["aroon_ind"] = ta.trend.AroonIndicator(
            close=taData[(coin + ":close")]
        ).aroon_indicator()

        # Indicator, so has value 0 or 1
        dataset["bbhi"] = ta.volatility.BollingerBands(
            close=taData[(coin + ":close")]
        ).bollinger_hband_indicator()
        dataset["bbli"] = ta.volatility.BollingerBands(
            close=taData[(coin + ":close")]
        ).bollinger_lband_indicator()
        dataset["kchi"] = ta.volatility.KeltnerChannel(
            high=taData[(coin + ":high")],
            low=taData[(coin + ":low")],
            close=taData[(coin + ":close")],
        ).keltner_channel_hband_indicator()
        dataset["kcli"] = ta.volatility.KeltnerChannel(
            high=taData[(coin + ":high")],
            low=taData[(coin + ":low")],
            close=taData[(coin + ":close")],
        ).keltner_channel_lband_indicator()

        # Has a mean of 50
        dataset["stoch_rsi"] = ta.momentum.StochRSIIndicator(
            close=taData[(coin + ":close")]
        ).stochrsi()
        dataset["stoch_rsi_d"] = ta.momentum.StochRSIIndicator(
            close=taData[(coin + ":close")]
        ).stochrsi_d()
        dataset["stoch_rsi_k"] = ta.momentum.StochRSIIndicator(
            close=taData[(coin + ":close")]
        ).stochrsi_k()
        dataset["uo"] = ta.momentum.UltimateOscillator(
            high=taData[(coin + ":high")],
            low=taData[(coin + ":low")],
            close=taData[(coin + ":close")],
        ).ultimate_oscillator()
        dataset["adx"] = ta.trend.ADXIndicator(
            high=taData[(coin + ":high")],
            low=taData[(coin + ":low")],
            close=taData[(coin + ":close")],
        ).adx()
        dataset["mass_index"] = ta.trend.MassIndex(
            high=taData[(coin + ":high")], low=taData[(coin + ":low")]
        ).mass_index()
        dataset["aroon_up"] = ta.trend.AroonIndicator(
            close=taData[(coin + ":close")]
        ).aroon_up()
        dataset["aroon_down"] = ta.trend.AroonIndicator(
            close=taData[(coin + ":close")]
        ).aroon_down()
        dataset["stc"] = ta.trend.STCIndicator(close=taData[(coin + ":close")]).stc()

        # Lot of NaN values
        # ta.trend.PSARIndicator(high=df[(coin + ':high')], low=df[(coin + ':low')], close=df[(coin + ':close')]).psar_down()

        dataset = dataset.add_prefix(coin + ":")

        # Drop first 50 rows from dataset
        dataset = dataset.iloc[50:]

        with NameSpace("binance"):
            streams = [
                Stream.source(dataset[c].tolist(), dtype="float").rename(c)
                for c in dataset.columns
            ]

        # This is everything the agent gets to see, when making decisions
        feed = DataFeed(streams)

        # Compiles all the given stream together
        feed.compile()

        # Print feed for debugging
        # print(feed.next())
        # print(feed.next())
        # print(feed.next())

        # === REWARDSCHEME ===
        # RiskAdjustedReturns rewards depends on return_algorithm and its parameters.
        # The risk-free rate is the return that you can expect from taking on zero risk.
        # A target return is what an investor would want to make from any capital invested in the asset.

        # SimpleProfit() or RiskAdjustedReturns() or PBR()
        # reward_scheme = RiskAdjustedReturns(return_algorithm='sortino')#, risk_free_rate=0, target_returns=0)
        # reward_scheme = RiskAdjustedReturns(return_algorithm='sharpe', risk_free_rate=0, target_returns=0, window_size=config["window_size"])

        reward_scheme = SimpleProfit(window_size=config["window_size"])
        # reward_scheme = PBR(price=p)

        # === ACTIONSCHEME ===
        # SimpleOrders() or ManagedRiskOrders() or BSH()

        # ManagedRiskOrders is bad, with default settings!
        # To use ManagedRiskOrders use settings like these:
        # ManagedRiskOrders(stop = [0.02], take = [0.03], trade_sizes=2,)

        action_scheme = ManagedRiskOrders(durations=[100])

        # action_scheme = SimpleOrders()

        # BSH only works with PBR as reward_scheme
        # action_scheme = BSH(cash=cash,asset=asset).attach(reward_scheme)

        # === RENDERER ===
        # Uses the OHCLV data passed to envData
        renderer_feed = DataFeed(
            [
                Stream.source(envData[c].tolist(), dtype="float").rename(c)
                for c in envData
            ]
        )

        # === RESULT ===
        environment = default.create(
            feed=feed,
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            renderer_feed=renderer_feed,
            renderer=PlotlyTradingChart(),  # PositionChangeChart()
            window_size=config["window_size"],  # part of OBSERVER
            max_allowed_loss=config["max_allowed_loss"],  # STOPPER
        )
        return environment

    # Register the name for the environment
    register_env("TradingEnv", create_env)

    ######################
    ###   PARAMETERS   ###
    ######################

    # === ALGORITHMS ===
    # RL ALGO ranking, algorithms found on https://docs.ray.io/en/master/rllib-algorithms.html
    # SimpleProfit & simpleOrders, amount = 1, maxIteration = 100, window_size = 10, max_allowed_loss = 0.95, gamma = 0
    #    ALGO   Test    Train
    # 1. IMPALA 95K     80k
    # 2. A2C    95k     80k
    # 3. PPO    85k     4M
    # 4. A3C    85k     85k
    # 5. DQN    80k     140k
    # 6. APPO   80k     75k

    # SimpleProfit & ManagedRiskOrders(durations=[100]), amount = 1, maxIteration = 100, window_size = 10, max_allowed_loss = 0.95, gamma = 0, BAT 4h 900 rows of data
    #    ALGO   Net Worth
    # 4. PPO    421k
    # 2. A3C    432k
    # 1. A2C    456k
    # 6. APPO   372k
    # 5. IMPALA 403k
    # 3. DQN    424k

    # Declare string and trainer of algorithm used by Ray
    algo = "A2C"

    # === LSTM ===
    # Change to True for LSTM
    lstm = False

    # Declare when training can stop
    # Never more than 200
    maxIter = 5

    # === TRADING ENVIRONMENT CONFIG ===
    # Lookback window for the TradingEnv
    # Increasing this too much can result in errors and overfitting, also increases the duration necessary for training
    # Value needs to be bigger than 1, otherwise it will take nothing in consideration
    window_size = 10

    # 1 meaning he cant lose anything 0 meaning it can lose everything
    # Setting a high value results in quicker training time, but could result in overfitting
    # Needs to be bigger than 0.2 otherwise test environment will not render correctly.
    max_allowed_loss = 0.95

    # === CONFIG FOR AGENT ===
    config = {
        # === ENV Parameters ===
        "env": "TradingEnv",
        "env_config": {
            "window_size": window_size,
            "max_allowed_loss": max_allowed_loss,
            # Use the train set data
            "train": True,
        },
        # === RLLib parameters ===
        # https://docs.ray.io/en/master/rllib-training.html#common-parameters
        # === Settings for Rollout Worker processes ===
        # Number of rollout worker actors to create for parallel sampling. Setting
        # this to 0 will force rollouts to be done in the trainer actor.
        # Increasing this increases parallelism, but can result in OOM errors
        # "num_workers" : 2,                     # Amount of CPU cores - 1
        "num_gpus": 1,
        # === Environment Settings ===
        # Discount factor of the MDP.
        # Lower gamma values will put more weight on short-term gains, whereas higher gamma values will put more weight towards long-term gains.
        "gamma": 0,  # default = 0.99
        # Higher lr fits training model better, but causes overfitting
        # "lr" : 0.01,                           # default = 0.00005
        # Decreases performance
        # "clip_rewards": True,
        # "observation_filter": "MeanStdFilter",
        # "lambda": 0.72,
        # "vf_loss_coeff": 0.5,
        # "entropy_coeff": 0.01,
        # "batch_mode": "complete_episodes",
        # === Debug Settings ===
        "log_level": "WARN",  # "WARN" or "DEBUG" for more info
        "ignore_worker_failures": True,
        # === Deep Learning Framework Settings ===
        "framework": "torch",  # Can be tf, tfe and torch
        # === Custom Metrics ===
        "callbacks": {"on_episode_end": get_net_worth},
        # === LSTM parameters ===
        # Currently not in use
        # https://docs.ray.io/en/master/rllib-models.html#default-model-config-settings
        "model": {
            "use_lstm": lstm,  # default = False
            # "max_seq_len" : 10,                # default = 20
            # "lstm_cell_size" : 32,             # default = 256
            # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
            # "lstm_use_prev_action": False,     # default = False
            # Whether to feed r_{t-1} to LSTM.
            # "lstm_use_prev_reward": False,     # default = False
        },
    }

    # === Scheduler ===
    # Currenlty not in use
    # https://docs.ray.io/en/master/tune/api_docs/schedulers.html
    asha_scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="episode_reward_mean",
        mode="max",
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1,
    )

    # === tune.run for Training ===
    # https://docs.ray.io/en/master/tune/api_docs/execution.html
    analysis = tune.run(
        algo,
        # https://docs.ray.io/en/master/tune/api_docs/stoppers.html
        # stop = ExperimentPlateauStopper(metric="episode_reward_mean", std=0.1, top=10, mode="max", patience=0),
        stop={"training_iteration": maxIter},
        # stop = {"episode_len_mean" : trainLength - 1},
        config=config,
        checkpoint_at_end=True,
        metric="episode_reward_mean",
        mode="max",
        checkpoint_freq=1,  # Necesasry to declare, in combination with Stopper
        checkpoint_score_attr="episode_reward_mean",
        # resume=True
        # scheduler=asha_scheduler,
        # max_failures=5,
    )

    # === ANALYSIS FOR TESTING ===
    # https://docs.ray.io/en/master/tune/api_docs/analysis.html
    # Get checkpoint based on highest episode_reward_mean
    checkpoint_path = analysis.get_best_checkpoint(
        trial=analysis.get_best_trial("episode_reward_mean"),
        metric="episode_reward_mean",
        mode="max",
    )

    print("Checkpoint path at:")
    print(checkpoint_path)

    # === ALGORITHM SELECTION ===
    # Get the correct trainer for the algorithm
    if algo == "IMPALA":
        algoTr = impala.ImpalaTrainer
    if algo == "PPO":
        algoTr = ppo.PPOTrainer
    if algo == "APPO":
        algoTr = appo.APPOTrainer
    if algo == "DQN":
        algoTr = dqn.DQNTrainer
    if algo == "A2C":
        algoTr = a2c.A2CTrainer
    if algo == "A3C":
        algoTr = a3c.A3CTrainer

    # === CREATE THE AGENT ===
    agent = algoTr(env="TradingEnv", config=config,)

    # Restore agent using best episode reward mean
    agent.restore(checkpoint_path)

    # Instantiate the testing environment
    # Must have same settings for window_size and max_allowed_loss as the training env
    test_env = create_env(
        {
            "window_size": window_size,
            "max_allowed_loss": max_allowed_loss,
            # Use the test set data
            "train": False,
        }
    )

    train_env = create_env(
        {
            "window_size": window_size,
            "max_allowed_loss": max_allowed_loss,
            # Use the training set data
            "train": True,
        }
    )

    print(testData)

    # === Render the environments ===
    render_env(test_env, agent, lstm, testData, coin)
    # render_env(train_env, agent, lstm, trainData)


def render_env(env, agent, lstm, data, asset):
    # Run until done == True
    episode_reward = 0
    done = False
    obs = env.reset()

    # Get state, necessary for LSTM
    # https://github.com/ray-project/ray/issues/13026
    if lstm == True:
        state = agent.get_policy().model.get_initial_state()

    # Start with initial capital
    networth = [100000]

    while not done:
        # Using LSTM
        if lstm == True:
            action, state, logit = agent.compute_action(
                observation=obs, prev_action=1.0, prev_reward=0.0, state=state
            )
        # Without LSTM
        else:
            action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        networth.append(info["net_worth"])

    # Render the test environment
    env.render()

    benchmark(comparison_list=networth, data_used=data, coin=asset)


# === CALLBACK ===
def get_net_worth(info):
    # info is a dict containing: env, policy and
    # info["episode"] is an evaluation episode
    episode = info["episode"]
    # DeprecationWarning: `callbacks dict interface` has been deprecated. Use `a class extending rllib.agents.callbacks.DefaultCallbacks` instead.
    episode.custom_metrics["net_worth"] = episode.last_info_for()["net_worth"]


if __name__ == "__main__":
    # To prevent CUDNN_STATUS_ALLOC_FAILED error
    tf.config.experimental.set_memory_growth(
        tf.config.experimental.list_physical_devices("GPU")[0], True
    )

    start()

    #   tensorboard --logdir=C:\Users\Stephan\ray_results\PPO
    #   tensorboard --logdir=C:\Users\Stephan\ray_results\APPO
    #   tensorboard --logdir=C:\Users\Stephan\ray_results\IMPALA
    #   tensorboard --logdir=C:\Users\Stephan\ray_results\A2C
