import numpy as np
import pandas as pd
import ray
import matplotlib.pyplot as plt
import tensortrade.env.default as default
import ta
import tensorflow as tf
import torch

import ray.rllib.agents.impala as impala
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ppo.appo as appo
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.dqn.apex as apex

from ray import tune
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler

from tensortrade.env.default.actions import TensorTradeActionScheme, SimpleOrders, ManagedRiskOrders, BSH
from tensortrade.env.default.rewards import TensorTradeRewardScheme, SimpleProfit, RiskAdjustedReturns, PBR
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger
from tensortrade.env.generic import ActionScheme, TradingEnv, Renderer
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.core import Clock
from tensortrade.oms.instruments import ExchangePair, Instrument
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.orders import (
    Order,
    proportion_order,
    TradeSide,
    TradeType
)
from BinanceData import fetch4Hour

# info is a dict containing: env, policy and episode
# info["episode"] is an evaluation episode
def on_episode_end(info):
    episode = info["episode"]
    # DeprecationWarning: `callbacks dict interface` has been deprecated. Use `a class extending rllib.agents.callbacks.DefaultCallbacks` instead. 
    episode.custom_metrics["net_worth"] = episode.last_info_for()["net_worth"]
    #episode.hist_data["net_worth"] = episode.last_info_for()["net_worth"]

def start():

    # Define custom instruments
    # Precision of BAT on Binance is 4 (because 4 numbers follow the comma)
    BAT = Instrument('BAT', 4, 'Basic Attention Token')
    coin = "BAT"
    coinAsset = BAT

    def create_env(config):           

        # 365 days results in 2500 rows of 4hour candle data
        # 1000 days are 6000 rows
        data = fetch4Hour(symbol=(coin + "USDT"), days=1000)

        # Divide the data in test (last 20%) and training (first 80%)
        dataEnd = (int)(len(data)*0.2)

        # Use config param to decide which data set to use
        if config["train"] == True:
            df = data[-2000:-400]
        else:
            df = data[-400:]

        # Used for renderer en Exchange 
        price_history = df[['date', 'open', 'high', 'low', 'close', 'volume']]  

        # OBSERVER
        p =  Stream.source(price_history['close'].tolist(), dtype="float").rename(("USD-" + coin))

        # Define the Exchange, with the options
        # Commission on Binance is 0.075% on the lowest level, using BNB (https://www.binance.com/en/fee/schedule)
        #binance_options = ExchangeOptions(commission=0.0075, min_trade_price=10.0)
        binance = Exchange("binance", service=execute_order)( #, options=binance_options
            p
        )

        # ORDER MANAGEMENT SYSTEM
        #start with 100.000 usd and 0 assets
        cash = Wallet(binance, 100000 * USD)
        asset = Wallet(binance, 0 * coinAsset)

        portfolio = Portfolio(USD, [
            cash,
            asset
        ])

        # Add RSI to dataset
        #df['RSI'] = ta.momentum.RSIIndicator(close = df['close']).rsi()
        #rsi[rsi < 40] = -1
        #rsi[(rsi >= 40) & (rsi <= 60)] = 0
        #rsi[rsi < 60] = 1
        #df['RSI'] = rsi

        df = ta.add_best_ta_features2(df, high = 'high', low = 'low', close = 'close', volume = 'volume', fillna=True)
        #df = ta.add_trend_ta(df, high = 'high', low = 'low', close = 'close', fillna=True)

        # OBSERVER
        # Delete date, since this is not necessary for the feed
        df.drop(columns=['date', 'open', 'high', 'low', 'close', 'volume'], inplace=True)
        #df.drop(columns=['date'], inplace=True)

        with NameSpace("binance"):
            streams = [Stream.source(df[c].tolist(), dtype="float").rename(c) for c in df.columns]

        # This is everything the agent gets to see, when making decisions
        feed = DataFeed(streams)

        # Compiles all the given stream together
        #feed.compile()

        # === REWARDSCHEME === 
        # RiskAdjustedReturns rewards depends on return_algorithm and its parameters.
        # The risk-free rate is the return that you can expect from taking on zero risk.
        # A target return is what an investor would want to make from any capital invested in the asset.

        # SimpleProfit() or RiskAdjustedReturns() or PBR()
        #reward_scheme = RiskAdjustedReturns(return_algorithm='sortino', risk_free_rate=0, target_returns=0)
        #reward_scheme = RiskAdjustedReturns(return_algorithm='sharpe', risk_free_rate=0, target_returns=0, window_size=config["window_size"])
        #reward_scheme = SimpleProfit(window_size=config["window_size"])      
        reward_scheme = PBR(price=p)

        # === ACTIONSCHEME ===
        # SimpleOrders() or ManagedRiskOrders() or BSH()
        #action_scheme = ManagedRiskOrders(trade_sizes=[1, 1/2])
        #action_scheme = SimpleOrders()     
        action_scheme = BSH(cash=cash,asset=asset).attach(reward_scheme) 

        # === RENDERER ===
        # Uses the OHCLV data passed to price_history
        renderer_feed = DataFeed(
            [Stream.source(price_history[c].tolist(), dtype="float").rename(c) for c in price_history]
        )

        # === RESULT === 
        environment = default.create(
            feed=feed,
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            renderer_feed=renderer_feed,
            renderer= PlotlyTradingChart(),             #PositionChangeChart()
            window_size=config["window_size"],          #part of OBSERVER
            max_allowed_loss=config["max_allowed_loss"] #STOPPER
        )
        return environment

    # Register the name for the environment
    register_env("TradingEnv", create_env)

    ######################
    ###   PARAMETERS   ###
    ######################

    # Lookback window for the TradingEnv
    # Increasing this too much, can result in errors (also makes it slow)
    window_size = 2000
    # 1 meaning he cant lose anything 0 meaning it can lose everything
    max_allowed_loss = 0.6

    # Change to True for LSTM
    lstm = False

    config = {

    # === ENV Parameters === 
    "env" : "TradingEnv",
    "env_config" : {"window_size" : window_size,
                    "max_allowed_loss" : max_allowed_loss,
                    "train" : True,
                    },
    
    # RLLib parameters 
    # https://docs.ray.io/en/master/rllib-training.html#common-parameters
    # === Settings for Rollout Worker processes ===

    # Number of rollout worker actors to create for parallel sampling. Setting
    # this to 0 will force rollouts to be done in the trainer actor.
    # Increasing this increases parallelism, but can result in OOM errors
    #"num_workers" : 2,                  # max is 11
    
    # Training batch size, if applicable. Should be >= rollout_fragment_length.
    # Making this too large makes it very slow, but it does not result in OOM errors
    #"train_batch_size" : 6000,           # default = 4000

    # === Environment Settings ===
    # Discount factor of the MDP.
    # Lower gamma values will put more weight on short-term gains, whereas higher gamma values will put more weight towards long-term gains. 
    #"gamma" : 0.55,       # default = 0.99

    # Whether to clip rewards during Policy's postprocessing.
    # True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
    #"clip_rewards" : False,              # default = None

    # Higher lr fits training model better, but causes overfitting  
    #"lr" : 0.00041,                      # default = 5e-5

    # === Debug Settings ===
    "log_level" : "WARN",               # "WARN" or "DEBUG" for more info
    "ignore_worker_failures" : True,

    # === Deep Learning Framework Settings ===
    "framework" : "torch",              # Can be tf, tfe and torch 

    # === Advanced Rollout Settings ===
    # Element-wise observation filter, either "NoFilter" or "MeanStdFilter".
    # MeanStdFilter Keeps track of a running mean for seen states
    #"observation_filter" : "NoFilter",    # default = "NoFilter

    # === IMPALA PARAMETERS ===
    # https://docs.ray.io/en/master/rllib-algorithms.html#impala 

    # === Custom Metrics === 
    "callbacks": {
        "on_episode_end": on_episode_end,
        },

    # === LSTM parameters ===
    # https://docs.ray.io/en/master/rllib-models.html#default-model-config-settings
    "model" : {
        "use_lstm" : lstm,             # default = False
        #"max_seq_len" : 10,            # default = 20
        #"lstm_cell_size" : 32,         # default = 256
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        #"lstm_use_prev_action": False,  # default = False
        # Whether to feed r_{t-1} to LSTM.
        #"lstm_use_prev_reward": False,  # default = False
        },
    }

    # === Scheduler ===
    # https://docs.ray.io/en/master/tune/api_docs/schedulers.html
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        metric='episode_reward_mean',
        mode='max',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1
       )

    ###########################
    ### ALGORITHM SELECTION ###
    ###########################
    
    # Declare string and trainer of algorithm used by Ray
    # IMPALA / PPO / APPO
    algo = "IMPALA"
    
    # RL ALGO ranking, based on https://docs.ray.io/en/master/rllib-algorithms.html
    # 1. PPO
    # 2. IMPALA
    # 3. APPO   (it is often better to use standard PPO or IMPALA.)
    # 4. APE-X  (Better than DQN & A3C)
    # 5. DQN    (Better than A3C)
    # 6. A2C, A3C

    # Get the correct trainer for the algorithm
    if (algo == "IMPALA"):
        algoTr = impala.ImpalaTrainer
    if (algo == "PPO"):
        algoTr = ppo.PPOTrainer
    if (algo == "APPO"):
        algoTr = appo.APPOTrainer
    if (algo == "APEX"):
        algoTr = apex.ApexTrainer
    if (algo == "DQN"):
        algoTr = dqn.DQNTrainer

    # Define the agent
    agent = algoTr(
        env="TradingEnv",
        config=config,
    )

    # === tune.run for Training ===
    # https://docs.ray.io/en/master/tune/api_docs/execution.html
    analysis = tune.run(
        algo,
        stop={
            #"episode_reward_mean": 500,
            "training_iteration": 10, # 300 is enough
        },
        config=config,
        checkpoint_at_end=True,
        metric = "episode_reward_mean",
        mode = "max",
        #resume=True
        #scheduler=asha_scheduler,
        #max_failures=
    )

    # === ANALYSIS FOR TESTING ===
    # https://docs.ray.io/en/master/tune/api_docs/analysis.html
    # Get checkpoint based on highest episode_reward_mean
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial(metric = "episode_reward_mean", mode = "max"),
        metric="episode_reward_mean",
    )
    checkpoint_path = checkpoints[0][0]

    # Restore agent using best episode reward mean
    agent.restore(checkpoint_path)

    # Instantiate the testing environment 
    # Must have same settings for window_size and max_allowed_loss as the training env
    env = create_env({
        "window_size": window_size,
        "max_allowed_loss": max_allowed_loss,
        "train" : False
    })

    # Run until done == True
    episode_reward = 0
    done = False
    obs = env.reset()

    # LSTM (https://github.com/ray-project/ray/issues/13026)
    if (lstm == True):
        state = agent.get_policy().model.get_initial_state()

    while not done:
        # LSTM
        if (lstm == True):
            action, state, logit = agent.compute_action(observation=obs, prev_action=1.0, prev_reward = 0.0, state = state)
        else:
            action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    # Render the test environment
    env.render()


if __name__ == '__main__':
    # To prevent CUDNN_STATUS_ALLOC_FAILED error
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

    start()
    
    #   tensorboard --logdir=C:\Users\Stephan\ray_results\IMPALA+