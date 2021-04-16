import numpy as np
import pandas as pd
import ray
import ray.rllib.agents.ppo as ppo
import matplotlib.pyplot as plt
import tensortrade.env.default as default
import ta
import tensorflow as tf

from gym.spaces import Discrete
from ray import tune
from ray.tune.registry import register_env
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest import ConcurrencyLimiter

from tensortrade.env.default.actions import TensorTradeActionScheme, SimpleOrders, ManagedRiskOrders, BSH
from tensortrade.env.default.rewards import TensorTradeRewardScheme, SimpleProfit, RiskAdjustedReturns, PBR
from tensortrade.env.default.renderers import PlotlyTradingChart, FileLogger
from tensortrade.env.generic import ActionScheme, TradingEnv, Renderer
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.feed.core import Stream, DataFeed, NameSpace
from tensortrade.core import Clock
from tensortrade.oms.instruments import ExchangePair, Instrument
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.orders import (
    Order,
    proportion_order,
    TradeSide,
    TradeType
)
from tensortrade.agents import DQNAgent
from tensortrade.data.cdd import CryptoDataDownload

def start():
    def create_env(config):

        cdd = CryptoDataDownload()
        # Head for beginning of data
        # Tail for end of data
        if config["train"] == True:
            df = cdd.fetch("Bitfinex", "USD", "BTC", "d")[-600:-100]
        else: #for testing
            df = cdd.fetch("Bitfinex", "USD", "BTC", "d")[-100:]
        price_history = df[['date', 'open', 'high', 'low', 'close', 'volume']]  # chart data

        # OBSERVER
        p =  Stream.source(price_history['close'].tolist(), dtype="float").rename("USD-BTC")

        bitfinex = Exchange("bitfinex", service=execute_order)(
            p
        )

        # ORDER MANAGEMENT SYSTEM
        #start with 100.000 usd and 0 assets
        cash = Wallet(bitfinex, 100000 * USD)
        asset = Wallet(bitfinex, 0 * BTC)

        portfolio = Portfolio(USD, [
            cash,
            asset
        ])

        # OBSERVER
        df.drop(columns=['date', 'open', 'high', 'low', 'close', 'volume'], inplace=True)

        with NameSpace("bitfinex"):
            streams = [Stream.source(df[c].tolist(), dtype="float").rename(c) for c in df.columns]

        feed = DataFeed(streams)

        # REWARDSCHEME
        #use PBR as reward_scheme
        #'SimpleProfit' object has no attribute 'on_action'
        reward_scheme = PBR(price=p) #SimpleProfit()#RiskAdjustedReturns()

        # ACTIONSCHEME
        #use BSH as action_scheme
        action_scheme = BSH(#ManagedRiskOrders()
            cash=cash,
            asset=asset
        ).attach(reward_scheme) 

        # RENDERER
        renderer_feed = DataFeed(
            [Stream.source(price_history[c].tolist(), dtype="float").rename(c) for c in price_history]
        )

        #create the environment
        environment = default.create(
            feed=feed,
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            renderer_feed=renderer_feed,
            renderer= PlotlyTradingChart(), #PositionChangeChart(), #
            window_size=config["window_size"], #part of OBSERVER
            max_allowed_loss=config["max_allowed_loss"] #STOPPER
        )
        return environment

    register_env("TradingEnv", create_env)

    #Now that the environment is registered;
    #We can run the training algorithm using the Proximal Policy Optimization (PPO) algorithm implemented in rllib.

    ######################
    ###   PARAMETERS   ###
    ######################

    # Lookback window for the TradingEnv
    #window_size = 100
    # 1 meaning he cant lose anything 0 meaning it can lose everything
    #max_allowed_loss = 0.5

    config = {

    #For RLLib, PPO and Env
    "env" : "TradingEnv",
    "env_config" : {"window_size" : tune.randint(1000,10000),
                    "max_allowed_loss" : tune.uniform(0.3,1),
                    "train" : True,
                    },
    
    # RLLib parameters (https://docs.ray.io/en/master/rllib-training.html#common-parameters)
    # === Settings for Rollout Worker processes ===

    # Number of rollout worker actors to create for parallel sampling. Setting
    # this to 0 will force rollouts to be done in the trainer actor.
    "num_workers" : 10,                  # max is 11

    # truncate_episodes: Each produced batch (when calling
    #   RolloutWorker.sample()) will contain exactly `rollout_fragment_length`
    #   steps. This mode guarantees evenly sized batches, but increases
    #   variance as the future return must now be estimated at truncation
    #   boundaries.
    # complete_episodes: Each unroll happens exactly over one episode, from
    #   beginning to end. Data collection will not stop unless the episode
    #   terminates or a configured horizon (hard or soft) is hit.
    "batch_mode" : "complete_episodes",  # default = "truncate_episodes"

    # === Settings for the Trainer process ===
    # Number of GPUs to allocate to the trainer process. Does not work for all algorithms.
    "num_gpus" : 1,                     # default = 0
    
    # Training batch size, if applicable. Should be >= rollout_fragment_length.
    "train_batch_size" : tune.uniform(5000,10000),           # default = 4000

    # === Environment Settings ===
    # Discount factor of the MDP.
    # Lower gamma values will put more weight on short-term gains, whereas higher gamma values will put more weight towards long-term gains. 
    "gamma" : tune.uniform(0.3, 1.0),       # default = 0.99

    # Whether to clip rewards during Policy's postprocessing.
    # True: r=sign(r): Fixed rewards -1.0, 1.0, or 0.0.
    "clip_rewards" : tune.choice([None,False,True]),              # default = None

    # Higher lr fits training model better, but causes overfitting  
    "lr" : tune.uniform(0.000005, 0.0005),                      # default = 5e-5

    # === Debug Settings ===
    "log_level" : "WARN",               # Can be "DEBUG" for more info
    "ignore_worker_failures" : True,

    # === Deep Learning Framework Settings ===
    "framework" : "torch",              # Can be tf, tfe and torch

    # === Advanced Rollout Settings ===
    # Element-wise observation filter, either "NoFilter" or "MeanStdFilter".
    "observation_filter" : tune.choice(["MeanStdFilter", "NoFilter"]),    # default = "NoFilter

    # PPO-specific (https://docs.ray.io/en/master/rllib-algorithms.html#ppo)    
    # The GAE (lambda) parameter.
    "lambda" : tune.uniform(0.5,2.0),                    # default = 1.0

    # Learning rate schedule.
    #"lr_schedule" : [[0, 1e-1], [int(1e2), 1e-2], [int(1e3), 1e-3], [int(1e4), 1e-4], [int(1e5), 1e-5], [int(1e6), 1e-6], [int(1e7), 1e-7] ],

    # Coefficient of the value function loss.
    "vf_loss_coeff" : tune.uniform(0.5,2.0),              # default = 1.0

    # Coefficient of the entropy regularizer.
    "entropy_coeff" : tune.uniform(0,0.1),             # default = 0

    # PPO clip parameter.
    "clip_param" : tune.uniform(0.2,0.5),                 # default = 0.3

    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param" : tune.uniform(10000,100000),            # default = 10

    # LSTM parameters (https://docs.ray.io/en/master/rllib-models.html#default-model-config-settings)
    "model" : {
        #"use_lstm" : True,             # default = False
        #"max_seq_len" : 10,            # default = 20
        #"lstm_cell_size" : 32,         # default = 256
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        "lstm_use_prev_action": False,  # default = False
        # Whether to feed r_{t-1} to LSTM.
        "lstm_use_prev_reward": False,  # default = False
        },
    }

    # https://docs.ray.io/en/master/tune/examples/optuna_example.html For Optuna implementation
    # https://docs.ray.io/en/master/tune/api_docs/suggestion.html#tune-optuna for basics
    optuna_search = OptunaSearch(
        metric="episode_reward_mean",
        mode="max")
    
    # https://docs.ray.io/en/master/tune/api_docs/suggestion.html#limiter for ConcurrencyLimiter
    #algo = ConcurrencyLimiter(optuna_search, max_concurrent = 4)

    # SCHEDULER?

    analysis = tune.run(
        "PPO",
        stop = {
          #"episode_reward_mean": 500,
          "training_iteration": 35, # 300 is enough
        },
        config = config,
        checkpoint_at_end = True,
        # For finding best hyperparameters
        num_samples = 30, #70 for 24h 
        search_alg = optuna_search,
        metric = "episode_reward_mean",
        mode = "max"
        #resources_per_trial={"gpu": 1, "cpu": 10} ???
    )
    # use tune.utils.wait_for_gpu for gpu memory issues

    print("Best hyperparameters found were: ", analysis.best_config)

    # https://docs.ray.io/en/master/tune/api_docs/analysis.html
    # Get checkpoint based on highest episode_reward_mean
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial(metric = "episode_reward_mean", mode = "max"),
        metric="episode_reward_mean",
    )
    checkpoint_path = checkpoints[0][0]

if __name__ == '__main__':
    # To prevent CUDNN_STATUS_ALLOC_FAILED error
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
     
    start()
