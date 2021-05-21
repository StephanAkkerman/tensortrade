import numpy as np
import pandas as pd
import ray
import ray.rllib.agents.ppo as ppo
import matplotlib.pyplot as plt
import tensortrade.env.default as default
#import ta
import tensorflow as tf
#import torch

from ray import tune
from ray.tune.registry import register_env
from ray.tune.suggest.optuna import OptunaSearch

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
from tensortrade.data.cdd import CryptoDataDownload

def start():

    def create_env(config):

        cdd = CryptoDataDownload()
        # 2269 days of BTC daily candles available
        # amount of training data needs to match small market cap coin for fair comparison
        # 80% train
        if config["train"] == True:
            df = cdd.fetch("Bitfinex", "USD", "BTC", "d")[-1000:-200]
        # 20% test
        else: #for testing
            df = cdd.fetch("Bitfinex", "USD", "BTC", "d")[-1000:-200]

        price_history = df[['date', 'open', 'high', 'low', 'close', 'volume']]  

        # Add RSI to dataset
        #df['RSI'] = ta.momentum.RSIIndicator(close = df['close']).rsi()

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

        dataset=df

        #dataset = ta.add_best_ta_features2(df, high = 'high', low = 'low', close = 'close', volume = 'volume', fillna=True)
        #dataset = ta.add_trend_ta(df, high = 'high', low = 'low', close = 'close', fillna=True)

        # OBSERVER
        # Delete data and unix, since these columns are not necessary for the feed
        dataset.drop(columns=['date', 'unix'], inplace=True)

        with NameSpace("bitfinex"):
            streams = [Stream.source(dataset[c].tolist(), dtype="float").rename(c) for c in dataset.columns]

        feed = DataFeed(streams)

        # REWARDSCHEME
        #use PBR as reward_scheme
        #other options are SimpleProfit() or RiskAdjustedReturns()
        #reward_scheme = PBR(price=p) 
        reward_scheme = SimpleProfit()

        # ACTIONSCHEME
        #use BSH as action_scheme
        #ManagedRiskOrders() is also possible
        #action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme) 
        action_scheme = ManagedRiskOrders()

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

    # RL ALGO ranking, based on https://docs.ray.io/en/master/rllib-algorithms.html
    # 1. PPO
    # 2. IMPALA
    # 3. APPO   (it is often better to use standard PPO or IMPALA.)
    # 4. APE-X  (Better than DQN & A3C)
    # 5. DQN    (Better than A3C)
    # 6. A2C, A3C

    # Lookback window for the TradingEnv
    # Increasing this too much can result in errors
    window_size = 800
    # 1 meaning he cant lose anything 0 meaning it can lose everything
    max_allowed_loss = 0.8

    # Change to True for LSTM
    lstm = False

    config = {

    #For RLLib, PPO and Env
    "env" : "TradingEnv",
    "env_config" : {"window_size" : window_size,
                    "max_allowed_loss" : max_allowed_loss,
                    "train" : True,
                    },
    
    # RLLib parameters (https://docs.ray.io/en/master/rllib-training.html#common-parameters)
    # === Settings for Rollout Worker processes ===

    # Number of rollout worker actors to create for parallel sampling. Setting
    # this to 0 will force rollouts to be done in the trainer actor.
    # Increasing this increases parallelism, but can result in OOM errors
    #"num_workers" : 2,                  # max is 11

    # truncate_episodes: Each produced batch (when calling
    #   RolloutWorker.sample()) will contain exactly `rollout_fragment_length`
    #   steps. This mode guarantees evenly sized batches, but increases
    #   variance as the future return must now be estimated at truncation
    #   boundaries.
    # complete_episodes: Each unroll happens exactly over one episode, from
    #   beginning to end. Data collection will not stop unless the episode
    #   terminates or a configured horizon (hard or soft) is hit.
    #"batch_mode" : "complete_episodes",  # default = "truncate_episodes"

    # === Settings for the Trainer process ===
    # Number of GPUs to allocate to the trainer process. Does not work for all algorithms.
    #"num_gpus" : 1,                     # default = 0
    
    # Training batch size, if applicable. Should be >= rollout_fragment_length.
    # Making this too large makes it very slow, but it does not result in OOM errors
    #"train_batch_size" : 128,           # default = 4000

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
    #"observation_filter" : "NoFilter",    # default = "NoFilter

    # PPO-specific (https://docs.ray.io/en/master/rllib-algorithms.html#ppo)    
    # The GAE (lambda) parameter.
    # Always between 0 and 1, close to 1.
    # Usually small, 0.1 to 0.3.
    #"lambda" : 0.657,                    # default = 1.0

    # Learning rate schedule.
    #"lr_schedule" : [[0, 1e-1], [int(1e2), 1e-2], [int(1e3), 1e-3], [int(1e4), 1e-4], [int(1e5), 1e-5], [int(1e6), 1e-6], [int(1e7), 1e-7] ],

    # Coefficient of the entropy regularizer.
    #"entropy_coeff" : 0.022,             # default = 0

    # PPO clip parameter.
    # How far can the new policy go from the old policy while still profiting (improving the objective function)?
    #"clip_param" : 0.42,                 # default = 0.3

    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    #"vf_clip_param" : 98250.0,            # default = 10

    # LSTM parameters (https://docs.ray.io/en/master/rllib-models.html#default-model-config-settings)
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

    analysis = tune.run(
        "PPO",
        stop={
            #"episode_reward_mean": 500,
            "training_iteration": 2, # 300 is enough
        },
        config=config,
        checkpoint_at_end=True,
    )

    # https://docs.ray.io/en/master/tune/api_docs/analysis.html
    # Get checkpoint based on highest episode_reward_mean
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial(metric = "episode_reward_mean", mode = "max"),
        metric="episode_reward_mean",
    )
    checkpoint_path = checkpoints[0][0]

    # Restore agent (for renderer)
    agent = ppo.PPOTrainer(
        env="TradingEnv",
        config=config,
    )
    agent.restore(checkpoint_path)

    # Instantiate the environment (must have same settings for window_size and max_allowed_loss as the training env)
    env = create_env({
        "window_size": window_size,
        "max_allowed_loss": max_allowed_loss,
        "train" : False
    })

    # Run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()

    #LSTM (https://github.com/ray-project/ray/issues/13026)
    if (lstm == True):
        state = agent.get_policy().model.get_initial_state()

    #Do this till the stop in tune.run
    while not done:
        #LSTM
        if (lstm == True):
            action, state, logit = agent.compute_action(observation=obs, prev_action=1.0, prev_reward = 0.0, state = state)
        else:
            action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    env.render()

if __name__ == '__main__':
    # To prevent CUDNN_STATUS_ALLOC_FAILED error
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

    start()
    