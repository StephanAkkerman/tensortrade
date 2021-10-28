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
from BinanceData import fetchData

# === Define custom instruments ===
# Precision of BAT on Binance is 4 (because 4 numbers follow the comma)
BAT = Instrument('BAT', 4, 'Basic Attention Token')

def start():

    # === Coin used in this run ===
    coin = "BAT"
    coinAsset = BAT

    # amount=1 -> 500 rows of data
    # Max 4500 rows in total for BAT
    candles = fetchData(symbol=(coin + "USDT"), amount=9, timeframe='4h')

    # Add prefix in case of multiple assets
    data = candles.add_prefix(coin + ":")

    # Divide the data in test (last 20%) and training (first 80%)
    dataEnd = (int)(len(data)*0.2)

    trainLength = (len(data) - dataEnd)

    # Print the amount of rows that are used for training and testing
    print("Training on " + (str)(trainLength) + " rows")
    print("Testing on " + (str)(dataEnd) + " rows")

    def create_env(config):

        # Use config param to decide which data set to use
        if config["train"] == True:
            df = data[:-dataEnd]
            envData = candles[:-dataEnd]
        else:
            df = data[-dataEnd:]
            envData = candles[-dataEnd:]
            
        # === OBSERVER ===
        p =  Stream.source(df[(coin + ':close')].tolist(), dtype="float").rename(("USD-" + coin))

        # === EXCHANGE ===
        # Commission on Binance is 0.075% on the lowest level, using BNB (https://www.binance.com/en/fee/schedule)
        binance_options = ExchangeOptions(commission=0.0075, min_trade_price=10.0)
        binance = Exchange("binance", service=execute_order, options=binance_options)(
            p
        )

        # === ORDER MANAGEMENT SYSTEM === 
        # Start with 100.000 usd and 0 assets
        cash = Wallet(binance, 100000 * USD)
        asset = Wallet(binance, 0 * coinAsset)

        portfolio = Portfolio(USD, [
            cash,
            asset
        ])
        ta.add_all_ta_features
        
        # === OBSERVER ===
        dataset = pd.DataFrame()

        # Use log-returns instead of raw OHLCV. This is a refined version of naive standarization
        # log(current_price / previous_price) = log(current_price) - log(previous_price)
        # If log value below 0 current_price > previous_price
        # Above 0 means current_price < previous_price 
        dataset['log_open'] = np.log(df[(coin + ':open')]) - np.log(df[(coin + ':open')].shift(1))
        dataset['log_low'] = np.log(df[(coin + ':low')]) - np.log(df[(coin + ':low')].shift(1))
        dataset['log_high'] = np.log(df[(coin + ':high')]) - np.log(df[(coin + ':high')].shift(1))
        dataset['log_close'] = np.log(df[(coin + ':close')]) - np.log(df[(coin + ':close')].shift(1))
        dataset['log_vol'] = np.log(df[(coin + ':volume')]) - np.log(df[(coin + ':volume')].shift(1))

        # === TECHNICAL ANALYSIS ===        
        # Technical analysis indicators that have a mean independent of OHCLV
        # RSI mean is always 50
        dataset['RSI'] = ta.momentum.RSIIndicator(close = df[(coin + ':close')]).rsi()

        BB_low = ta.volatility.BollingerBands(close = df[(coin + ':close')], window = 20).bollinger_lband()
        BB_mid = ta.volatility.BollingerBands(close = df[(coin + ':close')], window = 20).bollinger_mavg()
        BB_high = ta.volatility.BollingerBands(close = df[(coin + ':close')], window = 20).bollinger_hband()

        # Difference between close price and bollinger band
        dataset['BB_low'] =  np.log(BB_low) - np.log(df[(coin + ':close')])
        dataset['BB_mid'] =  np.log(BB_mid) - np.log(df[(coin + ':close')])
        dataset['BB_high'] =  np.log(BB_high) - np.log(df[(coin + ':close')])

        dataset = dataset.add_prefix(coin + ":")

        with NameSpace("binance"):
            streams = [Stream.source(dataset[c].tolist(), dtype="float").rename(c) for c in dataset.columns]

        # This is everything the agent gets to see, when making decisions
        feed = DataFeed(streams)

        # Compiles all the given stream together
        feed.compile()

        # Print feed for debugging
        print(feed.next())
        print(feed.next())
        print(feed.next())

        # === REWARDSCHEME === 
        # RiskAdjustedReturns rewards depends on return_algorithm and its parameters.
        # The risk-free rate is the return that you can expect from taking on zero risk.
        # A target return is what an investor would want to make from any capital invested in the asset.

        # SimpleProfit() or RiskAdjustedReturns() or PBR()
        #reward_scheme = RiskAdjustedReturns(return_algorithm='sortino')#, risk_free_rate=0, target_returns=0)
        #reward_scheme = RiskAdjustedReturns(return_algorithm='sharpe', risk_free_rate=0, target_returns=0, window_size=config["window_size"])

        reward_scheme = SimpleProfit(window_size=config["window_size"])      
        #reward_scheme = PBR(price=p)

        # === ACTIONSCHEME ===
        # SimpleOrders() or ManagedRiskOrders() or BSH()
        # ManagedRiskOrders stops halfway with trading, when test set is large
        # ManagedRiskOrders is bugged, with default settings!
        #action_scheme = ManagedRiskOrders(trade_sizes=[1])

        # durations=[10]    bad performance
        # durations=[50]    240k Better than [10]
        # durations=[100]   400k
        # durations=[500]   400k worse performance compared to [100]
        # durations=[1000]  400k but weird behavior

        action_scheme = ManagedRiskOrders(durations=[100])

        #action_scheme = SimpleOrders()     
        
        #BSH only works with PBR as reward_scheme
        #action_scheme = BSH(cash=cash,asset=asset).attach(reward_scheme) 

        # === RENDERER ===
        # Uses the OHCLV data passed to envData
        renderer_feed = DataFeed(
            [Stream.source(envData[c].tolist(), dtype="float").rename(c) for c in envData]
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

    # === ALGORITHMS ===
    # RL ALGO ranking, based on https://docs.ray.io/en/master/rllib-algorithms.html
    # Ranking made using BSH and PBR, amount = 1, maxIteration = 100, window_size = 10, max_allowed_loss = 0.95
    # 1. PPO 
    # 2. APPO 
    # 3. A2C
    # 4. A3C       
    # 5. IMPALA         
    # 6. DQN      
    
    # SimpleProfit & simpleOrders, amount = 1, maxIteration = 100, window_size = 10, max_allowed_loss = 0.95, gamma = 0
    #    ALGO   Test    Train
    # 1. IMPALA 95K     80k
    # 2. A2C    95k     80k
    # 3. PPO    85k     4M
    # 4. A3C    85k     85k
    # 5. DQN    80k     140k
    # 6. APPO   80k     75k
    
    # SimpleProfit & ManagedRiskOrders, amount = 1, maxIteration = 100, window_size = 10, max_allowed_loss = 0.95, gamma = 0
    #    ALGO   Test    Train
    # 1. PPO    112k    375k
    # 2. A3C    106k    300k
    # 3. A2C    100k    330k
    # 4. APPO   95k     285k
    # 5. IMPALA 95k     250k
    # 6. DQN    90k     266k  

    # Declare string and trainer of algorithm used by Ray
    algo = 'PPO'

    # === LSTM === 
    # Change to True for LSTM
    lstm = False

    # Declare when training can stop
    # Never more than 200
    maxIter = 100

    # === TRADING ENVIRONMENT CONFIG === 
    # Lookback window for the TradingEnv
    # Increasing this too much can result in errors and overfitting, also increases the duration necessary for training
    # Value needs to be bigger than 1, otherwise it will take nothing in consideration
    window_size = 10

    # 1 meaning he cant lose anything 0 meaning it can lose everything
    # Setting a high value results in quicker training time, but could result in overfitting
    # Needs to be bigger than 0.2 otherwise test environment will not render correctly.
    # 0.3 -> 90k
    # 0.4 -> 80k
    # 0.5 -> 70k
    # 0.6 -> 85k
    # 0.7 -> 75k
    # 0.8 -> 75k
    # 0.9 -> 80k
    # 1.0 -> 65k

    max_allowed_loss = 0.95
    
    # === CONFIG FOR AGENT ===
    config = {

    # === ENV Parameters === 
    "env" : "TradingEnv",
    "env_config" : {"window_size" : window_size,
                    "max_allowed_loss" : max_allowed_loss,
                    # Use the train set data
                    "train" : True,
                    },
    
    # === RLLib parameters ===
    # https://docs.ray.io/en/master/rllib-training.html#common-parameters

    # === Settings for Rollout Worker processes ===
    # Number of rollout worker actors to create for parallel sampling. Setting
    # this to 0 will force rollouts to be done in the trainer actor.
    # Increasing this increases parallelism, but can result in OOM errors
    #"num_workers" : 2,                     # Amount of CPU cores - 1
    "num_gpus": 1,

    # === Environment Settings ===
    # Discount factor of the MDP.
    # Lower gamma values will put more weight on short-term gains, whereas higher gamma values will put more weight towards long-term gains. 
    "gamma" : 0,       # default = 0.99

    # Higher lr fits training model better, but causes overfitting  
    #"lr" : 0.01,                           # default = 0.00005

    #Decreases performance
    #"clip_rewards": True, 
    #"observation_filter": "MeanStdFilter",
    #"lambda": 0.72,
    #"vf_loss_coeff": 0.5,

    #"entropy_coeff": 0.01,
    #"batch_mode": "complete_episodes",

    # === Debug Settings ===
    "log_level" : "WARN",                   # "WARN" or "DEBUG" for more info
    "ignore_worker_failures" : True,

    # === Deep Learning Framework Settings ===
    "framework" : "torch",                  # Can be tf, tfe and torch 

    # === Custom Metrics === 
    "callbacks": {"on_episode_end": get_net_worth},

    # === LSTM parameters ===
    # Currently not in use
    # https://docs.ray.io/en/master/rllib-models.html#default-model-config-settings
    "model" : {
        "use_lstm" : lstm,                  # default = False
        #"max_seq_len" : 10,                # default = 20
        #"lstm_cell_size" : 32,             # default = 256
        # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
        #"lstm_use_prev_action": False,     # default = False
        # Whether to feed r_{t-1} to LSTM.
        #"lstm_use_prev_reward": False,     # default = False
        },
    }

    # === Scheduler ===
    # Currenlty not in use
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

    # === tune.run for Training ===
    # https://docs.ray.io/en/master/tune/api_docs/execution.html
    analysis = tune.run(
        algo,
        # https://docs.ray.io/en/master/tune/api_docs/stoppers.html
        #stop = ExperimentPlateauStopper(metric="episode_reward_mean", std=0.1, top=10, mode="max", patience=0),
        stop = {"training_iteration": maxIter},
        #stop = {"episode_len_mean" : trainLength - 1},
        config=config,
        checkpoint_at_end=True,
        metric = "episode_reward_mean",
        mode = "max", 
        checkpoint_freq = 1,  #Necesasry to declare, in combination with Stopper
        checkpoint_score_attr = "episode_reward_mean",
        #resume=True
        #scheduler=asha_scheduler,
        #max_failures=5,
    )

    # === ANALYSIS FOR TESTING ===
    # https://docs.ray.io/en/master/tune/api_docs/analysis.html
    # Get checkpoint based on highest episode_reward_mean
    checkpoint_path = analysis.get_best_checkpoint(trial=analysis.get_best_trial("episode_reward_mean"), metric="episode_reward_mean", mode="max") 
    
    print("Checkpoint path at:")
    print(checkpoint_path)

    # === ALGORITHM SELECTION ===   
    # Get the correct trainer for the algorithm
    if (algo == "IMPALA"):
        algoTr = impala.ImpalaTrainer
    if (algo == "PPO"):
        algoTr = ppo.PPOTrainer
    if (algo == "APPO"):
        algoTr = appo.APPOTrainer
    if (algo == "DQN"):
        algoTr = dqn.DQNTrainer
    if (algo == "A2C"):
        algoTr = a2c.A2CTrainer
    if (algo == "A3C"):
        algoTr = a3c.A3CTrainer

    # === CREATE THE AGENT === 
    agent = algoTr(
        env="TradingEnv",
        config=config,
    )

    # Restore agent using best episode reward mean
    agent.restore(checkpoint_path)

    # Instantiate the testing environment
    # Must have same settings for window_size and max_allowed_loss as the training env
    test_env = create_env({
        "window_size": window_size,
        "max_allowed_loss": max_allowed_loss,
        # Use the test set data
        "train" : False
    })

    train_env = create_env({
        "window_size": window_size,
        "max_allowed_loss": max_allowed_loss,
        # Use the training set data
        "train" : True
    })

    # === Render the training data ===
    render_env(test_env, agent, lstm)
    render_env(train_env, agent, lstm)

def render_env(env, agent, lstm):
    # Run until done == True
    episode_reward = 0
    done = False
    obs = env.reset()

    # Get state, necessary for LSTM 
    # https://github.com/ray-project/ray/issues/13026
    if (lstm == True):
        state = agent.get_policy().model.get_initial_state()

    while not done:
        # Using LSTM
        if (lstm == True):
            action, state, logit = agent.compute_action(observation=obs, prev_action=1.0, prev_reward = 0.0, state = state)
        # Without LSTM
        else:
            action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    # Render the test environment
    env.render()

# === CALLBACK ===
def get_net_worth(info):
    # info is a dict containing: env, policy and 
    # info["episode"] is an evaluation episode
    episode = info["episode"]
    # DeprecationWarning: `callbacks dict interface` has been deprecated. Use `a class extending rllib.agents.callbacks.DefaultCallbacks` instead. 
    episode.custom_metrics["net_worth"] = episode.last_info_for()["net_worth"]

if __name__ == '__main__':
    # To prevent CUDNN_STATUS_ALLOC_FAILED error
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

    start()

    #   tensorboard --logdir=C:\Users\Stephan\ray_results\PPO
    #   tensorboard --logdir=C:\Users\Stephan\ray_results\IMPALA   
    