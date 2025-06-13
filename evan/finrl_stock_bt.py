import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

sys.path.append(os.path.abspath(".."))

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

train = pd.read_csv('train_data.csv')
trade = pd.read_csv('trade_data.csv')

# If you are not using the data generated from part 1 of this tutorial, make sure 
# it has the columns and index in the form that could be make into the environment. 
# Then you can comment and skip the following lines.
train = train.set_index(train.columns[0])
train.index.names = ['']
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']

if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c", device='cpu') if if_using_a2c else None
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo", device='cpu') if if_using_ppo else None
trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}


e_trade_gym = StockTradingEnv(df = train, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()

df_account_value_a2c, df_actions_a2c, trajectories_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c, 
    environment = e_trade_gym, deterministic=False) if if_using_a2c else (None, None)

df_account_value_ddpg, df_actions_ddpg, trajectories_ddpg = DRLAgent.DRL_prediction(
    model=trained_ddpg, 
    environment = e_trade_gym) if if_using_ddpg else (None, None)

df_account_value_ppo, df_actions_ppo, trajectories_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo, 
    environment = e_trade_gym, deterministic=False) if if_using_ppo else (None, None)

df_account_value_td3, df_actions_td3, trajectories_td3 = DRLAgent.DRL_prediction(
    model=trained_td3, 
    environment = e_trade_gym) if if_using_td3 else (None, None)

df_account_value_sac, df_actions_sac, trajectories_sac = DRLAgent.DRL_prediction(
    model=trained_sac, 
    environment = e_trade_gym, deterministic=False) if if_using_sac else (None, None)

# df = pd.DataFrame(trajectories_sac)
# df.to_pickle("../data/trajectories_train_sac.pkl")
#
# df = pd.DataFrame(trajectories_ppo)
# df.to_pickle("../data/trajectories_train_ppo.pkl")

print(type(trajectories_sac))
print(list(trajectories_sac.keys()))

#########
# 儲存trajectories的地方
#########

trajectories_sac = [{
    k: np.array(v).reshape(-1) if k in ["dones", "terminals"] else np.array(v)
    for k, v in trajectories_sac.items()
}]
with open("../data/trajectories_train_sac-v0.pkl", "wb") as f:
    pickle.dump(trajectories_sac, f)

trajectories_ppo = [{
    k: np.array(v).reshape(-1) if k in ["dones", "terminals"] else np.array(v)
    for k, v in trajectories_ppo.items()
}]
with open("../data/trajectories_train_ppo-v0.pkl", "wb") as f:
    pickle.dump(trajectories_ppo, f)

def process_df_for_mvo(df):
    return df.pivot(index="date", columns="tic", values="close")

# Codes in this section partially refer to Dr G A Vijayalakshmi Pai

# https://www.kaggle.com/code/vijipai/lesson-5-mean-variance-optimization-of-portfolios/notebook

def StockReturnsComputing(StockPrice, Rows, Columns): 
    import numpy as np 
    StockReturn = np.zeros([Rows-1, Columns]) 
    for j in range(Columns):        # j: Assets 
        for i in range(Rows-1):     # i: Daily Prices 
            StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100 
            
    return StockReturn

StockData = process_df_for_mvo(train)
TradeData = process_df_for_mvo(trade)

TradeData.to_numpy()

#compute asset returns
arStockPrices = np.asarray(StockData)
[Rows, Cols]=arStockPrices.shape
arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

#compute mean returns and variance covariance matrix of returns
meanReturns = np.mean(arReturns, axis = 0)
covReturns = np.cov(arReturns, rowvar=False)
 
#set precision for printing results
np.set_printoptions(precision=3, suppress = True)

#display mean returns and variance-covariance matrix of returns
print('Mean returns of assets in k-portfolio 1\n', meanReturns)
print('Variance-Covariance matrix of returns\n', covReturns)

from pypfopt.efficient_frontier import EfficientFrontier

ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
raw_weights_mean = ef_mean.max_sharpe()
cleaned_weights_mean = ef_mean.clean_weights()
mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(len(cleaned_weights_mean))])
print(mvo_weights)

LastPrice = np.array([1/p for p in StockData.tail(1).to_numpy()[0]])
Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
print(Initial_Portfolio)

Portfolio_Assets = TradeData @ Initial_Portfolio
MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
print(MVO_result)

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2020-07-01'
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2021-10-29'

# df_dji = YahooDownloader(
#     start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, ticker_list=["dji"]
# ).fetch_data()
df_ndx = YahooDownloader(
    start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, ticker_list=["^NDX"]
).fetch_data()

df_ndx = df_ndx[["date", "close"]]
fst_day = df_ndx["close"][0]
ndx = pd.merge(
    df_ndx["date"],
    df_ndx["close"].div(fst_day).mul(1000000),
    how="outer",
    left_index=True,
    right_index=True,
).set_index("date")

print(f"ndx: {ndx}")

df_result_a2c = (
    df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
    if if_using_a2c
    else None
)
df_result_ddpg = (
    df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
    if if_using_ddpg
    else None
)
df_result_ppo = (
    df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
    if if_using_ppo
    else None
)
df_result_td3 = (
    df_account_value_td3.set_index(df_account_value_td3.columns[0])
    if if_using_td3
    else None
)
df_result_sac = (
    df_account_value_sac.set_index(df_account_value_sac.columns[0])
    if if_using_sac
    else None
)


df_account_value_a2c.to_csv("a2c_account_value.csv")
result = pd.DataFrame(
    {
        # "a2c": df_result_a2c["account_value"] if if_using_a2c else None,
        # "ddpg": df_result_ddpg["account_value"] if if_using_ddpg else None,
        # "ppo": df_result_ppo["account_value"] if if_using_ppo else None,
        # "td3": df_result_td3["account_value"] if if_using_td3 else None,
        "sac": df_result_sac["account_value"] if if_using_sac else None,
        "mvo": MVO_result["Mean Var"],
        "ndx": ndx["close"],
    }
)

print(result)

plt.rcParams["figure.figsize"] = (15,5)
plt.figure()
result.plot()
plt.savefig("finrl_plt.png", dpi=300, bbox_inches="tight")

