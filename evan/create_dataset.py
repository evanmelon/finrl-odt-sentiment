import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from env_stocktrading import StockTradingEnv
from models import DRLAgent
from config import INDICATORS, TRAINED_MODEL_DIR

trade = pd.read_csv('train_data.csv')
trade = trade.set_index(trade.columns[0])
trade.index.names = ['']
trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac")

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

e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', **env_kwargs)

obs_list, next_obs_list, action_list, reward_list, terminal_list = DRLAgent.data_prediction(
    model=trained_sac, 
    environment = e_trade_gym)
print(f'obs: {len(obs_list)}')
print(f'next obs: {len(next_obs_list)}')
print(f'action: {len(action_list)}')
print(f'reward: {len(reward_list)}')
print(f'ternimal: {len(terminal_list)}')
data_dict = {
    "observations": obs_list,
    "next_observations": next_obs_list,
    "actions": action_list,
    "rewards": reward_list,
    "terminals": terminal_list
}
print(data_dict['rewards'][0])
print(data_dict['observations'][0])
print(data_dict['next_observations'][0])
print(data_dict['actions'][0])