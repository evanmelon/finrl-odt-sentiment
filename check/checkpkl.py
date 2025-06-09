import pickle
import pprint
import numpy as np


# 替換成你的 .pkl 檔案路徑
# file_path = rf"data/ant-expert-v2.pkl"
# file_path = rf"data/agent_sac.pkl"
file_path = rf"../data/trajectories_ppo.pkl"
file_path_1 = rf"../data/trajectories_ppo-v0.pkl"

# 開啟並讀取 .pkl 檔案
with open(file_path, "rb") as file:
    data = pickle.load(file)
with open(file_path_1, "rb") as file:
    data1 = pickle.load(file)
print(data[0]["observations"])
print(data1[0]["observations"])
