import os
import pandas as pd
import matplotlib.pyplot as plt

# **第一步：讀取 benchmark 數據**
evan_path = "./results/account_value_test_finrl-dt_train_offline_a2c_100_.csv"  # 你的 benchmark 檔案名稱
evan_df = pd.read_csv(evan_path, index_col=1, parse_dates=True)  # 設定日期索引

# **第二步：讀取策略結果**
results_map = {
    "A2C": pd.read_csv("results_csv/A2C.csv", index_col=0, parse_dates=True),
    # "DDPG": pd.read_csv("results_csv/DDPG.csv", index_col=0, parse_dates=True),
    # "PPO": pd.read_csv("results_csv/PPO.csv", index_col=0, parse_dates=True),
    # "TD3": pd.read_csv("results_csv/TD3.csv", index_col=0, parse_dates=True),
    # "SAC": pd.read_csv("results_csv/SAC.csv", index_col=0, parse_dates=True),
}

# td3_df = pd.read_csv("results_csv/TD3.csv", index_col=0, parse_dates=True)
a2c_df = pd.read_csv("results_csv/A2C.csv", index_col=0, parse_dates=True)
# finrl_td3_df = pd.read_csv("./finrl_result_csv/old/td3_account_value.csv", index_col=1, parse_dates=True)
# # **第三步：確保 benchmark 與策略數據的時間範圍一致**
# for strategy_name, df in results_map.items():
#     results_map[strategy_name] = df.reindex(evan_df.index).fillna(method='ffill')  # 確保索引對齊
# 讓所有策略的日期範圍與 benchmark 完全一致
# common_index = evan_df.index  # 取得 benchmark 的索引
#
# for strategy_name, df in results_map.items():
#     # 只保留 benchmark 和 strategy 共同的時間範圍
#     results_map[strategy_name] = df.loc[df.index.intersection(common_index)]
#
# # 讓 benchmark 也只保留與 strategy 相同的時間範圍
# evan_df = evan_df.loc[df.index.intersection(common_index)]
evan_df = evan_df.loc[a2c_df.index]

# **設定圖表**
plt.figure(figsize=(16, 9))

# **定義顏色與線條樣式**
color_palette = plt.get_cmap('tab10').colors
line_styles = ['-', '--', '-.', ':']

# **畫出 benchmark（黑色實線）**
plt.plot(evan_df.index, evan_df.iloc[:, 0], label="ODT", linestyle="-", color="black", linewidth=2)

# **畫出 TD3 原始數據**
plt.plot(a2c_df.index, a2c_df.iloc[:, 0], label="A2C", linestyle="--", color=color_palette[0])

# **畫出 TD3 變化版數據（假設是第 2 個欄位）**
plt.plot(a2c_df.index, a2c_df.iloc[:, 1], label="DT LoRA GPT2", linestyle="-.", color=color_palette[1])

# # **畫出 FinRL TD3 數據**
# plt.plot(finrl_td3_df.index, finrl_td3_df.iloc[:, 1], label="FinRL TD3", linestyle=":", color=color_palette[2])

# **設定標題與軸標籤**
plt.title("A2C Strategy, DT LoRA GPT2, ODT", fontsize=20, fontweight='bold')
plt.xlabel("Date", fontsize=16, fontweight='bold')
plt.ylabel("Total Asset Value ($)", fontsize=16, fontweight='bold')

# **設定 x 軸刻度**
plt.xticks(a2c_df.index[0::30], rotation=45)

# **設定圖例**
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)

# **儲存圖片**
save_name = "finrldt_vs_odt_a2c.png"
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.show()

print(f"Results saved as {save_name}")
