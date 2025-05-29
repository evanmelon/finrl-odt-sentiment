import pandas as pd


df = pd.read_csv("../sentiment_data/analyst_ratings_processed.csv", index_col=0)
print(df)
# 確保是字串
df["date"] = df["date"].astype(str)

years = pd.to_numeric(df["date"].str[:4], errors='coerce')  # 無法轉的變成 NaN
years = years.dropna().astype(int)  # 再轉成 int


# 輸出結果
print("最早年份：", years.min())
print("最晚年份：", years.max())

