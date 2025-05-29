import pandas as pd

NAS_100_TICKER = {
    "AAL",
    "AAPL",
    "ADBE",
    "ADI",
    "ADP",
    "ADSK",
    "ALGN",
    "AMAT",
    "AMD",
    "AMGN",
    "AMZN",
    "ASML",
    "BIDU",
    "BIIB",
    "BKNG",
    "BMRN",
    "CDNS",
    "CHKP",
    "CMCSA",
    "COST",
    "CSCO",
    "CSX",
    "CTAS",
    "CTSH",
    "DLTR",
    "EA",
    "EBAY",
    "EXPE",
    "FAST",
    "GILD",
    "GOOGL",
    "HAS",
    "HSIC",
    "IDXX",
    "ILMN",
    "INCY",
    "INTC",
    "INTU",
    "ISRG",
    "JBHT",
    "KLAC",
    "LBTYK",
    "LRCX",
    "LULU",
    "MAR",
    "MCHP",
    "MDLZ",
    "MELI",
    "MNST",
    "MSFT",
    "MU",
    "NFLX",
    "NTAP",
    "NTES",
    "NVDA",
    "ORLY",
    "PAYX",
    "PCAR",
    "PEP",
    "QCOM",
    "REGN",
    "ROST",
    "SBUX",
    "SIRI",
    "SNPS",
    "SWKS",
    "TCOM",
    "TMUS",
    "TTWO",
    "TXN",
    "UAL",
    "ULTA",
    "VRSN",
    "VRTX",
    "WBA",
    "WDC",
    "WYNN",
    "XEL"
}
df = pd.read_csv("../news_data/analyst_ratings_processed.csv", index_col=0)

# 1) 這些代號在資料裡「出現過幾筆紀錄」？
rows_with_target = df["stock"].isin(NAS_100_TICKER).sum()
print(f"rows with target: {rows_with_target}")

# 2) 共有「多少種」目標代號真的有出現在資料裡？（不重複計算）
unique_targets_present = df.loc[df["stock"].isin(NAS_100_TICKER), "stock"].nunique()
print(f"unique targets present: {unique_targets_present}")

# 3) 哪些目標代號根本沒出現在資料裡？
missing_symbols = NAS_100_TICKER - set(df["stock"].unique())
print(f"missing symbols: {missing_symbols}")

# 4) 如果想直接拿到那部分資料
print(df)
subset_df = df[df["stock"].isin(NAS_100_TICKER)].reset_index(drop=True)
print(f"subset df: {subset_df}")
subset_df.to_csv("../news_data/inNAS100_analyst_ratings_processed.csv")

# years = pd.to_numeric(df["date"].str[:4], errors='coerce')  # 無法轉的變成 NaN
# years = years.dropna().astype(int)  # 再轉成 int
#
#
# # 輸出結果
# print("最早年份：", years.min())
# print("最晚年份：", years.max())
#
# # 篩選出 NAS_100_TICKER 中有出現在 df["stock"] 的資料
# filtered_df = df[df["stock"].isin(NAS_100_TICKER)]
#
# # 計算每個 stock 出現的次數
# stock_counts = filtered_df["stock"].value_counts().sort_index()
#
# # 輸出結果
# print(stock_counts)
#
#
# # 建立完整的計數表，預設值為 0
# all_counts = pd.Series(0, index=sorted(NAS_100_TICKER))
#
# # 更新實際出現的數量
# all_counts.update(stock_counts)
#
# # 輸出
# print(all_counts)
#
# # 將 Series 轉成 DataFrame 並加上欄位名稱
# df = all_counts.to_frame(name="news_count")
#
# # # 加上 index 名稱（假設是股票代碼）
# # df.index.name = "ticker"
#
# # 把原本的 index（ticker）變成欄位，並新增自動整數 index
# df = df.reset_index()  # index becomes a column
#
# # 重命名欄位
# df.columns = ["ticker", "news_count"]
#
# # 輸出 CSV，自動加上 index（0, 1, 2,...）
# df.to_csv("news_count.csv", index=True)
# # all_counts.to_csv("news_count.csv")
#
