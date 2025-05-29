import pandas as pd
NAS_100_TICKER = [
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
]


# 讀入原始資料
df = pd.read_csv("../sentiment_data/inNAS100_analyst_ratings_precessed_with_full_scores.csv", parse_dates=["date"])

# 處理股票欄位為多個 ticker
df["stock"] = df["stock"].astype(str)
df["stock_list"] = df["stock"].str.split(",")
df = df.explode("stock_list")
df["stock_list"] = df["stock_list"].str.strip().str.replace('"', '')

# 建立分數 triplet
df["score_triplet"] = df.apply(lambda row: [
    row["score_positive"],
    row["score_negative"],
    row["score_neutral"]
], axis=1)

# 每日 sentiment_score list，固定順序，不足填 [-1, -1, -1]
def daily_score_list(group):
    score_map = {row["stock_list"]: row["score_triplet"] for _, row in group.iterrows()}
    ordered_score_list = [score_map.get(stock, [-1, -1, -1]) for stock in NAS_100_TICKER]
    return pd.Series({
        "sentiment_score": ordered_score_list
    })

grouped = df.groupby("date").apply(daily_score_list).reset_index()

# 儲存為 .pkl 檔
grouped.to_pickle("../sentiment_data/daily_sentiment_list.pkl")
