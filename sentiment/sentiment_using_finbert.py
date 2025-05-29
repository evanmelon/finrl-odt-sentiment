import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import Dataset  # NEW


# ===== 步驟 1：讀取資料 =====
df = pd.read_csv("../news_data/inNAS100_analyst_ratings_processed.csv")

df = df.dropna(subset=["title"]).reset_index(drop=True)


# ===== 2. 載入模型與 pipeline =====
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)

# ===== 3. 建立 Hugging Face Dataset =====
hf_dataset = Dataset.from_pandas(df[["title"]])  # 只要 title 欄

# ===== 4. 批次跑 sentiment analysis（每筆會回傳三個 label）=====
results = finbert(hf_dataset["title"], batch_size=32)

# ===== 5. 整理結果 =====
# results 是 list[list[dict]] 結構 → 展平成每筆三個分數
df["score_positive"] = [r[0]["score"] for r in results]
df["score_negative"] = [r[1]["score"] for r in results]
df["score_neutral"]  = [r[2]["score"] for r in results]

# 同時找出最大分數對應的情緒當作預測分類
df["sentiment"] = [max(r, key=lambda x: x["score"])["label"] for r in results]

# ===== 6. 儲存結果 =====
df.to_csv("../sentiment_data/inNAS100_analyst_ratings_precessed_with_full_scores.csv", index=False)
print("完成分析，已儲存至 inNAS100_analyst_ratings_precessed_with_full_scores.csv")
