import pickle
import numpy as np
import pandas as pd

# with open('../sentiment_data/daily_sentiment.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# # print(data)
# print(data.iloc[1].values[1])

with open('../sentiment_data/daily_sentiment_dict.pkl', 'rb') as f:
    data = pickle.load(f)

# print(data)
print(data.iloc[1].values[1])

with open('../sentiment_data/daily_sentiment_list.pkl', 'rb') as f:
    data = pickle.load(f)

# print(data)
print(data.iloc[1].values[0])
print(data.iloc[1].values[1])
