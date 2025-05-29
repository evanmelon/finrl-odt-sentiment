import pandas as pd

train = pd.read_csv('../train_data_NAS.csv')

train = train.set_index(train.columns[0])
train.index.names = ['']


pd.Series(train.tic.unique()).to_csv("tic_sort.csv", index=False)

