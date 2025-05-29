import pickle
import numpy as np
import pandas as pd

with open('data/test_a2c_trajectory_2025-04-15_17-19-31.pkl', 'rb') as f:
    data = pickle.load(f)

sample = data[0]

print("Keys:", sample.keys())
print("\nobservations[0]:", sample['observations'][0])
print("\nactions[0]:", sample['actions'][0])
print("\nrewards[0]:", sample['rewards'][0])
print("\nterminals[0]:", sample['terminals'][0])
print("\nnext_observations[0]:", sample['next_observations'][0])

