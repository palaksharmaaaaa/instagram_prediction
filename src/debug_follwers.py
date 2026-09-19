import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "top_200_instagrammers.csv")

df = pd.read_csv(DATA_PATH)

print("\n========== FOLLOWERS DEBUG ==========")

print("\nFirst 20 Followers:")
print(df["Followers"].head(20))

print("\nFollowers Data Type:")
print(df["Followers"].dtype)

print("\nMinimum Followers:")
print(df["Followers"].min())

print("\nMaximum Followers:")
print(df["Followers"].max())

print("\nAll columns:")
print(df.columns.tolist())

print("====================================")