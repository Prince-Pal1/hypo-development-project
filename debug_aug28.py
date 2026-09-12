import pandas as pd
df = pd.read_parquet("/Users/prince/algo-trading/data/historical/XAUUSD_5m.parquet")
df_day = df.loc["2026-08-28"]
eval_time = pd.Timestamp("2026-08-28 07:00:00")
hist = df_day[df_day.index < eval_time]
print(f"High: {hist['high'].max()} Low: {hist['low'].min()}")
post = df_day[df_day.index >= eval_time].head(20)
print(post[['open', 'high', 'low', 'close']])
