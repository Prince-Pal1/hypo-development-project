import datetime as dt

from src.data.condition_generator import ConditionGenerator

cg = ConditionGenerator()
start = dt.date(2026, 1, 1)
end = dt.date(2026, 9, 30)

parquet_path = "/Users/prince/strategy_development/other_related_projects_copied_database/algo-trading/data/historical/XAUUSD_1m.parquet"
res = cg.generate_conditions_from_parquet(parquet_path, start, end)

print(f"Generated {len(res)} conditions.")
print(res.head())
