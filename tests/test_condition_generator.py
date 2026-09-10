"""
Unit tests for the Condition Query & Event Mining Generator.
"""

import datetime as dt
import pandas as pd
import pytest
from pathlib import Path

from src.data.condition_generator import ConditionGenerator

PARQUET_PATH = Path("/Users/prince/algo-trading/data/historical/XAUUSD_1m.parquet")


def test_condition_generator_mining_and_query(tmp_path):
    db_file = tmp_path / "test_conditions.db"
    cg = ConditionGenerator(db_path=db_file)

    if not PARQUET_PATH.exists():
        pytest.skip(f"Historical parquet not found at {PARQUET_PATH}")

    # Generate condition dataset for 10 days in April 2024
    start_date = dt.date(2024, 4, 15)
    end_date = dt.date(2024, 4, 26)

    df_conditions = cg.generate_conditions_from_parquet(
        parquet_path=PARQUET_PATH,
        start_date=start_date,
        end_date=end_date,
        x_offset=3.5,
    )

    assert len(df_conditions) >= 5
    assert "range_width_pts" in df_conditions.columns
    assert "proximity_bias" in df_conditions.columns

    # Test SQL Condition Query: "Find days with range width > 15 points"
    filtered = cg.query_conditions("range_width_pts > 15.0")
    assert len(filtered) > 0
    assert (filtered["range_width_pts"] > 15.0).all()

    # Test SQL Condition Query: "Find days where price was within 3.5 points of extreme"
    near_extreme = cg.query_conditions("is_within_offset = 1")
    assert len(near_extreme) >= 1
    assert (near_extreme["distance_to_extreme"] <= 3.5).all()
