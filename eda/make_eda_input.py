"""Creates data/interim/athletes_with_target.parquet: the raw table + the target column, minus names.
Used as the input of the EDA notebook. Run from the project root:  python eda/make_eda_input.py"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import LIFTS, RAW_CSV  # noqa: E402

raw = pd.read_csv(RAW_CSV, low_memory=False)
raw["total_lift"] = raw[LIFTS].sum(axis=1, min_count=len(LIFTS))   # NaN if any lift is missing
out = ROOT / "data" / "interim" / "athletes_with_target.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
raw.drop(columns=["name"]).to_parquet(out, index=False)             # drop athlete names (personal data)
print(f"wrote {out.relative_to(ROOT)}: {len(raw):,} rows, {raw.total_lift.notna().sum():,} with total_lift")
