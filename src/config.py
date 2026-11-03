"""Single source of truth: paths, target, and the two feature versions.

Everything downstream (feature_repo/definitions.py, training, plots) imports from here,
so a feature name is never typed in two places.
"""
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "athletes.csv"
FEATURES_DIR = ROOT / "data" / "features"           # feature tables + labels (parquet); Feast's offline source

TARGET = "total_lift"
LIFTS = ["deadlift", "candj", "snatch", "backsq"]   # total_lift = sum of these -> never features (leakage)
ENTITY = "athlete_id"
TS = "event_timestamp"

# The dataset is a static snapshot with no real timestamps. Feast requires one, so we
# stamp features "as of" FEATURE_TS and labels at LABEL_TS (after the features exist).
FEATURE_TS = datetime(2023, 1, 1)
LABEL_TS = datetime(2023, 3, 1)

SPLIT_SEED = 42
TEST_SIZE = 0.2

# --- Feature versions -------------------------------------------------------------
# v1 = "basic": raw body/demographic columns only.
# v2 = v1 + engineered features (bmi, training history, survey flags).
V1_NUMERIC = ["age", "height", "weight", "is_male"]
V1_CATEGORICAL = ["region"]
V2_EXTRA_NUMERIC = [
    "bmi", "howlong_years",
    "bg_college_sports", "bg_no_athletic",
    "exp_started_with_coach", "exp_level1_cert", "exp_trains_others",
    "sched_multi_per_day", "eat_weigh_measure", "eat_paleo",
]
FEATURE_VERSIONS = {
    "v1": {"numeric": V1_NUMERIC, "categorical": V1_CATEGORICAL},
    "v2": {"numeric": V1_NUMERIC + V2_EXTRA_NUMERIC, "categorical": V1_CATEGORICAL},
}
