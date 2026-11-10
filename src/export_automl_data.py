"""Export the v2 training table from the feature store as CSVs for AutoML tools (DataRobot, H2O).

Run (venv active, `feast apply` already done):   python src/export_automl_data.py
Writes data/automl/athletes_v2_train.csv and athletes_v2_test.csv

Same athletes and same 80/20 split as the Part A experiments, so Part A and Part B
can be compared on the same test athletes. athlete_id and event_timestamp are dropped
(an ID is not a feature; AutoML would treat it as one).
"""
from config import ENTITY, ROOT, TARGET, TS
from feature_store_demo import get_split

OUT = ROOT / "data" / "automl"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train, test = get_split("v2")
    for name, df in [("train", train), ("test", test)]:
        df = df.drop(columns=[ENTITY, TS])
        df.to_csv(OUT / f"athletes_v2_{name}.csv", index=False)
        print(f"wrote data/automl/athletes_v2_{name}.csv: {len(df):,} rows, {df.shape[1]} columns")
    print("target:", TARGET, "| columns:", list(df.columns))


if __name__ == "__main__":
    main()
