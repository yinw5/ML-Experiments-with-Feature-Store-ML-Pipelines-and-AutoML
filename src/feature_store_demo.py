"""Step 2: register the feature repo, pull training sets for v1/v2, materialize, query online.

Run:  python src/feature_store_demo.py
Also exposes get_training_frame() which the training pipeline (next session) will import.
"""
import subprocess
import sys
from datetime import datetime

import pandas as pd
from feast import FeatureStore
from sklearn.model_selection import train_test_split

from config import ENTITY, FEATURE_VERSIONS, FEATURES_DIR, ROOT, SPLIT_SEED, TARGET, TEST_SIZE

REPO = ROOT / "feature_repo"


def get_store() -> FeatureStore:
    return FeatureStore(repo_path=str(REPO))


def get_training_frame(version: str, store: FeatureStore | None = None) -> pd.DataFrame:
    """Point-in-time-correct training set for one feature version (features + label)."""
    store = store or get_store()
    entity_df = pd.read_parquet(FEATURES_DIR / "labels.parquet")   # id + event_timestamp + label
    job = store.get_historical_features(entity_df=entity_df, features=store.get_feature_service(f"athlete_model_{version}"))
    return job.to_df()


def get_split(version: str):
    """Same athletes in train/test for every version (split on ids, fixed seed)."""
    df = get_training_frame(version)
    ids = pd.read_parquet(FEATURES_DIR / "labels.parquet")[ENTITY]
    train_ids, _ = train_test_split(ids, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    mask = df[ENTITY].isin(set(train_ids))
    return df[mask].reset_index(drop=True), df[~mask].reset_index(drop=True)


if __name__ == "__main__":
    subprocess.run(["feast", "apply"], cwd=REPO, check=True)          # register definitions
    store = get_store()
    print("feature views:", [fv.name for fv in store.list_feature_views()])
    print("feature services:", [fs.name for fs in store.list_feature_services()])

    for v in FEATURE_VERSIONS:
        df = get_training_frame(v)
        print(f"\n[{v}] training frame {df.shape}; columns: {[c for c in df.columns if c not in (ENTITY, 'event_timestamp', TARGET)]}")
        print(df.head(3).to_string())
        tr, te = get_split(v)
        print(f"[{v}] split -> train {len(tr):,} / test {len(te):,}")

    # Online path: copy latest values offline -> online, then look one athlete up by key.
    store.materialize_incremental(end_date=datetime(2023, 2, 1))
    some_id = int(pd.read_parquet(FEATURES_DIR / "labels.parquet")[ENTITY].iloc[0])
    online = store.get_online_features(features=store.get_feature_service("athlete_model_v2"),
                                       entity_rows=[{ENTITY: some_id}]).to_dict()
    print("\nonline lookup (v2) for athlete", some_id, "->", {k: v[0] for k, v in online.items()})
