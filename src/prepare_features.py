"""Step 1: raw athletes.csv -> cleaned table -> two feature-version parquet files + labels.

Run:  python src/prepare_features.py

Design choices (explain these in the video):
  * ONE cleaning step shared by both versions, so v1 and v2 are trained on the exact same
    athletes. Then any metric difference comes from the features, not from different rows.
  * Features are written to parquet files = Feast's "offline store" source.
  * Labels (total_lift) are NOT in the feature views; they go in a separate table that
    later becomes Feast's "entity dataframe". Features and labels live apart on purpose.
"""
import numpy as np
import pandas as pd

from config import (ENTITY, FEATURE_TS, FEATURE_VERSIONS, LABEL_TS, LIFTS, RAW_CSV,
                    FEATURES_DIR, TARGET, TS)

HOWLONG_YEARS = {"Less than 6 months": 0.25, "6-12 months": 0.75, "1-2 years": 1.5,
                 "2-4 years": 3.0, "4+ years": 5.0}


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Course cleaning recipe: drop incomplete rows + physiologically implausible outliers."""
    df = raw.dropna(subset=[ENTITY, "region", "age", "weight", "height", "gender", "eat",
                            "background", "experience", "schedule", "howlong", *LIFTS]).copy()
    df = df.drop_duplicates(subset=ENTITY)
    df[ENTITY] = df[ENTITY].astype("int64")
    df = df[(df.weight < 1500) & (df.gender != "--") & (df.age >= 18)
            & df.height.between(48, 96, inclusive="neither")]
    # Sex-specific deadlift cap (the original snippet's operator precedence made the
    # female cap a no-op; this is the intended behaviour).
    cap = np.where(df.gender == "Female", 636, 1105)
    df = df[(df.deadlift > 0) & (df.deadlift <= cap)]
    df = df[df.candj.between(0, 395, inclusive="right") & df.snatch.between(0, 496, inclusive="right")
            & df.backsq.between(0, 1069, inclusive="right")]
    df = df[(df[LIFTS] >= 20).all(axis=1)]   # drop implausibly tiny lifts (EDA finding F7)
    for col in ["background", "experience", "schedule", "howlong", "eat"]:
        df = df[~df[col].str.contains("Decline to answer", na=False)]
    return df


def howlong_to_years(answer: str) -> float:
    """Some athletes ticked several answers; keep the longest."""
    years = [HOWLONG_YEARS[a] for a in str(answer).split("|") if a in HOWLONG_YEARS]
    return max(years) if years else np.nan


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All candidate features (v1 columns + v2 extras). Versions are column subsets of this."""
    out = pd.DataFrame({ENTITY: df[ENTITY]})
    out["age"], out["height"], out["weight"] = df.age.astype(float), df.height.astype(float), df.weight.astype(float)
    out["is_male"] = (df.gender == "Male").astype(int)
    out["region"] = df.region.astype(str)
    # ---- v2 engineered features ----
    out["bmi"] = 703 * df.weight / df.height ** 2          # imperial units (lbs, inches)
    out["howlong_years"] = df.howlong.map(howlong_to_years)
    flags = {"bg_college_sports": ("background", "I played college sports"),
             "bg_no_athletic": ("background", "I have no athletic background"),
             "exp_started_with_coach": ("experience", "with a coach"),
             "exp_level1_cert": ("experience", "Level 1 certificate"),
             "exp_trains_others": ("experience", "I train other people"),
             "sched_multi_per_day": ("schedule", "multiple workouts in a day"),
             "eat_weigh_measure": ("eat", "I weigh and measure"),
             "eat_paleo": ("eat", "Paleo")}
    for name, (col, phrase) in flags.items():
        out[name] = df[col].str.contains(phrase, regex=False).astype(int)
    return out.dropna()


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_CSV, low_memory=False)
    df = clean(raw)
    feats = build_features(df)
    df = df[df[ENTITY].isin(feats[ENTITY])]
    print(f"raw {len(raw):,} -> cleaned {len(feats):,} athletes")

    for version, spec in FEATURE_VERSIONS.items():
        cols = [ENTITY, *spec["numeric"], *spec["categorical"]]
        table = feats[cols].copy()
        table[TS] = pd.Timestamp(FEATURE_TS)
        table.to_parquet(FEATURES_DIR / f"athlete_features_{version}.parquet", index=False)
        print(f"  {version}: {len(cols) - 1} features")

    labels = pd.DataFrame({ENTITY: df[ENTITY], TARGET: df[LIFTS].sum(axis=1)})
    labels[TS] = pd.Timestamp(LABEL_TS)
    labels.to_parquet(FEATURES_DIR / "labels.parquet", index=False)
    print(f"  labels: {len(labels):,} rows; {TARGET} mean={labels[TARGET].mean():.0f}")


if __name__ == "__main__":
    main()
