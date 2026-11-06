"""Train the SAME algorithm on 2 feature versions x 2 hyperparameter sets = 4 experiments.

Run (from the project root, with the venv active and `feast apply` already done):
    python src/train_pipeline.py

For every run it:
  1. asks Feast for the training table of one feature version (athlete_model_v1 / _v2)
  2. trains a GradientBoostingRegressor with one hyperparameter set
  3. measures error on the held-out test athletes (same athletes in every run)
  4. measures energy / CO2 with CodeCarbon
  5. logs parameters, metrics, plots, the git commit and the data fingerprint to MLflow
Afterwards:  mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # draw plots to files, no window needed
import matplotlib.pyplot as plt            # noqa: E402
import mlflow                              # noqa: E402
import numpy as np                         # noqa: E402
import pandas as pd                        # noqa: E402
import yaml                                # noqa: E402
from codecarbon import OfflineEmissionsTracker      # noqa: E402
from sklearn.compose import ColumnTransformer       # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.pipeline import Pipeline               # noqa: E402
from sklearn.preprocessing import OneHotEncoder     # noqa: E402

from config import FEATURE_VERSIONS, ROOT, SPLIT_SEED, TARGET   # noqa: E402
from feature_store_demo import get_split                         # noqa: E402

EXPERIMENT = "athlete-total-lift"
REPORTS = ROOT / "reports"

# Same algorithm, two settings. "small" is quick and simple; "large" has more, deeper trees
# with a smaller learning rate (slower, usually more accurate, uses more energy).
HYPERPARAMETERS = {
    "small": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.10},
    "large": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
}


def git_info() -> dict:
    """Which code produced this run? (commit id + whether there are unsaved changes)."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    try:
        return {"git_commit": run("rev-parse", "--short", "HEAD") or "unknown",
                "git_uncommitted_changes": str(bool(run("status", "--porcelain")))}
    except OSError:
        return {"git_commit": "unknown", "git_uncommitted_changes": "unknown"}


def data_fingerprints() -> dict:
    """Which data produced this run? DVC's fingerprints (md5) of the feature files, from dvc.lock."""
    lock = ROOT / "dvc.lock"
    if not lock.exists():
        return {}
    outs = yaml.safe_load(lock.read_text())["stages"]["prepare_features"]["outs"]
    return {Path(o["path"]).stem: o["md5"] for o in outs}


def build_model(version: str, params: dict) -> Pipeline:
    """One-hot encode region (text), pass numbers through, then gradient boosting."""
    categorical = FEATURE_VERSIONS[version]["categorical"]
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)],
                             remainder="passthrough", verbose_feature_names_out=False)
    return Pipeline([("prep", prep),
                     ("model", GradientBoostingRegressor(random_state=SPLIT_SEED, **params))])


def make_plots(out_dir: Path, y_true, y_pred, pipe: Pipeline) -> list[Path]:
    """Three plots per run: predicted vs actual, residuals, feature importance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    resid = np.asarray(y_true) - y_pred
    paths = []

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.25)
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(xlabel="actual total_lift", ylabel="predicted total_lift", title="Predicted vs actual")
    paths.append(out_dir / "pred_vs_actual.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=110); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].hist(resid, bins=50)
    axs[0].set(xlabel="actual - predicted", title="Residuals")
    axs[1].scatter(y_pred, resid, s=4, alpha=0.25)
    axs[1].axhline(0, color="r", ls="--", lw=1)
    axs[1].set(xlabel="predicted total_lift", ylabel="residual", title="Residuals vs predicted")
    paths.append(out_dir / "residuals.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=110); plt.close(fig)

    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp.plot.barh(ax=ax)
    ax.set(xlabel="importance", title="Top features")
    paths.append(out_dir / "feature_importance.png")
    fig.tight_layout(); fig.savefig(paths[-1], dpi=110); plt.close(fig)
    return paths


def run_experiment(version: str, hp_name: str, params: dict) -> dict:
    run_name = f"{version}_{hp_name}"
    train, test = get_split(version)                         # <- the feature store hands us the data
    features = FEATURE_VERSIONS[version]["numeric"] + FEATURE_VERSIONS[version]["categorical"]
    X_train, y_train, X_test, y_test = train[features], train[TARGET], test[features], test[TARGET]

    tracker = OfflineEmissionsTracker(country_iso_code="USA", project_name=run_name, log_level="error",
                                      output_dir=str(REPORTS / "emissions"), output_file="emissions.csv")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"feature_version": version, "feature_service": f"athlete_model_{version}",
                         "hyperparameter_set": hp_name, "algorithm": "GradientBoostingRegressor",
                         **git_info(),
                         "data_md5_features": data_fingerprints().get(f"athlete_features_{version}", "unknown"),
                         "data_md5_labels": data_fingerprints().get("labels", "unknown")})
        mlflow.log_params({**params, "n_features": len(features), "random_state": SPLIT_SEED,
                           "n_train": len(X_train), "n_test": len(X_test)})

        pipe = build_model(version, params)
        tracker.start()                                      # carbon measurement covers fit + predict
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        seconds = time.perf_counter() - t0
        emissions_kg = tracker.stop() or 0.0
        energy_kwh = tracker.final_emissions_data.energy_consumed

        men = (X_test["is_male"] == 1).to_numpy()           # error by group (EDA finding F5 / hypothesis H2)
        metrics = {"rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                   "mae": float(mean_absolute_error(y_test, pred)),
                   "mae_men": float(mean_absolute_error(y_test[men], pred[men])),
                   "mae_women": float(mean_absolute_error(y_test[~men], pred[~men])),
                   "r2": float(r2_score(y_test, pred)),
                   "train_r2": float(r2_score(y_train, pipe.predict(X_train))),
                   "fit_predict_seconds": seconds,
                   "energy_kwh": float(energy_kwh), "emissions_kg_co2e": float(emissions_kg)}
        mlflow.log_metrics(metrics)
        for p in make_plots(REPORTS / "runs" / run_name, y_test, pred, pipe):
            mlflow.log_artifact(str(p))
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="pickle")
        return {"run_name": run_name, "run_id": run.info.run_id, "feature_version": version,
                "hyperparameter_set": hp_name, **params, **metrics}


def main() -> None:
    (REPORTS / "emissions").mkdir(parents=True, exist_ok=True)   # CodeCarbon needs this folder to exist
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT)

    results = []
    for version in FEATURE_VERSIONS:                         # v1, v2
        for hp_name, params in HYPERPARAMETERS.items():      # small, large
            print(f"\n=== {version} + {hp_name} {params} ===")
            res = run_experiment(version, hp_name, params)
            print(f"rmse {res['rmse']:.1f} | mae {res['mae']:.1f} | r2 {res['r2']:.3f} | "
                  f"{res['fit_predict_seconds']:.1f}s | {res['emissions_kg_co2e']:.2e} kg CO2e")
            results.append(res)

    table = pd.DataFrame(results)
    table.to_csv(REPORTS / "results.csv", index=False)
    cols = ["run_name", "rmse", "mae", "r2", "train_r2", "fit_predict_seconds", "energy_kwh", "emissions_kg_co2e"]
    print("\n", table[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nOpen the tracking UI:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
