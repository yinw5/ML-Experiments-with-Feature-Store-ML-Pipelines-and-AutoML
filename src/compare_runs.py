"""Compare the 4 experiments: numbers (table), plots, and carbon. Reads reports/results.csv.

Run (after train_pipeline.py):   python src/compare_runs.py
Writes reports/comparison.md and reports/comparison_*.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg   # noqa: E402
import matplotlib.pyplot as plt    # noqa: E402
import pandas as pd                # noqa: E402

from config import ROOT            # noqa: E402

REPORTS = ROOT / "reports"
ORDER = ["v1_small", "v1_large", "v2_small", "v2_large"]
COLORS = {"v1": "#9aa5b1", "v2": "#4c78a8"}          # grey = basic features, blue = engineered


def pct(new, old):
    return (new - old) / old * 100


def metric_bars(t: pd.DataFrame) -> None:
    """Quantitative: error, carbon, and the accuracy-vs-carbon trade-off."""
    colors = [COLORS[r[:2]] for r in t.run_name]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].bar(t.run_name, t.rmse, color=colors)
    axs[0].set(title="Test error (RMSE, lower is better)", ylabel="lbs")
    for i, v in enumerate(t.rmse):
        axs[0].text(i, v + 1, f"{v:.0f}", ha="center")
    axs[0].set_ylim(0, t.rmse.max() * 1.15)
    mg = t.emissions_kg_co2e * 1e6                    # kg -> milligrams
    axs[1].bar(t.run_name, mg, color=colors)
    axs[1].set(title="Carbon per run (milligrams CO2e)", ylabel="mg CO2e")
    for i, v in enumerate(mg):
        axs[1].text(i, v * 1.02, f"{v:.1f}", ha="center")
    axs[2].scatter(mg, t.rmse, c=colors, s=90)
    for x, y, n in zip(mg, t.rmse, t.run_name):
        axs[2].annotate(n, (x, y), textcoords="offset points", xytext=(6, 6))
    axs[2].set(title="Accuracy vs carbon (bottom-left is best)", xlabel="mg CO2e", ylabel="RMSE")
    axs[2].margins(0.2)
    for ax in axs[:2]:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(REPORTS / "comparison_metrics.png", dpi=120)
    plt.close(fig)


def composite(name: str, title: str, size=(10, 9)) -> None:
    """Qualitative: the same plot from all 4 runs in a 2x2 grid (rows = feature version)."""
    fig, axs = plt.subplots(2, 2, figsize=size)
    for ax, run in zip(axs.ravel(), ORDER):
        ax.imshow(mpimg.imread(REPORTS / "runs" / run / f"{name}.png"))
        ax.set_title(run, fontsize=12)
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORTS / f"comparison_{name}.png", dpi=120)
    plt.close(fig)


def write_markdown(t: pd.DataFrame) -> None:
    g = t.set_index("run_name")
    feat_small = pct(g.loc["v2_small", "rmse"], g.loc["v1_small", "rmse"])
    feat_large = pct(g.loc["v2_large", "rmse"], g.loc["v1_large", "rmse"])
    hp_v1 = pct(g.loc["v1_large", "rmse"], g.loc["v1_small", "rmse"])
    hp_v2 = pct(g.loc["v2_large", "rmse"], g.loc["v2_small", "rmse"])
    co2_v1 = g.loc["v1_large", "emissions_kg_co2e"] / g.loc["v1_small", "emissions_kg_co2e"]
    co2_v2 = g.loc["v2_large", "emissions_kg_co2e"] / g.loc["v2_small", "emissions_kg_co2e"]
    show = t[["run_name", "rmse", "mae", "r2", "train_r2", "mae_men", "mae_women",
              "fit_predict_seconds", "energy_kwh", "emissions_kg_co2e"]].copy()
    show["emissions_mg_co2e"] = show.pop("emissions_kg_co2e") * 1e6
    table_md = show.to_markdown(index=False, floatfmt=".4g")
    md = f"""# Comparison of the 4 experiments

Same algorithm (GradientBoostingRegressor), same athletes, same 80/20 split. Only the feature version
and the hyperparameters change.

{table_md}

Error columns are in lbs on the test athletes. `train_r2` minus `r2` shows overfitting.

## What changed the result?

- **Feature version (v2 vs v1):** RMSE changes by {feat_small:+.1f}% with the small settings and {feat_large:+.1f}% with the large settings.
- **Hyperparameters (large vs small):** RMSE changes by {hp_v1:+.1f}% on v1 and {hp_v2:+.1f}% on v2.
- **Carbon of the large settings vs small:** {co2_v1:.1f}x on v1, {co2_v2:.1f}x on v2.

## Plots

- `comparison_metrics.png`: error, carbon, and accuracy vs carbon.
- `comparison_pred_vs_actual.png`, `comparison_residuals.png`, `comparison_feature_importance.png`: the same plot for all 4 runs.
- Per-run plots: `reports/runs/<run>/`. Raw numbers: `reports/results.csv`. Carbon log: `reports/emissions/emissions.csv`.

Note: carbon figures are small and noisy (CodeCarbon estimates energy from CPU use over a few seconds). Compare them as ratios between runs, not as absolute values.
"""
    (REPORTS / "comparison.md").write_text(md)


def main() -> None:
    t = pd.read_csv(REPORTS / "results.csv").set_index("run_name").loc[ORDER].reset_index()
    metric_bars(t)
    composite("pred_vs_actual", "Predicted vs actual (top row: v1, bottom row: v2)")
    composite("residuals", "Residuals (top row: v1, bottom row: v2)", size=(11, 8))
    composite("feature_importance", "Top features (top row: v1, bottom row: v2)")
    write_markdown(t)
    print((REPORTS / "comparison.md").read_text())


if __name__ == "__main__":
    main()
