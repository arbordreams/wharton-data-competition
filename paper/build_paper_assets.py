import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


FIGURE_MAP = {
    "cv_oof_model_comparison.png": "fig_cv_oof_model_comparison.png",
    "holdout_model_comparison.png": "fig_holdout_model_comparison.png",
    "holdout_confusion_matrix.png": "fig_holdout_confusion_matrix.png",
    "holdout_reliability_plot.png": "fig_holdout_reliability_plot.png",
    "holdout_selected_vs_baseline.png": "fig_holdout_selected_vs_baseline.png",
    "blend_weights.png": "fig_blend_weights.png",
    "top_power_rankings.png": "fig_top_power_rankings.png",
    "round1_home_win_probabilities.png": "fig_round1_probabilities.png",
    "feature_importance_logreg.png": "fig_feature_importance_logreg.png",
    "feature_importance_rf.png": "fig_feature_importance_rf.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication paper assets from model outputs.")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory containing model outputs.")
    parser.add_argument("--paper-dir", default="paper", help="Paper directory.")
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def save_tabular(df: pd.DataFrame, path: Path, float_fmt: str = "%.4f") -> None:
    latex = df.to_latex(index=False, escape=True, float_format=float_fmt.__mod__, na_rep="")
    path.write_text(latex, encoding="utf-8")


def prettify_label(value: object) -> str:
    return str(value).replace("_", " ")


def bounded_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    p = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def ece_10(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 11)
    bucket = np.digitize(y_prob, bins, right=False) - 1
    bucket = np.clip(bucket, 0, 9)
    error = 0.0
    n = len(y_true)
    for idx in range(10):
        mask = bucket == idx
        if not np.any(mask):
            continue
        frac = float(mask.sum()) / n
        error += frac * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return float(error)


def bootstrap_log_loss_delta_ci(
    y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray, n_bootstrap: int = 10000, seed: int = 42
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = bounded_log_loss(y_true[idx], y_a[idx]) - bounded_log_loss(y_true[idx], y_b[idx])
    return float(samples.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def bootstrap_delta_samples(
    y_true: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    n_bootstrap: int = 20000,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = bounded_log_loss(y_true[idx], y_a[idx]) - bounded_log_loss(y_true[idx], y_b[idx])
    return samples


def block_bootstrap_delta_samples(
    y_true: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    block_size: int = 12,
    n_bootstrap: int = 20000,
    seed: int = 4242,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    block = max(2, min(block_size, n))
    max_start = max(1, n - block + 1)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx_parts: list[np.ndarray] = []
        total = 0
        while total < n:
            start = int(rng.integers(0, max_start))
            segment = np.arange(start, min(start + block, n))
            idx_parts.append(segment)
            total += len(segment)
        idx = np.concatenate(idx_parts)[:n]
        samples[i] = bounded_log_loss(y_true[idx], y_a[idx]) - bounded_log_loss(y_true[idx], y_b[idx])
    return samples


def create_delta_bootstrap_distribution(
    path: Path,
    iid_samples: np.ndarray,
    block_samples: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(
        float(min(iid_samples.min(), block_samples.min())),
        float(max(iid_samples.max(), block_samples.max())),
        60,
    )
    ax.hist(
        iid_samples,
        bins=bins,
        density=True,
        alpha=0.45,
        color="#4c78a8",
        edgecolor="white",
        label="IID bootstrap",
    )
    ax.hist(
        block_samples,
        bins=bins,
        density=True,
        alpha=0.35,
        color="#f58518",
        edgecolor="white",
        label="Block bootstrap",
    )
    ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2, label="No improvement")
    ax.set_xlabel("Delta Log Loss (Selected - Baseline)")
    ax.set_ylabel("Density")
    ax.set_title("Estimated Test-Set Delta Log-Loss Distribution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def create_pipeline_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    stages = [
        (0.4, 0.8, 2.0, 1.2, "Season Data\n(whl_2025.csv)"),
        (2.9, 0.8, 2.1, 1.2, "Pregame\nFeature Engine"),
        (5.5, 0.8, 2.1, 1.2, "CV + Blend\nSelection"),
        (8.1, 0.8, 2.1, 1.2, "Holdout\nEvaluation"),
        (10.7, 0.8, 1.0, 1.2, "Round 1\nScoring"),
    ]

    for x, y, w, h, label in stages:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.2,
            edgecolor="#224466",
            facecolor="#d7e6f5",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2.0, y + h / 2.0, label, ha="center", va="center", fontsize=10)

    arrows = [
        (2.4, 1.4, 0.45, 0),
        (5.0, 1.4, 0.45, 0),
        (7.6, 1.4, 0.45, 0),
        (10.2, 1.4, 0.45, 0),
    ]
    for x, y, dx, dy in arrows:
        ax.arrow(x, y, dx, dy, width=0.01, head_width=0.12, head_length=0.12, color="#224466")

    ax.text(
        6.0,
        0.25,
        "Leak-safe chronology: features computed before each game; holdout untouched during tuning.",
        ha="center",
        va="center",
        fontsize=9,
        color="#223344",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def create_holdout_probability_density(path: Path, holdout_predictions: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for cls, color, label in [
        (1, "#0b6e4f", "Actual home win"),
        (0, "#b22222", "Actual home loss"),
    ]:
        values = holdout_predictions.loc[holdout_predictions["home_win"] == cls, "prob_final"].to_numpy()
        ax.hist(
            values,
            bins=np.linspace(0.0, 1.0, 21),
            alpha=0.45,
            color=color,
            edgecolor="white",
            label=label,
            density=True,
        )
    ax.set_xlabel("Predicted Home-Win Probability")
    ax.set_ylabel("Density")
    ax.set_title("Holdout Probability Distribution by True Outcome")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    paper_dir = Path(args.paper_dir)
    figures_dir = paper_dir / "figures"
    tables_dir = paper_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    required_outputs = [
        "holdout_metrics.csv",
        "holdout_predictions.csv",
        "model_cv_summary.csv",
        "model_blend_summary.csv",
        "team_power_rankings.csv",
        "round1_home_win_probabilities.csv",
        "run_metadata.json",
    ]
    for name in required_outputs:
        require_file(outputs_dir / name)

    visual_dir = outputs_dir / "visuals"
    for src_name, dst_name in FIGURE_MAP.items():
        src = visual_dir / src_name
        require_file(src)
        shutil.copy2(src, figures_dir / dst_name)

    holdout = pd.read_csv(outputs_dir / "holdout_metrics.csv")
    holdout_predictions = pd.read_csv(outputs_dir / "holdout_predictions.csv")
    cv = pd.read_csv(outputs_dir / "model_cv_summary.csv")
    blend = pd.read_csv(outputs_dir / "model_blend_summary.csv")
    rankings = pd.read_csv(outputs_dir / "team_power_rankings.csv")
    round1 = pd.read_csv(outputs_dir / "round1_home_win_probabilities.csv")
    with (outputs_dir / "run_metadata.json").open("r", encoding="utf-8") as infile:
        metadata = json.load(infile)

    create_pipeline_diagram(figures_dir / "fig_pipeline_overview.png")
    create_holdout_probability_density(
        figures_dir / "fig_holdout_probability_density.png", holdout_predictions
    )

    baseline = holdout.loc[holdout["model"] == "baseline_home_rate"].iloc[0]
    selected = holdout.loc[holdout["model"] == metadata["selected_model"]].iloc[0]
    summary_df = pd.DataFrame(
        [
            {
                "metric": "Total games",
                "value": metadata["n_games"],
            },
            {
                "metric": "Development games",
                "value": metadata["n_dev"],
            },
            {
                "metric": "Holdout games",
                "value": metadata["n_holdout"],
            },
            {
                "metric": "Selected model",
                "value": prettify_label(metadata["selected_model"]),
            },
            {
                "metric": "Selected prediction source",
                "value": prettify_label(metadata.get("selected_prediction_source", "blend_final")),
            },
            {
                "metric": "Selected holdout log loss",
                "value": float(selected["log_loss"]),
            },
            {
                "metric": "Baseline holdout log loss",
                "value": float(baseline["log_loss"]),
            },
            {
                "metric": "Delta log loss vs baseline",
                "value": float(selected["log_loss"] - baseline["log_loss"]),
            },
        ]
    )
    save_tabular(summary_df, tables_dir / "table_experiment_summary.tex", float_fmt="%.6f")

    holdout_table = holdout[["model", "accuracy", "log_loss", "brier", "auc"]].copy()
    holdout_table["delta_log_loss_vs_baseline"] = holdout_table["log_loss"] - float(baseline["log_loss"])
    holdout_table["model"] = holdout_table["model"].map(prettify_label)
    save_tabular(holdout_table, tables_dir / "table_holdout_metrics.tex", float_fmt="%.4f")

    oof_table = cv[cv["fold"].astype(str) == "OOF"][
        ["model", "accuracy", "log_loss", "brier", "auc"]
    ].copy()
    oof_table["model"] = oof_table["model"].map(prettify_label)
    save_tabular(oof_table, tables_dir / "table_oof_metrics.tex", float_fmt="%.4f")

    blend_table = blend.copy()
    recent_selector_mask = blend_table["component"].astype(str).str.startswith("recent_selector_")
    blend_table.loc[recent_selector_mask, "weight"] = pd.NA
    blend_table.loc[recent_selector_mask, "cv_brier"] = pd.NA
    blend_table["component"] = blend_table["component"].map(prettify_label)
    blend_table["selected"] = blend_table["selected"].map(lambda x: "yes" if bool(x) else "no")
    save_tabular(blend_table, tables_dir / "table_blend_components.tex", float_fmt="%.4f")

    top_rank = rankings.head(10).copy()
    save_tabular(top_rank, tables_dir / "table_top10_rankings.tex", float_fmt="%.4f")

    round1_table = round1.sort_values("home_win_prob", ascending=False).head(8).copy()
    save_tabular(round1_table, tables_dir / "table_top_round1_probs.tex", float_fmt="%.4f")

    y = holdout_predictions["home_win"].to_numpy(dtype=float)
    p_final = holdout_predictions["prob_final"].to_numpy(dtype=float)
    p_base = holdout_predictions["baseline_prob"].to_numpy(dtype=float)
    baseline_ll = float(baseline["log_loss"])
    selected_ll = float(selected["log_loss"])

    delta_mean, delta_low, delta_high = bootstrap_log_loss_delta_ci(
        y_true=y, y_a=p_final, y_b=p_base, n_bootstrap=10000, seed=42
    )
    iid_samples = bootstrap_delta_samples(y_true=y, y_a=p_final, y_b=p_base, n_bootstrap=20000, seed=42)
    block_samples = block_bootstrap_delta_samples(
        y_true=y,
        y_a=p_final,
        y_b=p_base,
        block_size=12,
        n_bootstrap=20000,
        seed=4242,
    )
    create_delta_bootstrap_distribution(
        figures_dir / "fig_delta_bootstrap_distribution.png",
        iid_samples=iid_samples,
        block_samples=block_samples,
    )

    iid_mean = float(iid_samples.mean())
    iid_low = float(np.quantile(iid_samples, 0.025))
    iid_high = float(np.quantile(iid_samples, 0.975))
    iid_prob_beat = float(np.mean(iid_samples < 0.0))

    block_mean = float(block_samples.mean())
    block_low = float(np.quantile(block_samples, 0.025))
    block_high = float(np.quantile(block_samples, 0.975))
    block_prob_beat = float(np.mean(block_samples < 0.0))

    expected_df = pd.DataFrame(
        [
            {"metric": "holdout_selected_log_loss", "value": selected_ll},
            {"metric": "holdout_baseline_log_loss", "value": baseline_ll},
            {"metric": "iid_delta_log_loss_mean", "value": iid_mean},
            {"metric": "iid_delta_log_loss_ci_2_5", "value": iid_low},
            {"metric": "iid_delta_log_loss_ci_97_5", "value": iid_high},
            {"metric": "iid_prob_selected_beats_baseline", "value": iid_prob_beat},
            {"metric": "block_delta_log_loss_mean", "value": block_mean},
            {"metric": "block_delta_log_loss_ci_2_5", "value": block_low},
            {"metric": "block_delta_log_loss_ci_97_5", "value": block_high},
            {"metric": "block_prob_selected_beats_baseline", "value": block_prob_beat},
            {
                "metric": "projected_selected_log_loss_mean_assuming_baseline_stable",
                "value": baseline_ll + iid_mean,
            },
            {
                "metric": "projected_selected_log_loss_pessimistic95_assuming_baseline_stable",
                "value": baseline_ll + iid_high,
            },
            {
                "metric": "projected_selected_log_loss_optimistic95_assuming_baseline_stable",
                "value": baseline_ll + iid_low,
            },
        ]
    )
    expected_df["metric"] = expected_df["metric"].map(prettify_label)
    save_tabular(expected_df, tables_dir / "table_expected_test_performance.tex", float_fmt="%.6f")

    uncertainty_df = pd.DataFrame(
        [
            {"metric": "holdout_home_win_rate", "value": float(y.mean())},
            {"metric": "mean_predicted_prob_final", "value": float(p_final.mean())},
            {"metric": "mean_predicted_prob_baseline", "value": float(p_base.mean())},
            {"metric": "ece10_final", "value": ece_10(y, p_final)},
            {"metric": "ece10_baseline", "value": ece_10(y, p_base)},
            {"metric": "delta_log_loss_mean_bootstrap", "value": delta_mean},
            {"metric": "delta_log_loss_ci_2_5", "value": delta_low},
            {"metric": "delta_log_loss_ci_97_5", "value": delta_high},
            {"metric": "iid_prob_selected_beats_baseline", "value": iid_prob_beat},
            {"metric": "block_prob_selected_beats_baseline", "value": block_prob_beat},
        ]
    )
    uncertainty_df["metric"] = uncertainty_df["metric"].map(prettify_label)
    save_tabular(uncertainty_df, tables_dir / "table_holdout_uncertainty.tex", float_fmt="%.6f")

    high_conf_errors = holdout_predictions[
        (holdout_predictions["pred_final"] != holdout_predictions["home_win"])
        & (
            (holdout_predictions["prob_final"] >= 0.70)
            | (holdout_predictions["prob_final"] <= 0.30)
        )
    ][["game_id", "home_team", "away_team", "home_win", "prob_final"]].copy()
    high_conf_errors = high_conf_errors.sort_values("prob_final", ascending=False).head(10)
    save_tabular(high_conf_errors, tables_dir / "table_high_confidence_errors.tex", float_fmt="%.4f")

    metadata_path = tables_dir / "paper_metrics.json"
    metadata_path.write_text(json.dumps({"run_metadata": metadata}, indent=2), encoding="utf-8")

    print(f"Paper assets updated in: {paper_dir.resolve()}")


if __name__ == "__main__":
    main()
