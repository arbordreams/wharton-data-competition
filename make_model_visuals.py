import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

MODEL_LABELS = {
    "baseline_home_rate": "Baseline (Home Rate)",
    "elo_only": "Elo Only",
    "elo_shrunk_home_rate": "Elo Shrunk",
    "elo_only_recent_guard": "Elo Guard",
    "elo_shrunk_recent_guard": "Elo-Shrunk Guard",
    "elo_only_recent_objective": "Elo Objective",
    "elo_shrunk_recent_objective": "Elo-Shrunk Objective",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "blend_uncalibrated": "Blend (Uncalibrated)",
    "blend_final": "Blend (Final)",
    "blend_recent_objective": "Blend Objective",
    "stack_uncalibrated": "Stack (Uncalibrated)",
    "stack_final": "Stack (Final)",
    "stack_recent_objective": "Stack Objective",
    "blend_calibrated_platt": "Blend (Platt)",
    "blend_calibrated_isotonic": "Blend (Isotonic)",
    "blend_platt_calibrated": "Blend (Platt)",
    "blend_isotonic_calibrated": "Blend (Isotonic)",
}

plt.style.use("seaborn-v0_8-whitegrid")


def pretty_model_name(value: object) -> str:
    key = str(value)
    return MODEL_LABELS.get(key, key.replace("_", " "))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visual summaries for WHSDSC model outputs.")
    parser.add_argument("--input-dir", default="outputs", help="Directory containing model output CSV files.")
    parser.add_argument("--output-dir", default="outputs/visuals", help="Directory to save generated visuals.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as infile:
        while True:
            chunk = infile.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def load_required_frames(input_dir: Path) -> tuple[pd.DataFrame, ...]:
    required = [
        "model_cv_summary.csv",
        "holdout_metrics.csv",
        "holdout_predictions.csv",
        "model_blend_summary.csv",
        "feature_importance.csv",
        "team_power_rankings.csv",
        "round1_home_win_probabilities.csv",
        "run_metadata.json",
    ]
    missing = [name for name in required if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {input_dir}: {missing}")

    cv_df = pd.read_csv(input_dir / "model_cv_summary.csv")
    holdout_metrics_df = pd.read_csv(input_dir / "holdout_metrics.csv")
    holdout_preds_df = pd.read_csv(input_dir / "holdout_predictions.csv")
    blend_df = pd.read_csv(input_dir / "model_blend_summary.csv")
    feature_df = pd.read_csv(input_dir / "feature_importance.csv")
    ranking_df = pd.read_csv(input_dir / "team_power_rankings.csv")
    round1_df = pd.read_csv(input_dir / "round1_home_win_probabilities.csv")
    with (input_dir / "run_metadata.json").open("r", encoding="utf-8") as infile:
        metadata = json.load(infile)

    return cv_df, holdout_metrics_df, holdout_preds_df, blend_df, feature_df, ranking_df, round1_df, metadata


def save_cv_oof_metrics(cv_df: pd.DataFrame, output_dir: Path) -> None:
    oof = cv_df[cv_df["fold"].astype(str) == "OOF"].copy()
    if oof.empty:
        return

    oof = oof[~oof["model"].astype(str).str.startswith("blend_")].copy()
    oof = oof.sort_values("log_loss", ascending=True)
    oof["model_label"] = oof["model"].map(pretty_model_name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].barh(oof["model_label"], oof["log_loss"], color="#4c78a8")
    axes[0].set_title("OOF Log Loss (Lower Better)")
    axes[0].set_xlabel("Log Loss")

    axes[1].barh(oof["model_label"], oof["accuracy"], color="#f58518")
    axes[1].set_title("OOF Accuracy (Higher Better)")
    axes[1].set_xlabel("Accuracy")
    axes[1].set_xlim(0, 1)

    fig.suptitle("Cross-Validation OOF Model Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "cv_oof_model_comparison.png", dpi=180)
    plt.close(fig)


def save_holdout_metrics(holdout_metrics_df: pd.DataFrame, selected_model: str, output_dir: Path) -> None:
    plot_df = holdout_metrics_df[holdout_metrics_df["model"] != "baseline_home_rate"].copy()
    plot_df = plot_df.sort_values("log_loss", ascending=True)
    plot_df["model_label"] = plot_df["model"].map(pretty_model_name)

    selected_colors = [
        "#2f5597" if model == selected_model else "#54a24b"
        for model in plot_df["model"].tolist()
    ]
    selected_colors_acc = [
        "#2f5597" if model == selected_model else "#e45756"
        for model in plot_df["model"].tolist()
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(plot_df["model_label"], plot_df["log_loss"], color=selected_colors)
    axes[0].set_title("Holdout Log Loss (Lower Better)")
    axes[0].set_xlabel("Log Loss")

    axes[1].barh(plot_df["model_label"], plot_df["accuracy"], color=selected_colors_acc)
    axes[1].set_title("Holdout Accuracy (Higher Better)")
    axes[1].set_xlabel("Accuracy")
    axes[1].set_xlim(0, 1)

    fig.suptitle(f"Untouched Holdout Performance (Selected: {pretty_model_name(selected_model)})")
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_model_comparison.png", dpi=180)
    plt.close(fig)


def save_selected_vs_baseline(
    holdout_metrics_df: pd.DataFrame,
    selected_model: str,
    output_dir: Path,
) -> None:
    baseline = holdout_metrics_df.loc[holdout_metrics_df["model"] == "baseline_home_rate"]
    selected = holdout_metrics_df.loc[holdout_metrics_df["model"] == selected_model]
    if baseline.empty or selected.empty:
        return

    baseline_row = baseline.iloc[0]
    selected_row = selected.iloc[0]
    metrics = ["log_loss", "brier", "accuracy"]
    baseline_vals = [float(baseline_row[m]) for m in metrics]
    selected_vals = [float(selected_row[m]) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bars_base = ax.bar(
        x - width / 2,
        baseline_vals,
        width=width,
        label=pretty_model_name("baseline_home_rate"),
        color="#b0b0b0",
    )
    bars_selected = ax.bar(
        x + width / 2,
        selected_vals,
        width=width,
        label=pretty_model_name(selected_model),
        color="#4c78a8",
    )
    ax.set_xticks(x, labels=["Log Loss", "Brier", "Accuracy"])
    ax.set_ylim(0, 1)
    ax.set_title("Selected Model vs Baseline (Holdout)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    for bars in (bars_base, bars_selected):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_selected_vs_baseline.png", dpi=180)
    plt.close(fig)


def save_confusion_matrix(holdout_preds_df: pd.DataFrame, output_dir: Path) -> None:
    tp = int(((holdout_preds_df["home_win"] == 1) & (holdout_preds_df["pred_final"] == 1)).sum())
    tn = int(((holdout_preds_df["home_win"] == 0) & (holdout_preds_df["pred_final"] == 0)).sum())
    fp = int(((holdout_preds_df["home_win"] == 0) & (holdout_preds_df["pred_final"] == 1)).sum())
    fn = int(((holdout_preds_df["home_win"] == 1) & (holdout_preds_df["pred_final"] == 0)).sum())

    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title("Holdout Confusion Matrix (Final Model)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_confusion_matrix.png", dpi=180)
    plt.close(fig)


def save_reliability_plot(holdout_preds_df: pd.DataFrame, output_dir: Path) -> None:
    probs = holdout_preds_df["prob_final"].to_numpy()
    y = holdout_preds_df["home_win"].to_numpy()
    n_bins = 8
    # Quantile bins avoid unstable empty/near-empty fixed bins in a narrow probability range.
    bin_idx = pd.qcut(probs, q=n_bins, labels=False, duplicates="drop")
    x_vals: list[float] = []
    y_vals: list[float] = []
    for b in sorted(pd.unique(bin_idx)):
        mask = bin_idx == b
        x_vals.append(float(probs[mask].mean()))
        y_vals.append(float(y[mask].mean()))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(x_vals, y_vals, marker="o", color="#4c78a8", label="Model")
    ax.set_title("Holdout Reliability Plot")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Home Win Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_reliability_plot.png", dpi=180)
    plt.close(fig)


def save_blend_weights(blend_df: pd.DataFrame, output_dir: Path) -> None:
    comp = blend_df[
        (~blend_df["component"].astype(str).str.startswith("blend_"))
        & (~blend_df["component"].astype(str).str.startswith("recent_selector_"))
    ].copy()
    comp = comp[pd.to_numeric(comp["weight"], errors="coerce").notna()].copy()
    if comp.empty:
        return
    comp = comp.sort_values("weight", ascending=False)
    comp["component_label"] = comp["component"].map(pretty_model_name)

    fig, ax = plt.subplots(figsize=(7, 4))
    palette = ["#4c78a8", "#f58518", "#54a24b", "#eeca3b", "#72b7b2", "#b279a2"]
    colors = [palette[i % len(palette)] for i in range(len(comp))]
    ax.bar(comp["component_label"], comp["weight"], color=colors)
    ax.set_ylim(0, 1)
    ax.set_title("Final Blend Weights")
    ax.set_ylabel("Weight")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "blend_weights.png", dpi=180)
    plt.close(fig)


def save_power_rankings(ranking_df: pd.DataFrame, output_dir: Path, top_n: int = 12) -> None:
    top = ranking_df.sort_values("rank").head(top_n).copy()
    top = top.sort_values("power_score", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["team"], top["power_score"], color="#72b7b2")
    ax.set_title(f"Top {top_n} Teams by Power Score")
    ax.set_xlabel("Power Score")
    fig.tight_layout()
    fig.savefig(output_dir / "top_power_rankings.png", dpi=180)
    plt.close(fig)


def save_round1_probs(round1_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = round1_df.copy()
    plot_df["matchup"] = plot_df["home_team"] + " vs " + plot_df["away_team"]
    plot_df = plot_df.sort_values("home_win_prob", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(plot_df["matchup"], plot_df["home_win_prob"], color="#b279a2")
    ax.set_title("Round 1 Predicted Home Win Probabilities")
    ax.set_xlabel("Home Win Probability")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "round1_home_win_probabilities.png", dpi=180)
    plt.close(fig)


def save_feature_importance(feature_df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    for imp_type, filename, title, color in [
        ("logreg_abs_coef", "feature_importance_logreg.png", "Top Logistic Coefficient Importances", "#4c78a8"),
        ("rf_feature_importance", "feature_importance_rf.png", "Top Random Forest Feature Importances", "#f58518"),
    ]:
        subset = feature_df[feature_df["importance_type"] == imp_type].copy()
        if subset.empty:
            continue
        top = subset.nlargest(top_n, "importance_value").sort_values("importance_value", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        ax.barh(top["feature"], top["importance_value"], color=color)
        ax.set_title(title)
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        cv_df,
        holdout_metrics_df,
        holdout_preds_df,
        blend_df,
        feature_df,
        ranking_df,
        round1_df,
        metadata,
    ) = load_required_frames(input_dir)

    selected_model = metadata.get("selected_model", "blend_uncalibrated")

    save_cv_oof_metrics(cv_df, output_dir)
    save_holdout_metrics(holdout_metrics_df, selected_model, output_dir)
    save_selected_vs_baseline(holdout_metrics_df, selected_model, output_dir)
    save_confusion_matrix(holdout_preds_df, output_dir)
    save_reliability_plot(holdout_preds_df, output_dir)
    save_blend_weights(blend_df, output_dir)
    save_power_rankings(ranking_df, output_dir)
    save_round1_probs(round1_df, output_dir)
    save_feature_importance(feature_df, output_dir)

    visual_files = sorted([path for path in output_dir.glob("*.png") if path.is_file()])
    visual_manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "matplotlib_version": matplotlib.__version__,
        "visual_hashes": {path.name: sha256_file(path) for path in visual_files},
    }
    with (output_dir / "visual_manifest.json").open("w", encoding="utf-8") as outfile:
        json.dump(visual_manifest, outfile, indent=2)

    print(f"Saved visuals to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
