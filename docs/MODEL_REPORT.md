# Model Report (Current Run)

This report summarizes the latest tuned `PredictionModel_V3.py` run optimized for both competition scoring and presentation clarity.

## Run Setup
- Seed: `42`
- Elo config: `K=5`, home advantage `+80`
- Feature smoothing prior: `8` games
- Feature set size: `185` pregame features (leak-safe, chronology-only, including opponent-adjusted and upset-risk signals)
- Supervised models:
  - Logistic regression (`C=1.0`)
  - Random forest (`n_estimators=350`, `max_depth=8`, `min_samples_leaf=10`, `n_jobs=1`)
  - HistGradientBoosting (`learning_rate=0.05`, `max_depth=3`, `max_iter=450`, `min_samples_leaf=20`)
- Supervised training window: last `700` development games
- Elo shrinkage candidate:
  - `elo_shrink_alpha=0.55` (selected on development OOF only)
- Final selector: `elo_shrunk_recent_objective` (prediction source `elo_shrunk_home_rate`)

## Data Split
- Total games: `1312`
- Development: `1115` (first 85%)
- Holdout: `197` (last 15%, untouched during tuning)

## Holdout Results
From `outputs/holdout_metrics.csv`:

- Selected model (`elo_shrunk_recent_objective`):
  - Accuracy: `0.6041`
  - Log loss: `0.6720`
  - Brier: `0.2395`
  - AUC: `0.5284`
- Baseline (`baseline_home_rate`):
  - Accuracy: `0.6041`
  - Log loss: `0.6759`
  - Brier: `0.2414`

Delta vs baseline:
- Log loss: `-0.00387` (better)
- Brier: `-0.00185` (better)
- Accuracy: tied

## OOF CV Snapshot
From `outputs/model_cv_summary.csv` (OOF rows, sorted by log loss):

1. `blend_uncalibrated`: `0.6820`
2. `elo_shrunk_home_rate`: `0.6850`
3. `stack_uncalibrated`: `0.6852`
4. `elo_only`: `0.6868`

Note:
- Some high-capacity variants still underperform in OOF (`hist_gradient_boosting`, isotonic calibration).
- Final model selection prioritizes robust late-fold behavior and holdout-safe protocol, not only average OOF rank.

## Round 1 Prediction Summary
From `outputs/round1_home_win_probabilities.csv`:

- Matchups scored: `16`
- Mean home win probability: `0.6176`
- Min / Max: `0.5817 / 0.6622`

Top 5 strongest home favorites:
1. `brazil vs kazakhstan` (`0.6622`)
2. `netherlands vs mongolia` (`0.6592`)
3. `peru vs rwanda` (`0.6481`)
4. `thailand vs oman` (`0.6311`)
5. `india vs usa` (`0.6286`)

## Power Ranking Snapshot
Top 10 from `outputs/team_power_rankings.csv`:
1. brazil
2. netherlands
3. peru
4. thailand
5. india
6. pakistan
7. china
8. panama
9. iceland
10. philippines

## Reproducibility Artifacts
- `outputs/run_metadata.json`: run parameters, selected model, shrinkage alpha
- `outputs/run_manifest.json`: command, env versions, input/output SHA-256 hashes
- `outputs/visuals/visual_manifest.json`: hashes for generated charts
- Determinism check: `./run_all.sh --seed 42 --verify-repro` passed (artifact hashes matched across reruns)

## Presentation Notes
- For stakeholder slides, prioritize:
  1. `outputs/visuals/holdout_selected_vs_baseline.png`
  2. `outputs/visuals/holdout_model_comparison.png`
  3. `outputs/visuals/round1_home_win_probabilities.png`
  4. `outputs/visuals/top_power_rankings.png`
