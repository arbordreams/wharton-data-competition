# WHSDSC 2026 Hockey Prediction Pipeline

Production-ready, leak-safe modeling package for the Wharton Data Science Competition (Phase 1a).

## What this repository contains
- End-to-end chronological feature engineering and model training.
- Holdout-safe model selection based on probability quality (`log_loss`, `brier`).
- Round 1 matchup probability generation.
- Automated visualization and paper artifact generation.
- Reproducibility metadata with input/output SHA-256 hashes.

## Repository structure
- `PredictionModel_V3.py` - main training, validation, selection, and Round 1 scoring pipeline.
- `make_model_visuals.py` - generates figures from model output artifacts.
- `run_all.sh` - one-command pipeline runner with optional reproducibility check.
- `requirements.txt` - Python dependencies.
- `outputs/` - generated prediction artifacts, metrics, manifests, and visuals.
- `docs/` - model and reproducibility documentation.
- `paper/` - publication assets and compiled paper PDF.

## Requirements
- Python 3.10+
- `pip`
- (Optional, for paper rebuild) LaTeX toolchain with `latexmk`.

Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

## Quick start
Run full pipeline:
```bash
./run_all.sh
```

Equivalent manual run:
```bash
python3 PredictionModel_V3.py
python3 make_model_visuals.py
```

## CLI options
Model pipeline options:
```bash
python3 PredictionModel_V3.py \
  --season-file whl_2025.csv \
  --round1-file "Wharton Download/WHSDSC_2026_CompetitionPackage-20260208T051745Z-1-001/WHSDSC_2026_CompetitionPackage/WHSDSC_Rnd1_matchups.xlsx" \
  --output-dir outputs \
  --seed 42
```

Runner options:
```bash
./run_all.sh --help
```

## Reproducibility
Run deterministic verification:
```bash
./run_all.sh --verify-repro
```

Core reproducibility artifacts:
- `outputs/run_manifest.json` (command, environment, input/output hashes)
- `outputs/run_metadata.json` (selection and run metadata)
- `outputs/visuals/visual_manifest.json` (visual hash manifest)

Note: timestamped metadata fields change run-to-run by design.

## Main outputs
- `outputs/round1_home_win_probabilities.csv`
- `outputs/team_power_rankings.csv`
- `outputs/holdout_metrics.csv`
- `outputs/model_cv_summary.csv`
- `outputs/model_blend_summary.csv`
- `outputs/holdout_predictions.csv`
- `outputs/feature_importance.csv`
- `outputs/visuals/*.png`

## Paper build
Rebuild paper:
```bash
./paper/build_paper.sh
```

Publication PDF:
- `paper/WHSDSC_2026_paper.pdf`

## Data
Primary inputs in this repository:
- `whl_2025.csv`
- `Wharton Download/.../WHSDSC_Rnd1_matchups.xlsx`

The pipeline validates required columns and fails fast if inputs are incompatible.
