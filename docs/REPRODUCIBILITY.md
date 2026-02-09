# Reproducibility Guide

## 1) Install dependencies
```bash
python3 -m pip install -r requirements.txt
```

## 2) Run pipeline
```bash
python3 PredictionModel_V3.py --seed 42
python3 make_model_visuals.py
```

## 3) Verify manifests
- `outputs/run_manifest.json`
- `outputs/visuals/visual_manifest.json`

These include SHA-256 hashes for generated artifacts.

## 4) Compare deterministic outputs across two runs
Run the model twice with the same seed and compare hashes of prediction artifacts:

```bash
python3 PredictionModel_V3.py --seed 42
python3 - <<'PY'
import json
with open('outputs/run_manifest.json') as f:
    m=json.load(f)
for k,v in sorted(m['output_hashes'].items()):
    if k.endswith('round1_home_win_probabilities.csv') or k.endswith('team_power_rankings.csv') or k.endswith('holdout_metrics.csv'):
        print(k, v)
PY
```

Repeat and confirm hashes match for those files.

## Notes
- `run_metadata.json` and `run_manifest.json` include timestamps and will differ each run.
- Results can change if dependency versions differ; keep versions stable and documented.
- Input file hashes are stored in `run_manifest.json` to prove dataset consistency.
