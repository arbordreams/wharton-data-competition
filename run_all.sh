#!/usr/bin/env bash
set -euo pipefail

SEED=42
SEASON_FILE="whl_2025.csv"
ROUND1_FILE="Wharton Download/WHSDSC_2026_CompetitionPackage-20260208T051745Z-1-001/WHSDSC_2026_CompetitionPackage/WHSDSC_Rnd1_matchups.xlsx"
OUTPUT_DIR="outputs"
VERIFY_REPRO=0

usage() {
  cat <<'EOF'
Usage: ./run_all.sh [options]

Options:
  --seed N                Random seed (default: 42)
  --season-file PATH      Season CSV file (default: whl_2025.csv)
  --round1-file PATH      Round 1 matchups file (default: official xlsx path)
  --output-dir PATH       Output directory (default: outputs)
  --verify-repro          Re-run pipeline and verify deterministic output hashes
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --season-file)
      SEASON_FILE="$2"
      shift 2
      ;;
    --round1-file)
      ROUND1_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --verify-repro)
      VERIFY_REPRO=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

run_model() {
  echo "==> Running model pipeline..."
  python3 PredictionModel_V3.py \
    --seed "$SEED" \
    --season-file "$SEASON_FILE" \
    --round1-file "$ROUND1_FILE" \
    --output-dir "$OUTPUT_DIR"
}

run_visuals() {
  echo "==> Generating visuals..."
  python3 make_model_visuals.py \
    --input-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/visuals"
}

capture_hashes() {
  local target_json="$1"
  python3 - "$OUTPUT_DIR" "$target_json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
dest = Path(sys.argv[2])

tracked = [
    out / "round1_home_win_probabilities.csv",
    out / "team_power_rankings.csv",
    out / "holdout_metrics.csv",
    out / "model_cv_summary.csv",
    out / "model_blend_summary.csv",
    out / "holdout_predictions.csv",
    out / "feature_importance.csv",
]
tracked.extend(sorted((out / "visuals").glob("*.png")))

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

missing = [str(p) for p in tracked if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing expected artifact(s): {missing}")

hashes = {str(p): sha256_file(p) for p in tracked}
dest.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
PY
}

verify_hashes() {
  local baseline_json="$1"
  python3 - "$OUTPUT_DIR" "$baseline_json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
baseline_path = Path(sys.argv[2])

baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

current = {}
for key in baseline:
    p = Path(key)
    if not p.exists():
        print(f"Missing artifact after re-run: {p}")
        sys.exit(1)
    current[key] = sha256_file(p)

mismatches = [k for k in baseline if baseline[k] != current[k]]
if mismatches:
    print("Reproducibility check failed. Mismatched files:")
    for k in mismatches:
        print(f"- {k}")
    sys.exit(1)

print("Reproducibility check passed: tracked artifact hashes match.")
PY
}

print_summary() {
  python3 - "$OUTPUT_DIR" <<'PY'
import json
import sys
import pandas as pd
from pathlib import Path

out = Path(sys.argv[1])
holdout = pd.read_csv(out / "holdout_metrics.csv")
meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))

selected = meta["selected_model"]
selected_row = holdout[holdout["model"] == selected].iloc[0]
baseline_row = holdout[holdout["model"] == "baseline_home_rate"].iloc[0]

print("==> Final Summary")
print(f"Selected model: {selected} ({meta.get('selected_prediction_source')})")
print(f"Holdout accuracy: {selected_row['accuracy']:.4f}")
print(f"Holdout log_loss: {selected_row['log_loss']:.4f}")
print(f"Holdout brier:    {selected_row['brier']:.4f}")
print(f"Baseline log_loss:{baseline_row['log_loss']:.4f}")
print(f"Delta vs baseline:{selected_row['log_loss'] - baseline_row['log_loss']:+.6f}")
print(f"Round 1 output:   {out / 'round1_home_win_probabilities.csv'}")
print(f"Visuals dir:      {out / 'visuals'}")
PY
}

run_model
run_visuals

if [[ "$VERIFY_REPRO" -eq 1 ]]; then
  echo "==> Running reproducibility verification..."
  SNAPSHOT_FILE="$(mktemp -t whs_hashes.XXXXXX.json)"
  trap 'rm -f "$SNAPSHOT_FILE"' EXIT
  capture_hashes "$SNAPSHOT_FILE"
  run_model
  run_visuals
  verify_hashes "$SNAPSHOT_FILE"
fi

print_summary
