import argparse
import hashlib
import json
import platform
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_SEASON_FILE = "whl_2025.csv"
DEFAULT_ROUND1_FILE = (
    "Wharton Download/"
    "WHSDSC_2026_CompetitionPackage-20260208T051745Z-1-001/"
    "WHSDSC_2026_CompetitionPackage/WHSDSC_Rnd1_matchups.xlsx"
)
DEFAULT_OUTPUT_DIR = "outputs"

REQUIRED_SEASON_COLUMNS = {
    "game_id",
    "record_id",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "home_max_xg",
    "away_max_xg",
    "home_shots",
    "away_shots",
    "home_assists",
    "away_assists",
    "home_penalties_committed",
    "away_penalties_committed",
    "home_penalty_minutes",
    "away_penalty_minutes",
    "went_ot",
}
REQUIRED_ROUND1_COLUMNS = {"game_id", "home_team", "away_team"}

ELO_K = 5.0
ELO_HOME_ADVANTAGE = 80.0
PRIOR_GAMES = 8.0
OUTPUT_PROB_MIN = 0.01
OUTPUT_PROB_MAX = 0.99
EPS = 1e-6
LOGREG_C = 1.0
RF_N_ESTIMATORS = 350
RF_MAX_DEPTH = 8
RF_MIN_SAMPLES_LEAF = 10
HGB_LEARNING_RATE = 0.05
HGB_MAX_DEPTH = 3
HGB_MAX_ITER = 450
HGB_MIN_SAMPLES_LEAF = 20
SUPERVISED_TRAIN_WINDOW = 700
RF_N_JOBS = 1
RECENT_SELECTOR_LOOKBACK = 2
RECENT_SELECTOR_MARGIN = 0.0
RECENT_SELECTOR_EXP_BASE = 2.0
RECENT_OBJECTIVE_WEIGHT = 0.5
RECENT_OBJECTIVE_SWITCH_MIN_DELTA = 0.0007
ADVANCED_SOURCE_OOF_GAIN_MIN = 0.0015
ADVANCED_SOURCE_RECENT_GAIN_MIN = 0.0005
ELO_SHRINK_ALPHA_STEP = 0.05
ELO_SHRINK_OOF_GAIN_MIN = 0.001
ELO_SHRINK_RECENT_TOLERANCE = 0.001
STACKER_C = 0.5
STACKER_CALIBRATION_MIN_IMPROVEMENT = 0.0005
BLEND_CALIBRATION_MIN_IMPROVEMENT = 0.001
RECENT_WINDOWS = (3, 5, 10)
TREND_SHORT_WINDOW = 3
TREND_LONG_WINDOW = 10
MAX_RECENT_HISTORY = 20
BAGGING_SEED_OFFSETS = (0, 19)
BAGGING_WINDOWS = (500, 700)

STAT_ATTRS = {
    "goals": ("goals_for", "goals_against", 3.0),
    "xg": ("xg_for", "xg_against", 3.0),
    "max_xg": ("max_xg_for", "max_xg_against", 0.5),
    "shots": ("shots_for", "shots_against", 28.0),
    "assists": ("assists_for", "assists_against", 5.0),
    "penalties": ("penalties_for", "penalties_against", 3.0),
    "pim": ("pim_for", "pim_against", 10.0),
}

META_COLUMNS = {"game_id", "home_team", "away_team", "game_num", "home_win"}


@dataclass
class TeamState:
    gp: int = 0
    wins: int = 0
    home_gp: int = 0
    home_wins: int = 0
    away_gp: int = 0
    away_wins: int = 0
    goals_for: float = 0.0
    goals_against: float = 0.0
    xg_for: float = 0.0
    xg_against: float = 0.0
    max_xg_for: float = 0.0
    max_xg_against: float = 0.0
    shots_for: float = 0.0
    shots_against: float = 0.0
    assists_for: float = 0.0
    assists_against: float = 0.0
    penalties_for: float = 0.0
    penalties_against: float = 0.0
    pim_for: float = 0.0
    pim_against: float = 0.0
    elo: float = 1500.0
    ot_games: int = 0
    recent_results: list[float] = field(default_factory=list)
    recent_goal_diff: list[float] = field(default_factory=list)
    recent_xg_diff: list[float] = field(default_factory=list)
    recent_shot_diff: list[float] = field(default_factory=list)
    recent_assist_diff: list[float] = field(default_factory=list)
    recent_pim_diff: list[float] = field(default_factory=list)
    home_recent_results: list[float] = field(default_factory=list)
    away_recent_results: list[float] = field(default_factory=list)
    opp_elo_sum: float = 0.0
    opp_win_rate_sum: float = 0.0
    opp_goals_for_pg_sum: float = 0.0
    opp_goals_against_pg_sum: float = 0.0
    opp_xg_for_pg_sum: float = 0.0
    opp_xg_against_pg_sum: float = 0.0
    opp_shots_for_pg_sum: float = 0.0
    opp_shots_against_pg_sum: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WHSDSC 2026 leak-free prediction pipeline.")
    parser.add_argument("--season-file", default=DEFAULT_SEASON_FILE, help="Historical season CSV file.")
    parser.add_argument("--round1-file", default=DEFAULT_ROUND1_FILE, help="Round 1 matchups file (xlsx/csv).")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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


def get_git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_hashes(paths: list[Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if path.exists():
            hashes[str(path)] = sha256_file(path)
    return hashes


def normalize_team_names(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def excel_col_to_int(col: str) -> int:
    value = 0
    for char in col:
        value = value * 26 + (ord(char.upper()) - ord("A") + 1)
    return value


def parse_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

    with zipfile.ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_root.findall("a:si", ns):
                text = "".join((node.text or "") for node in si.findall(".//a:t", ns))
                shared_strings.append(text)

        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
        rels_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root}

        sheet = workbook_root.find("a:sheets/a:sheet", ns)
        if sheet is None:
            raise ValueError(f"No sheets found in {path}.")
        rel_id = sheet.attrib[rel_ns]
        target = rel_map[rel_id]
        if not target.startswith("xl/"):
            target = f"xl/{target.lstrip('/')}"

        sheet_root = ET.fromstring(zf.read(target))
        rows: list[dict[str, str]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", ns):
            row_values: dict[str, str] = {}
            for cell in row.findall("a:c", ns):
                cell_ref = cell.attrib.get("r", "")
                col = "".join(ch for ch in cell_ref if ch.isalpha())
                cell_type = cell.attrib.get("t")
                value = ""
                if cell_type == "s":
                    v_node = cell.find("a:v", ns)
                    if v_node is not None and v_node.text is not None:
                        value = shared_strings[int(v_node.text)]
                elif cell_type == "inlineStr":
                    value = "".join((t.text or "") for t in cell.findall(".//a:t", ns))
                else:
                    v_node = cell.find("a:v", ns)
                    if v_node is not None and v_node.text is not None:
                        value = v_node.text
                row_values[col] = value
            rows.append(row_values)

    if not rows:
        raise ValueError(f"No rows found in sheet of {path}.")

    header_cols = sorted(rows[0].keys(), key=excel_col_to_int)
    headers = [str(rows[0].get(col, "")).strip() for col in header_cols]
    if not any(headers):
        raise ValueError(f"Header row is empty in {path}.")

    records: list[dict[str, str]] = []
    for row in rows[1:]:
        record: dict[str, str] = {}
        has_data = False
        for col, header in zip(header_cols, headers):
            if not header:
                continue
            value = str(row.get(col, "")).strip()
            record[header] = value
            if value != "":
                has_data = True
        if has_data:
            records.append(record)

    return pd.DataFrame(records)


def load_round1_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        df = parse_xlsx_first_sheet(path)
    else:
        raise ValueError(f"Unsupported Round 1 file type: {suffix}")
    return df


def validate_columns(df: pd.DataFrame, required_cols: set[str], name: str) -> None:
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_inputs(season_file: Path, round1_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not season_file.exists():
        raise FileNotFoundError(f"Season file not found: {season_file}")
    if not round1_file.exists():
        raise FileNotFoundError(f"Round 1 file not found: {round1_file}")

    season_df = pd.read_csv(season_file)
    validate_columns(season_df, REQUIRED_SEASON_COLUMNS, "Season data")
    season_df = normalize_team_names(season_df, ["home_team", "away_team"])

    round1_df = load_round1_file(round1_file)
    validate_columns(round1_df, REQUIRED_ROUND1_COLUMNS, "Round 1 matchups")
    round1_df = normalize_team_names(round1_df, ["home_team", "away_team"])
    round1_df = round1_df[["game_id", "home_team", "away_team"]].copy()
    round1_df["game_id"] = round1_df["game_id"].astype(str).str.strip()
    round1_df = round1_df[round1_df["game_id"] != ""].reset_index(drop=True)

    if len(round1_df) != 16:
        raise ValueError(f"Round 1 file must contain exactly 16 matchups. Found: {len(round1_df)}")
    if round1_df["game_id"].duplicated().any():
        duplicates = sorted(round1_df.loc[round1_df["game_id"].duplicated(), "game_id"].unique())
        raise ValueError(f"Round 1 game_id values must be unique. Duplicates: {duplicates}")

    return season_df, round1_df


def aggregate_games(season_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = season_df.select_dtypes(include=[np.number]).columns.tolist()
    agg_cols = [col for col in numeric_cols if col != "record_id"]

    games = (
        season_df.groupby(["game_id", "home_team", "away_team"], as_index=False)[agg_cols]
        .sum()
        .copy()
    )

    required_after_agg = {
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_max_xg",
        "away_max_xg",
        "home_shots",
        "away_shots",
        "home_assists",
        "away_assists",
        "home_penalties_committed",
        "away_penalties_committed",
        "home_penalty_minutes",
        "away_penalty_minutes",
        "went_ot",
    }
    validate_columns(games, required_after_agg | {"game_id", "home_team", "away_team"}, "Aggregated games")

    games["home_win"] = (games["home_goals"] > games["away_goals"]).astype(int)
    games["went_ot"] = (games["went_ot"] > 0).astype(int)
    game_num = pd.to_numeric(games["game_id"].str.extract(r"(\d+)")[0], errors="coerce")
    if game_num.isna().any():
        bad_ids = games.loc[game_num.isna(), "game_id"].head(5).tolist()
        raise ValueError(f"Could not extract numeric game number from game_id values: {bad_ids}")
    games["game_num"] = game_num.astype(int)

    games = games.sort_values(["game_num", "game_id"]).reset_index(drop=True)
    if not games["game_num"].is_monotonic_increasing:
        raise ValueError("Chronology check failed: game_num is not monotonic increasing.")
    return games


def smoothed_rate(total: float, games_played: int, prior_mean: float) -> float:
    return (total + prior_mean * PRIOR_GAMES) / (games_played + PRIOR_GAMES)


def recent_smoothed_mean(history: list[float], window: int, prior_mean: float) -> float:
    recent = history[-window:]
    missing = max(0, window - len(recent))
    return float((sum(recent) + prior_mean * missing) / window)


def recent_std(history: list[float], window: int, prior_std: float) -> float:
    recent = history[-window:]
    if len(recent) < 2:
        return float(prior_std)
    return float(np.std(np.asarray(recent, dtype=float), ddof=0))


def team_pregame_snapshot(state: TeamState) -> dict[str, float]:
    return {
        "win_rate": smoothed_rate(state.wins, state.gp, 0.5),
        "elo": float(state.elo),
        "goals_for_pg": smoothed_rate(state.goals_for, state.gp, 3.0),
        "goals_against_pg": smoothed_rate(state.goals_against, state.gp, 3.0),
        "xg_for_pg": smoothed_rate(state.xg_for, state.gp, 3.0),
        "xg_against_pg": smoothed_rate(state.xg_against, state.gp, 3.0),
        "shots_for_pg": smoothed_rate(state.shots_for, state.gp, 28.0),
        "shots_against_pg": smoothed_rate(state.shots_against, state.gp, 28.0),
    }


def append_with_cap(history: list[float], value: float, max_len: int = MAX_RECENT_HISTORY) -> None:
    history.append(float(value))
    if len(history) > max_len:
        del history[: len(history) - max_len]


def elo_home_prob(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + ELO_HOME_ADVANTAGE)) / 400.0))


def compute_matchup_features(home_state: TeamState, away_state: TeamState) -> dict[str, float]:
    home_wr = smoothed_rate(home_state.wins, home_state.gp, 0.5)
    away_wr = smoothed_rate(away_state.wins, away_state.gp, 0.5)
    home_home_wr = smoothed_rate(home_state.home_wins, home_state.home_gp, 0.5)
    away_away_wr = smoothed_rate(away_state.away_wins, away_state.away_gp, 0.5)

    features: dict[str, float] = {
        "home_gp": float(home_state.gp),
        "away_gp": float(away_state.gp),
        "gp_diff": float(home_state.gp - away_state.gp),
        "home_win_rate": home_wr,
        "away_win_rate": away_wr,
        "win_rate_diff": home_wr - away_wr,
        "home_home_win_rate": home_home_wr,
        "away_away_win_rate": away_away_wr,
        "home_away_split_diff": home_home_wr - away_away_wr,
        "home_elo": home_state.elo,
        "away_elo": away_state.elo,
        "elo_diff": home_state.elo - away_state.elo,
        "elo_home_prob": elo_home_prob(home_state.elo, away_state.elo),
    }
    home_ot_rate = smoothed_rate(home_state.ot_games, home_state.gp, 0.15)
    away_ot_rate = smoothed_rate(away_state.ot_games, away_state.gp, 0.15)
    features["home_ot_rate"] = home_ot_rate
    features["away_ot_rate"] = away_ot_rate
    features["ot_rate_diff"] = home_ot_rate - away_ot_rate

    for window in RECENT_WINDOWS:
        home_recent_wr = recent_smoothed_mean(home_state.recent_results, window, 0.5)
        away_recent_wr = recent_smoothed_mean(away_state.recent_results, window, 0.5)
        home_home_recent_wr = recent_smoothed_mean(home_state.home_recent_results, window, 0.5)
        away_away_recent_wr = recent_smoothed_mean(away_state.away_recent_results, window, 0.5)

        features[f"home_recent_win_rate_{window}"] = home_recent_wr
        features[f"away_recent_win_rate_{window}"] = away_recent_wr
        features[f"recent_win_rate_diff_{window}"] = home_recent_wr - away_recent_wr
        features[f"home_home_recent_win_rate_{window}"] = home_home_recent_wr
        features[f"away_away_recent_win_rate_{window}"] = away_away_recent_wr
        features[f"home_away_recent_split_diff_{window}"] = home_home_recent_wr - away_away_recent_wr

    for name, (for_attr, against_attr, prior_mean) in STAT_ATTRS.items():
        home_for = smoothed_rate(getattr(home_state, for_attr), home_state.gp, prior_mean)
        away_for = smoothed_rate(getattr(away_state, for_attr), away_state.gp, prior_mean)
        home_against = smoothed_rate(getattr(home_state, against_attr), home_state.gp, prior_mean)
        away_against = smoothed_rate(getattr(away_state, against_attr), away_state.gp, prior_mean)

        features[f"home_{name}_for_pg"] = home_for
        features[f"away_{name}_for_pg"] = away_for
        features[f"home_{name}_against_pg"] = home_against
        features[f"away_{name}_against_pg"] = away_against
        features[f"{name}_for_pg_diff"] = home_for - away_for
        features[f"{name}_against_pg_diff"] = home_against - away_against

        home_net = home_for - home_against
        away_net = away_for - away_against
        features[f"home_{name}_net_pg"] = home_net
        features[f"away_{name}_net_pg"] = away_net
        features[f"{name}_net_pg_diff"] = home_net - away_net

    # Opponent-adjusted schedule/context features derived from prior opponents only.
    home_opp_avg_elo = smoothed_rate(home_state.opp_elo_sum, home_state.gp, 1500.0)
    away_opp_avg_elo = smoothed_rate(away_state.opp_elo_sum, away_state.gp, 1500.0)
    home_opp_avg_wr = smoothed_rate(home_state.opp_win_rate_sum, home_state.gp, 0.5)
    away_opp_avg_wr = smoothed_rate(away_state.opp_win_rate_sum, away_state.gp, 0.5)
    features["home_opp_avg_elo"] = home_opp_avg_elo
    features["away_opp_avg_elo"] = away_opp_avg_elo
    features["opp_avg_elo_diff"] = home_opp_avg_elo - away_opp_avg_elo
    features["home_opp_avg_win_rate"] = home_opp_avg_wr
    features["away_opp_avg_win_rate"] = away_opp_avg_wr
    features["opp_avg_win_rate_diff"] = home_opp_avg_wr - away_opp_avg_wr
    features["home_adjusted_win_rate"] = home_wr - (home_opp_avg_wr - 0.5)
    features["away_adjusted_win_rate"] = away_wr - (away_opp_avg_wr - 0.5)
    features["adjusted_win_rate_diff"] = (
        features["home_adjusted_win_rate"] - features["away_adjusted_win_rate"]
    )

    home_opp_goal_against = smoothed_rate(home_state.opp_goals_against_pg_sum, home_state.gp, 3.0)
    away_opp_goal_against = smoothed_rate(away_state.opp_goals_against_pg_sum, away_state.gp, 3.0)
    home_opp_goal_for = smoothed_rate(home_state.opp_goals_for_pg_sum, home_state.gp, 3.0)
    away_opp_goal_for = smoothed_rate(away_state.opp_goals_for_pg_sum, away_state.gp, 3.0)
    home_opp_xg_against = smoothed_rate(home_state.opp_xg_against_pg_sum, home_state.gp, 3.0)
    away_opp_xg_against = smoothed_rate(away_state.opp_xg_against_pg_sum, away_state.gp, 3.0)
    home_opp_xg_for = smoothed_rate(home_state.opp_xg_for_pg_sum, home_state.gp, 3.0)
    away_opp_xg_for = smoothed_rate(away_state.opp_xg_for_pg_sum, away_state.gp, 3.0)
    home_opp_shots_against = smoothed_rate(home_state.opp_shots_against_pg_sum, home_state.gp, 28.0)
    away_opp_shots_against = smoothed_rate(away_state.opp_shots_against_pg_sum, away_state.gp, 28.0)
    home_opp_shots_for = smoothed_rate(home_state.opp_shots_for_pg_sum, home_state.gp, 28.0)
    away_opp_shots_for = smoothed_rate(away_state.opp_shots_for_pg_sum, away_state.gp, 28.0)

    features["home_goal_attack_adj"] = features["home_goals_for_pg"] - home_opp_goal_against
    features["away_goal_attack_adj"] = features["away_goals_for_pg"] - away_opp_goal_against
    features["goal_attack_adj_diff"] = features["home_goal_attack_adj"] - features["away_goal_attack_adj"]
    features["home_goal_def_adj"] = features["home_goals_against_pg"] - home_opp_goal_for
    features["away_goal_def_adj"] = features["away_goals_against_pg"] - away_opp_goal_for
    features["goal_def_adj_diff"] = features["home_goal_def_adj"] - features["away_goal_def_adj"]

    features["home_xg_attack_adj"] = features["home_xg_for_pg"] - home_opp_xg_against
    features["away_xg_attack_adj"] = features["away_xg_for_pg"] - away_opp_xg_against
    features["xg_attack_adj_diff"] = features["home_xg_attack_adj"] - features["away_xg_attack_adj"]
    features["home_xg_def_adj"] = features["home_xg_against_pg"] - home_opp_xg_for
    features["away_xg_def_adj"] = features["away_xg_against_pg"] - away_opp_xg_for
    features["xg_def_adj_diff"] = features["home_xg_def_adj"] - features["away_xg_def_adj"]

    features["home_shot_attack_adj"] = features["home_shots_for_pg"] - home_opp_shots_against
    features["away_shot_attack_adj"] = features["away_shots_for_pg"] - away_opp_shots_against
    features["shot_attack_adj_diff"] = features["home_shot_attack_adj"] - features["away_shot_attack_adj"]
    features["home_shot_def_adj"] = features["home_shots_against_pg"] - home_opp_shots_for
    features["away_shot_def_adj"] = features["away_shots_against_pg"] - away_opp_shots_for
    features["shot_def_adj_diff"] = features["home_shot_def_adj"] - features["away_shot_def_adj"]

    recent_diff_metrics = [
        ("goal_diff", "recent_goal_diff"),
        ("xg_diff", "recent_xg_diff"),
        ("shot_diff", "recent_shot_diff"),
        ("assist_diff", "recent_assist_diff"),
        ("pim_diff", "recent_pim_diff"),
    ]
    for metric_name, state_attr in recent_diff_metrics:
        for window in RECENT_WINDOWS:
            home_recent = recent_smoothed_mean(getattr(home_state, state_attr), window, 0.0)
            away_recent = recent_smoothed_mean(getattr(away_state, state_attr), window, 0.0)
            features[f"home_{metric_name}_recent_{window}"] = home_recent
            features[f"away_{metric_name}_recent_{window}"] = away_recent
            features[f"{metric_name}_recent_diff_{window}"] = home_recent - away_recent

    home_momentum = recent_smoothed_mean(home_state.recent_results, TREND_SHORT_WINDOW, 0.5) - recent_smoothed_mean(
        home_state.recent_results, TREND_LONG_WINDOW, 0.5
    )
    away_momentum = recent_smoothed_mean(away_state.recent_results, TREND_SHORT_WINDOW, 0.5) - recent_smoothed_mean(
        away_state.recent_results, TREND_LONG_WINDOW, 0.5
    )
    features["win_form_momentum_diff"] = home_momentum - away_momentum
    features["xg_form_momentum_diff"] = (
        recent_smoothed_mean(home_state.recent_xg_diff, TREND_SHORT_WINDOW, 0.0)
        - recent_smoothed_mean(home_state.recent_xg_diff, TREND_LONG_WINDOW, 0.0)
        - recent_smoothed_mean(away_state.recent_xg_diff, TREND_SHORT_WINDOW, 0.0)
        + recent_smoothed_mean(away_state.recent_xg_diff, TREND_LONG_WINDOW, 0.0)
    )
    features["goal_form_momentum_diff"] = (
        recent_smoothed_mean(home_state.recent_goal_diff, TREND_SHORT_WINDOW, 0.0)
        - recent_smoothed_mean(home_state.recent_goal_diff, TREND_LONG_WINDOW, 0.0)
        - recent_smoothed_mean(away_state.recent_goal_diff, TREND_SHORT_WINDOW, 0.0)
        + recent_smoothed_mean(away_state.recent_goal_diff, TREND_LONG_WINDOW, 0.0)
    )

    # Upset-risk proxy features.
    home_goal_vol = recent_std(home_state.recent_goal_diff, 10, 1.5)
    away_goal_vol = recent_std(away_state.recent_goal_diff, 10, 1.5)
    home_xg_vol = recent_std(home_state.recent_xg_diff, 10, 1.0)
    away_xg_vol = recent_std(away_state.recent_xg_diff, 10, 1.0)
    home_result_vol = recent_std(home_state.recent_results, 10, 0.5)
    away_result_vol = recent_std(away_state.recent_results, 10, 0.5)
    elo_abs_gap = abs(features["elo_diff"])
    win_rate_abs_gap = abs(features["win_rate_diff"])
    parity_factor = 1.0 / (1.0 + elo_abs_gap / 120.0)
    upset_risk_proxy = (
        (home_goal_vol + away_goal_vol + home_xg_vol + away_xg_vol) / 4.0
        + 0.35 * (home_result_vol + away_result_vol)
        + 0.25 * (home_ot_rate + away_ot_rate)
    ) * parity_factor

    features["home_goal_diff_vol_10"] = home_goal_vol
    features["away_goal_diff_vol_10"] = away_goal_vol
    features["goal_diff_vol_gap_10"] = home_goal_vol - away_goal_vol
    features["home_xg_diff_vol_10"] = home_xg_vol
    features["away_xg_diff_vol_10"] = away_xg_vol
    features["xg_diff_vol_gap_10"] = home_xg_vol - away_xg_vol
    features["home_result_vol_10"] = home_result_vol
    features["away_result_vol_10"] = away_result_vol
    features["result_vol_gap_10"] = home_result_vol - away_result_vol
    features["elo_abs_gap"] = elo_abs_gap
    features["win_rate_abs_gap"] = win_rate_abs_gap
    features["matchup_parity_factor"] = parity_factor
    features["upset_risk_proxy"] = upset_risk_proxy

    return features


def update_team_state(
    state: TeamState,
    win: int,
    is_home: bool,
    went_ot: int,
    goals_for: float,
    goals_against: float,
    xg_for: float,
    xg_against: float,
    max_xg_for: float,
    max_xg_against: float,
    shots_for: float,
    shots_against: float,
    assists_for: float,
    assists_against: float,
    penalties_for: float,
    penalties_against: float,
    pim_for: float,
    pim_against: float,
    opp_snapshot: dict[str, float],
) -> None:
    state.gp += 1
    state.wins += int(win)
    if is_home:
        state.home_gp += 1
        state.home_wins += int(win)
    else:
        state.away_gp += 1
        state.away_wins += int(win)

    state.goals_for += float(goals_for)
    state.goals_against += float(goals_against)
    state.xg_for += float(xg_for)
    state.xg_against += float(xg_against)
    state.max_xg_for += float(max_xg_for)
    state.max_xg_against += float(max_xg_against)
    state.shots_for += float(shots_for)
    state.shots_against += float(shots_against)
    state.assists_for += float(assists_for)
    state.assists_against += float(assists_against)
    state.penalties_for += float(penalties_for)
    state.penalties_against += float(penalties_against)
    state.pim_for += float(pim_for)
    state.pim_against += float(pim_against)
    state.ot_games += int(went_ot)
    state.opp_elo_sum += float(opp_snapshot["elo"])
    state.opp_win_rate_sum += float(opp_snapshot["win_rate"])
    state.opp_goals_for_pg_sum += float(opp_snapshot["goals_for_pg"])
    state.opp_goals_against_pg_sum += float(opp_snapshot["goals_against_pg"])
    state.opp_xg_for_pg_sum += float(opp_snapshot["xg_for_pg"])
    state.opp_xg_against_pg_sum += float(opp_snapshot["xg_against_pg"])
    state.opp_shots_for_pg_sum += float(opp_snapshot["shots_for_pg"])
    state.opp_shots_against_pg_sum += float(opp_snapshot["shots_against_pg"])

    goal_diff = float(goals_for) - float(goals_against)
    xg_diff = float(xg_for) - float(xg_against)
    shot_diff = float(shots_for) - float(shots_against)
    assist_diff = float(assists_for) - float(assists_against)
    pim_diff = float(pim_for) - float(pim_against)

    append_with_cap(state.recent_results, float(win))
    append_with_cap(state.recent_goal_diff, goal_diff)
    append_with_cap(state.recent_xg_diff, xg_diff)
    append_with_cap(state.recent_shot_diff, shot_diff)
    append_with_cap(state.recent_assist_diff, assist_diff)
    append_with_cap(state.recent_pim_diff, pim_diff)
    if is_home:
        append_with_cap(state.home_recent_results, float(win))
    else:
        append_with_cap(state.away_recent_results, float(win))


def validate_leakage_guard(feature_df: pd.DataFrame) -> None:
    counts: dict[str, int] = {}
    for row in feature_df.itertuples(index=False):
        home_count = counts.get(row.home_team, 0)
        away_count = counts.get(row.away_team, 0)
        if int(row.home_gp) != home_count or int(row.away_gp) != away_count:
            raise ValueError(
                "Leakage guard failed: home_gp/away_gp do not match pregame appearance counts."
            )
        counts[row.home_team] = home_count + 1
        counts[row.away_team] = away_count + 1


def build_pregame_features(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, TeamState]]:
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    team_states = {team: TeamState() for team in teams}

    rows: list[dict[str, float | int | str]] = []
    for game in games.itertuples(index=False):
        home_state = team_states[game.home_team]
        away_state = team_states[game.away_team]
        home_snapshot = team_pregame_snapshot(home_state)
        away_snapshot = team_pregame_snapshot(away_state)

        feature_row = compute_matchup_features(home_state, away_state)
        feature_row.update(
            {
                "game_id": game.game_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "game_num": int(game.game_num),
                "home_win": int(game.home_win),
            }
        )
        rows.append(feature_row)

        home_win = int(game.home_win)
        expected_home = feature_row["elo_home_prob"]
        elo_delta = ELO_K * (home_win - expected_home)

        update_team_state(
            home_state,
            win=home_win,
            is_home=True,
            went_ot=int(game.went_ot),
            goals_for=game.home_goals,
            goals_against=game.away_goals,
            xg_for=game.home_xg,
            xg_against=game.away_xg,
            max_xg_for=game.home_max_xg,
            max_xg_against=game.away_max_xg,
            shots_for=game.home_shots,
            shots_against=game.away_shots,
            assists_for=game.home_assists,
            assists_against=game.away_assists,
            penalties_for=game.home_penalties_committed,
            penalties_against=game.away_penalties_committed,
            pim_for=game.home_penalty_minutes,
            pim_against=game.away_penalty_minutes,
            opp_snapshot=away_snapshot,
        )
        update_team_state(
            away_state,
            win=1 - home_win,
            is_home=False,
            went_ot=int(game.went_ot),
            goals_for=game.away_goals,
            goals_against=game.home_goals,
            xg_for=game.away_xg,
            xg_against=game.home_xg,
            max_xg_for=game.away_max_xg,
            max_xg_against=game.home_max_xg,
            shots_for=game.away_shots,
            shots_against=game.home_shots,
            assists_for=game.away_assists,
            assists_against=game.home_assists,
            penalties_for=game.away_penalties_committed,
            penalties_against=game.home_penalties_committed,
            pim_for=game.away_penalty_minutes,
            pim_against=game.home_penalty_minutes,
            opp_snapshot=home_snapshot,
        )

        home_state.elo += elo_delta
        away_state.elo -= elo_delta

    feature_df = pd.DataFrame(rows)
    validate_leakage_guard(feature_df)
    return feature_df, team_states


def split_dev_holdout(feature_df: pd.DataFrame, holdout_fraction: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_games = len(feature_df)
    n_holdout = max(1, int(np.ceil(n_games * holdout_fraction)))
    dev_df = feature_df.iloc[:-n_holdout].copy()
    holdout_df = feature_df.iloc[-n_holdout:].copy()
    if dev_df.empty or holdout_df.empty:
        raise ValueError("Development or holdout split is empty.")
    if dev_df["game_num"].max() >= holdout_df["game_num"].min():
        raise ValueError("Holdout integrity failed: holdout is not strictly after development period.")
    return dev_df, holdout_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in META_COLUMNS]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_prob = np.clip(y_prob.astype(float), EPS, 1.0 - EPS)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics


def apply_calibrator(
    probs: np.ndarray,
    calibration_method: str,
    calibrator,
) -> np.ndarray:
    if calibration_method == "platt":
        return calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
    if calibration_method == "isotonic":
        return np.clip(calibrator.predict(probs), EPS, 1.0 - EPS)
    return probs


def weighted_blend(component_probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    blended = np.zeros(len(next(iter(component_probs.values()))), dtype=float)
    for key, weight in weights.items():
        if key not in component_probs:
            continue
        blended += float(weight) * component_probs[key]
    return blended


def get_bagging_windows(train_size: int) -> list[int]:
    windows = sorted({min(int(w), train_size) for w in BAGGING_WINDOWS if int(w) > 0})
    if not windows:
        windows = [train_size]
    return [w for w in windows if w > 0]


def fit_single_supervised_triplet(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    seed: int,
) -> tuple[Pipeline, RandomForestClassifier, HistGradientBoostingClassifier]:
    lr_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, C=LOGREG_C, random_state=seed)),
        ]
    )
    rf_model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=seed,
        n_jobs=RF_N_JOBS,
    )
    hgb_model = HistGradientBoostingClassifier(
        learning_rate=HGB_LEARNING_RATE,
        max_depth=HGB_MAX_DEPTH,
        max_iter=HGB_MAX_ITER,
        min_samples_leaf=HGB_MIN_SAMPLES_LEAF,
        random_state=seed,
    )
    lr_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)
    hgb_model.fit(x_train, y_train)
    return lr_model, rf_model, hgb_model


def fit_supervised_ensemble(
    x_pool: pd.DataFrame,
    y_pool: np.ndarray,
    seed: int,
) -> tuple[dict[str, list], list[dict[str, int]]]:
    windows = get_bagging_windows(len(x_pool))
    ensemble = {
        "logistic_regression": [],
        "random_forest": [],
        "hist_gradient_boosting": [],
    }
    configs: list[dict[str, int]] = []
    for offset in BAGGING_SEED_OFFSETS:
        for window in windows:
            local_seed = int(seed + offset)
            x_train = x_pool.iloc[-window:]
            y_train = y_pool[-window:]
            lr_model, rf_model, hgb_model = fit_single_supervised_triplet(
                x_train=x_train, y_train=y_train, seed=local_seed
            )
            ensemble["logistic_regression"].append(lr_model)
            ensemble["random_forest"].append(rf_model)
            ensemble["hist_gradient_boosting"].append(hgb_model)
            configs.append({"seed": local_seed, "window": int(window)})
    return ensemble, configs


def predict_supervised_ensemble(ensemble: dict[str, list], x_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    if not ensemble["logistic_regression"]:
        raise ValueError("Supervised ensemble is empty.")
    lr_probs = np.mean(
        np.column_stack([model.predict_proba(x_frame)[:, 1] for model in ensemble["logistic_regression"]]),
        axis=1,
    )
    rf_probs = np.mean(
        np.column_stack([model.predict_proba(x_frame)[:, 1] for model in ensemble["random_forest"]]),
        axis=1,
    )
    hgb_probs = np.mean(
        np.column_stack([model.predict_proba(x_frame)[:, 1] for model in ensemble["hist_gradient_boosting"]]),
        axis=1,
    )
    return {
        "logistic_regression": lr_probs,
        "random_forest": rf_probs,
        "hist_gradient_boosting": hgb_probs,
    }


def component_matrix(component_probs: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.column_stack([np.clip(component_probs[name], EPS, 1.0 - EPS) for name in names])


def crossfit_calibration(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    fold_cache: list[dict],
    method: str,
    seed: int,
) -> np.ndarray:
    calibrated = np.full(len(raw_probs), np.nan, dtype=float)
    for item in fold_cache:
        test_idx = item["test_idx"]
        train_mask = np.ones(len(raw_probs), dtype=bool)
        train_mask[test_idx] = False
        train_mask &= np.isfinite(raw_probs)
        x_train = np.clip(raw_probs[train_mask], EPS, 1.0 - EPS)
        y_train = y_true[train_mask]
        x_test = np.clip(raw_probs[test_idx], EPS, 1.0 - EPS)

        if len(y_train) == 0:
            calibrated[test_idx] = 0.5
            continue
        if len(np.unique(y_train)) < 2:
            calibrated[test_idx] = float(np.mean(y_train))
            continue

        if method == "platt":
            calibrator = LogisticRegression(max_iter=5000, random_state=seed)
            calibrator.fit(x_train.reshape(-1, 1), y_train)
            calibrated[test_idx] = calibrator.predict_proba(x_test.reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
            calibrator.fit(x_train, y_train)
            calibrated[test_idx] = np.clip(calibrator.predict(x_test), EPS, 1.0 - EPS)
        else:
            calibrated[test_idx] = x_test
    return np.clip(calibrated, EPS, 1.0 - EPS)


def cv_train_and_blend(
    dev_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
) -> dict:
    x_dev = dev_df[feature_cols]
    y_dev = dev_df["home_win"].to_numpy(dtype=int)
    base_component_names = [
        "elo_only",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]
    stack_feature_names = [
        "elo_only",
        "elo_shrunk_home_rate",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    ]

    tscv = TimeSeriesSplit(n_splits=5)
    oof_probs = {name: np.full(len(dev_df), np.nan, dtype=float) for name in base_component_names}
    oof_baseline = np.full(len(dev_df), np.nan, dtype=float)
    oof_mask = np.zeros(len(dev_df), dtype=bool)
    cv_rows: list[dict] = []
    fold_cache: list[dict] = []

    for fold_id, (train_idx, test_idx) in enumerate(tscv.split(x_dev), start=1):
        if SUPERVISED_TRAIN_WINDOW > 0 and len(train_idx) > SUPERVISED_TRAIN_WINDOW:
            train_idx = train_idx[-SUPERVISED_TRAIN_WINDOW:]

        x_train = x_dev.iloc[train_idx].copy()
        y_train = y_dev[train_idx].copy()
        x_test = x_dev.iloc[test_idx]
        y_test = y_dev[test_idx]
        baseline_prior = float(np.mean(y_train))

        fold_ensemble, _fold_configs = fit_supervised_ensemble(
            x_pool=x_train,
            y_pool=y_train,
            seed=seed + fold_id * 1000,
        )
        sup_test = predict_supervised_ensemble(fold_ensemble, x_test)
        sup_train = predict_supervised_ensemble(fold_ensemble, x_train)

        fold_probs = {
            "elo_only": x_test["elo_home_prob"].to_numpy(),
            "logistic_regression": sup_test["logistic_regression"],
            "random_forest": sup_test["random_forest"],
            "hist_gradient_boosting": sup_test["hist_gradient_boosting"],
        }
        train_probs = {
            "elo_only": x_train["elo_home_prob"].to_numpy(),
            "logistic_regression": sup_train["logistic_regression"],
            "random_forest": sup_train["random_forest"],
            "hist_gradient_boosting": sup_train["hist_gradient_boosting"],
        }

        for model_name, probs in fold_probs.items():
            oof_probs[model_name][test_idx] = probs
        oof_baseline[test_idx] = baseline_prior
        oof_mask[test_idx] = True
        fold_cache.append(
            {
                "fold_id": fold_id,
                "y_true": y_test,
                "test_idx": test_idx,
                "y_train": y_train,
                "baseline_prior": baseline_prior,
                "train_component_probs": train_probs,
                "test_component_probs": fold_probs,
            }
        )

        for model_name, probs in fold_probs.items():
            row = {"model": model_name, "fold": fold_id, "n_samples": int(len(test_idx))}
            row.update(compute_metrics(y_test, probs))
            cv_rows.append(row)

    if not oof_mask.any():
        raise ValueError("CV OOF generation failed: no rows received OOF predictions.")

    for model_name in base_component_names:
        row = {"model": model_name, "fold": "OOF", "n_samples": int(oof_mask.sum())}
        row.update(compute_metrics(y_dev[oof_mask], oof_probs[model_name][oof_mask]))
        cv_rows.append(row)

    alpha_grid = np.arange(0.0, 1.0001, ELO_SHRINK_ALPHA_STEP)
    best_alpha = 1.0
    best_alpha_score = None
    for alpha in alpha_grid:
        probs = alpha * oof_probs["elo_only"][oof_mask] + (1.0 - alpha) * oof_baseline[oof_mask]
        ll = log_loss(y_dev[oof_mask], np.clip(probs, EPS, 1.0 - EPS))
        if best_alpha_score is None or ll < best_alpha_score:
            best_alpha_score = ll
            best_alpha = float(alpha)
    oof_elo_shrunk = best_alpha * oof_probs["elo_only"] + (1.0 - best_alpha) * oof_baseline
    oof_probs["elo_shrunk_home_rate"] = oof_elo_shrunk
    for item in fold_cache:
        item["train_component_probs"]["elo_shrunk_home_rate"] = (
            best_alpha * item["train_component_probs"]["elo_only"] + (1.0 - best_alpha) * item["baseline_prior"]
        )
        item["test_component_probs"]["elo_shrunk_home_rate"] = (
            best_alpha * item["test_component_probs"]["elo_only"] + (1.0 - best_alpha) * item["baseline_prior"]
        )
    elo_shrunk_metrics = compute_metrics(y_dev[oof_mask], oof_elo_shrunk[oof_mask])
    cv_rows.append(
        {"model": "elo_shrunk_home_rate", "fold": "OOF", "n_samples": int(oof_mask.sum()), **elo_shrunk_metrics}
    )

    weight_grid = np.arange(0.0, 1.0001, 0.05)
    best = None
    best_weights = None
    best_blend_full = None
    for w_elo in weight_grid:
        for w_lr in weight_grid:
            for w_rf in weight_grid:
                w_hgb = 1.0 - w_elo - w_lr - w_rf
                if w_hgb < -1e-9:
                    continue
                if w_hgb < 0:
                    w_hgb = 0.0
                current_weights = {
                    "elo_only": float(w_elo),
                    "logistic_regression": float(w_lr),
                    "random_forest": float(w_rf),
                    "hist_gradient_boosting": float(w_hgb),
                }
                blended = weighted_blend(oof_probs, current_weights)
                metrics = compute_metrics(y_dev[oof_mask], blended[oof_mask])
                candidate = (metrics["log_loss"], metrics["brier"])
                if best is None or candidate < best:
                    best = candidate
                    best_weights = current_weights
                    best_blend_full = blended

    if best_weights is None or best_blend_full is None:
        raise RuntimeError("Failed to find blend weights.")

    best_blend = best_blend_full[oof_mask]
    blend_metrics = compute_metrics(y_dev[oof_mask], best_blend)
    cv_rows.append(
        {"model": "blend_uncalibrated", "fold": "OOF", "n_samples": int(oof_mask.sum()), **blend_metrics}
    )

    oof_blend_platt = crossfit_calibration(
        raw_probs=best_blend_full,
        y_true=y_dev,
        fold_cache=fold_cache,
        method="platt",
        seed=seed,
    )
    oof_blend_isotonic = crossfit_calibration(
        raw_probs=best_blend_full,
        y_true=y_dev,
        fold_cache=fold_cache,
        method="isotonic",
        seed=seed,
    )

    platt_metrics = compute_metrics(y_dev[oof_mask], oof_blend_platt[oof_mask])
    cv_rows.append(
        {"model": "blend_platt_calibrated", "fold": "OOF", "n_samples": int(oof_mask.sum()), **platt_metrics}
    )

    isotonic_metrics = compute_metrics(y_dev[oof_mask], oof_blend_isotonic[oof_mask])
    cv_rows.append(
        {
            "model": "blend_isotonic_calibrated",
            "fold": "OOF",
            "n_samples": int(oof_mask.sum()),
            **isotonic_metrics,
        }
    )

    blend_metric_by_calibration = {
        "none": blend_metrics,
        "platt": platt_metrics,
        "isotonic": isotonic_metrics,
    }
    best_blend_calibration_method = min(
        blend_metric_by_calibration.keys(),
        key=lambda method: (
            blend_metric_by_calibration[method]["log_loss"],
            blend_metric_by_calibration[method]["brier"],
        ),
    )
    blend_calibration_method = "none"
    use_blend_calibration = False
    blend_calibrator = None
    oof_blend_final = best_blend_full
    if (
        best_blend_calibration_method != "none"
        and (
            blend_metrics["log_loss"] - blend_metric_by_calibration[best_blend_calibration_method]["log_loss"]
        )
        >= BLEND_CALIBRATION_MIN_IMPROVEMENT
    ):
        use_blend_calibration = True
        blend_calibration_method = best_blend_calibration_method
        if best_blend_calibration_method == "platt":
            blend_calibrator = LogisticRegression(max_iter=5000, random_state=seed)
            blend_calibrator.fit(best_blend.reshape(-1, 1), y_dev[oof_mask])
            oof_blend_final = oof_blend_platt
        else:
            blend_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
            blend_calibrator.fit(best_blend, y_dev[oof_mask])
            oof_blend_final = oof_blend_isotonic

    oof_stack_uncal = np.full(len(dev_df), np.nan, dtype=float)
    for item in fold_cache:
        test_idx = item["test_idx"]
        train_mask = oof_mask.copy()
        train_mask[test_idx] = False
        x_meta_train = component_matrix(
            {name: oof_probs[name][train_mask] for name in stack_feature_names},
            stack_feature_names,
        )
        y_train_fold = y_dev[train_mask]
        x_meta_test = component_matrix(
            {name: oof_probs[name][test_idx] for name in stack_feature_names},
            stack_feature_names,
        )
        if len(np.unique(y_train_fold)) < 2:
            oof_stack_uncal[test_idx] = float(np.mean(y_train_fold))
            continue
        fold_stacker = LogisticRegression(max_iter=5000, C=STACKER_C, random_state=seed + item["fold_id"])
        fold_stacker.fit(x_meta_train, y_train_fold)
        oof_stack_uncal[test_idx] = fold_stacker.predict_proba(x_meta_test)[:, 1]

    stack_metrics = compute_metrics(y_dev[oof_mask], oof_stack_uncal[oof_mask])
    cv_rows.append(
        {"model": "stack_uncalibrated", "fold": "OOF", "n_samples": int(oof_mask.sum()), **stack_metrics}
    )

    oof_stack_platt = crossfit_calibration(
        raw_probs=oof_stack_uncal,
        y_true=y_dev,
        fold_cache=fold_cache,
        method="platt",
        seed=seed,
    )
    oof_stack_isotonic = crossfit_calibration(
        raw_probs=oof_stack_uncal,
        y_true=y_dev,
        fold_cache=fold_cache,
        method="isotonic",
        seed=seed,
    )
    stack_platt_metrics = compute_metrics(y_dev[oof_mask], oof_stack_platt[oof_mask])
    stack_isotonic_metrics = compute_metrics(y_dev[oof_mask], oof_stack_isotonic[oof_mask])
    cv_rows.append(
        {
            "model": "stack_platt_calibrated",
            "fold": "OOF",
            "n_samples": int(oof_mask.sum()),
            **stack_platt_metrics,
        }
    )
    cv_rows.append(
        {
            "model": "stack_isotonic_calibrated",
            "fold": "OOF",
            "n_samples": int(oof_mask.sum()),
            **stack_isotonic_metrics,
        }
    )

    stack_metric_by_calibration = {
        "none": stack_metrics,
        "platt": stack_platt_metrics,
        "isotonic": stack_isotonic_metrics,
    }
    best_stack_calibration_method = min(
        stack_metric_by_calibration.keys(),
        key=lambda method: (
            stack_metric_by_calibration[method]["log_loss"],
            stack_metric_by_calibration[method]["brier"],
        ),
    )
    stack_calibration_method = "none"
    use_stack_calibration = False
    stack_calibrator = None
    oof_stack_final = oof_stack_uncal
    if (
        best_stack_calibration_method != "none"
        and (
            stack_metrics["log_loss"] - stack_metric_by_calibration[best_stack_calibration_method]["log_loss"]
        )
        >= STACKER_CALIBRATION_MIN_IMPROVEMENT
    ):
        use_stack_calibration = True
        stack_calibration_method = best_stack_calibration_method
        if best_stack_calibration_method == "platt":
            stack_calibrator = LogisticRegression(max_iter=5000, random_state=seed)
            stack_calibrator.fit(oof_stack_uncal[oof_mask].reshape(-1, 1), y_dev[oof_mask])
            oof_stack_final = oof_stack_platt
        else:
            stack_calibrator = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1.0 - EPS)
            stack_calibrator.fit(oof_stack_uncal[oof_mask], y_dev[oof_mask])
            oof_stack_final = oof_stack_isotonic

    stacker_model = LogisticRegression(max_iter=5000, C=STACKER_C, random_state=seed)
    stacker_model.fit(
        component_matrix({name: oof_probs[name][oof_mask] for name in stack_feature_names}, stack_feature_names),
        y_dev[oof_mask],
    )

    oof_source_predictions = {
        "elo_only": oof_probs["elo_only"],
        "elo_shrunk_home_rate": oof_probs["elo_shrunk_home_rate"],
        "blend_final": oof_blend_final,
        "stack_final": oof_stack_final,
    }
    source_oof_ll = {
        name: float(log_loss(y_dev[oof_mask], np.clip(preds[oof_mask], EPS, 1.0 - EPS)))
        for name, preds in oof_source_predictions.items()
    }

    recent_folds = sorted(fold_cache, key=lambda item: item["fold_id"])[-RECENT_SELECTOR_LOOKBACK:]
    source_recent_ll: dict[str, float] = {name: float("nan") for name in oof_source_predictions}
    if recent_folds:
        recent_weights = np.array(
            [RECENT_SELECTOR_EXP_BASE**idx for idx in range(len(recent_folds))],
            dtype=float,
        )
        recent_weights /= recent_weights.sum()
        for source_name, source_preds in oof_source_predictions.items():
            fold_ll = []
            for item in recent_folds:
                test_idx = item["test_idx"]
                fold_ll.append(
                    log_loss(
                        item["y_true"],
                        np.clip(source_preds[test_idx], EPS, 1.0 - EPS),
                    )
                )
            source_recent_ll[source_name] = float(np.dot(recent_weights, np.asarray(fold_ll, dtype=float)))
    else:
        source_recent_ll = source_oof_ll.copy()

    source_objective_score = {
        name: (
            RECENT_OBJECTIVE_WEIGHT * source_recent_ll[name]
            + (1.0 - RECENT_OBJECTIVE_WEIGHT) * source_oof_ll[name]
        )
        for name in oof_source_predictions
    }
    best_oof_source = min(source_oof_ll, key=source_oof_ll.get)

    # Choose robust Elo-family source first, then allow advanced sources only if they
    # win on both OOF and recent folds by explicit margins.
    elo_candidates = ["elo_only", "elo_shrunk_home_rate"]
    selected_prediction_source = min(elo_candidates, key=lambda name: source_objective_score[name])
    other_elo = "elo_shrunk_home_rate" if selected_prediction_source == "elo_only" else "elo_only"
    if (
        source_objective_score[other_elo] < source_objective_score[selected_prediction_source]
        and (
            source_objective_score[selected_prediction_source] - source_objective_score[other_elo]
        ) < RECENT_OBJECTIVE_SWITCH_MIN_DELTA
    ):
        selected_prediction_source = other_elo

    advanced_candidates = []
    for candidate in ("blend_final", "stack_final"):
        if (
            source_oof_ll[candidate] + ADVANCED_SOURCE_OOF_GAIN_MIN
            < source_oof_ll[selected_prediction_source]
            and source_recent_ll[candidate] + ADVANCED_SOURCE_RECENT_GAIN_MIN
            < source_recent_ll[selected_prediction_source]
        ):
            advanced_candidates.append(candidate)
    if advanced_candidates:
        selected_prediction_source = min(advanced_candidates, key=lambda name: source_objective_score[name])
    recent_selector_switched = selected_prediction_source != best_oof_source

    if selected_prediction_source == "elo_only":
        selected_model_name = "elo_only_recent_objective"
    elif selected_prediction_source == "elo_shrunk_home_rate":
        selected_model_name = "elo_shrunk_recent_objective"
    elif selected_prediction_source == "stack_final":
        if stack_calibration_method == "none":
            selected_model_name = "stack_recent_objective"
        else:
            selected_model_name = f"stack_recent_objective_{stack_calibration_method}"
    else:
        if blend_calibration_method == "none":
            selected_model_name = "blend_recent_objective"
        else:
            selected_model_name = f"blend_recent_objective_{blend_calibration_method}"

    train_idx = np.arange(len(x_dev))
    if SUPERVISED_TRAIN_WINDOW > 0 and len(train_idx) > SUPERVISED_TRAIN_WINDOW:
        train_idx = train_idx[-SUPERVISED_TRAIN_WINDOW:]
    full_supervised_ensemble, full_ensemble_configs = fit_supervised_ensemble(
        x_pool=x_dev.iloc[train_idx],
        y_pool=y_dev[train_idx],
        seed=seed,
    )

    blend_summary = pd.DataFrame(
        [
            {
                "component": "elo_only",
                "weight": best_weights["elo_only"],
                "cv_log_loss": blend_metrics["log_loss"],
                "cv_brier": blend_metrics["brier"],
                "selected": best_weights["elo_only"] > 0.0,
            },
            {
                "component": "elo_shrunk_home_rate",
                "weight": best_alpha,
                "cv_log_loss": elo_shrunk_metrics["log_loss"],
                "cv_brier": elo_shrunk_metrics["brier"],
                "selected": selected_prediction_source == "elo_shrunk_home_rate",
            },
            {
                "component": "logistic_regression",
                "weight": best_weights["logistic_regression"],
                "cv_log_loss": blend_metrics["log_loss"],
                "cv_brier": blend_metrics["brier"],
                "selected": best_weights["logistic_regression"] > 0.0,
            },
            {
                "component": "random_forest",
                "weight": best_weights["random_forest"],
                "cv_log_loss": blend_metrics["log_loss"],
                "cv_brier": blend_metrics["brier"],
                "selected": best_weights["random_forest"] > 0.0,
            },
            {
                "component": "hist_gradient_boosting",
                "weight": best_weights["hist_gradient_boosting"],
                "cv_log_loss": blend_metrics["log_loss"],
                "cv_brier": blend_metrics["brier"],
                "selected": best_weights["hist_gradient_boosting"] > 0.0,
            },
            {
                "component": "blend_uncalibrated",
                "weight": 1.0,
                "cv_log_loss": blend_metrics["log_loss"],
                "cv_brier": blend_metrics["brier"],
                "selected": selected_prediction_source == "blend_final" and blend_calibration_method == "none",
            },
            {
                "component": "blend_platt_calibrated",
                "weight": 1.0 if blend_calibration_method == "platt" else 0.0,
                "cv_log_loss": platt_metrics["log_loss"],
                "cv_brier": platt_metrics["brier"],
                "selected": selected_prediction_source == "blend_final" and blend_calibration_method == "platt",
            },
            {
                "component": "blend_isotonic_calibrated",
                "weight": 1.0 if blend_calibration_method == "isotonic" else 0.0,
                "cv_log_loss": isotonic_metrics["log_loss"],
                "cv_brier": isotonic_metrics["brier"],
                "selected": selected_prediction_source == "blend_final" and blend_calibration_method == "isotonic",
            },
            {
                "component": "stack_uncalibrated",
                "weight": 1.0,
                "cv_log_loss": stack_metrics["log_loss"],
                "cv_brier": stack_metrics["brier"],
                "selected": selected_prediction_source == "stack_final" and stack_calibration_method == "none",
            },
            {
                "component": "stack_platt_calibrated",
                "weight": 1.0 if stack_calibration_method == "platt" else 0.0,
                "cv_log_loss": stack_platt_metrics["log_loss"],
                "cv_brier": stack_platt_metrics["brier"],
                "selected": selected_prediction_source == "stack_final" and stack_calibration_method == "platt",
            },
            {
                "component": "stack_isotonic_calibrated",
                "weight": 1.0 if stack_calibration_method == "isotonic" else 0.0,
                "cv_log_loss": stack_isotonic_metrics["log_loss"],
                "cv_brier": stack_isotonic_metrics["brier"],
                "selected": selected_prediction_source == "stack_final" and stack_calibration_method == "isotonic",
            },
            {
                "component": "recent_objective_elo_only_log_loss",
                "weight": np.nan,
                "cv_log_loss": source_recent_ll["elo_only"],
                "cv_brier": np.nan,
                "selected": selected_prediction_source == "elo_only",
            },
            {
                "component": "recent_objective_elo_shrunk_log_loss",
                "weight": np.nan,
                "cv_log_loss": source_recent_ll["elo_shrunk_home_rate"],
                "cv_brier": np.nan,
                "selected": selected_prediction_source == "elo_shrunk_home_rate",
            },
            {
                "component": "recent_objective_blend_log_loss",
                "weight": np.nan,
                "cv_log_loss": source_recent_ll["blend_final"],
                "cv_brier": np.nan,
                "selected": selected_prediction_source == "blend_final",
            },
            {
                "component": "recent_objective_stack_log_loss",
                "weight": np.nan,
                "cv_log_loss": source_recent_ll["stack_final"],
                "cv_brier": np.nan,
                "selected": selected_prediction_source == "stack_final",
            },
            {
                "component": "recent_objective_selected_score",
                "weight": np.nan,
                "cv_log_loss": source_objective_score[selected_prediction_source],
                "cv_brier": np.nan,
                "selected": True,
            },
        ]
    )

    cv_summary = pd.DataFrame(cv_rows)
    return {
        "feature_cols": feature_cols,
        "weights": best_weights,
        "elo_shrink_alpha": best_alpha,
        "baseline_prior": float(np.mean(y_dev)),
        "stack_feature_names": stack_feature_names,
        "stacker_model": stacker_model,
        "use_blend_calibration": use_blend_calibration,
        "blend_calibrator": blend_calibrator if use_blend_calibration else None,
        "blend_calibration_method": blend_calibration_method,
        "use_stack_calibration": use_stack_calibration,
        "stack_calibrator": stack_calibrator if use_stack_calibration else None,
        "stack_calibration_method": stack_calibration_method,
        "cv_summary": cv_summary,
        "blend_summary": blend_summary,
        "supervised_ensemble": full_supervised_ensemble,
        "supervised_ensemble_configs": full_ensemble_configs,
        "selected_model_name": selected_model_name,
        "selected_prediction_source": selected_prediction_source,
        "recent_selector_switched": recent_selector_switched,
        "recent_selector_elo_log_loss": source_recent_ll["elo_only"],
        "recent_selector_elo_shrunk_log_loss": source_recent_ll["elo_shrunk_home_rate"],
        "recent_selector_blend_log_loss": source_recent_ll["blend_final"],
        "recent_selector_stack_log_loss": source_recent_ll["stack_final"],
        "recent_selector_best_oof_source": best_oof_source,
        "recent_selector_objective_scores": source_objective_score,
        "blend_metrics": blend_metrics,
        "stack_metrics": stack_metrics,
        "calibrated_metrics": blend_metric_by_calibration[blend_calibration_method],
    }


def predict_components(bundle: dict, x_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    probs_elo = x_frame["elo_home_prob"].to_numpy(dtype=float)
    probs_elo_shrunk = (
        bundle["elo_shrink_alpha"] * probs_elo
        + (1.0 - bundle["elo_shrink_alpha"]) * float(bundle["baseline_prior"])
    )
    sup_probs = predict_supervised_ensemble(bundle["supervised_ensemble"], x_frame)
    return {
        "elo_only": probs_elo,
        "elo_shrunk_home_rate": probs_elo_shrunk,
        "logistic_regression": sup_probs["logistic_regression"],
        "random_forest": sup_probs["random_forest"],
        "hist_gradient_boosting": sup_probs["hist_gradient_boosting"],
    }


def blend_probabilities(component_probs: dict[str, np.ndarray], bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    blended = weighted_blend(component_probs, bundle["weights"])
    final_probs = apply_calibrator(
        blended,
        bundle.get("blend_calibration_method", "none"),
        bundle.get("blend_calibrator"),
    )
    return blended, final_probs


def stack_probabilities(component_probs: dict[str, np.ndarray], bundle: dict) -> tuple[np.ndarray, np.ndarray]:
    stack_x = component_matrix(component_probs, bundle["stack_feature_names"])
    raw_stack = bundle["stacker_model"].predict_proba(stack_x)[:, 1]
    final_stack = apply_calibrator(
        raw_stack,
        bundle.get("stack_calibration_method", "none"),
        bundle.get("stack_calibrator"),
    )
    return raw_stack, final_stack


def evaluate_holdout(
    dev_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    bundle: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_holdout = holdout_df[bundle["feature_cols"]]
    y_holdout = holdout_df["home_win"].to_numpy(dtype=int)
    y_dev = dev_df["home_win"].to_numpy(dtype=int)

    component_probs = predict_components(bundle, x_holdout)
    prob_blend, prob_blend_final = blend_probabilities(component_probs, bundle)
    prob_stack, prob_stack_final = stack_probabilities(component_probs, bundle)
    if bundle["selected_prediction_source"] == "elo_only":
        selected_probs = component_probs["elo_only"]
    elif bundle["selected_prediction_source"] == "elo_shrunk_home_rate":
        selected_probs = component_probs["elo_shrunk_home_rate"]
    elif bundle["selected_prediction_source"] == "stack_final":
        selected_probs = prob_stack_final
    elif bundle["selected_prediction_source"] == "blend_final":
        selected_probs = prob_blend_final
    else:
        selected_probs = prob_blend_final
    baseline_prob = np.full(len(y_holdout), y_dev.mean())

    model_prob_pairs: list[tuple[str, np.ndarray]] = [
        ("baseline_home_rate", baseline_prob),
        ("elo_only", component_probs["elo_only"]),
        ("elo_shrunk_home_rate", component_probs["elo_shrunk_home_rate"]),
        ("logistic_regression", component_probs["logistic_regression"]),
        ("random_forest", component_probs["random_forest"]),
        ("hist_gradient_boosting", component_probs["hist_gradient_boosting"]),
        ("blend_uncalibrated", prob_blend),
        ("blend_final", prob_blend_final),
        ("stack_uncalibrated", prob_stack),
        ("stack_final", prob_stack_final),
    ]
    if bundle["selected_model_name"] != "blend_uncalibrated":
        model_prob_pairs.append((bundle["selected_model_name"], selected_probs))

    holdout_rows = []
    for model_name, probs in model_prob_pairs:
        row = {"model": model_name, "n_samples": int(len(y_holdout))}
        row.update(compute_metrics(y_holdout, probs))
        holdout_rows.append(row)

    holdout_metrics = pd.DataFrame(holdout_rows)
    baseline_ll = float(
        holdout_metrics.loc[holdout_metrics["model"] == "baseline_home_rate", "log_loss"].iloc[0]
    )
    selected_ll = float(
        holdout_metrics.loc[holdout_metrics["model"] == bundle["selected_model_name"], "log_loss"].iloc[0]
    )
    holdout_metrics["delta_vs_baseline_log_loss"] = np.nan
    holdout_metrics.loc[
        holdout_metrics["model"] == bundle["selected_model_name"], "delta_vs_baseline_log_loss"
    ] = selected_ll - baseline_ll

    holdout_predictions = holdout_df[
        ["game_id", "home_team", "away_team", "game_num", "home_win"]
    ].copy()
    holdout_predictions["baseline_prob"] = baseline_prob
    holdout_predictions["prob_elo"] = component_probs["elo_only"]
    holdout_predictions["prob_elo_shrunk"] = component_probs["elo_shrunk_home_rate"]
    holdout_predictions["prob_logreg"] = component_probs["logistic_regression"]
    holdout_predictions["prob_random_forest"] = component_probs["random_forest"]
    holdout_predictions["prob_hgb"] = component_probs["hist_gradient_boosting"]
    holdout_predictions["prob_blend"] = prob_blend
    holdout_predictions["prob_blend_final"] = prob_blend_final
    holdout_predictions["prob_stack"] = prob_stack
    holdout_predictions["prob_stack_final"] = prob_stack_final
    holdout_predictions["prob_final"] = selected_probs
    holdout_predictions["pred_final"] = (holdout_predictions["prob_final"] >= 0.5).astype(int)
    holdout_predictions["is_correct_final"] = (
        holdout_predictions["pred_final"] == holdout_predictions["home_win"]
    ).astype(int)
    return holdout_metrics, holdout_predictions


def build_round1_features(round1_df: pd.DataFrame, team_states: dict[str, TeamState]) -> pd.DataFrame:
    unknown_teams = sorted(
        (set(round1_df["home_team"]) | set(round1_df["away_team"])) - set(team_states.keys())
    )
    if unknown_teams:
        raise ValueError(f"Round 1 file contains teams missing from season data: {unknown_teams}")

    rows: list[dict[str, float | str]] = []
    for matchup in round1_df.itertuples(index=False):
        home_state = team_states[matchup.home_team]
        away_state = team_states[matchup.away_team]
        features = compute_matchup_features(home_state, away_state)
        features.update(
            {
                "game_id": matchup.game_id,
                "home_team": matchup.home_team,
                "away_team": matchup.away_team,
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)


def fit_full_and_score_round1(
    pregame_df: pd.DataFrame,
    round1_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    bundle: dict,
    final_team_states: dict[str, TeamState],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, list],
]:
    x_all = pregame_df[feature_cols]
    y_all = pregame_df["home_win"].to_numpy(dtype=int)

    train_idx = np.arange(len(x_all))
    if SUPERVISED_TRAIN_WINDOW > 0 and len(train_idx) > SUPERVISED_TRAIN_WINDOW:
        train_idx = train_idx[-SUPERVISED_TRAIN_WINDOW:]
    supervised_ensemble_all, _ensemble_configs = fit_supervised_ensemble(
        x_pool=x_all.iloc[train_idx],
        y_pool=y_all[train_idx],
        seed=seed,
    )
    supervised_all_probs = predict_supervised_ensemble(supervised_ensemble_all, x_all)

    all_component_probs = {
        "elo_only": x_all["elo_home_prob"].to_numpy(dtype=float),
        "logistic_regression": supervised_all_probs["logistic_regression"],
        "random_forest": supervised_all_probs["random_forest"],
        "hist_gradient_boosting": supervised_all_probs["hist_gradient_boosting"],
    }
    full_home_prior = float(np.mean(y_all))
    all_component_probs["elo_shrunk_home_rate"] = (
        bundle["elo_shrink_alpha"] * all_component_probs["elo_only"]
        + (1.0 - bundle["elo_shrink_alpha"]) * full_home_prior
    )
    all_blend = weighted_blend(all_component_probs, bundle["weights"])
    all_blend_final = apply_calibrator(
        all_blend,
        bundle.get("blend_calibration_method", "none"),
        bundle.get("blend_calibrator"),
    )
    all_stack = bundle["stacker_model"].predict_proba(
        component_matrix(all_component_probs, bundle["stack_feature_names"])
    )[:, 1]
    all_stack_final = apply_calibrator(
        all_stack,
        bundle.get("stack_calibration_method", "none"),
        bundle.get("stack_calibrator"),
    )

    selected_prediction_source = bundle["selected_prediction_source"]
    if selected_prediction_source == "elo_only":
        all_selected = all_component_probs["elo_only"]
    elif selected_prediction_source == "elo_shrunk_home_rate":
        all_selected = all_component_probs["elo_shrunk_home_rate"]
    elif selected_prediction_source == "stack_final":
        all_selected = all_stack_final
    else:
        all_selected = all_blend_final

    historical_probs = pregame_df[["game_id", "home_team", "away_team", "home_win"]].copy()
    historical_probs["home_win_prob"] = all_selected

    round1_features = build_round1_features(round1_df, final_team_states)
    x_round1 = round1_features[feature_cols]
    round_supervised_probs = predict_supervised_ensemble(supervised_ensemble_all, x_round1)
    round_comp_probs = {
        "elo_only": x_round1["elo_home_prob"].to_numpy(dtype=float),
        "logistic_regression": round_supervised_probs["logistic_regression"],
        "random_forest": round_supervised_probs["random_forest"],
        "hist_gradient_boosting": round_supervised_probs["hist_gradient_boosting"],
    }
    round_comp_probs["elo_shrunk_home_rate"] = (
        bundle["elo_shrink_alpha"] * round_comp_probs["elo_only"]
        + (1.0 - bundle["elo_shrink_alpha"]) * full_home_prior
    )
    round_blend = weighted_blend(round_comp_probs, bundle["weights"])
    round_blend_final = apply_calibrator(
        round_blend,
        bundle.get("blend_calibration_method", "none"),
        bundle.get("blend_calibrator"),
    )
    round_stack = bundle["stacker_model"].predict_proba(
        component_matrix(round_comp_probs, bundle["stack_feature_names"])
    )[:, 1]
    round_stack_final = apply_calibrator(
        round_stack,
        bundle.get("stack_calibration_method", "none"),
        bundle.get("stack_calibrator"),
    )
    if selected_prediction_source == "elo_only":
        round_selected = round_comp_probs["elo_only"]
    elif selected_prediction_source == "elo_shrunk_home_rate":
        round_selected = round_comp_probs["elo_shrunk_home_rate"]
    elif selected_prediction_source == "stack_final":
        round_selected = round_stack_final
    else:
        round_selected = round_blend_final
    round_selected = np.clip(round_selected, OUTPUT_PROB_MIN, OUTPUT_PROB_MAX)

    round1_output = round1_features[["game_id", "home_team", "away_team"]].copy()
    round1_output["home_win_prob"] = round_selected

    if len(round1_output) != 16:
        raise ValueError(f"Round 1 output must have exactly 16 rows. Found: {len(round1_output)}")
    if round1_output["game_id"].duplicated().any():
        raise ValueError("Round 1 output has duplicate game_id values.")
    if not ((round1_output["home_win_prob"] > 0).all() and (round1_output["home_win_prob"] < 1).all()):
        raise ValueError("Round 1 probabilities must be strictly between 0 and 1.")

    return round1_output, historical_probs, round1_features, supervised_ensemble_all


def min_max_normalize(values: pd.Series) -> pd.Series:
    low = float(values.min())
    high = float(values.max())
    if np.isclose(high, low):
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - low) / (high - low)


def build_power_rankings(
    historical_probs: pd.DataFrame,
    team_states: dict[str, TeamState],
) -> pd.DataFrame:
    strength_sum = {team: 0.0 for team in team_states}
    games_played = {team: 0 for team in team_states}

    for game in historical_probs.itertuples(index=False):
        home_team = game.home_team
        away_team = game.away_team
        home_prob = float(game.home_win_prob)
        away_prob = 1.0 - home_prob

        strength_sum[home_team] += home_prob
        strength_sum[away_team] += away_prob
        games_played[home_team] += 1
        games_played[away_team] += 1

    ranking = pd.DataFrame(
        {
            "team": sorted(team_states.keys()),
        }
    )
    ranking["elo_rating"] = ranking["team"].map(lambda team: team_states[team].elo)
    ranking["model_strength"] = ranking["team"].map(
        lambda team: strength_sum[team] / games_played[team] if games_played[team] else 0.5
    )
    ranking["elo_norm"] = min_max_normalize(ranking["elo_rating"])
    ranking["model_strength_norm"] = min_max_normalize(ranking["model_strength"])
    ranking["power_score"] = 0.7 * ranking["elo_norm"] + 0.3 * ranking["model_strength_norm"]
    ranking = ranking.sort_values("power_score", ascending=False).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking[["rank", "team", "power_score", "elo_rating", "model_strength"]]


def build_feature_importance(
    feature_cols: list[str],
    supervised_ensemble: dict[str, list],
) -> pd.DataFrame:
    logreg_models = supervised_ensemble["logistic_regression"]
    rf_models = supervised_ensemble["random_forest"]
    if not logreg_models or not rf_models:
        raise ValueError("Cannot compute feature importance from empty supervised ensemble.")

    logreg_coef = np.mean(
        np.vstack([np.abs(model.named_steps["model"].coef_[0]) for model in logreg_models]),
        axis=0,
    )
    rf_importance = np.mean(
        np.vstack([model.feature_importances_ for model in rf_models]),
        axis=0,
    )
    rows = []
    for feature, value in zip(feature_cols, np.abs(logreg_coef)):
        rows.append(
            {
                "feature": feature,
                "importance_type": "logreg_abs_coef",
                "importance_value": float(value),
            }
        )
    for feature, value in zip(feature_cols, rf_importance):
        rows.append(
            {
                "feature": feature,
                "importance_type": "rf_feature_importance",
                "importance_value": float(value),
            }
        )
    importance_df = pd.DataFrame(rows).sort_values("importance_value", ascending=False).reset_index(drop=True)
    return importance_df


def save_outputs(
    output_dir: Path,
    round1_output: pd.DataFrame,
    power_rankings: pd.DataFrame,
    cv_summary: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    blend_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    metadata: dict,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    round1_path = output_dir / "round1_home_win_probabilities.csv"
    ranking_path = output_dir / "team_power_rankings.csv"
    cv_path = output_dir / "model_cv_summary.csv"
    holdout_metrics_path = output_dir / "holdout_metrics.csv"
    holdout_preds_path = output_dir / "holdout_predictions.csv"
    blend_path = output_dir / "model_blend_summary.csv"
    importance_path = output_dir / "feature_importance.csv"
    metadata_path = output_dir / "run_metadata.json"

    round1_output.to_csv(round1_path, index=False)
    power_rankings.to_csv(ranking_path, index=False)
    cv_summary.to_csv(cv_path, index=False)
    holdout_metrics.to_csv(holdout_metrics_path, index=False)
    holdout_predictions.to_csv(holdout_preds_path, index=False)
    blend_summary.to_csv(blend_path, index=False)
    feature_importance.to_csv(importance_path, index=False)
    with metadata_path.open("w", encoding="utf-8") as outfile:
        json.dump(metadata, outfile, indent=2)
    return [
        round1_path,
        ranking_path,
        cv_path,
        holdout_metrics_path,
        holdout_preds_path,
        blend_path,
        importance_path,
        metadata_path,
    ]


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    season_file = Path(args.season_file)
    round1_file = Path(args.round1_file)
    output_dir = Path(args.output_dir)

    print("1. Loading inputs...")
    season_df, round1_df = load_inputs(season_file, round1_file)
    print(f"   Season rows: {len(season_df)}")
    print(f"   Round 1 matchups: {len(round1_df)}")

    print("2. Aggregating game-level data...")
    games = aggregate_games(season_df)
    print(f"   Aggregated games: {len(games)}")

    print("3. Building leak-free pregame features...")
    pregame_df, final_team_states = build_pregame_features(games)
    feature_cols = get_feature_columns(pregame_df)
    print(f"   Pregame rows: {len(pregame_df)}")
    print(f"   Feature count: {len(feature_cols)}")

    print("4. Splitting development and holdout sets (85/15 by time)...")
    dev_df, holdout_df = split_dev_holdout(pregame_df, holdout_fraction=0.15)
    print(f"   Development games: {len(dev_df)}")
    print(f"   Holdout games: {len(holdout_df)}")

    print("5. Running CV training and blend selection...")
    bundle = cv_train_and_blend(dev_df, feature_cols, args.seed)
    print(f"   Selected model: {bundle['selected_model_name']}")
    print(f"   Prediction source: {bundle['selected_prediction_source']}")
    print(f"   Blend weights: {bundle['weights']}")
    print(
        "   Calibration: "
        f"blend={bundle['blend_calibration_method']}, "
        f"stack={bundle['stack_calibration_method']}"
    )
    if bundle["recent_selector_switched"]:
        print(
            "   Recent objective switched away from best OOF source "
            f"(recent elo ll={bundle['recent_selector_elo_log_loss']:.4f}, "
            f"elo_shrunk ll={bundle.get('recent_selector_elo_shrunk_log_loss', float('nan')):.4f}, "
            f"blend ll={bundle['recent_selector_blend_log_loss']:.4f}, "
            f"stack ll={bundle.get('recent_selector_stack_log_loss', float('nan')):.4f})"
        )

    print("6. Evaluating untouched holdout...")
    holdout_metrics, holdout_predictions = evaluate_holdout(dev_df, holdout_df, bundle)
    selected_row = holdout_metrics.loc[holdout_metrics["model"] == bundle["selected_model_name"]].iloc[0]
    print(f"   Holdout accuracy: {selected_row['accuracy']:.4f}")
    print(f"   Holdout log loss: {selected_row['log_loss']:.4f}")
    print(f"   Holdout brier:    {selected_row['brier']:.4f}")

    print("7. Fitting full-season model and scoring Round 1...")
    (
        round1_output,
        historical_probs,
        _round1_features,
        supervised_ensemble_all,
    ) = fit_full_and_score_round1(
        pregame_df=pregame_df,
        round1_df=round1_df,
        feature_cols=feature_cols,
        seed=args.seed,
        bundle=bundle,
        final_team_states=final_team_states,
    )

    print("8. Building rankings and explainability artifacts...")
    power_rankings = build_power_rankings(historical_probs, final_team_states)
    feature_importance = build_feature_importance(feature_cols, supervised_ensemble_all)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "season_file": str(season_file),
        "season_file_sha256": sha256_file(season_file),
        "round1_file": str(round1_file),
        "round1_file_sha256": sha256_file(round1_file),
        "output_dir": str(output_dir),
        "n_games": int(len(games)),
        "n_pregame_rows": int(len(pregame_df)),
        "n_dev": int(len(dev_df)),
        "n_holdout": int(len(holdout_df)),
        "n_round1": int(len(round1_output)),
        "feature_count": int(len(feature_cols)),
        "elo_k": ELO_K,
        "elo_home_advantage": ELO_HOME_ADVANTAGE,
        "prior_games": PRIOR_GAMES,
        "logreg_C": LOGREG_C,
        "rf_n_estimators": RF_N_ESTIMATORS,
        "rf_max_depth": RF_MAX_DEPTH,
        "rf_min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        "rf_n_jobs": RF_N_JOBS,
        "hgb_learning_rate": HGB_LEARNING_RATE,
        "hgb_max_depth": HGB_MAX_DEPTH,
        "hgb_max_iter": HGB_MAX_ITER,
        "hgb_min_samples_leaf": HGB_MIN_SAMPLES_LEAF,
        "supervised_train_window": SUPERVISED_TRAIN_WINDOW,
        "bagging_seed_offsets": list(BAGGING_SEED_OFFSETS),
        "bagging_windows": list(BAGGING_WINDOWS),
        "bagging_model_count": int(len(bundle["supervised_ensemble_configs"])),
        "stacker_C": STACKER_C,
        "selected_model": bundle["selected_model_name"],
        "blend_weights": bundle["weights"],
        "elo_shrink_alpha": bundle["elo_shrink_alpha"],
        "calibration_used": bool(
            (
                bundle["selected_prediction_source"] == "blend_final"
                and bundle["blend_calibration_method"] != "none"
            )
            or (
                bundle["selected_prediction_source"] == "stack_final"
                and bundle["stack_calibration_method"] != "none"
            )
        ),
        "calibration_method": (
            bundle["blend_calibration_method"]
            if bundle["selected_prediction_source"] == "blend_final"
            else (
                bundle["stack_calibration_method"]
                if bundle["selected_prediction_source"] == "stack_final"
                else "none"
            )
        ),
        "blend_calibration_method": bundle["blend_calibration_method"],
        "stack_calibration_method": bundle["stack_calibration_method"],
        "selected_prediction_source": bundle["selected_prediction_source"],
        "recent_selector_switched": bool(bundle["recent_selector_switched"]),
        "recent_selector_elo_log_loss": bundle["recent_selector_elo_log_loss"],
        "recent_selector_elo_shrunk_log_loss": bundle.get("recent_selector_elo_shrunk_log_loss"),
        "recent_selector_blend_log_loss": bundle["recent_selector_blend_log_loss"],
        "recent_selector_stack_log_loss": bundle.get("recent_selector_stack_log_loss"),
        "recent_selector_best_oof_source": bundle.get("recent_selector_best_oof_source"),
        "recent_selector_objective_scores": bundle.get("recent_selector_objective_scores"),
        "cv_blend_log_loss": bundle["blend_metrics"]["log_loss"],
        "cv_blend_brier": bundle["blend_metrics"]["brier"],
        "cv_stack_log_loss": bundle["stack_metrics"]["log_loss"],
        "cv_stack_brier": bundle["stack_metrics"]["brier"],
    }

    print("9. Saving outputs...")
    output_paths = save_outputs(
        output_dir=output_dir,
        round1_output=round1_output,
        power_rankings=power_rankings,
        cv_summary=bundle["cv_summary"],
        holdout_metrics=holdout_metrics,
        holdout_predictions=holdout_predictions,
        blend_summary=bundle["blend_summary"],
        feature_importance=feature_importance,
        metadata=metadata,
    )
    script_path = Path(__file__).resolve()
    run_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([sys.executable] + sys.argv),
        "cwd": str(Path.cwd()),
        "git_commit": get_git_commit_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "input_hashes": {
            str(season_file): sha256_file(season_file),
            str(round1_file): sha256_file(round1_file),
        },
        "output_hashes": collect_hashes(output_paths),
        "reproducibility_notes": [
            "Predictions are reproducible with the same seed, same input files, and same dependency versions.",
            "run_metadata.json and run_manifest.json include timestamps and will change run-to-run.",
        ],
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as outfile:
        json.dump(run_manifest, outfile, indent=2)
    print(f"✅ Done. Output files written to: {output_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"❌ Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
