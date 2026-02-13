import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
try:
    df = pd.read_csv('whl_2025.csv')
    new_matches = pd.read_csv('WHSDSC_Rnd1_matchupsv2.csv')
except FileNotFoundError:
    print("Error: CSV files not found. Please upload 'whl_2025.csv' and 'WHSDSC_Rnd1_matchupsv2.csv'")
    exit()

def clean_team_names(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df

df = clean_team_names(df, ['home_team', 'away_team', 'home_off_line', 'away_off_line'])
new_matches = clean_team_names(new_matches, ['home_team', 'away_team'])

# ---------------------------------------------------------
# 2. GLOBAL RATING (Static Elo)
# ---------------------------------------------------------
def calculate_global_ratings(games_df, iterations=50, base_rating=1500, k_factor=10, home_advantage=50):
    teams = set(games_df['home_team']).union(set(games_df['away_team']))
    ratings = {team: base_rating for team in teams}
    calc_df = games_df[['home_team', 'away_team', 'target_home_win']].copy()
    
    for _ in range(iterations):
        current_ratings = ratings.copy()
        for idx, row in calc_df.iterrows():
            home, away, result = row['home_team'], row['away_team'], row['target_home_win']
            r_home = current_ratings.get(home, base_rating)
            r_away = current_ratings.get(away, base_rating)
            
            dr = (r_away - (r_home + home_advantage))
            expected = 1 / (1 + 10 ** (dr / 400))
            
            ratings[home] += k_factor * (result - expected)
            ratings[away] += k_factor * ((1 - result) - (1 - expected))
            
    return ratings

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING: AGGREGATION
# ---------------------------------------------------------
def categorize_situation(row):
    home_line = str(row.get('home_off_line', ''))
    away_line = str(row.get('away_off_line', ''))
    if 'empty_net' in home_line or 'empty_net' in away_line: return 'EN'
    elif 'PP' in home_line or 'PP' in away_line: return 'ST'
    else: return 'EV'

df['situation'] = df.apply(categorize_situation, axis=1)

def get_situation_sums(df, situation_label):
    mask = df['situation'] == situation_label
    agg_funcs = {
        'home_xg': 'sum', 'away_xg': 'sum',
        'home_goals': 'sum', 'away_goals': 'sum',
        'home_shots': 'sum', 'away_shots': 'sum',
        'home_assists': 'sum', 'away_assists': 'sum',
        'home_penalties_committed': 'sum', 'away_penalties_committed': 'sum',
        'home_max_xg': 'max', 'away_max_xg': 'max'
    }
    final_agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
    
    if not final_agg_funcs: return pd.DataFrame()

    agg = df[mask].groupby('game_id').agg(final_agg_funcs)
    agg.columns = [f'{c}_{situation_label}' for c in agg.columns]
    return agg

game_base = df.groupby('game_id').agg({
    'home_team': 'first', 'away_team': 'first',
    'home_goals': 'sum', 'away_goals': 'sum'
})
game_base['target_home_win'] = (game_base['home_goals'] > game_base['away_goals']).astype(int)

stats_ev = get_situation_sums(df, 'EV')
stats_st = get_situation_sums(df, 'ST')
# We calculate EN stats but effectively ignore them for skill ratings
stats_en = get_situation_sums(df, 'EN') 

game_data = game_base.join([stats_ev, stats_st, stats_en]).fillna(0)

# ---------------------------------------------------------
# 4. TEAM STATS (Process Metrics + Goalie GSAx)
# ---------------------------------------------------------
def get_team_stats(games_df):
    # Calculate GSAx: (Shots Expected) - (Goals Allowed)
    games_df['home_GSAx'] = (games_df.get('away_xg_EV', 0) + games_df.get('away_xg_ST', 0)) - \
                            (games_df.get('away_goals_EV', 0) + games_df.get('away_goals_ST', 0))
    
    games_df['away_GSAx'] = (games_df.get('home_xg_EV', 0) + games_df.get('home_xg_ST', 0)) - \
                            (games_df.get('home_goals_EV', 0) + games_df.get('home_goals_ST', 0))
    
    metrics = ['xg', 'shots', 'assists', 'max_xg']
    
    # Process Home
    home_cols = ['home_team', 'home_GSAx']
    home_rename = {'home_team': 'team', 'home_GSAx': 'goalie_gsax'}
    
    for m in metrics:
        if f'home_{m}_EV' in games_df.columns: home_cols.append(f'home_{m}_EV'); home_rename[f'home_{m}_EV'] = f'{m}_ev'
        if f'home_{m}_ST' in games_df.columns: home_cols.append(f'home_{m}_ST'); home_rename[f'home_{m}_ST'] = f'{m}_st'
    
    # Separate Penalties Committed vs Drawn
    if 'home_penalties_committed_ST' in games_df.columns: 
        home_cols.append('home_penalties_committed_ST')
        home_rename['home_penalties_committed_ST'] = 'penalties_committed_st'
    if 'away_penalties_committed_ST' in games_df.columns: 
        home_cols.append('away_penalties_committed_ST')
        home_rename['away_penalties_committed_ST'] = 'penalties_drawn_st'
        
    home_df = games_df[home_cols].copy().rename(columns=home_rename)

    # Process Away
    away_cols = ['away_team', 'away_GSAx']
    away_rename = {'away_team': 'team', 'away_GSAx': 'goalie_gsax'}
    
    for m in metrics:
        if f'away_{m}_EV' in games_df.columns: away_cols.append(f'away_{m}_EV'); away_rename[f'away_{m}_EV'] = f'{m}_ev'
        if f'away_{m}_ST' in games_df.columns: away_cols.append(f'away_{m}_ST'); away_rename[f'away_{m}_ST'] = f'{m}_st'
        
    if 'away_penalties_committed_ST' in games_df.columns: 
        away_cols.append('away_penalties_committed_ST')
        away_rename['away_penalties_committed_ST'] = 'penalties_committed_st'
    if 'home_penalties_committed_ST' in games_df.columns: 
        away_cols.append('home_penalties_committed_ST')
        away_rename['home_penalties_committed_ST'] = 'penalties_drawn_st'
        
    away_df = games_df[away_cols].copy().rename(columns=away_rename)
    
    return pd.concat([home_df, away_df]).groupby('team').mean()

# ---------------------------------------------------------
# 5. ENSEMBLE MODEL DEFINITION & TRAINING
# ---------------------------------------------------------
print("\n--- Preparing Data and Models ---")

train_games, test_games = train_test_split(game_data, test_size=0.2, random_state=42)

# Calculate Ratings on Train Data Only
global_ratings_train = calculate_global_ratings(train_games)
team_profiles_train = get_team_stats(train_games)
team_profiles_train.columns = [f'rating_{c}' for c in team_profiles_train.columns]

def add_diff_features(games_df, profiles, ratings_dict):
    merged = games_df.merge(profiles, left_on='home_team', right_index=True, how='left')
    merged = merged.rename(columns={c: f'home_{c}' for c in profiles.columns})
    merged = merged.merge(profiles, left_on='away_team', right_index=True, how='left')
    merged = merged.rename(columns={c: f'away_{c}' for c in profiles.columns})
    
    merged['home_global_rating'] = merged['home_team'].map(ratings_dict)
    merged['away_global_rating'] = merged['away_team'].map(ratings_dict)
    
    feature_list = []
    # 1. Global Elo Diff
    merged['diff_global_rating'] = merged['home_global_rating'] - merged['away_global_rating']
    feature_list.append('diff_global_rating')
    
    # 2. Stats Diff
    for col in profiles.columns:
        diff_name = f'diff_{col}'
        merged[diff_name] = merged[f'home_{col}'] - merged[f'away_{col}']
        feature_list.append(diff_name)
        
    return merged, feature_list

# Prepare Train/Test with Diffs
train_prepared, feature_cols = add_diff_features(train_games, team_profiles_train, global_ratings_train)
test_prepared, _ = add_diff_features(test_games, team_profiles_train, global_ratings_train)

# Fill NaNs
mean_ratings = team_profiles_train.mean()
fill_values = {f'{side}_{k}': v for side in ['home', 'away'] for k, v in mean_ratings.items()}
fill_values['home_global_rating'] = 1500
fill_values['away_global_rating'] = 1500
diff_fill_values = {c: 0 for c in feature_cols}

train_prepared = train_prepared.fillna(fill_values).fillna(diff_fill_values)
test_prepared = test_prepared.fillna(fill_values).fillna(diff_fill_values)

X_train = train_prepared[feature_cols]
y_train = train_prepared['target_home_win']
X_test = test_prepared[feature_cols]
y_test = test_prepared['target_home_win']

# --- DEFINE MODELS FOR ENSEMBLE ---

# 1. XGBoost (Standard Params that work well generally)
xgb_model = xgb.XGBClassifier(
    n_estimators=100, 
    max_depth=3, 
    learning_rate=0.05, 
    eval_metric='logloss'
)

# 2. HistGradientBoosting
hgb = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_depth=3,
    max_iter=450,
    min_samples_leaf=20,
    random_state=42
)

# 3. RandomForest
rf = RandomForestClassifier(
    n_estimators=350,
    max_depth=8,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42
)

# 4. Logistic Regression (Standardized)
lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, random_state=42))

# --- CREATE VOTING ENSEMBLE ---
print("Training Ensemble (XGB + HGB + RF + LogReg)...")
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('hgb', hgb),
        ('rf', rf),
        ('lr', lr)
    ],
    voting='soft' # Average probabilities
)

ensemble.fit(X_train, y_train)

preds = ensemble.predict(X_test)
probs = ensemble.predict_proba(X_test)[:, 1]

print(f"Test Accuracy: {accuracy_score(y_test, preds):.3f}")
print(f"Log Loss: {log_loss(y_test, probs):.4f}")

# ---------------------------------------------------------
# 6. FINAL PREDICTION & RANKING
# ---------------------------------------------------------
print("\n--- Generating Final Predictions ---")
# 1. Full Data Ratings
final_global_ratings = calculate_global_ratings(game_data)
team_profiles_full = get_team_stats(game_data)
team_profiles_full.columns = [f'rating_{c}' for c in team_profiles_full.columns]

# 2. Retrain Ensemble on Full Data
full_prepared, _ = add_diff_features(game_data, team_profiles_full, final_global_ratings)
full_prepared = full_prepared.fillna(fill_values).fillna(diff_fill_values)
ensemble.fit(full_prepared[feature_cols], full_prepared['target_home_win'])

# 3. Power Score
avg_profile = team_profiles_full.mean()
avg_global_rating = np.mean(list(final_global_ratings.values()))
power_score_rows = []

for team in team_profiles_full.index.unique():
    row = {'team': team}
    row['diff_global_rating'] = final_global_ratings.get(team, 1500) - avg_global_rating
    for col in team_profiles_full.columns:
        row[f'diff_{col}'] = team_profiles_full.loc[team, col] - avg_profile[col]
    power_score_rows.append(row)

power_score_df = pd.DataFrame(power_score_rows)
power_score_df = power_score_df.reindex(columns=['team'] + feature_cols, fill_value=0)

probs = ensemble.predict_proba(power_score_df[feature_cols])[:, 1]
power_score_df['Model_Win_Prob'] = probs

ranking = power_score_df[['team', 'Model_Win_Prob', 'diff_global_rating']].sort_values('Model_Win_Prob', ascending=False)
ranking.to_csv('rst_ensemble_power_rankings.csv', index=False)
print("Power Rankings saved.")

# 4. Predict New Matches
new_matches_prepared, _ = add_diff_features(new_matches, team_profiles_full, final_global_ratings)
new_matches_prepared = new_matches_prepared.fillna(fill_values).fillna(diff_fill_values)

probs = ensemble.predict_proba(new_matches_prepared[feature_cols])[:, 1]
new_matches['home_win_prob'] = probs
new_matches['predicted_winner'] = np.where(probs > 0.5, new_matches['home_team'], new_matches['away_team'])
new_matches.to_csv('rst_ensemble_round_1_predictions.csv', index=False)
print("Predictions saved.")