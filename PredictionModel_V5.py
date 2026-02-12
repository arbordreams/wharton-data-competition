import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import shap


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
try:
    df = pd.read_csv('whl_2025.csv')
    new_matches = pd.read_csv('WHSDSC_Rnd1_matchupsv2.csv')
except FileNotFoundError:
    print("Error: CSV files not found.")
    exit()

def clean_team_names(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
    return df

df = clean_team_names(df, ['home_team', 'away_team'])
new_matches = clean_team_names(new_matches, ['home_team', 'away_team'])

# ---------------------------------------------------------
# 2. GLOBAL STRENGTH RATING (Iterative Elo-like)
# ---------------------------------------------------------
def calculate_global_ratings(games_df, iterations=100, base_rating=1500, k_factor=10, home_advantage=50):
    """
    Calculates a static 'Global Rating' for each team.
    Since games are independent, we iterate over the SAME dataset multiple times
    until the ratings stabilize. This finds the rating that best explains ALL results.
    """
    teams = set(games_df['home_team']).union(set(games_df['away_team']))
    ratings = {team: base_rating for team in teams}
    
    # We create a copy to not mess up the order or index
    calc_df = games_df[['home_team', 'away_team', 'target_home_win']].copy()
    
    for _ in range(iterations):
        # We process all games in the batch
        # To avoid 'order bias', we can shuffle, but for convergence just iterating is usually fine
        # with small K-factor.
        total_change = 0
        
        # Temp dictionary to store updates so updates within one iteration don't affect each other
        # (This makes it truly independent/simultaneous)
        current_ratings = ratings.copy()
        
        for idx, row in calc_df.iterrows():
            home = row['home_team']
            away = row['away_team']
            result = row['target_home_win']
            
            r_home = current_ratings.get(home, base_rating)
            r_away = current_ratings.get(away, base_rating)
            
            # Expected Home Win Prob
            dr = (r_away - (r_home + home_advantage))
            expected = 1 / (1 + 10 ** (dr / 400))
            
            # Update (accumulate changes)
            ratings[home] += k_factor * (result - expected)
            ratings[away] += k_factor * ((1-result) - (1-expected))
            
    return ratings

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING: AGGREGATION
# ---------------------------------------------------------
def categorize_situation(row):
    home_line = str(row['home_off_line'])
    away_line = str(row['away_off_line'])
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
stats_en = get_situation_sums(df, 'EN')
game_data = game_base.join([stats_ev, stats_st, stats_en]).fillna(0)

# ---------------------------------------------------------
# 4. TEAM STATS (Process Stats + Goalie)
# ---------------------------------------------------------
def get_team_stats(games_df):
    # GSAx Calculation
    games_df['home_GSAx'] = (games_df['away_xg_EV'] + games_df['away_xg_ST']) - (games_df['away_goals_EV'] + games_df['away_goals_ST'])
    games_df['away_GSAx'] = (games_df['home_xg_EV'] + games_df['home_xg_ST']) - (games_df['home_goals_EV'] + games_df['home_goals_ST'])
    
    metrics = ['xg', 'shots', 'assists', 'max_xg']
    
    # Process Home
    home_cols = ['home_team', 'home_GSAx']
    home_rename = {'home_team': 'team', 'home_GSAx': 'goalie_gsax'}
    for m in metrics:
        if f'home_{m}_EV' in games_df.columns: home_cols.append(f'home_{m}_EV'); home_rename[f'home_{m}_EV'] = f'{m}_ev'
        if f'home_{m}_ST' in games_df.columns: home_cols.append(f'home_{m}_ST'); home_rename[f'home_{m}_ST'] = f'{m}_st'
    if 'home_penalties_committed_ST' in games_df.columns: home_cols.append('home_penalties_committed_ST'); home_rename['home_penalties_committed_ST'] = 'penalties_committed_st'
    if 'away_penalties_committed_ST' in games_df.columns: home_cols.append('away_penalties_committed_ST'); home_rename['away_penalties_committed_ST'] = 'penalties_drawn_st'
    home_df = games_df[home_cols].copy().rename(columns=home_rename)

    # Process Away
    away_cols = ['away_team', 'away_GSAx']
    away_rename = {'away_team': 'team', 'away_GSAx': 'goalie_gsax'}
    for m in metrics:
        if f'away_{m}_EV' in games_df.columns: away_cols.append(f'away_{m}_EV'); away_rename[f'away_{m}_EV'] = f'{m}_ev'
        if f'away_{m}_ST' in games_df.columns: away_cols.append(f'away_{m}_ST'); away_rename[f'away_{m}_ST'] = f'{m}_st'
    if 'away_penalties_committed_ST' in games_df.columns: away_cols.append('away_penalties_committed_ST'); away_rename['away_penalties_committed_ST'] = 'penalties_committed_st'
    if 'home_penalties_committed_ST' in games_df.columns: away_cols.append('home_penalties_committed_ST'); away_rename['home_penalties_committed_ST'] = 'penalties_drawn_st'
    away_df = games_df[away_cols].copy().rename(columns=away_rename)
    
    return pd.concat([home_df, away_df]).groupby('team').mean()

# ---------------------------------------------------------
# 5. MODEL EVALUATION
# ---------------------------------------------------------
train_games, test_games = train_test_split(game_data, test_size=0.2, random_state=42)

# A. Calculate Global Ratings (Using ONLY Train data)
global_ratings = calculate_global_ratings(train_games)

# B. Calculate Stats Ratings (Using ONLY Train data)
team_profiles_train = get_team_stats(train_games)
team_profiles_train.columns = [f'rating_{c}' for c in team_profiles_train.columns]

def add_features(games_df, profiles, ratings_dict):
    # Merge Stats
    merged = games_df.merge(profiles, left_on='home_team', right_index=True, how='left')
    merged = merged.rename(columns={c: f'home_{c}' for c in profiles.columns})
    merged = merged.merge(profiles, left_on='away_team', right_index=True, how='left')
    merged = merged.rename(columns={c: f'away_{c}' for c in profiles.columns})
    
    # Map Global Ratings
    merged['home_global_rating'] = merged['home_team'].map(ratings_dict)
    merged['away_global_rating'] = merged['away_team'].map(ratings_dict)
    
    return merged

train_prepared = add_features(train_games, team_profiles_train, global_ratings)
test_prepared = add_features(test_games, team_profiles_train, global_ratings)

# Fill NaNs
mean_ratings = team_profiles_train.mean()
fill_values = {f'{side}_{k}': v for side in ['home', 'away'] for k, v in mean_ratings.items()}
# Default global rating is 1500
fill_values['home_global_rating'] = 1500
fill_values['away_global_rating'] = 1500

train_prepared = train_prepared.fillna(fill_values)
test_prepared = test_prepared.fillna(fill_values)

# Features
feature_cols = [c for c in train_prepared.columns if 'rating_' in c] + ['home_global_rating', 'away_global_rating']
print(f"Features used: {feature_cols}")
X_train = train_prepared[feature_cols]
y_train = train_prepared['target_home_win']
X_test = test_prepared[feature_cols]
y_test = test_prepared['target_home_win']

# Train
# model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
# model.fit(X_train, y_train)
# print(f"Test Accuracy: {accuracy_score(y_test, model.predict(X_test)):.3f}")
# Alternative: XGBoost (If installed)
model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, eval_metric='logloss')  # 0.57, 0.719
#model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss')  # 0.57, 0.704
#model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss')  # 0.559, 0.712
#model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, eval_metric='logloss')  # 0.578, 0.738
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(f"Test Accuracy: {accuracy_score(y_test, preds):.3f}")
print(f"Log Loss: {log_loss(y_test, probs):.4f}")


# Importances
importances = model.feature_importances_
feature_importance_dict = dict(zip(X_train.columns, importances))
print("\n--- Feature Importance (Top 10) ---")
for feat, score in sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{feat}: {score:.4f}")

# ---------------------------------------------------------
# 6. FINAL PREDICTION
# ---------------------------------------------------------
print("\n--- Generating Final Predictions ---")
# 1. Global Ratings on Full Data
final_global_ratings = calculate_global_ratings(game_data)

# 2. Stats on Full Data
team_profiles_full = get_team_stats(game_data)
team_profiles_full.columns = [f'rating_{c}' for c in team_profiles_full.columns]

# 3. Retrain
full_prepared = add_features(game_data, team_profiles_full, final_global_ratings).fillna(fill_values)
model.fit(full_prepared[feature_cols], full_prepared['target_home_win'])

# 4. Predict
new_matches_prepared = add_features(new_matches, team_profiles_full, final_global_ratings).fillna(fill_values)
probs = model.predict_proba(new_matches_prepared[feature_cols])[:, 1]
new_matches['home_win_prob'] = probs
new_matches['predicted_winner'] = np.where(probs > 0.5, new_matches['home_team'], new_matches['away_team'])

new_matches.to_csv('rst_round_1_predictions_global.csv', index=False)
print("Predictions saved to 'rst_round_1_predictions_global.csv'")

# ---------------------------------------------------------
# 7. POWER RANKINGS
# ---------------------------------------------------------

print("\n--- Generating Power Scores ---")
# 1. Full Data Ratings
final_global_ratings = calculate_global_ratings(game_data)
team_profiles_full = get_team_stats(game_data)
team_profiles_full.columns = [f'rating_{c}' for c in team_profiles_full.columns]

# 2. Retrain on Full
full_prepared = add_features(game_data, team_profiles_full, final_global_ratings).fillna(fill_values)
model.fit(full_prepared[feature_cols], full_prepared['target_home_win'])

# 3. Create Hypothetical "Team vs Average" Matchups
avg_profile = team_profiles_full.mean()
avg_global_rating = np.mean(list(final_global_ratings.values()))

power_score_rows = []
teams = team_profiles_full.index.unique()

for team in teams:
    row = {'team': team}
    # Set Home Team = The Team
    for col in team_profiles_full.columns:
        row[f'home_{col}'] = team_profiles_full.loc[team, col]
    row['home_global_rating'] = final_global_ratings.get(team, 1500)
    
    # Set Away Team = Average
    for col in team_profiles_full.columns:
        # Map rating_xg_ev -> away_rating_xg_ev
        row[f'away_{col}'] = avg_profile[col]
    row['away_global_rating'] = avg_global_rating
    
    power_score_rows.append(row)

power_score_df = pd.DataFrame(power_score_rows)
# Predict win probability
probs = model.predict_proba(power_score_df[feature_cols])[:, 1]
power_score_df['Model_Win_Prob'] = probs

# Create Final Ranking Table
final_ranking = power_score_df[['team', 'Model_Win_Prob', 'home_global_rating', 'home_rating_xg_ev']].copy()
final_ranking.columns = ['Team', 'Power_Score', 'Global_Elo', 'Avg_EV_xG']
final_ranking = final_ranking.sort_values('Power_Score', ascending=False)
final_ranking['Rank'] = range(1, len(final_ranking) + 1)

print(final_ranking.head())
final_ranking.to_csv('rst_team_power_rankings_model_implied.csv', index=False)