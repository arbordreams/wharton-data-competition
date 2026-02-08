import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier  # The Scikit-Learn equivalent of XGBoost
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- CONFIGURATION ---
DATA_FILE = "whl_2025.csv"

print("1. Loading & Aggregating All Game Stats...")

# Load Data
try:
    df_raw = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"❌ Error: {DATA_FILE} not found.")
    exit()

# 1. IDENTIFY ALL NUMERIC FEATURES
numeric_cols = df_raw.select_dtypes(include=['float64', 'int64']).columns.tolist()
exclude = ['record_id', 'game_id'] 
agg_cols = [c for c in numeric_cols if c not in exclude]

# 2. AGGREGATE PER GAME
games = df_raw.groupby(['game_id', 'home_team', 'away_team'], as_index=False)[agg_cols].sum()

# 3. DEFINE TARGET
games['home_win'] = (games['home_goals'] > games['away_goals']).astype(int)

# --- NEW: SAVE TO CSV ---
games.to_csv("aggregated_games.csv", index=False)
print("✅ Saved aggregated game data to 'aggregated_games.csv'")

# --- PREPARE TRAINING DATA ---
print("2. Preparing Dataset (80/20 Split)...")

drop_cols = ['home_goals', 'away_goals', 'home_win', 'game_id', 'home_team', 'away_team']
X_cols = [c for c in games.columns if c not in drop_cols]

#X = games[X_cols]
#y = games['home_win']

# --- CUSTOM TEST SET: 32 SPECIFIC GAMES ---
print("2. Isolating the 32 Test Games...")

# 1. Load the 32 Matchups
try:
    # Ensure this filename matches your upload exactly
    matchups = pd.read_csv("WHSDSC_Rnd1_matchupsv2.csv")
    
    # Create a unique ID for every game in both dataframes to ensure perfect matching
    # Format: "HomeTeam_vs_AwayTeam" (stripping whitespace to be safe)
    matchups['match_id'] = matchups['home_team'].str.strip() + "_vs_" + matchups['away_team'].str.strip()
    games['match_id'] = games['home_team'].str.strip() + "_vs_" + games['away_team'].str.strip()
    
    # Create the list of 32 games we want to find
    target_matches = set(matchups['match_id'].unique())
    print(f"   Targeting {len(target_matches)} unique matchups from file.")

    # 2. Split the Data
    # is_test is TRUE only if the game is in your list of 32
    games['is_test'] = games['match_id'].isin(target_matches)

    train_df = games[~games['is_test']]
    test_df = games[games['is_test']]

    # 3. Define X and y
    X_train = train_df[X_cols]
    y_train = train_df['home_win']

    X_test = test_df[X_cols]
    y_test = test_df['home_win']

    print(f"   ✅ Training Set: {len(X_train)} games")
    print(f"   ✅ Test Set:     {len(X_test)} games (Should be exactly 32)")
    
    if len(X_test) != 32:
        print(f"   ⚠️ WARNING: Found {len(X_test)} games instead of 32. Check team names for typos.")

except FileNotFoundError:
    print("❌ Error: Matchups file not found. Make sure 'WHSDSC_Rnd1_matchupsv2.csv' is uploaded.")
    exit()

train_df.to_csv("aggregated_train.csv", index=False)
test_df.to_csv("aggregated_test.csv", index=False)


X_train = train_df[X_cols]
y_train = train_df['home_win']
X_test = test_df[X_cols]   # Matches your specific file
y_test = test_df['home_win']

print(f"   Training Games: {len(X_train)}")
print(f"   Test Games:     {len(X_test)} (From Matchups File)")

print(f"   Training Rows: {len(X_train)}")
print(f"   Testing Rows:  {len(X_test)}")

# --- TRAIN MODEL (GRADIENT BOOSTING) ---
print("3. Training Gradient Boosting Classifier...")

# random forest
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
# # --- EVALUATE ---
# train_acc = accuracy_score(y_train, rf.predict(X_train))
# test_acc = accuracy_score(y_test, rf.predict(X_test))


# Gradient Boosting parameters (similar to XGBoost defaults)
gb = GradientBoostingClassifier(
   n_estimators=100,      # Number of boosting stages
   learning_rate=0.1,     # How much each tree contributes (shrinks overfitting)
   max_depth=3,           # Limits tree complexity
   random_state=42
)

gb.fit(X_train, y_train)

# --- EVALUATE ---
train_acc = accuracy_score(y_train, gb.predict(X_train))
test_acc = accuracy_score(y_test, gb.predict(X_test))

print(f"\n📊 Training Accuracy: {train_acc:.2%}")
print(f"📊 Test Accuracy:     {test_acc:.2%}")
print(f"   Gap:               {train_acc - test_acc:.2%}")

# ==========================================
# 2. CALCULATE & PRINT ELO RATINGS
# ==========================================
print("2. Calculating and Ranking Elo Ratings...")

K_FACTOR = 20
all_teams = pd.concat([games['home_team'], games['away_team']]).unique()
elo_ratings = {team: 1500 for team in all_teams}

h_elos = []
a_elos = []
elo_diffs = []

for idx, row in games.iterrows():
    h = row['home_team']
    a = row['away_team']
    
    curr_h = elo_ratings[h]
    curr_a = elo_ratings[a]
    
    h_elos.append(curr_h)
    a_elos.append(curr_a)
    elo_diffs.append(curr_h - curr_a)
    
    h_win = row['home_win']
    prob_home_win = 1 / (1 + 10 ** ((curr_a - curr_h) / 400))
    change = K_FACTOR * (h_win - prob_home_win)
    
    elo_ratings[h] += change
    elo_ratings[a] -= change

games['home_elo'] = h_elos
games['away_elo'] = a_elos
games['elo_diff'] = elo_diffs

# ==========================================
# 3. PRINT THE RANKINGS
# ==========================================
print("\n🏆 FINAL ELO STANDINGS:")
print(f"{'Rank':<5} | {'Team':<20} | {'Rating':<10}")
print("-" * 40)

elo_df = pd.DataFrame(list(elo_ratings.items()), columns=['Team', 'Elo_Rating'])
elo_df = elo_df.sort_values('Elo_Rating', ascending=False).reset_index(drop=True)

for i, row in elo_df.head(20).iterrows():
    print(f"{i+1:<5} | {row['Team']:<20} | {row['Elo_Rating']:.2f}")

elo_df.to_csv("final_elo_rankings.csv", index=False)
print("-" * 40)
print("✅ Rankings saved to 'final_elo_rankings.csv'")

# --- FEATURE IMPORTANCE ---

# print("\n⭐ Key Features Driving Wins (random forest")
# importances = pd.DataFrame({
#     'Feature': X_cols,
#     'Importance': rf.feature_importances_
# }).sort_values('Importance', ascending=False)

print("\n⭐ Key Features Driving Wins (Gradient Boosting):")
importances = pd.DataFrame({
    'Feature': X_cols,
    'Importance': gb.feature_importances_
}).sort_values('Importance', ascending=False)

print(importances.head(10))

# --- SAVE OUTPUTS ---
games.to_csv("games_all_features.csv", index=False)
importances.to_csv("feature_importance.csv", index=False)
print("\n✅ Data and Analysis saved to CSV.")