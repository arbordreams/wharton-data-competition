import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# Automatically grab every number column
numeric_cols = df_raw.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Exclude IDs (they are numbers but not stats)
exclude = ['record_id', 'game_id'] 
agg_cols = [c for c in numeric_cols if c not in exclude]

# 2. AGGREGATE PER GAME
# Summing all shifts to get Game Totals
games = df_raw.groupby(['game_id', 'home_team', 'away_team'], as_index=False)[agg_cols].sum()

# 3. DEFINE TARGET
# Did Home Team Win? (1 = Yes, 0 = No)
games['home_win'] = (games['home_goals'] > games['away_goals']).astype(int)

# --- PREPARE TRAINING DATA ---
print("2. Preparing Dataset (80/20 Split)...")

# Remove the answers (Goals) from the test
# We want to know if OTHER stats (Shots, xG, Penalties) predict the win
drop_cols = ['home_goals', 'away_goals', 'home_win', 'game_id', 'home_team', 'away_team']
X_cols = [c for c in games.columns if c not in drop_cols]

X = games[X_cols]
y = games['home_win']

# --- THE MODIFICATION ---
# Explicitly setting 0.8 (80%) for training and 0.2 (20%) for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    train_size=0.8, 
    test_size=0.2, 
    random_state=1
)

print(f"   Training Rows: {len(X_train)}")
print(f"   Testing Rows:  {len(X_test)}")

# --- TRAIN MODEL ---
print("3. Training Random Forest...")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# --- EVALUATE ---

train_acc = accuracy_score(y_train, rf.predict(X_train))
test_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"\n📊 Training Accuracy: {train_acc:.2%} (Should be lower than 99%)")
print(f"📊 Test Accuracy:     {test_acc:.2%} (Should be close to Train)")
print(f"   Gap:               {train_acc - test_acc:.2%}")

# ==========================================
# 2. CALCULATE & PRINT ELO RATINGS
# ==========================================
print("2. Calculating and Ranking Elo Ratings...")

K_FACTOR = 20  # Sensitivity

# 1. INITIALIZE (Crucial Step: Create the dictionary first)
# Get unique list of all teams
all_teams = pd.concat([games['home_team'], games['away_team']]).unique()
elo_ratings = {team: 1500 for team in all_teams}

# Lists to store history for training features
h_elos = []
a_elos = []
elo_diffs = []

# 2. LOOP (Update ratings game-by-game)
for idx, row in games.iterrows():
    h = row['home_team']
    a = row['away_team']
    
    # Get current (pre-game) rating
    curr_h = elo_ratings[h]
    curr_a = elo_ratings[a]
    
    # Save for features
    h_elos.append(curr_h)
    a_elos.append(curr_a)
    elo_diffs.append(curr_h - curr_a)
    
    # Calculate Winner & Update
    h_win = row['home_win']
    prob_home_win = 1 / (1 + 10 ** ((curr_a - curr_h) / 400))
    change = K_FACTOR * (h_win - prob_home_win)
    
    elo_ratings[h] += change
    elo_ratings[a] -= change

# Add to dataframe
games['home_elo'] = h_elos
games['away_elo'] = a_elos
games['elo_diff'] = elo_diffs

# ==========================================
# 3. PRINT THE RANKINGS (The Requested Part)
# ==========================================
print("\n🏆 FINAL ELO STANDINGS:")
print(f"{'Rank':<5} | {'Team':<20} | {'Rating':<10}")
print("-" * 40)

# Convert dictionary to DataFrame for sorting
elo_df = pd.DataFrame(list(elo_ratings.items()), columns=['Team', 'Elo_Rating'])
elo_df = elo_df.sort_values('Elo_Rating', ascending=False).reset_index(drop=True)

# Print Top 20 to screen
for i, row in elo_df.head(20).iterrows():
    print(f"{i+1:<5} | {row['Team']:<20} | {row['Elo_Rating']:.2f}")

# Save to CSV
elo_df.to_csv("final_elo_rankings.csv", index=False)
print("-" * 40)
print("✅ Rankings saved to 'final_elo_rankings.csv'")

#print(f"\n📊 Model Accuracy: {acc:.2%}")

# --- FEATURE IMPORTANCE ---
print("\n⭐ Key Features Driving Wins:")
importances = pd.DataFrame({
    'Feature': X_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(importances.head(10))

# --- SAVE OUTPUTS ---
games.to_csv("games_all_features.csv", index=False)
importances.to_csv("feature_importance.csv", index=False)
print("\n✅ Data and Analysis saved to CSV.")