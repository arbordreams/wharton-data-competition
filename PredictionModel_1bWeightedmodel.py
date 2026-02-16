import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
try:
    df = pd.read_csv('whl_2025.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: whl_2025.csv not found. Please upload the file.")
    exit()

# ---------------------------------------------------------
# 2. PREPARE SHIFT-LEVEL DATA
# ---------------------------------------------------------
def prep_shift_data(df):
    # 1. Home Offense Shifts
    home_rows = df.copy()
    home_rows = home_rows.rename(columns={
        'home_team': 'off_team', 'away_team': 'def_team',
        'home_off_line': 'off_line', 'away_def_pairing': 'def_pair',
        'home_xg': 'xg', 'home_shots': 'shots', 'home_goals': 'goals',
        'home_assists': 'assists', 'home_max_xg': 'max_xg',
        'home_penalties_committed': 'penalties_committed'
    })
    home_rows['is_home'] = 1
    
    # 2. Away Offense Shifts
    away_rows = df.copy()
    away_rows = away_rows.rename(columns={
        'away_team': 'off_team', 'home_team': 'def_team',
        'away_off_line': 'off_line', 'home_def_pairing': 'def_pair',
        'away_xg': 'xg', 'away_shots': 'shots', 'away_goals': 'goals',
        'away_assists': 'assists', 'away_max_xg': 'max_xg',
        'away_penalties_committed': 'penalties_committed'
    })
    away_rows['is_home'] = 0
    
    # Map Penalties Drawn
    home_rows['penalties_drawn'] = df['away_penalties_committed']
    away_rows['penalties_drawn'] = df['home_penalties_committed']

    cols = ['game_id', 'off_team', 'def_team', 'off_line', 'def_pair', 
            'xg', 'toi', 'is_home', 
            'shots', 'goals', 'assists', 'max_xg', 'penalties_committed', 'penalties_drawn']
    
    shifts = pd.concat([home_rows[cols], away_rows[cols]], ignore_index=True)
    
    # Filter Even Strength Only
    def is_ev(line):
        s = str(line).lower()
        return not ('pp' in s or 'kill' in s or 'empty' in s or 'nan' in s)

    shifts = shifts[shifts['off_line'].apply(is_ev)]
    shifts = shifts[shifts['toi'] > 0]
    
    return shifts

shift_df = prep_shift_data(df)
print(f"Total Even Strength Shifts: {len(shift_df)}")

# ---------------------------------------------------------
# 3. SPLIT DATA (80/20)
# ---------------------------------------------------------
unique_games = shift_df['game_id'].unique()
train_games, test_games = train_test_split(unique_games, test_size=0.2, random_state=42)

train_df = shift_df[shift_df['game_id'].isin(train_games)].copy()
test_df = shift_df[shift_df['game_id'].isin(test_games)].copy()

# ---------------------------------------------------------
# 4. FEATURE ENGINEERING (SPLIT OFFENSE BY LINE)
# ---------------------------------------------------------
# Metrics specific to the OFFENSIVE LINE
line_metrics = ['xg', 'goals', 'shots', 'assists', 'max_xg', 'penalties_committed']

# Metrics specific to the DEFENSIVE TEAM (Context)
def_metrics = ['xg', 'goals', 'shots', 'assists', 'max_xg']

def get_ratings(source_df):
    # 1. OFFENSE RATINGS: Group by Team AND Line
    # This creates specific ratings for "Team A - Line 1", "Team A - Line 2", etc.
    line_ratings = source_df.groupby(['off_team', 'off_line'])[line_metrics].mean()
    line_ratings = line_ratings.rename(columns={m: f'rating_{m}' for m in line_metrics})
    
    # 2. DEFENSE RATINGS: Group by Team (Opponent Strength)
    def_ratings = source_df.groupby('def_team')[def_metrics].mean()
    def_ratings = def_ratings.rename(columns={m: f'rating_{m}_against' for m in def_metrics})
    
    # 3. GLOBAL CONTEXT (GSAx + Elo) - Calculated at Team Level
    # We need intermediate team stats to calc Elo/GSAx
    team_off_agg = source_df.groupby('off_team')['goals'].mean()
    team_def_agg = source_df.groupby('def_team')['goals'].mean()
    
    # Global Elo (Team Level)
    # Formula: 1500 + (Avg Goals For - Avg Goals Against) * 500
    # We map this to the defense dataframe for convenience
    def_ratings['rating_global'] = 1500 + (team_off_agg - team_def_agg) * 500
    
    # GSAx (Goals Saved Above Expected) - Team Level
    def_ratings['rating_goalie_gsax'] = def_ratings['rating_xg_against'] - def_ratings['rating_goals_against']
    
    # Fill any NaNs (e.g., new teams)
    line_ratings = line_ratings.fillna(0)
    def_ratings = def_ratings.fillna(0)
    
    return line_ratings, def_ratings

off_ratings, def_ratings = get_ratings(train_df)

# List of columns we expect after merging
feature_names = [
    # Offense (Line Level)
    'rating_xg', 'rating_goals', 'rating_shots', 'rating_assists', 'rating_max_xg', 'rating_penalties_committed',
    # Defense (Team Level)
    'rating_xg_against', 'rating_goals_against', 'rating_shots_against', 'rating_assists_against', 'rating_max_xg_against',
    # Context
    'rating_global', 'rating_goalie_gsax'
]

def add_features(data, o_ratings, d_ratings):
    # 1. Merge Offense Ratings (Matches on Team + Line)
    data = data.merge(o_ratings, left_on=['off_team', 'off_line'], right_index=True, how='left')
    
    # 2. Merge Defense Ratings (Matches on Team)
    data = data.merge(d_ratings, left_on='def_team', right_index=True, how='left')
    
    # 3. Rename columns for clarity (prefixing)
    # The merge already created the columns, we just need to ensure formatting for diffs
    # In this specific setup, we don't need to rename much because we named them in get_ratings
    
    # 4. Calculate Diffs (Offense Rating - Defense Rating) where applicable
    # We compare Line XG vs Team XG Against
    data['diff_xg'] = data['rating_xg'] - data['rating_xg_against']
    data['diff_goals'] = data['rating_goals'] - data['rating_goals_against']
    data['diff_shots'] = data['rating_shots'] - data['rating_shots_against']
    
    return data.fillna(0) # Fill NaNs for unknown lines

train_data = add_features(train_df, off_ratings, def_ratings)
test_data = add_features(test_df, off_ratings, def_ratings)

# ---------------------------------------------------------
# 5. PREPARE ML MATRICES
# ---------------------------------------------------------
le_line = LabelEncoder()
all_lines = pd.concat([shift_df['off_line'], shift_df['def_pair']]).unique()
le_line.fit(all_lines)

# Construct Feature List
# We use the raw ratings + the calculated diffs
final_features = feature_names + ['diff_xg', 'diff_goals', 'diff_shots', 'off_line_code', 'def_pair_code', 'toi']

def get_X_y(data):
    X = data.copy()
    X['off_line_code'] = le_line.transform(data['off_line'])
    X['def_pair_code'] = le_line.transform(data['def_pair'])
    return X[final_features], data['xg'], data['toi']

X_train_raw, y_train, w_train = get_X_y(train_data)
X_test_raw, y_test, w_test = get_X_y(test_data)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
print(f"Model Training on {X_train.shape[1]} features.")

# ---------------------------------------------------------
# 6. TRAIN WEIGHTED ENSEMBLE
# ---------------------------------------------------------
print("\n--- Optimizing Ensemble Weights ---")

gbm = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
hgb = HistGradientBoostingRegressor(learning_rate=0.05, max_depth=3, max_iter=200, random_state=42, verbose=0)
rf = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
lr = Ridge(alpha=1.0, random_state=42)

estimators = [('gbm', gbm), ('hgb', hgb), ('rf', rf), ('lr', lr)]
model_scores = []

# Train Individual Models
for name, model_instance in estimators:
    model_instance.fit(X_train, y_train, sample_weight=w_train)
    pred = model_instance.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    
    weight_score = 1 / (rmse ** 2)
    model_scores.append(weight_score)
    print(f"  > {name.upper()} RMSE: {rmse:.4f} (Score: {weight_score:.2f})")

# Normalize weights
total_score = sum(model_scores)
final_weights = [s / total_score for s in model_scores]

print("\n--- Final Model Importance (Weights) ---")
for name, w in zip(['GBM', 'HGB', 'RF', 'Ridge'], final_weights):
    print(f"  > {name}: {w*100:.1f}%")

# Create Final Model
model = VotingRegressor(estimators=estimators, weights=final_weights)
model.fit(X_train, y_train, sample_weight=w_train)

final_preds = model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
print(f"\nFinal Weighted Ensemble RMSE: {final_rmse:.4f}")

# ---------------------------------------------------------
# 7. GENERATE LINE 1 vs LINE 2 DISPARITY (UPDATED)
# ---------------------------------------------------------
print("\n--- Generating Rankings (Line Specific) ---")
# Baseline opponent (Average Defense)
avg_def_stats = def_ratings.mean()
avg_def_code = int(X_train_raw['def_pair_code'].mode()[0])
avg_toi = shift_df['toi'].mean()

sim_rows = []
unique_teams = df['home_team'].unique() # Get list of teams
target_lines = ['first_off', 'second_off'] 

for team in unique_teams:
    for line in target_lines:
        # Check if we have data for this specific line
        if (team, line) in off_ratings.index:
            line_stat = off_ratings.loc[(team, line)]
            
            row = {'team': team, 'line': line}
            
            # Fill Offense Features (From Specific Line)
            for m in line_metrics:
                row[f'rating_{m}'] = line_stat[f'rating_{m}']
            
            # Fill Defense Features (From Average Opponent)
            for m in def_metrics:
                row[f'rating_{m}_against'] = avg_def_stats[f'rating_{m}_against']
            
            # Fill Context (From Average Opponent)
            row['rating_global'] = avg_def_stats['rating_global']
            row['rating_goalie_gsax'] = avg_def_stats['rating_goalie_gsax']
            
            # Calculate Diffs
            row['diff_xg'] = row['rating_xg'] - row['rating_xg_against']
            row['diff_goals'] = row['rating_goals'] - row['rating_goals_against']
            row['diff_shots'] = row['rating_shots'] - row['rating_shots_against']
            
            # Categorical & TOI
            row['off_line_code'] = le_line.transform([line])[0]
            row['def_pair_code'] = avg_def_code
            row['toi'] = avg_toi
            
            sim_rows.append(row)

sim_df_raw = pd.DataFrame(sim_rows)
sim_X_scaled = scaler.transform(sim_df_raw[final_features])
sim_df_raw['model_xg'] = model.predict(sim_X_scaled)

results = []
for team in unique_teams:
    data = sim_df_raw[sim_df_raw['team'] == team]
    # Ensure we have both lines
    if len(data) == 2:
        l1 = data[data['line'] == 'first_off']['model_xg'].values[0]
        l2 = data[data['line'] == 'second_off']['model_xg'].values[0]
        
        ratio = l1 / l2 if l2 > 0 else 1.0
        results.append({
            'Team': team, 
            'First_Line_xG': l1, 
            'Second_Line_xG': l2, 
            'Disparity_Ratio': ratio
        })

results_df = pd.DataFrame(results).sort_values('Disparity_Ratio', ascending=False)
results_df['Rank'] = range(1, len(results_df)+1)

# Calculate the baseline variation
std_y = y_test.std()

print(f"Model RMSE:      {rmse:.4f}")
print(f"Baseline Std Y:  {std_y:.4f}")

if rmse < std_y:
    print("✅ GREAT: The model is finding real patterns!")
    print(f"   (It is {(1 - rmse/std_y)*100:.1f}% better than random guessing)")
else:
    print("❌ BAD: The model is failing. It's worse than just guessing the average.")
print(results_df.head(10))
results_df.to_csv('phase_1b_line_specific_disparity.csv', index=False)
print("Rankings saved to phase_1b_line_specific_disparity.csv")