"""
Purpose: Project 5
Details: Conducts design of experiments with Pc, MR, and area ratio for GOX/GCH4 thruster and uses machine learning techniques to do ANOVA and create predictive models
Author: Syam Evani
"""

# Standard imports
import os
import random

# General imports
import numpy as np
import itertools
import pandas as pd

# NFL imports
import nfl_data_py as nfl

# Plotting imports
import seaborn as sns
import matplotlib.pyplot as plt

# ML utils
from scipy.interpolate import griddata
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Regressors imports
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, BayesianRidge

# NN imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Local imports
from scraper import get_defense_data

# Process flow
# - Iterate through years of offense_data to form a training set
# - Make ML model (age, avg pass yards allowed, avg run yards allowed, avg run tds allowed, avg pass yards allowed --> ppr points)

# Inputs
GENERATE_DATA = False
player_display_name = "Patrick Mahomes"
seasons = [2021, 2022, 2023]

#--------------------------------------------------------------------
# Load Data
#--------------------------------------------------------------------
# Generate datasets
if GENERATE_DATA:
    offense_data = nfl.import_weekly_data(seasons, downcast=True)
    offense_data.to_csv(os.path.join(os.getenv('HOME'),'repos','ff-ml','input','offense_data.csv'), index=False)
    get_defense_data(seasons)

# Load data
offense_data = pd.read_csv(os.path.join(os.getenv('HOME'),'repos','ff-ml','input','offense_data.csv'))
defense_data = pd.read_csv(os.path.join(os.getenv('HOME'),'repos','ff-ml','input','defense_data.csv'))

# Filter the DataFrame for the specific player across all seasons and weeks
player_data = offense_data.loc[(offense_data['player_display_name'] == player_display_name) & (offense_data['season'].isin(seasons))]

# Extract the 'fantasy_points_ppr' column and assign it to y_data
filtered_player_data = player_data[['position', 'season', 'recent_team', 'week', 'opponent_team', 'fantasy_points_ppr']]

# Map full team names to their abbreviations used in filtered_player_data['opponent_team']
team_name_mapping = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS',
    'Washington Football Team': 'WAS'
}

# Replace team names in defense_data to match the abbreviations in filtered_player_data
defense_data['Tm'] = defense_data['Tm'].replace(team_name_mapping)

# Define a function to get fantasy points allowed for a given row in filtered_player_data
def get_fantasy_points_allowed(row, position):
    # Find the corresponding row in defense_data
    defense_row = defense_data[
        (defense_data['Tm'] == row['opponent_team']) &
        (defense_data['Season'] == row['season'])
    ]
    
    # Get the fantasy points allowed for the position
    if not defense_row.empty:
        return defense_row[f'{position} FantPt'].values[0]
    return None

# Replace with dynamics
position = "QB"  # Replace with actual logic to determine the position

# Apply the function to each row in filtered_player_data
filtered_player_data[f'{position}_FantPt_Allowed'] = filtered_player_data.apply(
    lambda row: get_fantasy_points_allowed(row, position),
    axis=1
)

# Print
print(filtered_player_data)

#--------------------------------------------------------------------
# Slice data into training and test
#--------------------------------------------------------------------
X = filtered_player_data[[f'{position}_FantPt_Allowed']].values.tolist()

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, filtered_player_data['fantasy_points_ppr'], test_size=0.2, random_state=42)

# Scaling the data for some delicate regressors
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

#--------------------------------------------------------------------
# Apply different regression approaches
#--------------------------------------------------------------------
# Init results dict
results = {
    "lr": [],            # Linear 
    "dtr": [],           # Decision tree 
    "rfr": [],           # Random forest 
    "svr": [],           # Support vector 
    "knn": [],           # K-nearest neighbors 
    "gbr": [],           # Gradient boosting 
    "xgbr": [],          # XGBoost 
    "catbr": [],         # CatBoost 
    "en": [],            # ElasticNet 
    "lasso": [],         # Lasso 
    "ridge": [],         # Ridge 
    "bayesianridge": [], # Bayesian ridge 
    "adaboost": []       # AdaBoost 
}

models = {}
predictions = {}
table = []
best_transformed_features = None

# Train different regression models
models["lr"] = LinearRegression().fit(X_train, y_train)
models["dtr"] = DecisionTreeRegressor().fit(X_train, y_train)
models["rfr"] = RandomForestRegressor().fit(X_train, y_train)
models["svr"] = SVR().fit(X_train, y_train)
models["knn"] = KNeighborsRegressor().fit(X_train, y_train)
models["gbr"] = GradientBoostingRegressor().fit(X_train, y_train)
models["xgbr"] = XGBRegressor().fit(X_train, y_train)
models["catbr"] = CatBoostRegressor(silent=True).fit(X_train, y_train)
models["en"] = ElasticNet().fit(X_train, y_train)
models["lasso"] = Lasso().fit(X_train, y_train)
models["ridge"] = Ridge().fit(X_train, y_train)
models["bayesianridge"] = BayesianRidge().fit(X_train, y_train)
models["adaboost"] = AdaBoostRegressor().fit(X_train, y_train)

# Predict and calculate MSE
for regressor in models:
    predictions[regressor] = models[regressor].predict(X_test)
    mse = mean_squared_error(y_test, predictions[regressor])

    # Store the results
    results[regressor].append((mse, predictions[regressor]))

#--------------------------------------------------------------------
# Post process and plot different regressors for comparison
#--------------------------------------------------------------------
# Init table for txt output
table = []

# Post-process different regression approaches
for regressor in results:
    # Extract the MSE and best predictions
    min_mse, best_predictions = results[regressor][0]

    # Update table
    table.append([regressor, min_mse])

    # Print the MSE
    print(f"Regressor: {regressor}")
    print(f"MSE: {min_mse}")

    # Plotting predictions against actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, best_predictions, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.xlabel("Actual Points")
    plt.ylabel("Predicted Points")
    plt.title(f"Predictions vs Actual for {regressor} with MSE: {"{:.5f}".format(min_mse)}")
    plt.savefig(os.path.join(os.getenv('HOME'), 'repos', 'ff-ml', 'output', 'regressors', regressor + '.png'))
    plt.close()

#--------------------------------------------------------------------
# Output summary table for various techniques
#--------------------------------------------------------------------
# Write the table to a text file
output_file = os.path.join(os.getenv('HOME'), 'repos', 'ff-ml', 'output', 'regressors', 'regressor_results.txt')
with open(output_file, 'w') as f:
    f.write(f"{'Regressor':<20} {'MSE':<20}\n")
    f.write("="*60 + "\n")
    for row in table:
        f.write(f"{row[0]:<20} {row[1]:<20.5f}\n")