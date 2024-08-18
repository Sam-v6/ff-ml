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
# - Iterate through years of data to form a training set
# - Make ML model (age, avg pass yards allowed, avg run yards allowed, avg run tds allowed, avg pass yards allowed --> ppr points)

# Init
player_display_name = "Breece Hall"
seasons = [2021, 2022, 2023]

# Get defense data
defense_df = get_defense_data(seasons)

# Load the full dataset
data = nfl.import_weekly_data(seasons, downcast=True)

# Filter the DataFrame for the specific player across all seasons and weeks
player_data = data.loc[(data['player_display_name'] == player_display_name) & (data['season'].isin(seasons))]

# Extract the 'fantasy_points_ppr' column and assign it to y_data
y_data = player_data[['season', 'recent_team', 'week', 'opponent_team', 'fantasy_points_ppr']]

# Display the result
print(y_data)