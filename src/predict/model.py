"""
Purpose: Creates xgb model
Author: Syam Evani
"""

# Standard imports
import os

# Local imports
from src.utils.load_data import download_and_save_season, generate_matchup_data

# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine learning utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold

# ML models
import xgboost as xgb

def prepare_data(matchup_df):
    # Drop columns
    cols_to_drop = ['season', 'week', 'game_id', 'home_team', 'away_team']
    matchup_df = matchup_df.drop(columns=cols_to_drop)

    # Convert to binary flags
    matchup_df['is_division_game'] = matchup_df['is_division_game'].astype(int)
    matchup_df['is_early_season'] = matchup_df['is_early_season'].astype(int)
    
    # Create features
    X = matchup_df.drop(columns=['actual_spread'])
    print("Features:", X.columns.tolist())
    y = matchup_df['actual_spread']

    # Scale features
    binary_cols = ['is_division_game', 'is_early_season']
    continuous_cols = [col for col in X.columns if col not in binary_cols]
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[continuous_cols] = scaler.fit_transform(X[continuous_cols])

    # Return
    return X_scaled, y, scaler  # Return scaler in case you want to transform future data

def train_xgboost_with_cv(X, y, season_col):
    # Split Train/Test
    train_idx = X[season_col] < 2023
    test_idx = X[season_col] == 2023

    X_train = X.loc[train_idx].drop(columns=[season_col])
    y_train = y.loc[train_idx]

    X_test = X.loc[test_idx].drop(columns=[season_col])
    y_test = y.loc[test_idx]

    # XGBoost Regressor base model
    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 6, 8],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    # 5-Fold Cross Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search with RMSE as scoring
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kfold,
        scoring='neg_root_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )

    print("STATUS: Starting Grid Search CV...")
    grid_search.fit(X_train, y_train)

    print(f"Best CV RMSE: {-grid_search.best_score_:.3f}")
    print(f"Best Params: {grid_search.best_params_}")

    # Evaluate on 2023 holdout test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"2023 Test RMSE: {rmse:.3f}")
    print(f"2023 Test MAE: {mae:.3f}")

    return best_model, X_train, X_test, y_test, y_pred

def plot_feature_importance(model, feature_names, top_n=15):
    importance = model.feature_importances_

    # Only keep features with non-zero importance
    non_zero_idx = importance > 0
    importance = importance[non_zero_idx]
    feature_names = feature_names[non_zero_idx]

    # Sort descending
    sorted_idx = importance.argsort()[::-1]
    top_n = min(top_n, len(sorted_idx))  # Adjust if fewer features

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importance[sorted_idx][:top_n][::-1])
    plt.yticks(range(top_n), feature_names[sorted_idx][:top_n][::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} XGBoost Feature Importances")
    plt.tight_layout()
    plt.show()

def export_2023_predictions(X_test, y_test, y_pred, matchup_df, output_file):
    """
    Export a CSV comparing actual and predicted spreads for 2023 season games,
    including prediction correctness for game outcome (winner).

    Args:
        X_test (pd.DataFrame): Features used for prediction (after scaling, still has 'week_number').
        y_test (pd.Series): Actual spreads.
        y_pred (np.array): Predicted spreads.
        matchup_df (pd.DataFrame): Original matchup_df including team names.
        output_file (str): Path to save the CSV output.
    """
    # Retrieve relevant info (home/away team, week, actual spread)
    test_season_df = matchup_df[matchup_df['season'] == 2023][['home_team', 'away_team', 'week', 'actual_spread']].reset_index(drop=True)

    # Build result DataFrame
    results_df = pd.DataFrame({
        'week': test_season_df['week'],
        'home_team': test_season_df['home_team'],
        'away_team': test_season_df['away_team'],
        'actual_spread': y_test.values,
        'predicted_spread': y_pred,
    })

    # Compute prediction error
    results_df['prediction_error'] = results_df['predicted_spread'] - results_df['actual_spread']

    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"2023 predictions exported to {output_file}")

def merge_spread_line_and_evaluate(predictions_file, games_file):
    """
    Evaluates:
    - Vegas win/loss correctness
    - Model win/loss correctness
    - Model closer to actual spread (ATS simplified)
    - Model beats Vegas in winner prediction
    """
    # Load data
    output_file = predictions_file
    predictions_df = pd.read_csv(predictions_file)
    games_df = pd.read_csv(games_file)

    # Filter to 2023 games only
    games_2023 = games_df[games_df['season'] == 2023][['home_team', 'away_team', 'week', 'spread_line']]

    # Merge spread_line into predictions
    merged_df = predictions_df.merge(
        games_2023,
        on=['home_team', 'away_team', 'week'],
        how='left'
    )

    # Prepare result columns
    vegas_prediction = []
    vegas_correct = []
    model_correct = []
    model_beats_spread = []
    model_beats_vegas = []

    for idx, row in merged_df.iterrows():
        actual_spread = row['actual_spread']
        predicted_spread = row['predicted_spread']
        vegas_spread = row['spread_line']

        # --- Vegas Winner Prediction ---
        if vegas_spread == 0:
            vegas_pred = 'Push'
        elif vegas_spread > 0:
            vegas_pred = 'Home'
        else:
            vegas_pred = 'Away'
        vegas_prediction.append(vegas_pred)

        # --- Actual Winner ---
        if actual_spread == 0:
            actual_result = 'Push'
        elif actual_spread > 0:
            actual_result = 'Home'
        else:
            actual_result = 'Away'

        # --- Vegas Correct Prediction ---
        if vegas_pred == 'Push' or actual_result == 'Push':
            vegas_correct_result = 'Push'
        elif vegas_pred == actual_result:
            vegas_correct_result = 'Correct'
        else:
            vegas_correct_result = 'Incorrect'
        vegas_correct.append(vegas_correct_result)

        # --- Model Correct Prediction ---
        if predicted_spread == 0 or actual_spread == 0:
            model_correct_result = 'Push'
        elif (predicted_spread > 0 and actual_spread > 0) or (predicted_spread < 0 and actual_spread < 0):
            model_correct_result = 'Correct'
        else:
            model_correct_result = 'Incorrect'
        model_correct.append(model_correct_result)

        # --- Model Beats Vegas on Spread Accuracy ---
        vegas_error = abs(vegas_spread - actual_spread)
        model_error = abs(predicted_spread - actual_spread)

        if model_error < vegas_error:
            model_ats_result = 'Beats Vegas'
        elif model_error == vegas_error:
            model_ats_result = 'Tie'
        else:
            model_ats_result = 'Loses Vegas'
        model_beats_spread.append(model_ats_result)

        # --- Model Beats Vegas on Winner Prediction ---
        if model_correct_result == 'Correct' and vegas_correct_result == 'Incorrect':
            model_vs_vegas_result = 'Beats Vegas'
        elif model_correct_result == 'Incorrect' and vegas_correct_result == 'Correct':
            model_vs_vegas_result = 'Loses to Vegas'
        elif model_correct_result == 'Correct' and vegas_correct_result == 'Correct':
            model_vs_vegas_result = 'Tie'
        elif model_correct_result == 'Incorrect' and vegas_correct_result == 'Incorrect':
            model_vs_vegas_result = 'Tie'
        else:
            model_vs_vegas_result = 'No Result'
        model_beats_vegas.append(model_vs_vegas_result)

    # Assign to DataFrame
    merged_df['vegas_prediction'] = vegas_prediction
    merged_df['vegas_correct'] = vegas_correct
    merged_df['model_correct'] = model_correct
    merged_df['model_beats_spread'] = model_beats_spread
    merged_df['model_beats_vegas'] = model_beats_vegas

    # Save to CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Vegas vs Model evaluation saved to {output_file}")

    # Print summary
    model_win_accuracy = (merged_df['model_correct'] == 'Correct').sum() / merged_df['model_correct'].count()
    vegas_win_accuracy = (merged_df['vegas_correct'] == 'Correct').sum() / merged_df['vegas_correct'].count()
    model_over_vegas_spread_accuracy = (merged_df['model_beats_spread'] == 'Beats Vegas').sum() / merged_df['model_beats_spread'].count()
    model_over_vegas_win_accuracy = ((merged_df['model_beats_vegas'] == 'Beats Vegas').sum() + (merged_df['model_beats_vegas'] == 'Tie').sum())/ merged_df['model_beats_vegas'].count()

    print(f'Model win accuracy: {model_win_accuracy}')
    print(f'Vegas win accuracy: {vegas_win_accuracy}')
    print(f'Model over vegas spread accuracy: {model_over_vegas_spread_accuracy}')
    print(f'Model over vegas win accuracy: {model_over_vegas_win_accuracy}')

    # Return
    return 0

def main():

    # Grab the data
    download_and_save_season()
    matchup_df = generate_matchup_data()

    # Prepare data
    X_scaled, y, scaler = prepare_data(matchup_df)

    # Add season back to X_scaled for splitting
    X_scaled['season'] = matchup_df['season'].values

    # Train XGBoost and predict 2023
    best_model, X_train, X_test, y_test, y_pred = train_xgboost_with_cv(X_scaled, y, 'season')

    # Feature Importance Plot
    plot_feature_importance(best_model, X_scaled.drop(columns=['season']).columns)

    # 2023 comparison
    export_2023_predictions(X_test, y_test, y_pred, matchup_df, output_file=os.path.join(os.getenv('FF_HOME'),'spread','output','2023_spread_predictions.csv'))
    merge_spread_line_and_evaluate(os.path.join(os.getenv('FF_HOME'),'spread','output','2023_spread_predictions.csv'), os.path.join(os.getenv('FF_HOME'),'spread','data','games.csv'))

    # Return
    return 0

main()
