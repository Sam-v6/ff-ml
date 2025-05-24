#!/usr/bin/env python

"""
Purpose: Grabs data from the nfl py
"""

# Base imports
import time
import os

# Common imports
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# Additional imports
import nfl_data_py as nfl

# Local imports
# None

# import_pbp_data() - import play-by-play data
# import_weekly_data() - import weekly player stats
# import_seasonal_data() - import seasonal player stats
# import_snap_counts() - import weekly snap count stats
# import_ngs_data() - import NGS advanced analytics
# import_qbr() - import QBR for NFL or college
# import_seasonal_pfr() - import advanced stats from PFR on a seasonal basis
# import_weekly_pfr() - import advanced stats from PFR on a weekly basis
# import_officials() - import details on game officials
# import_schedules() - import weekly teams schedules
# import_seasonal_rosters() - import yearly team rosters
# import_weekly_rosters() - import team rosters by week, including in-season updates
# import_players() - import descriptive data for all players
# import_depth_charts() - import team depth charts
# import_injuries() - import team injury reports
# import_ids() - import mapping of player ids for more major sites
# import_contracts() - import contract data
# import_win_totals() - import win total lines for teams
# import_sc_lines() - import weekly betting lines for teams
# import_draft_picks() - import draft pick history
# import_draft_values() - import draft value models by pick
# import_combine_data() - import combine stats
# import_ftn_data() - import FTN charting data
# see_pbp_cols() - return list of play-by-play columns
# see_weekly_cols() - return list of weekly stat columns
# import_team_desc() - import descriptive data for team viz
# cache_pbp() - save pbp files locally to allow for faster loading
# clean_nfl_data() - clean df by aligning common name diffs


def grab_online_data():

    # Define some values that will be useful for data gathering
    positions = ['QB', 'RB', 'WR', 'TE']
    year_list = list(range(2024, 2025))  # data starts in 1999
    REPORT_COLUMNS = True

    # Get data
    weekly_player_stats_df = nfl.import_weekly_data(years=year_list)                                     # Get weekly player stats
    weekly_player_stats_df = weekly_player_stats_df[weekly_player_stats_df['position'].isin(positions)]  # Filter to only the offensive positions

    weekly_snap_counts_df = nfl.import_snap_counts(years=year_list)
    print(weekly_snap_counts_df.head())

    # Save to PostgreSQL
    db_url = "postgresql://ffuser:ffpass@localhost:5432/ffdb"
    engine = create_engine(db_url)

    # Save the dataframe to a table called 'player_stats'
    weekly_player_stats_df.to_sql('player_stats', engine, if_exists='replace', index=False, method='multi', chunksize=1000)
    print("Data saved to PostgreSQL!")

    # Save some informational data what's contained in some of these calls
    if REPORT_COLUMNS:

        # Define common save path for this info
        save_path = os.path.join(os.getenv('FF_HOME'), 'data', 'nfl_data_source')

        # Save info about the columns
        if REPORT_COLUMNS:
            save_path = os.path.join(os.getenv('FF_HOME'), 'data', 'nfl_data_source')
            os.makedirs(save_path, exist_ok=True)
            file_name = "import_weekly_data_fcn_columns.txt"
            file_path = os.path.join(save_path, file_name)
            weekly_player_stats_df.columns.to_series().to_csv(file_path, index=False, header=True)

def query_db():
    engine = create_engine("postgresql://ffuser:ffpass@localhost:5432/ffdb")

    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM player_stats;"), conn)

    print('-------------------------------------------------')
    print('Flavor of SQL data:')
    print(df.head())

    return df


def generate_rb_features(df, window=3):
    # Filter to just RBs
    df_rb = df[df['position'] == 'RB'].copy()

    # Sort by player and week
    df_rb = df_rb.sort_values(['player_id', 'season', 'week'])

    # Avoid divide-by-zero for carries and receptions
    df_rb['yards_per_carry'] = np.where(
        df_rb['carries'] != 0, df_rb['rushing_yards'] / df_rb['carries'], np.nan
    )

    df_rb['yards_per_reception'] = np.where(
        df_rb['receptions'] != 0, df_rb['receiving_yards'] / df_rb['receptions'], np.nan
    )
    # Define columns to roll
    roll_cols = [
        "carries", "rushing_yards", "rushing_tds", "rushing_fumbles",
        "receptions", "targets", "receiving_yards", "receiving_tds",
        "receiving_fumbles", "fantasy_points_ppr", "yards_per_carry", "yards_per_reception"
    ]

    # Group by player and compute rolling average of past `window` games
    for col in roll_cols:
        df_rb[f'{col}_last{window}'] = (
            df_rb.groupby('player_id')[col]
                 .shift(1)  # Don't include current week
                 .rolling(window)
                 .mean()
                 .reset_index(0, drop=True)
        )

    # Drop any rows where we don't have enough history
    df_rb = df_rb.dropna(subset=[f'{col}_last{window}' for col in roll_cols])

    return df_rb


# Main
#grab_online_data()
df = query_db()
df_rb = generate_rb_features(df)
print(df_rb.head())