"""
Purpose: Loads data
Author: Syam Evani
"""

# Standard imports
import os

# General imports
import numpy as np
import pandas as pd

# NFL imports
import nfl_data_py as nfl

def download_and_save_season():

    # Export available columns in the play-by-play to a text file
    pbp_cols = nfl.see_pbp_cols()
    output_file = os.path.join(os.getenv('FF_HOME'), 'spread', 'data', 'pbp_cols.txt')
    with open(output_file, 'w') as f:
        for col in pbp_cols:
            f.write(f"{col}\n")

    for season in range(1999, 2024):
        # Define season(s) and local file path
        season_file = f'pbp_{season}.parquet'
        parquet_file = os.path.join(os.getenv('FF_HOME'),'spread','data',season_file)

        # Only download if file doesn't exist
        if not os.path.exists(parquet_file):
            print(f"STATUS: Downloading play-by-play data for {season}...")
            pbp_data = nfl.import_pbp_data([season], downcast=False)        # Downcast can save data if needed (choosing not to do it since we making parquets)
            pbp_data.to_parquet(parquet_file, index=False)
        else:
            print(f"Parquet file for {season} already exists. Skipping download.")

    # Return
    return 0

def load_team_weekly_stats(parquet_file, team_abbr):
    # Example usage:
    # parquet_file = os.path.join(os.getenv('FF_HOME'), 'spread', 'data', 'pbp_2023.parquet')
    # team_abbr = 'DET'  # Detroit Lions
    # weekly_stats_df = load_team_weekly_stats(parquet_file, team_abbr)

    # Load Parquet efficiently from disk
    pbp_data = pd.read_parquet(parquet_file)

    # Filter out plays without a posteam (like kickoffs, penalties, etc.)
    pbp_data = pbp_data[pbp_data['posteam'].notna()]

    # Build team offensive stats (team when on offense)
    team_offense = pbp_data[pbp_data['posteam'] == team_abbr]
    offense_weekly = team_offense.groupby(['season', 'week']).agg(
        passing_yards=('passing_yards', 'sum'),
        rushing_yards=('rushing_yards', 'sum'),
        interceptions=('interception', 'sum'),
        fumbles=('fumble_lost', 'sum'),
        points_scored=('posteam_score_post', 'max'),  # End of game score
        epa_total=('epa', 'sum'),
    ).reset_index()

    # Build team defensive stats (team when on defense)
    team_defense = pbp_data[pbp_data['defteam'] == team_abbr]
    defense_weekly = team_defense.groupby(['season', 'week']).agg(
        interceptions_def=('interception', 'sum'),
        fumbles_forced=('fumble_lost', 'sum'),
        points_allowed=('posteam_score_post', 'max'),
        epa_allowed=('epa', 'sum'),
    ).reset_index()

    # Merge offense & defense stats
    weekly_stats = pd.merge(offense_weekly, defense_weekly, on=['season', 'week'], how='inner')

    # Compute Turnover Differential (forced - lost)
    weekly_stats['turnover_margin'] = (weekly_stats['interceptions_def'] + weekly_stats['fumbles_forced']) - (weekly_stats['interceptions'] + weekly_stats['fumbles'])

    # Compute Point Differential
    weekly_stats['point_delta'] = weekly_stats['points_scored'] - weekly_stats['points_allowed']

    # Optional: Sort by week
    weekly_stats = weekly_stats.sort_values('week')

    return weekly_stats

def build_weekly_team_stats(pbp_data):
    pbp_data = pbp_data[pbp_data['posteam'].notna()]

    # Offense aggregates
    offense_agg = pbp_data.groupby(['season', 'week', 'posteam']).agg(
        passing_yards=('passing_yards', 'sum'),
        rushing_yards=('rushing_yards', 'sum'),
        interceptions=('interception', 'sum'),
        fumbles=('fumble_lost', 'sum'),
        points_scored=('posteam_score_post', 'max'),
        epa_total=('epa', 'sum'),
        rush_epa=('rush_attempt', lambda x: pbp_data.loc[x.index, 'epa'].sum()),
        pass_epa=('pass_attempt', lambda x: pbp_data.loc[x.index, 'epa'].sum()),
    ).reset_index().rename(columns={'posteam': 'team'})

    # Defense aggregates
    defense_agg = pbp_data.groupby(['season', 'week', 'defteam']).agg(
        interceptions_def=('interception', 'sum'),
        fumbles_forced=('fumble_lost', 'sum'),
        points_allowed=('posteam_score_post', 'max'),
        epa_allowed=('epa', 'sum'),
    ).reset_index().rename(columns={'defteam': 'team'})

    # Merge offense & defense
    weekly_stats = pd.merge(offense_agg, defense_agg, on=['season', 'week', 'team'], how='inner')

    # Turnover & point delta
    weekly_stats['turnover_margin'] = (weekly_stats['interceptions_def'] + weekly_stats['fumbles_forced']) - (weekly_stats['interceptions'] + weekly_stats['fumbles'])

    # Return
    return weekly_stats

def build_matchup_dataset(weekly_stats_df, schedule_df, prior_season_stats, debug_team=None, debug_weeks=[1]):
    feature_cols = ['passing_yards', 'rushing_yards', 'turnover_margin', 'points_scored', 'points_allowed', 'epa_total', 'rush_epa', 'pass_epa']

    blended_avgs = []
    for (season, team) in weekly_stats_df[['season', 'team']].drop_duplicates().itertuples(index=False):
        team_data = weekly_stats_df[(weekly_stats_df['season'] == season) & (weekly_stats_df['team'] == team)].copy()
        team_data = team_data.sort_values('week')

        prior_stats = prior_season_stats.get((season - 1, team), None)

        for col in feature_cols:
            team_data[f'rolling_{col}'] = team_data[col].expanding().mean().shift(1)

            blended_values = []
            for idx, row in team_data.iterrows():
                week = row['week']
                if prior_stats is not None and week <= 3:
                    weight_prior = (4 - week) / 3
                    weight_current = 1 - weight_prior
                    blended_value = (weight_prior * prior_stats[col]) + (weight_current * row[f'rolling_{col}'])

                    # ✅ Sanity Print for Debug Team & Week
                    if team == debug_team and week in debug_weeks:
                        print(f"[DEBUG] {team} Week {week} - Blending {col}:")
                        print(f"   prior_stat: {prior_stats[col]:.3f}")
                        print(f"   rolling_current: {row[f'rolling_{col}']:.3f}")
                        print(f"   weight_prior: {weight_prior:.2f}, weight_current: {weight_current:.2f}")
                        print(f"   blended_value: {blended_value:.3f}")

                else:
                    blended_value = row[f'rolling_{col}']

                blended_values.append(blended_value)

            team_data[col] = blended_values

        blended_avgs.append(team_data[['season', 'week', 'team'] + feature_cols])

    blended_avgs_df = pd.concat(blended_avgs)

    matchup_df = schedule_df.copy()

    # Merge Home/Away stats
    matchup_df = matchup_df.merge(
        blended_avgs_df.add_prefix('home_'),
        left_on=['season', 'week', 'home_team'],
        right_on=['home_season', 'home_week', 'home_team'],
        how='left'
    )
    matchup_df = matchup_df.merge(
        blended_avgs_df.add_prefix('away_'),
        left_on=['season', 'week', 'away_team'],
        right_on=['away_season', 'away_week', 'away_team'],
        how='left'
    )

    # Compute deltas
    for col in feature_cols:
        matchup_df[f'{col}_delta'] = matchup_df[f'home_{col}'] - matchup_df[f'away_{col}']

    # Contextual flags
    matchup_df['week_number'] = matchup_df['week']
    matchup_df['is_early_season'] = (matchup_df['week'] <= 3).astype(int)
    matchup_df['is_division_game'] = matchup_df['home_team'].str[:3] == matchup_df['away_team'].str[:3]

    # Target
    matchup_df['actual_spread'] = matchup_df['home_score'] - matchup_df['away_score']

    # Final feature selection
    keep_cols = ['season', 'week', 'game_id', 'home_team', 'away_team', 'week_number', 'is_early_season'] + \
                [f'{col}_delta' for col in feature_cols] + \
                ['is_division_game', 'actual_spread']

    
    # Return
    return matchup_df[keep_cols]

def compute_prior_season_stats(all_weekly_stats_df):
    feature_cols = ['passing_yards', 'rushing_yards', 'turnover_margin', 'points_scored', 'points_allowed', 'epa_total', 'rush_epa', 'pass_epa']
    prior_season_stats = {}

    for (season, team) in all_weekly_stats_df[['season', 'team']].drop_duplicates().itertuples(index=False):
        prior_data = all_weekly_stats_df[(all_weekly_stats_df['season'] == season - 1) & (all_weekly_stats_df['team'] == team)]
        if not prior_data.empty:
            prior_avg = prior_data[feature_cols].mean()
            prior_season_stats[(season, team)] = prior_avg.fillna(0)

    return prior_season_stats

def generate_matchup_data():
    all_matchup_dfs = []
    all_weekly_stats_df = []

    # Build out weekly stats throughout all szns
    for season in range(1999, 2024):

        # Load data
        parquet_file = os.path.join('data', f'pbp_{season}.parquet')
        print(f"Processing season {season}...")
        pbp_data = pd.read_parquet(parquet_file)

        # Build out the weekly stats df
        weekly_stats_df = build_weekly_team_stats(pbp_data)
        all_weekly_stats_df.append(weekly_stats_df)

    # Build prior season stats dict once
    all_weekly_stats_df_df = pd.concat(all_weekly_stats_df, ignore_index=True)
    prior_season_stats = compute_prior_season_stats(all_weekly_stats_df_df)

    # Build matchup data with blended prior stats
    for season in range(1999, 2024):
        parquet_file = os.path.join('data', f'pbp_{season}.parquet')
        pbp_data = pd.read_parquet(parquet_file)

        schedule_df = pbp_data[['season', 'week', 'game_id', 'home_team', 'away_team', 'home_score', 'away_score']].drop_duplicates()
        weekly_stats_df = all_weekly_stats_df_df[all_weekly_stats_df_df['season'] == season]

        matchup_df = build_matchup_dataset(weekly_stats_df, schedule_df, prior_season_stats, debug_team='KC', debug_weeks=[1,2,3])
        all_matchup_dfs.append(matchup_df)

    full_dataset = pd.concat(all_matchup_dfs, ignore_index=True)

    # Save or return
    # full_dataset.to_parquet('data/matchup_dataset_1999_2023_blended.parquet', index=False)

    return full_dataset