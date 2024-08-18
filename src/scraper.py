#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Purpose: Scrapes https://www.pro-football-reference.com/ for NFL data
Details: Adopted from https://github.com/Degentleman/NFL-Results-Scraper
Author: Syam Evani
"""

# Base imports
from urllib.request import urlopen
import time

# General imports
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from scipy import stats

# Local imports
# None

def get_defense_data(years):

    # URL to Format
    url_template = "https://www.pro-football-reference.com/years/{year}/fantasy-points-against-{position}.htm"

    # Initialize an empty DataFrame to store the final results
    nfl_df = pd.DataFrame()
    positions = ["RB", "TE", "WR", "QB"]

    for year in years:
        yearly_df = pd.DataFrame()  # Initialize a DataFrame for each year

        for position in positions:
            time.sleep(20)  # Wait for 5 seconds before making the next request

            # Form the full URL and get the soup
            url = url_template.format(year=year, position=position)
            html = urlopen(url)
            soup = BeautifulSoup(html, 'html.parser')

            # Parse the columns (dropping out some grouped headings)
            column_headers = [th.getText() for th in soup.findAll('thead', limit=1)[0].findAll('th')]
            items_to_remove = ['', 'Rushing', 'Passing', 'Receiving', 'Fantasy', 'Fantasy per Game']
            filtered_list = [item for item in column_headers if item not in items_to_remove]

            # Get the actual content
            data_rows = soup.findAll('tbody', limit=1)[0].findAll('tr')[0:]
            team_data = [[td.getText() for td in data_rows[i].findAll(['th', 'td'])] for i in range(len(data_rows))]

            # Turn yearly data into a DataFrame and drop duplicates
            year_df = pd.DataFrame(team_data, columns=filtered_list)
            year_df = year_df.map(lambda x: x.strip() if isinstance(x, str) else x)  # Strip whitespace
            year_df = year_df.drop_duplicates()  # Drop duplicates

            # Add Season column to track the year
            year_df['Season'] = year

            # Identify the second instance of the 'FantPt' column
            fantpt_columns = np.where(year_df.columns == 'FantPt')[0]
            if len(fantpt_columns) > 1:
                second_fantpt_index = fantpt_columns[1]  # Get the index of the second 'FantPt' column

                # Add the second instance of FantPt to the yearly_df
                year_df[f'{position} FantPt'] = year_df.iloc[:, second_fantpt_index].astype(float)

                # Select relevant columns to merge
                year_df = year_df[['Tm', 'Season', f'{position} FantPt']]

                # Merge or update the data with the yearly_df
                if yearly_df.empty:
                    yearly_df = year_df
                else:
                    yearly_df = pd.merge(yearly_df, year_df, on=['Tm', 'Season'], how='outer')

        # Combine the yearly_df into the main nfl_df, ensuring no duplication
        if nfl_df.empty:
            nfl_df = yearly_df
        else:
            nfl_df = pd.concat([nfl_df, yearly_df], ignore_index=True)

    # Display the final DataFrame
    # After processing all the data and creating the final nfl_df DataFrame
    nfl_df.to_csv('nfl_fantasy_points.csv', index=False)

get_defense_data([2021, 2022, 2023])