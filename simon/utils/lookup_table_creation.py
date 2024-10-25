"""
Functions for creating lookup tables with probabilities for different features.
"""

import pandas as pd
import numpy as np
from utils.utils import *
import warnings

warnings.filterwarnings("ignore")


def create_grid_year_lookup(bird_sightings:pd.DataFrame, on_year: int):
    """
    Create a lookup table for a given year and available grids.
    :param df: Dataframe.
    :param on_year: Year on which to calculate emergent filters.
    :return: Emergent Filters.
    """
    bird_sightings = bird_sightings[bird_sightings.date.str.contains(str(on_year))]
    bird_sightings['day_of_year'] = pd.to_datetime(bird_sightings.date).dt.dayofyear
    bird_sightings = bird_sightings[['name_species', 'eea_grid_id', 'day_of_year']]

    grid_list = bird_sightings.eea_grid_id.unique()
    day_list = range(1, 366)
    species_list = bird_sightings.name_species.unique()

    all_combinations = pd.MultiIndex.from_product([species_list, grid_list, day_list], names=['name_species', 'eea_grid_id', 'day_of_year'])
    all_combinations = pd.DataFrame(index=all_combinations).reset_index()

    by_days = bird_sightings.groupby(['day_of_year', 'eea_grid_id']).count().reset_index()
    by_days.rename(columns={'name_species':'total_sightings'}, inplace=True)
    by_days = all_combinations.merge(by_days, on=['eea_grid_id', 'day_of_year'], how='left')
    by_days['total_sightings'] = by_days['total_sightings'].fillna(0).astype(int)
    
    by_days_and_species = bird_sightings.groupby(['name_species', 'day_of_year', 'eea_grid_id']).size().reset_index()
    by_days_and_species.rename(columns={0:'n_sightings'}, inplace=True)

    total_df = by_days.merge(by_days_and_species, on=['name_species', 'eea_grid_id', 'day_of_year'], how='left')
    total_df['n_sightings'] = total_df['n_sightings'].fillna(0).astype(int)
    total_df['frequency'] = total_df.n_sightings / total_df.total_sightings
    total_df['frequency'] = total_df['frequency'].fillna(0)

    groups = total_df.groupby(['name_species', 'eea_grid_id'])
    result_df = pd.concat([circular_rolling(group) for _, group in groups])
    result_df.reset_index(drop=True, inplace=True)
    result_df.drop(columns=['total_sightings', 'n_sightings', 'frequency', 'frequency_rolling'], inplace=True)

    grid_year_lookup = result_df.groupby(['name_species', 'eea_grid_id', 'day_of_year'])['plausibility'].first().to_dict()
    return grid_year_lookup


def create_land_cover_lookup(bird_sightings: pd.DataFrame):
    """
    Create a lookup table for land cover percentages.
    
    :param bird_sightings: DataFrame with bird sightings.
    :param land_cover_columns: List of columns representing land cover percentages.
    :return: Lookup table with mean percentages for each land cover type per species.
    """
    land_cover_columns = [
        "urban_area_percent","industrial_area_percent","agriculture_area_percent","forest_area_percent","grassland_area_percent",
        "shrubland_area_percent","coastal_area_percent","rocky_area_percent","sparsley_vegetated_area_percent","burnt_area_percent","glacier_area_percent",
        "wetlands_area_percent","water_area_percent","unclassified_area_percent"
    ]
    landcover_lookup = bird_sightings.groupby('name_species')[land_cover_columns].mean().reset_index()
    landcover_lookup.columns = ['name_species'] + [f'mean_{col}' for col in land_cover_columns]
    return landcover_lookup


def assign_altitude_bins(df: pd.DataFrame, bin_size: int =100):
    """
    Assign altitude bins to each sighting based on specified bin size.
    
    :param df: DataFrame with bird sightings including an 'altitude' column.
    :param bin_size: The size of each altitude bin.
    :return: DataFrame with an additional 'altitude_bin' column.
    """
    min_altitude = df['altitude'].min().astype(int)
    max_altitude = df['altitude'].max().astype(int)
    
    bins = np.arange(min_altitude + 1, max_altitude + 1 + bin_size, bin_size)
    df['altitude_bin'] = pd.cut(df['altitude'], bins=bins, include_lowest=True, precision=0)
    return df


def create_altitude_lookup(bird_sightings: pd.DataFrame, bin_size: int = 50, min_sightings: int = 10, smoothing: float = 1.0):
    """
    Create a lookup table for altitude ranges with smoothed probabilities.
    
    :param bird_sightings: DataFrame with bird sightings.
    :param bin_size: The size of each altitude bin.
    :param min_sightings: Minimum number of sightings required for reliable probability estimation.
    :param smoothing: Smoothing factor to avoid zero probabilities (Laplace smoothing).
    :return: DataFrame representing the altitude lookup table.
    """
    bird_sightings = assign_altitude_bins(bird_sightings, bin_size=bin_size)
    sightings_per_bin = bird_sightings.groupby(['name_species', 'altitude_bin']).size().reset_index(name='n_sightings')
    total_sightings_per_species = bird_sightings.groupby('name_species').size().reset_index(name='total_sightings')

    sightings_per_bin = sightings_per_bin.merge(total_sightings_per_species, on='name_species')
    sightings_per_bin['smoothed_n_sightings'] = sightings_per_bin['n_sightings'] + smoothing
    sightings_per_bin['smoothed_total'] = sightings_per_bin['total_sightings'] + (smoothing * sightings_per_bin['altitude_bin'].nunique())
    sightings_per_bin['probability'] = sightings_per_bin['smoothed_n_sightings'] / sightings_per_bin['smoothed_total']
    sightings_per_bin = sightings_per_bin[sightings_per_bin['total_sightings'] >= min_sightings]
    
    altitude_lookup = sightings_per_bin.pivot_table(
        index='name_species',
        columns='altitude_bin',
        values='probability',
        fill_value=smoothing / (sightings_per_bin['total_sightings'] + (smoothing * sightings_per_bin['altitude_bin'].nunique()))
    ).reset_index()
    
    return altitude_lookup
