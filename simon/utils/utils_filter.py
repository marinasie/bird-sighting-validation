"""
Utils for emergent filters.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")


def is_altitude_plausible(row, altitude_lookup, threshold=0.05):
    """
    Determine if a bird sighting is plausible based on altitude data.
    
    :param row: A single record (Series) representing a bird sighting with 'name_species' and 'altitude'.
    :param altitude_lookup: DataFrame containing altitude probabilities per species.
    :param threshold: Probability threshold below which a sighting is considered implausible.
    :return: 1 if implausible, 0 if plausible.
    """
    species = row['name_species']
    altitude = row['altitude']
    
    species_data = altitude_lookup[altitude_lookup['name_species'] == species]
    
    if species_data.empty:
        return 1
    
    bins = species_data.columns[1:]
    bin_edges = [interval.left for interval in bins] + [bins[-1].right]
    altitude_bin = pd.IntervalIndex.from_arrays(bin_edges[:-1], bin_edges[1:]).get_indexer([altitude])[0]
    
    if altitude_bin == -1:
        return 1
    
    probability = species_data.iloc[0, altitude_bin + 1]
    return int(probability < threshold)


def is_grid_year_plausible(sighting, emergent_filters_lookup, threshold=0.05):
    """
    Check if a sighting is unlikely given the Emergent Filters.
    :param sighting: A sighting record (Series).
    :param emergent_filters_lookup: Dictionary with emergent filters lookup values.
    :param threshold: Threshold for plausibility.
    :return: Boolean indicating if the sighting is unlikely.
    """
    key = (sighting['name_species'], sighting['eea_grid_id'], sighting['date'])
    plausibility = emergent_filters_lookup.get(key, None)
    return plausibility is not None and plausibility < threshold


def is_landcover_plausible(row, land_cover_dict, threshold):
    """
    Determine if a bird sighting is an outlier based on land cover data.
    :param row: Series with land cover percentages for a sighting.
    :param land_cover_dict: Dictionary with mean land cover percentages per species.
    :param threshold: Threshold for determining outlier status.
    :return: 1 if outlier, 0 if not an outlier.
    """
    species = row['name_species']
    if species not in land_cover_dict:
        return 1 
    
    land_covers = land_cover_dict[species]
    deviation = sum((row[f'{col[5:]}'] - land_covers[col])**2 for col in land_covers)
    return 1 if deviation > threshold else 0


def evaluate_emergent_filters_grid_year(X_test:pd.DataFrame, y_test:pd.Series, lookup_table:dict, threshold:float=0.05):
    """
    Evaluate Emergent Filters on a test set.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :param path_to_lookup_table: Path to Emergent Filter lookup table.
    :return: Confusion matrix.
    """
    y_pred = X_test.apply(is_grid_year_plausible, args=(lookup_table, threshold), axis=1)
    return confusion_matrix(y_test, y_pred)


def evaluate_emergent_filters_landcover(X_test: pd.DataFrame, y_test: pd.Series, land_cover_lookup: pd.DataFrame, threshold: float = 0.1):
    """
    Evaluate if bird sightings are outliers based on land cover.
    :param X_test: DataFrame with test bird sightings data.
    :param y_test: Series with binary labels (0 for plausible, 1 for outlier).
    :param land_cover_lookup: DataFrame with mean land cover percentages per species.
    :param threshold: Threshold for outlier detection based on deviation.
    :return: Confusion matrix comparing true labels (y_test) and predicted labels (y_pred).
    """
    land_cover_dict = land_cover_lookup.set_index('name_species').to_dict('index')
    y_pred = X_test.apply(is_landcover_plausible, axis=1, land_cover_dict=land_cover_dict, threshold=threshold)
    return confusion_matrix(y_test, y_pred)


def evaluate_emergent_filters_altitude(X_test: pd.DataFrame, y_test: pd.Series, altitude_lookup: pd.DataFrame, threshold: float = 0.05):
    """
    Evaluate the altitude filter on a test dataset.
    :param X_test: DataFrame with test bird sightings data including 'name_species' and 'altitude'.
    :param y_test: Series with binary labels (0 for plausible, 1 for implausible).
    :param altitude_lookup: DataFrame containing altitude probabilities per species.
    :param threshold: Probability threshold below which a sighting is considered implausible.
    :return: Confusion matrix comparing true labels (y_test) and predicted labels (y_pred).
    """
    y_pred = X_test.apply(is_altitude_plausible, axis=1, altitude_lookup=altitude_lookup, threshold=threshold)
    return confusion_matrix(y_test, y_pred)


def calculate_metrics(conf_matrix):
    """
    Calculate metrics from a confusion matrix.
    :param conf_matrix: Confusion matrix.
    :return: Accuracy, Precision, Recall, F1 Score.
    """
    TP, FP = conf_matrix[0, 0], conf_matrix[0, 1]
    FN, TN = conf_matrix[1, 0], conf_matrix[1, 1]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, specificity, f1_score

