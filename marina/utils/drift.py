
import io
from typing import List
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *

def drift_report(reference: pd.DataFrame, current: pd.DataFrame, analyzed_features: List[str], to_json=False):
    """
    Generates a drift report between two dataframes.
    """
    report = Report(metrics=[DataDriftPreset(columns=analyzed_features)])
    report.run(reference_data=reference, current_data=current)
    if to_json:
        return json.loads(report.json())
    return report

def calculate_species_drift(data, reference, current, analyzed_features, categorical=False):
    """
    Calculates the drift for each species and each feature between a reference and a current dataframe.
    If categorical, the result only contains if a drift was detected or not. Otherwise, it contains the precise drift score.
    """
    drift_results = {}
    for species in data.name_species.unique():
        species_drift = {}
        report = drift_report(reference=reference[reference.name_species == species],
                              current=current[current.name_species == species],
                              analyzed_features=analyzed_features,
                              to_json=True)
        for metric in report['metrics']:
            if metric['metric'] == 'DataDriftTable':
                result = metric['result']
                drift_by_columns = result['drift_by_columns']
                for column_name, column_data in drift_by_columns.items():
                    if categorical:
                        drift_score = column_data['drift_detected']
                    else:
                        drift_score = column_data['drift_score']
                    species_drift[column_name] = drift_score
        drift_results[species] = species_drift
    return drift_results

def calculate_drift_over_years(data, current, years, analyzed_features):
    """
    Calculate drift for one feature of one species over the given years.
    """
    drift_results = []
    for year in years:
        reference = data[data.date.str.contains(year)]
        reference['date'] = pd.to_datetime(reference.date).dt.dayofyear
        report = drift_report(reference=reference, current=current, analyzed_features=analyzed_features, to_json=True)
        for metric in report['metrics']:
            if metric['metric'] == 'DataDriftTable':
                result = metric['result']
                drift_by_columns = result['drift_by_columns']
                for column_name, column_data in drift_by_columns.items():
                    if column_name == 'eea_grid_id':
                        drift_score = column_data['drift_score']
                        drift_results.append(drift_score)
    return drift_results
