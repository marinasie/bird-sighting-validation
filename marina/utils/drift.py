from typing import List
import json

import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import *


residential = ['Mittelspecht', 'Sperlingskauz', 'Dreizehenspecht', 'Auerhuhn', 'Alpenschneehuhn', 'Zitronenzeisig (Zitronengirlitz)']
short_distance_migrants = ['Haubentaucher', 'Schwarzkehlchen',  'Wasseramsel', 'Bergpieper', 'Seeadler', 'Rostgans']
long_distance_migrants = ['Wiesenpieper', 'Rohrammer', 'Singschwan', 'Flussuferläufer', 'Bergente', 'Berghänfling', 'Knäkente', 'Schwarzmilan', 'Steinschmätzer', 'Braunkehlchen', 'Gelbspötter', 'Orpheusspötter', 'Zwergohreule', 'Karmingimpel', 'Gänsegeier']



def replace_with_emojis(value):
    """
    Replaces boolean values with signs for better readability.
    """
    if value == True:
        return "✅"  # Check emoji for True
    elif value == False:
        return "-"  # Dash for False
    

def flags_to_emojis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces boolean values with signs for better readability.
    """
    for column in df.columns:
        if column != "species_name":
            df[column] = df[column].apply(replace_with_emojis)
    return df


def create_drift_table(drift_results: dict) -> pd.DataFrame:
    """
    Creates a table with drift results.
    """
    df = pd.DataFrame(drift_results).T.reset_index()
    df.rename(columns={'index': 'species_name'}, inplace=True)
    df = flags_to_emojis(df)
    return df


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, analyzed_features: List[str], to_json=False):
    """
    Generates a drift report between two dataframes.
    """
    report = Report(metrics=[DataDriftPreset(columns=analyzed_features)])
    report.run(reference_data=reference, current_data=current)
    if to_json:
        return json.loads(report.json())
    return report
