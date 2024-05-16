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


def create_drift_heatmap(drift_results: dict, analyzed_features: List) -> None:
    """
    Creates a heatmap with drift results.
    """
    df = pd.DataFrame(drift_results).T.reset_index()
    df.rename(columns={'index': 'species_name'}, inplace=True)
    pivot_df = df.pivot_table(index='species_name', values=analyzed_features)

    plt.figure(figsize=(10, 8))
    heatmap = sns.heatmap(pivot_df, cmap='YlOrBr', annot=True, cbar_kws={'label': 'Drift Score'})
    plt.title('Drift Scores of Bird Species for Different Features')
    plt.show()

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


def plot_change_points(data: pd.DataFrame, change_points: List[int], title: str, legend_title: str):
    """
    Plots the change points on a line plot, along with the time series data.
    """
    fig = px.line(data, x='date', y='n_sightings', title=title, color_discrete_sequence=['#55a630'])
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Number of Sightings')


    for idx in change_points:
        fig.add_vline(x=data['date'][idx], line_width=1, line_dash="dash", line_color="red")

    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', width=2, dash='dash'), showlegend=True, name=legend_title))
    fig.update_layout(xaxis_title='number of sightings', yaxis_title='date',
                    title_x=0.5,
                    font=dict(family="Aleo", size=15, color="#4d5f81"),
                    legend=dict(x=0.5, y=1, xanchor='center', yanchor='bottom', bgcolor="rgba(255, 255, 255, 0.6)"))
    fig.show()


def plot_change_points_per_year(data: pd.DataFrame, change_points: List[int], title: str, legend_title: str, show_first_sighting=False):
    """
    Plots the change points on a line plot, along with the time series data, creating one line per year and overlaying these.
    """
    data['date'] = pd.to_datetime(data['date'])  # Convert date strings to datetime objects
    data['day_of_year'] = data['date'].dt.dayofyear
    years = data['date'].dt.year.unique()

    fig = go.Figure()

    for year in years:
        year_data = data[data['date'].dt.year == year]
        color = px.colors.qualitative.Plotly[years.tolist().index(year)]  # Get a color from the Plotly color palette
        fig.add_trace(go.Scatter(x=year_data['day_of_year'], y=year_data['n_sightings'], mode='lines', name=str(year), line=dict(color=color)))

        if not show_first_sighting:
            for idx in change_points:
                if data['date'][idx].year == year:
                    fig.add_vline(x=data['day_of_year'][idx], line_width=1, line_dash="dash", line_color=color)

        if show_first_sighting:
            first_sighting_idx = year_data.index[year_data['n_sightings'] > 0].min()
            fig.add_vline(x=year_data['day_of_year'][first_sighting_idx], line_width=1, line_dash="dot", line_color=color)

    fig.update_xaxes(title_text='Day of Year')
    fig.update_yaxes(title_text='Number of Sightings')
    fig.update_layout(title=title, xaxis_title='Day of Year', yaxis_title='Number of Sightings',
                    title_x=0.5,
                    font=dict(family="Aleo", size=15, color="#4d5f81"),)
                    #legend=dict(x=0.5, y=1, xanchor='center', yanchor='bottom', bgcolor="rgba(255, 255, 255, 0.6)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='red', width=2, dash='dash'), showlegend=True, name=legend_title))

    fig.show()