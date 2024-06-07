"""
Outsources plotting functions used in the notebooks.
"""

import io
from typing import List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import mapping


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


def plot_habitat_grid_heatmap_side_by_side(old, new, years):
    """
    Plots two heatmaps side by side, showing the density of sightings in old and new data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=500)
    for i, gdf_grid in enumerate([old, new]):
        fig = px.choropleth_mapbox(gdf_grid,
                                    geojson=gdf_grid.geometry,
                                    locations=gdf_grid.index,
                                    color='density',
                                    color_continuous_scale='Viridis',
                                    range_color=(0, gdf_grid['density'].max()),
                                    mapbox_style="carto-positron",
                                    zoom=4.5,
                                    center={'lat': gdf_grid.geometry.centroid.y.mean(),
                                            'lon': gdf_grid.geometry.centroid.x.mean()},
                                    opacity=0.7,
                                    title=f'pygmy owl sighting density {years[i]}',
                                    hover_data={'eea_grid_id': True, 'density': True})

        fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, width=800,
                        font=dict(family="Aleo", size=20, color="#4d5f81"))
        img_bytes = fig.to_image(format="png", width=800, height=600, scale=2)
        img = plt.imread(io.BytesIO(img_bytes))
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.show()


def plot_habibat_difference_heatmap(old, new):
    """
    Plots a heatmap showing the difference in density of sightings between two dataframes.
    """
    difference = old.copy()
    difference['density_2022'] = new['density']
    difference['density'].fillna(0, inplace=True)
    difference['density_2022'].fillna(0, inplace=True)
    difference['density_diff'] = difference['density_2022'] - difference['density']
    difference['density_diff'] = difference['density_diff'].replace(0, np.nan)

    fig = px.choropleth_mapbox(difference, 
                            geojson=difference.geometry, 
                            locations=difference.index, 
                            color='density_diff',
                            color_continuous_scale='RdYlGn', 
                            range_color=(-difference['density_diff'].abs().max(), difference['density_diff'].abs().max()), 
                            mapbox_style="carto-positron",
                            zoom=4.5, 
                            center={'lat': difference.geometry.centroid.y.mean(), 
                                    'lon': difference.geometry.centroid.x.mean()},
                            opacity=0.7,
                            hover_data={'eea_grid_id': True, 'density_diff': True})

    fig.update_layout(margin={"r":0,"t":00,"l":0,"b":0}, width=500)
    fig.show()


def plot_new_inhabitated_grids(old, new):
    """
    Plots the grids that were not inhabited in old data but are in new data in green, and vice versa in red.
    """
    fig = go.Figure()

    difference = old.copy()
    difference['density_2022'] = new['density']
    difference['density_diff'] = difference['density_2022'] - difference['density']
    for color, condition in [('green', (difference['density'].isnull() & ~difference['density_2022'].isnull())),
                            ('red', (~difference['density'].isnull() & difference['density_2022'].isnull()))]:
        for idx in difference[condition].index:
            polygon = difference.loc[idx, 'geometry']
            geojson_polygon = mapping(polygon)
            fig.add_trace(go.Scattermapbox(
                mode="lines",
                lon=[point[0] for point in geojson_polygon["coordinates"][0]],
                lat=[point[1] for point in geojson_polygon["coordinates"][0]],
                fill="toself",
                fillcolor=color,
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False
            ))
    fig.update_layout(mapbox_style="carto-positron",
                    mapbox_zoom=4.5, 
                    mapbox_center={'lat': difference.geometry.centroid.y.mean(), 
                                    'lon': difference.geometry.centroid.x.mean()},
                    margin={"r":0,"t":0,"l":0,"b":0}, width=500)
    fig.show()


def emergent_filters_plot_plausibility_over_multiple_years(filter_dict, species, grid, sign='🦆', threshold=0.05):
    """
    Plot the plausibility of seeing a species in a grid over multiple years based on Emergent Filters.
    """
    fig = go.Figure()

    for year, data_dict in filter_dict.items():
        df = pd.DataFrame(data_dict.items(), columns=['species_grid_day', 'plausibility'])
        df[['name_species', 'eea_grid_id', 'day_of_year']] = pd.DataFrame(df['species_grid_day'].tolist(), index=df.index)
        df.drop(columns=['species_grid_day'], inplace=True)
        data = df[(df.name_species == species) & (df.eea_grid_id == grid)]
        line_name = f'{year}'
        fig.add_trace(go.Scatter(x=data['day_of_year'], y=data['plausibility'], mode='lines', name=line_name))

    fig.add_shape(type="rect", x0=0, x1=365, y0=0, y1=threshold, fillcolor="#dd752c", opacity=0.1)
    fig.add_shape(type="rect", x0=0, x1=365, y0=threshold, y1=data.plausibility.max(), fillcolor="#89b959", opacity=0.3)
    fig.add_annotation(x=365/2, y=threshold/2, text='flagged for review', showarrow=False, font=dict(color="#dd752c"))
    fig.add_annotation(x=365/2, y=threshold+0.02, text='OK', showarrow=False, font=dict(color="#667761"))

    fig.update_layout(title={'text': "{} Plausibility for seeing a {} in '{}' {}".format(sign, species, grid, sign),
                             'x': 0.5,'xanchor': 'center','yanchor': 'top', 'font_color': '#4d5f81'},
                      xaxis_title='Day of Year', yaxis_title='Likelihood',
                      font=dict(family="Aleo", size=15, color="#4d5f81"))
    fig.show()


def plot_model_performance_over_years(sensitivities, precisions, years, title):
    """
    Plots the sensitivity and precision of a model over multiple years.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=sensitivities, mode='lines+markers', name='Sensitivity', yaxis='y'))
    fig.add_trace(go.Scatter(x=years, y=precisions, mode='lines+markers', name='Precision', yaxis='y2'))
    fig.update_layout(yaxis=dict(title='Sensitivity', side='left'),
                    yaxis2=dict(title='Precision', side='right', overlaying='y'))
    fig.update_layout(font=dict(family="Aleo", size=10, color="#4d5f81"),
                    title=dict(x=0.5, y=0.95, font=dict(size=18), 
                    text=title, xanchor='center'),
                    margin=dict(l=50, r=70, t=50, b=50),
                    width=700, height=300)
    fig.show()


def plot_drift_over_years(habitat_drift_results, years, title):
    """
    Plots the habitat drift over multiple years.
    """
    df = pd.DataFrame({'year': years, 'drift score': habitat_drift_results})
    fig = px.line(df, x='year', y='drift score')
    fig.update_traces(line_color='#667761')
    fig.update_layout(font=dict(family="Aleo", size=10, color="#4d5f81"),
                    title=dict(x=0.5, y=0.95, font=dict(size=18), 
                    text=title, xanchor='center'),
                    margin=dict(l=50, r=50, t=50, b=50),
                    width=600, height=300)
    fig.show()


def plot_change_points(data: pd.DataFrame, change_points_dict: Dict[str, List[int]], title: str):
    """
    Plots the change points on a line plot, along with the time series data.
    """
    fig = px.line(data, x='date', y='n_sightings', title=title, color_discrete_sequence=['#55a630'])
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Number of Sightings')

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, (algo, points) in enumerate(change_points_dict.items()):
        for idx in points:
            fig.add_vline(x=data['date'][idx], line_width=1, line_dash="dash", line_color=colors[i])

        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color=colors[i], width=2, dash='dash'), showlegend=True, name=algo))

    fig.update_layout(xaxis_title='number of sightings', yaxis_title='date',
                    title_x=0.5,
                    font=dict(family="Aleo", size=15, color="#4d5f81"),
                    legend=dict(x=0.5, y=1, xanchor='center', yanchor='bottom', bgcolor="rgba(255, 255, 255, 0.6)"),
                    legend_orientation="h")
    fig.show()




def plot_time_series(data: pd.DataFrame, title: str):
    """
    Plots the change points on a line plot, along with the time series data.
    """
    fig = px.line(data, x='date', y='n_sightings', title=title, color_discrete_sequence=['#55a630'])
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Number of Sightings')
    fig.update_layout(width=800, height=300)
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