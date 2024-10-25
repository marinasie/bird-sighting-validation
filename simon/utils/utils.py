"""
Utils in general.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore")

def get_land_uses():
    return {
    1: 'urban',
    2: 'industrial',
    3: 'agriculture',
    4: 'forest',
    5: 'grassland',
    6: 'mediterranean_vegetation',
    7: 'shrubland',
    8: 'coastal',
    9: 'rocky_area',
    10: 'sparsely_vegetated',
    11: 'burnt_area',
    12: 'glacier',
    13: 'wetlands',
    14: 'water',
    15: 'unclassified_land',
    16: 'unclassified_water',
    17: 'unclassified'}


def get_land_uses_numeric():
    return {
    'urban': 1,
    'industrial': 2,
    'agriculture': 3,
    'forest': 4,
    'grassland': 5,
    'mediterranean_vegetation': 6,
    'shrubland': 7,
    'coastal': 8,
    'rocky_area': 9,
    'sparsely_vegetated': 10,
    'burnt_area': 11,
    'glacier': 12,
    'wetlands': 13,
    'water': 14,
    'unclassified_land': 15,
    'unclassified_water': 16,
    'unclassified': 17}


def get_precision_sensitivity(y_true, y_pred):
    """
    Calculate precision and sensitivity.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Precision and sensitivity.
    """
    precision = precision_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    return precision, sensitivity


def compute_metrics(conf_mat):
    TN, FP, FN, TP = conf_mat.ravel()
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
    return accuracy, precision, sensitivity, f1


def rename_land_use_columns(df:pd.DataFrame, land_uses:dict):
    """
    Rename one-hot-encoded land use columns from land_use_coord_numeric_<land_use_numerical> to luse_<land_use_str>.
    E.g. from <land_use_coord_numeric_1> to <luse_urban>.
    :param df: Dataframe.
    :param land_uses: Land use categories.
    :return: Dataframe.
    """
    for col in df.columns:
        if col.startswith('land_use_coord_numeric_'):
            number = int(col.split('_')[-1])
            key = land_uses.get(number, f'unknown_{number}')
            new_col = col.replace(f'land_use_coord_numeric_{number}', f'luse_{key}')
            df.rename(columns={col: new_col}, inplace=True)
    return df


def one_hot_encode_land_use(df:pd.DataFrame):
    """
    One-hot-encode land use feature.
    :param df: Dataframe.
    :return: Dataframe.
    """
    df = pd.get_dummies(df, columns=['land_cover_coord_numeric'], dtype=int)
    df = rename_land_use_columns(df, get_land_uses())
    return df


def prepare_df_for_outlier_detection(all_species_df:pd.DataFrame, features:list):
    """
    Prepare a dataframe for outlier detection.
    :param df: Dataframe.
    :return: Dataframe.
    """
    X_train_per_species = {}
    for name_species in all_species_df.name_species.unique():
        df = all_species_df[all_species_df.name_species == name_species]
        df = df[features]
        if 'land_use_coord_numeric' in df.columns:
            df = one_hot_encode_land_use(df)
        if 'total_count' in df.columns:
            df.total_count = df.total_count.fillna(1)
        scaler = MinMaxScaler()
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df = df.reindex(sorted(df.columns), axis=1)
        X_train_per_species[name_species] = df
    return X_train_per_species


def plot_fn_vs_fp(false_negatives, false_positives, x_parameter, x_parameter_name):
    """
    Plot False Negatives and False Positives vs. Threshold.
    :param false_negatives: False Negatives.
    :param false_positives: False Positives.
    :param x_parameter: X parameter.
    :param x_parameter_name: X parameter name.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_parameter, y=false_negatives, mode='lines', name='Undetected outlier (FN)', yaxis='y', line=dict(color='#d80032')))
    fig.add_trace(go.Scatter(x=x_parameter, y=false_positives, mode='lines', name='Correct sightings labeled as outlier (FP)', yaxis='y2', line=dict(color='green')))
    fig.update_layout(yaxis2=dict(title='Correct sightings labeled as outlier (FP)',overlaying='y', side='right', color='green'),
                    title='False Negatives and False Positives vs. Threshold',
                    xaxis=dict(title=x_parameter_name),
                    yaxis=dict(title='Undetected outlier (FN)', side='left', color='#d80032'),
                    legend=dict(x=0, y=1, traceorder='normal'),
                    font=dict(family="Aleo", size=13, color="#4d5f81"))
    fig.show()


def get_confusion_matrix_values(conf_matrix):
    """
    Get True Positives, True Negatives, False Positives, and False Negatives from a confusion matrix.
    :param conf_matrix: Confusion matrix.
    :return: TP, TN, FP, FN.
    """
    TP = conf_matrix[1, 1]  # True Positive
    TN = conf_matrix[0, 0]  # True Negative
    FP = conf_matrix[0, 1]  # False Positive
    FN = conf_matrix[1, 0]  # False Negative
    return TP, TN, FP, FN


def circular_rolling(group):
    """Helper function to calculate Emergent Filters.
    :param group: Grouped dataframe.
    :return: Dataframe.
    """
    df = group.copy()
    df = pd.concat([df.iloc[-19:], df, df.iloc[:17]])
    df['frequency_rolling'] = df.frequency.rolling(window=7, center=True).max()
    df['plausibility'] = df.frequency_rolling.rolling(window=30, center=True).mean()
    return df.iloc[19:].iloc[:-17]


def emergent_filters_plot_plausibility_over_year(data_dict, species, grid, year, sign='🦆', threshold=0.05):
    """
    Plot the plausibility of seeing a species in a grid over a year based on Emergent Filters.
    :param data_dict: Data dictionary.
    :param species: Species.
    :param grid: Grid.
    :param year: Year.
    :param sign: Sign.
    :param threshold: Threshold.
    """
    df = pd.DataFrame(data_dict.items(), columns=['species_grid_day', 'plausibility'])
    df[['name_species', 'eea_grid_id', 'day_of_year']] = pd.DataFrame(df['species_grid_day'].tolist(), index=df.index)
    df.drop(columns=['species_grid_day'], inplace=True)
    data = df[(df.name_species == species) & (df.eea_grid_id == grid)]
    fig = px.line(data, x='day_of_year', y='plausibility', color_discrete_sequence=['#4d5f81'])
    fig.add_shape(type="rect", x0=0, x1=365, y0=0, y1=threshold, fillcolor="#dd752c", opacity=0.1)
    fig.add_shape(type="rect", x0=0, x1=365, y0=threshold, y1=data.plausibility.max(), fillcolor="#89b959", opacity=0.3)
    fig.add_annotation(x=365/2, y=threshold/2, text='flagged for review', showarrow=False, font=dict(color="#dd752c"))
    fig.add_annotation(x=365/2, y=threshold+0.02, text='OK', showarrow=False, font=dict(color="#667761"))
    fig.update_layout(title={'text': "{} Plausibility for seeing a {} in {} in '{}' {}".format(sign, species, year, grid, sign),
                             'x': 0.5,'xanchor': 'center','yanchor': 'top', 'font_color': '#4d5f81'},
                      xaxis_title='Day of Year', yaxis_title='Likelihood',
                      font=dict(family="Aleo", size=15, color="#4d5f81"))
    fig.show()