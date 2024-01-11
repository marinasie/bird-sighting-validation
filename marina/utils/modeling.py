"""
Utils for modeling Emergent Filters and Outlier Detection algorithms.
"""

import pandas as pd

import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from pyod.models.auto_encoder import AutoEncoder
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
    df = pd.get_dummies(df, columns=['land_use_coord_numeric'], dtype=int)
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
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df = df.reindex(sorted(df.columns), axis=1)
        X_train_per_species[name_species] = df
    return X_train_per_species


def is_unlikely(sighting, emergent_filters_lookup, threshold=0.05):
    key = (sighting['name_species'], sighting['eea_grid_id'], sighting['date'])
    plausibility = emergent_filters_lookup.get(key, None)
    return plausibility is not None and plausibility < threshold


def evaluate_emergent_filters(X_test:pd.DataFrame, y_test:pd.Series, emergent_filters:dict, threshold:float):
    """
    Evaluate Emergent Filters on a test set.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :param path_to_lookup_table: Path to Emergent Filter lookup table.
    :return: Confusion matrix.
    """
    y_pred = X_test.apply(is_unlikely, args=(emergent_filters, threshold), axis=1)
    return confusion_matrix(y_test, y_pred)


def evaluate_dbscan(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series):
    """
    Evaluate DBSCAN on a test set.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :return: Confusion matrix.
    """
    dbscan = DBSCAN(eps=0.1, min_samples=5)
    _ = dbscan.fit(X_train)
    y_pred = dbscan.fit_predict(X_test)
    y_pred[y_pred > 0] = 0
    y_pred[y_pred == -1] = 1
    return confusion_matrix(y_test, y_pred)


def evaluate_isolation_forest(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series):
    """
    Evaluate Isolation Forest on a test set.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :return: Confusion matrix.
    """
    isolation_forest = IsolationForest(contamination=0.01, random_state=0)
    isolation_forest.fit(X_train)
    y_pred = isolation_forest.predict(X_test)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    return confusion_matrix(y_test, y_pred)


def evaluate_autoencoder(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series):
    """
    Evaluate Autoencoder on a test set.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :return: Confusion matrix.
    """
    ae = AutoEncoder(hidden_neurons=[5, 2, 5], batch_size=256, epochs=10, verbose=0)
    ae.fit(X_train)
    y_pred = ae.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def plot_fn_vs_fp(false_negatives, false_positives, x_parameter, x_parameter_name):
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


def get_confusion_matrix_values(conf_matrices):
    TPs = [mat[1, 1] for mat in conf_matrices]  # True Positives
    TNs = [mat[0, 0] for mat in conf_matrices]  # True Negatives
    FPs = [mat[0, 1] for mat in conf_matrices]  # False Positives
    FNs = [mat[1, 0] for mat in conf_matrices]  # False Negatives
    return TPs, TNs, FPs, FNs
