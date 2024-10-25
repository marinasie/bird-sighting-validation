"""
Utils for modeling.
"""

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import hdbscan
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from pyod.models.auto_encoder import AutoEncoder
from sklearn.neighbors import LocalOutlierFactor

from itertools import product

import warnings
warnings.filterwarnings("ignore")


# DBSCAN
# DBSCAN
def manual_grid_search_dbscan(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series, eps_values:list=[0.5], min_samples_values:list=[5]):
    """
    Perform a manual grid search for DBSCAN.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :param eps_values: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples_values: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    :return: Best result and all results.
    """
    combinations = []

    for eps, min_samples in product(eps_values, min_samples_values):
            print(f'eps: {eps}, min_samples: {min_samples}')
            conf_mats = evaluate_dbscan(X_train, X_test, y_test, eps, min_samples)
            combinations.append((eps, min_samples, conf_mats))
    
    return combinations


def evaluate_dbscan(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series, eps:float=0.5, min_samples:int=5):
    """
    Evaluate DBSCAN on a test set.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :return: Confusion matrix.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X_train)

    y_pred = dbscan.fit_predict(X_test)
    y_pred[y_pred > 0] = 0
    y_pred[y_pred == -1] = 1
    return confusion_matrix(y_test, y_pred)


# HDBSCAN
# HDBSCAN
def manual_grid_search_hdbscan(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series, min_cluster_size_values:list=[5], min_samples_values:list=[None]):
    """
    Perform a manual grid search for HDBSCAN.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :param min_cluster_size_values: The minimum size of clusters.
    :param min_samples_values: The number of samples in a neighbourhood for a point to be considered a core point.
    :return: Best result and all results.
    """
    combinations = []

    for min_cluster_size, min_samples in product(min_cluster_size_values, min_samples_values):
            print(f'min_cluster_size: {min_cluster_size}, min_samples: {min_samples}')
            conf_mats = evaluate_hdbscan(X_train, X_test, y_test, min_cluster_size, min_samples)
            combinations.append((min_cluster_size, min_samples, conf_mats))
    
    return combinations


def evaluate_hdbscan(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series, min_cluster_size:int=5, min_samples=None):
    """
    Evaluate DBSCAN on a test set.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :return: Confusion matrix.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, 
        min_samples=min_samples,
        prediction_data=True
    )
    clusterer.fit(X_train)
    y_pred, _ = hdbscan.approximate_predict(clusterer, X_test)
    y_pred = (y_pred == -1).astype(int)
    return confusion_matrix(y_test, y_pred)


# Isolation Forest
# Isolation Forest
def manual_grid_search_isolation_forest(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series,  n_estimator_values, contamination_values, max_features_values):
    """
    Perform a manual grid search for Isolation Forest.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :param contamination_values: The proportion of outliers in the data set.
    :param max_features_values: The number of features to draw from X to train each base estimator.
    :param n_estimator_values: The number of base estimators in the ensemble.
    :return: Best result and all results.
    """
    combinations = []

    for n_estimators, contamination, max_features  in product(n_estimator_values, contamination_values, max_features_values):
            print(f'n_estimators: {n_estimators}, contamination: {contamination}, max_feature: {max_features}')
            conf_mats = evaluate_isolation_forest(X_train, X_test, y_test, n_estimators, contamination, max_features)
            combinations.append((max_features, n_estimators, conf_mats))
        
    return combinations


def evaluate_isolation_forest(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series, n_estimators=100, contamination='auto', max_features=1):
    """
    Evaluate Isolation Forest on a test set.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :return: Confusion matrix.
    """
    isolation_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, max_features=max_features, verbose=0)
    isolation_forest.fit(X_train)

    y_pred = isolation_forest.predict(X_test)
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    return confusion_matrix(y_test, y_pred)


# Autoencoder
# Autoencoder
def manual_grid_search_autoencoder(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series, hidden_neurons_values:list=[[64, 32, 32, 64]], epochs_values:list=[100], batch_size_values:list=[32]):
    """
    Perform a manual grid search for Autoencoder.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :param batch_size_values: Number of samples per gradient update.
    :param epochs_values: Number of epochs to train the model.
    :param hidden_neurons: Number of neurons in each hidden layer.
    :return: Best result and all results.
    """
    combinations = []

    for hidden_neurons, epochs, batch_size,  in product(hidden_neurons_values, epochs_values, batch_size_values):
        print(f'hidden_neurons: {hidden_neurons}, epochs: {epochs}, batch_size: {batch_size}')
        conf_mats = evaluate_autoencoder(X_train, X_test, y_test, hidden_neurons, epochs, batch_size)
        combinations.append((hidden_neurons, epochs, batch_size, conf_mats))
    
    return combinations


def evaluate_autoencoder(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series, hidden_neurons=[64, 32, 32, 64], epochs:int=100, batch_size:int=32):
    """
    Evaluate Autoencoder on a test set.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :return: Confusion matrix.
    """
    ae = AutoEncoder(hidden_neurons=hidden_neurons, epochs=epochs, batch_size=batch_size, verbose=0)
    ae.fit(X_train)
    y_pred = ae.predict(X_test)

    return confusion_matrix(y_test, y_pred)


# One-Class SVM
# One-Class SVM
def manual_grid_search_oneclass_svm(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series, kernel_values:list=['rbf'], gamma_values:list=['scale'], nu_values:list=[0.5]):
    """
    Perform a manual grid search for One-Class SVM.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :param nu_values: Upper bounds on the fraction of training errors and a lower bounds of the fraction of support vectors.
    :param kernel_values: Specifies the kernel types to be used in the algorithm.
    :param gamma_values: Kernel coefficients for ‘rbf’, ‘poly’ and ‘sigmoid’.
    :return: Best result and all results.
    """
    combinations = []

    for kernel, gamma, nu in product(kernel_values, gamma_values, nu_values):
        print(f'kernel: {kernel}, gamma: {gamma}, nu: {nu}')
        conf_mats = evaluate_oneclass_svm(X_train, X_test, y_test, kernel, gamma,  nu)
        combinations.append((kernel, gamma, nu, conf_mats))
    
    return combinations


def evaluate_oneclass_svm(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series, kernel:str='rbf', gamma:str='scale', nu:int=0.5):
    """
    Evaluate One-Class SVM on a test set and calculate the F1 Score.

    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :param nu: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    :param kernel: Specifies the kernel type to be used in the algorithm.
    :param gamma: Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
    :return: F1 Score.
    """
    svm = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu, verbose=0)
    svm.fit(X_train)
    y_pred = svm.predict(X_test)
    y_pred = (y_pred == -1).astype(int)

    return confusion_matrix(y_test, y_pred)


# Local Outlier Factor
# Local Outlier Factor
def manual_grid_search_lof(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series, n_neighbors_values:list=[20], algorithm_values:list=['auto'], leaf_size_values:list=[30]):
    """
    Perform a manual grid search for Local Outlier Factor.
    :param X_train: Training set.
    :param X_test: Test set.
    :param y_test: Test labels.
    :param n_neighbors_values: Number of neighbors to use by default for kneighbors queries.
    :return: Best result and all results.
    """
    combinations = []

    for n_neighbors, algorithm, leaf_size in product(n_neighbors_values, algorithm_values, leaf_size_values):
        print(f'n_neighbors: {n_neighbors}, leaf_size: {leaf_size}')
        conf_mats = evaluate_lof(X_train, X_test, y_test, n_neighbors, algorithm, leaf_size)
        combinations.append((n_neighbors, leaf_size, conf_mats))
    
    return combinations


def evaluate_lof(X_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.Series, n_neighbors:int=20, algorithm:str='auto', leaf_size:int=30):
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        leaf_size=leaf_size,
        novelty=True,
    )
    lof.fit(X_train)
    y_pred = lof.predict(X_test)
    y_pred = (y_pred == -1).astype(int)

    return confusion_matrix(y_test, y_pred)