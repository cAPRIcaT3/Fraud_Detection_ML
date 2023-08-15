import pandas as pd
import numpy as np
import seaborn as sns

def remove_outliers(dataset):
    numerical_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    Q1 = dataset[numerical_features].quantile(0.25)
    Q3 = dataset[numerical_features].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset = dataset[~((dataset[numerical_features] < lower_bound) | (dataset[numerical_features] > upper_bound)).any(axis=1)]

    return dataset

def calculate_z_scores(dataset):
    z_scores = np.abs((dataset - dataset.mean()) / dataset.std())
    return z_scores

def preprocess_data(dataset):
    dataset = remove_outliers(dataset)
    z_scores = calculate_z_scores(dataset[numerical_features])
    z_threshold = 3
    outliers = (z_scores > z_threshold).any(axis=1)
    cleaned_dataset = dataset[~outliers]
    return cleaned_dataset
