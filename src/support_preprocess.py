import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import math
from itertools import combinations

from scipy.stats import chi2_contingency

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def chi2_test(df,  categories, target_variable, alpha = 0.05):
    """
    Performs a chi-squared test of independence for multiple categorical variables against a target variable.

    Parameters:
    - categories (list): A list of categorical variable names to test.
    - target_variable (str): The name of the target variable for the chi-squared test.
    - alpha (float, optional): The significance level for the test. Default is 0.05.

    Returns:
    - None: Prints results of the chi-squared test and displays contingency tables and expected frequencies.
    """

    for category in categories:

        print(f"We are evaluating the variable {category.upper()}")

        df_crosstab = pd.crosstab(df[category], df[target_variable])
        display(df_crosstab)
        chi2, p, dof, expected = chi2_contingency(df_crosstab)

        if p < alpha:
            print(f"For the category {category.upper()} there are significant differences, p = {p:.4f}")
            display(pd.DataFrame(expected, index=df_crosstab.index, columns=df_crosstab.columns).round())
        else:
            print(f"For the category {category.upper()} there are NO significant differences, p = {p:.4f}\n")
        print("--------------------------")


def scale_df(df, cols, method="robust", include_others=False):
    """
    Scale selected columns of a DataFrame using specified scaling method.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        cols (list): List of columns to scale.
        method (str): Scaling method, one of ["minmax", "robust", "standard"]. Defaults to "robust".
        include_others (bool): If True, include non-scaled columns in the output. Defaults to False.
    
    Returns:
        pd.DataFrame: DataFrame with scaled columns (and optionally unscaled columns).
        scaler: Scaler object used for scaling.
    """
    if method not in ["minmax", "robust", "standard"]:
        raise ValueError(f"Invalid method '{method}'. Choose from ['minmax', 'robust', 'standard'].")
    
    if not all(col in df.columns for col in cols):
        missing = [col for col in cols if col not in df.columns]
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # Select the scaler
    scaler = {
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "standard": StandardScaler()
    }[method]
    
    # Scale the selected columns
    scaled_data = scaler.fit_transform(df[cols])
    df_scaled = pd.DataFrame(scaled_data, columns=cols, index=df.index)
    
    # Include unscaled columns if requested
    if include_others:
        unscaled_cols = df.drop(columns=cols)
        df_scaled = pd.concat([df_scaled, unscaled_cols], axis=1)
    
    return df_scaled, scaler


def find_outliers(dataframe, cols, method="lof", random_state=42, n_est=100, contamination=0.01, n_neigh=20): 
    """
    Identifies outliers in a given dataset using Isolation Forest or Local Outlier Factor methods.

    Parameters:
    - dataframe (pd.DataFrame): The input dataframe containing the data to analyze.
    - cols (list): List of column names to be used for outlier detection.
    - method (str, optional): The method to use for detecting outliers. Options are "ifo" (Isolation Forest) or "lof" (Local Outlier Factor). Defaults to "lof".
    - random_state (int, optional): Random seed for reproducibility when using Isolation Forest. Defaults to 42.
    - n_est (int, optional): Number of estimators for the Isolation Forest model. Defaults to 100.
    - contamination (float, optional): The proportion of outliers in the dataset. Defaults to 0.01.
    - n_neigh (int, optional): Number of neighbors for the Local Outlier Factor model. Defaults to 20.

    Returns:
    - (tuple): A tuple containing:
    - pd.DataFrame: The original dataframe with an added column 'outlier' indicating the outlier status (-1 for outliers, 1 for inliers).
    - object: The trained model used for outlier detection.

    Recommendations:
    - `n_estimators` (Isolation Forest): `100-300`. More trees improve accuracy, rarely needed >500.
    - `contamination`: `0.01-0.1`. Adjust based on expected anomalies (higher if >10% anomalies).
    - `n_neighbors` (LOF): `10-50`. Low for local anomalies, high for large/noisy datasets.
    """


    df = dataframe.copy()

    if method == "ifo":  
        model = IsolationForest(random_state=random_state, n_estimators=n_est, contamination=contamination, n_jobs=-1)
        outliers = model.fit_predict(X=df[cols])

    elif method == "lof":
        model = LocalOutlierFactor(n_neighbors=n_neigh, contamination=contamination, n_jobs=-1)
        outliers = model.fit_predict(X=df[cols])

    else:
        raise ValueError("Unrecognized method. Use 'ifo', or 'lof'.")
    
    df = pd.concat([df, pd.DataFrame(outliers, columns=['outlier'])], axis=1)

    return df, model