"""
OmniStats Causal Module: Matching & Alignment
Implements Propensity Score Matching, Optimal Bipartite Matching, and Quantile Matching.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def propensity_score_match(df: pd.DataFrame, treatment_col: str, covariate_cols: list) -> pd.DataFrame:
    """
    Performs 1:1 nearest neighbor Propensity Score Matching (PSM) without replacement.
    
    Args:
        df: DataFrame containing the data.
        treatment_col: Name of the binary treatment column (1 for treated, 0 for control).
        covariate_cols: List of column names to be used as covariates.
        
    Returns:
        Matched DataFrame containing the paired treated and control units.
    """
    df = df.copy().reset_index(drop=True)
    
    # Fit logistic regression to get propensity scores
    X = df[covariate_cols]
    y = df[treatment_col]
    
    lr = LogisticRegression(solver='liblinear', random_state=42)
    lr.fit(X, y)
    df['propensity_score'] = lr.predict_proba(X)[:, 1]
    
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    
    if len(treated) == 0 or len(control) == 0:
        return pd.DataFrame(columns=df.columns)
        
    # Nearest neighbor matching
    nn = NearestNeighbors(n_neighbors=len(control), metric='euclidean')
    nn.fit(control[['propensity_score']].values)
    
    distances, indices = nn.kneighbors(treated[['propensity_score']].values)
    
    matched_controls = []
    used_controls = set()
    treated_indices = []
    
    for i in range(len(treated)):
        for j in range(len(control)):
            control_idx = indices[i, j]
            if control_idx not in used_controls:
                matched_controls.append(control_idx)
                used_controls.add(control_idx)
                treated_indices.append(i)
                break
                
    treated_matched = treated.iloc[treated_indices]
    control_matched = control.iloc[matched_controls]
    
    matched_df = pd.concat([treated_matched, control_matched]).sort_values(by=treatment_col, ascending=False).reset_index(drop=True)
    return matched_df

def optimal_bipartite_match(df: pd.DataFrame, treatment_col: str, covariate_cols: list) -> pd.DataFrame:
    """
    Performs optimal 1:1 bipartite matching between treated and control units using the Hungarian algorithm,
    minimizing the total Euclidean distance across all matched pairs.
    
    Args:
        df: DataFrame containing the data.
        treatment_col: Name of the binary treatment column.
        covariate_cols: List of covariates used for distance calculation.
        
    Returns:
        Matched DataFrame with exactly paired treated and control units.
    """
    df = df.copy().reset_index(drop=True)
    
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    if len(treated) == 0 or len(control) == 0:
        return pd.DataFrame(columns=df.columns)
        
    # Standardize covariates for distance calculation
    scaler = StandardScaler()
    X_treated = scaler.fit_transform(treated[covariate_cols])
    X_control = scaler.transform(control[covariate_cols])
    
    cost_matrix = cdist(X_treated, X_control, metric='euclidean')
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    treated_matched = treated.iloc[row_ind]
    control_matched = control.iloc[col_ind]
    
    matched_df = pd.concat([treated_matched, control_matched]).sort_values(by=treatment_col, ascending=False).reset_index(drop=True)
    return matched_df

def quantile_match(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Matches the distribution of a source array to a target array via quantile mapping.
    
    Args:
        source: 1D numpy array representing the source distribution.
        target: 1D numpy array representing the target distribution.
        
    Returns:
        1D numpy array of the source distribution aligned to the target distribution.
    """
    source_sorted = np.sort(source)
    target_sorted = np.sort(target)
    
    if len(source) <= 1:
        return source
        
    # Calculate empirical CDF percentiles for the source array
    percentiles = np.argsort(np.argsort(source)) / (len(source) - 1)
    
    # Interpolate to find the corresponding values in the target distribution
    matched_source = np.interp(percentiles, np.linspace(0, 1, len(target)), target_sorted)
    
    return matched_source
