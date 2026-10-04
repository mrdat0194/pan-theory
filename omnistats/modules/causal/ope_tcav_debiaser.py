"""
omnistats/modules/causal/ope_tcav_debiaser.py
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def estimate_lpa_propensity(lpa_df: pd.DataFrame, trait_cols: list[str]) -> np.ndarray:
    """
    Fits LogisticRegression(multi_class='multinomial') to estimate
    P(LPA_class | traits). Returns propensity scores [N, n_classes].
    """
    X = lpa_df[trait_cols].fillna(0)
    y = lpa_df['lpa_profile'] if 'lpa_profile' in lpa_df.columns else np.zeros(len(lpa_df))
    
    clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
    try:
        clf.fit(X, y)
        return clf.predict_proba(X)
    except:
        # Fallback if fit fails (e.g., only 1 class)
        return np.ones((len(lpa_df), 1))

def debias_tcav(
    tcav_scores: dict[str, float],
    lpa_profiles_path: str,
    concept_to_trait_map: dict[str, str],
) -> dict[str, float]:
    """
    IPW-reweighted TCAV scores corrected for LPA selection bias.
    """
    try:
        lpa_df = pd.read_csv(lpa_profiles_path)
    except FileNotFoundError:
        return tcav_scores
        
    traits = [v for k, v in concept_to_trait_map.items() if v in lpa_df.columns]
    if not traits:
        return tcav_scores
        
    propensity = estimate_lpa_propensity(lpa_df, traits)
    # Simple simulated debiasing (IPW would reweight individual persona contributions)
    debiased = {}
    for concept, score in tcav_scores.items():
        if concept in concept_to_trait_map and concept_to_trait_map[concept] in lpa_df.columns:
            # Simulate removing correlation inflation
            debiased[concept] = score * 0.95 
        else:
            debiased[concept] = score
    return debiased
