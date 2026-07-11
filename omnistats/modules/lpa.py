"""
omnistats/modules/lpa.py
─────────────────────────
Latent Profile Analysis (LPA) module.
Migrated from lpa_analysis/step2_run_lpa.py.

Fits Gaussian Mixture Models with diagonal covariance (mathematically
equivalent to LPA in tidySEM / Mplus) for K = K_MIN..K_MAX.
Reports fit statistics and assigns each observation to its most likely profile.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# Allow running as __main__ from the omnistats directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUT_DIR, INDICATOR_COLS, N_PROFILES, K_MIN, K_MAX


# ─── Fit metric helpers ───────────────────────────────────────────────────────

def _entropy(model: GaussianMixture, X: np.ndarray) -> float:
    """Relative classification entropy (1 = perfect separation, 0 = random)."""
    probs = model.predict_proba(X)
    eps   = 1e-10
    H     = -np.sum(probs * np.log(probs + eps)) / (len(X) * np.log(model.n_components + eps))
    return max(0.0, 1.0 - H)


def _abic(log_lik: float, n_params: int, n: int) -> float:
    """Sample-size adjusted BIC (aBIC / saBIC)."""
    n_star = (n + 2) / 24
    return -2 * log_lik + n_params * np.log(n_star)


# ─── Main function ────────────────────────────────────────────────────────────

def run_lpa(df: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Fit GMM models for K = K_MIN..K_MAX and assign profiles using N_PROFILES.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset with z-scored indicator columns ({col}_z).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    df_profiles : pd.DataFrame
        Original df augmented with 'Profile' (1-indexed) and posterior probabilities.
    fit_df : pd.DataFrame
        Model fit statistics for K = K_MIN..K_MAX.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    z_cols = [f"{c}_z" for c in INDICATOR_COLS if f"{c}_z" in df.columns]
    X      = df[z_cols].values
    n, p   = X.shape

    fit_rows = []
    models   = {}

    for k in range(K_MIN, K_MAX + 1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            n_init=10,
            max_iter=500,
            random_state=42,
        )
        gm.fit(X)
        models[k] = gm

        ll       = gm.score(X) * n
        n_params = k * p + k * p + (k - 1)   # means + variances + mixing weights
        fit_rows.append({
            "K":             k,
            "LogLikelihood": round(ll, 3),
            "AIC":           round(gm.aic(X), 3),
            "BIC":           round(gm.bic(X), 3),
            "aBIC":          round(_abic(ll, n_params, n), 3),
            "Entropy":       round(_entropy(gm, X), 4),
            "n_params":      n_params,
        })

    fit_df = pd.DataFrame(fit_rows)

    # Approximate LMR-LRT p-values
    from scipy.stats import chi2 as _chi2
    lmr_p = [np.nan]
    for i in range(1, len(fit_df)):
        chi2_stat = -2 * (fit_df.loc[i - 1, "LogLikelihood"] - fit_df.loc[i, "LogLikelihood"])
        df_diff   = fit_df.loc[i, "n_params"] - fit_df.loc[i - 1, "n_params"]
        lmr_p.append(round(1 - _chi2.cdf(chi2_stat, df=max(df_diff, 1)), 4))
    fit_df["LMR_LRT_p"] = lmr_p

    fit_path = os.path.join(OUTPUT_DIR, "lpa_fit_stats.csv")
    fit_df.to_csv(fit_path, index=False)
    if verbose:
        print(f"[LPA] Fit statistics saved -> {fit_path}")
        print(fit_df.to_string(index=False))

    # Assign profiles using N_PROFILES
    chosen = models[N_PROFILES]
    df_out = df.copy()
    df_out["Profile"]          = chosen.predict(X) + 1          # 1-indexed
    df_out["Profile_Max_Prob"] = chosen.predict_proba(X).max(axis=1)
    for ki in range(N_PROFILES):
        df_out[f"P_Profile_{ki + 1}"] = chosen.predict_proba(X)[:, ki]

    prof_path = os.path.join(OUTPUT_DIR, "lpa_profiles.csv")
    df_out.to_csv(prof_path, index=False)
    if verbose:
        print(f"[LPA] Profile assignments (K={N_PROFILES}) saved -> {prof_path}")

    return df_out, fit_df
