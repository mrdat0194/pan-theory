"""
step2_run_lpa.py
────────────────
Run Latent Profile Analysis (LPA) using sklearn GaussianMixture
with diagonal covariance (equivalent to LPA in tidySEM/Mplus).

Fits K = K_MIN … K_MAX models, computes:
  - AIC, BIC, Entropy, Approximate BIC (aBIC)
  - LMR-LRT approximation (chi-square difference in log-likelihood)

Saves:
  - outputs/lpa_fit_stats.csv   — model comparison table (review to choose K)
  - outputs/lpa_profiles.csv    — full dataset with assigned profile labels
"""
import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from config import OUTPUT_DIR, INDICATOR_COLS, N_PROFILES, K_MIN, K_MAX


def entropy(model: GaussianMixture, X: np.ndarray) -> float:
    """Relative entropy (classification entropy) — higher is better separation."""
    probs = model.predict_proba(X)
    # Avoid log(0)
    eps = 1e-10
    H = -np.sum(probs * np.log(probs + eps)) / (len(X) * np.log(model.n_components + eps))
    return max(0.0, 1.0 - H)   # 1 = perfect separation, 0 = random


def approximate_bic(log_lik: float, n_params: int, n: int) -> float:
    """Sample-size adjusted BIC (aBIC / saBIC)."""
    n_star = (n + 2) / 24
    return -2 * log_lik + n_params * np.log(n_star)


def run_lpa(df: pd.DataFrame) -> pd.DataFrame:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    z_cols = [f"{c}_z" for c in INDICATOR_COLS]
    X = df[z_cols].values
    n, p = X.shape

    fit_rows = []
    models   = {}

    for k in range(K_MIN, K_MAX + 1):
        gm = GaussianMixture(
            n_components=k,
            covariance_type="diag",   # LPA = diagonal (within-class variances, no covariances)
            n_init=10,
            max_iter=500,
            random_state=42,
        )
        gm.fit(X)
        models[k] = gm

        ll       = gm.score(X) * n      # total log-likelihood
        n_params = k * p + k * p + (k - 1)  # means + variances + mixing weights
        aic      = gm.aic(X)
        bic      = gm.bic(X)
        abic     = approximate_bic(ll, n_params, n)
        ent      = entropy(gm, X)

        fit_rows.append({
            "K": k,
            "LogLikelihood": round(ll, 3),
            "AIC": round(aic, 3),
            "BIC": round(bic, 3),
            "aBIC": round(abic, 3),
            "Entropy": round(ent, 4),
            "n_params": n_params,
        })

    fit_df = pd.DataFrame(fit_rows)

    # ── LMR-LRT p-value (approximate: chi-sq difference) ────────────────────
    lmr_p = [np.nan]
    for i in range(1, len(fit_df)):
        chi2_stat = -2 * (fit_df.loc[i - 1, "LogLikelihood"] - fit_df.loc[i, "LogLikelihood"])
        df_diff   = fit_df.loc[i, "n_params"] - fit_df.loc[i - 1, "n_params"]
        from scipy.stats import chi2
        p_val = 1 - chi2.cdf(chi2_stat, df=max(df_diff, 1))
        lmr_p.append(round(p_val, 4))
    fit_df["LMR_LRT_p"] = lmr_p

    fit_path = os.path.join(OUTPUT_DIR, "lpa_fit_stats.csv")
    fit_df.to_csv(fit_path, index=False)
    print(f"[Step 2] Fit statistics saved → {fit_path}")
    print(fit_df.to_string(index=False))

    # ── Assign profile labels using the chosen K ─────────────────────────────
    chosen_model = models[N_PROFILES]
    df = df.copy()
    df["Profile"]     = chosen_model.predict(X) + 1    # 1-indexed
    df["Profile_Max_Prob"] = chosen_model.predict_proba(X).max(axis=1)

    # Attach per-profile probabilities
    for k_idx in range(N_PROFILES):
        df[f"P_Profile_{k_idx + 1}"] = chosen_model.predict_proba(X)[:, k_idx]

    out_path = os.path.join(OUTPUT_DIR, "lpa_profiles.csv")
    df.to_csv(out_path, index=False)
    print(f"[Step 2] Profile assignments (K={N_PROFILES}) saved → {out_path}")

    return df, fit_df


if __name__ == "__main__":
    df_in = pd.read_csv(os.path.join(OUTPUT_DIR, "lpa_input.csv"))
    run_lpa(df_in)
