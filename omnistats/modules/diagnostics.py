"""
omnistats/modules/diagnostics.py
──────────────────────────────────
Stage 0: Pre-Flight Diagnostics (VALIDATE)

Implements:
1. Linear Algebra & Condition Number Diagnostics (SVD, Rank, Covariance Determinant)
2. Non-Parametric Kernel MMD Two-Sample Test (Gretton et al., 2008 / arXiv:0805.2368)
3. Sample Ratio Mismatch (SRM) Chi-Square Test
4. Convexity & Gram Matrix Conditioning (Bauschke & Combettes, 2011)
5. Distributional Assumptions (D'Agostino-Pearson Normality, Levene Variance Homogeneity)
"""
import numpy as np
import pandas as pd
import scipy.stats as scipy_stats
from scipy.spatial.distance import pdist, squareform

def compute_rbf_mmd(X1: np.ndarray, X2: np.ndarray, gamma: float = None) -> float:
    """
    Computes Maximum Mean Discrepancy (MMD^2) with an RBF kernel (Gretton et al., 2008 / arXiv:0805.2368).
    
    # --- Educational Manual Implementation ---
    # def manual_rbf_kernel(X, Y, gamma):
    #     dist_sq = np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=-1)
    #     return np.exp(-gamma * dist_sq)
    # K_11 = manual_rbf_kernel(X1, X1, gamma)
    # K_22 = manual_rbf_kernel(X2, X2, gamma)
    # K_12 = manual_rbf_kernel(X1, X2, gamma)
    # mmd_sq = np.mean(K_11) + np.mean(K_22) - 2 * np.mean(K_12)
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    
    if gamma is None:
        # Median heuristic for gamma
        combined = np.vstack([X1, X2])
        dists = pdist(combined, metric='sqeuclidean')
        median_dist = np.median(dists) if len(dists) > 0 else 1.0
        gamma = 1.0 / (median_dist + 1e-8)
        
    K_11 = np.exp(-gamma * squareform(pdist(X1, metric='sqeuclidean')))
    K_22 = np.exp(-gamma * squareform(pdist(X2, metric='sqeuclidean')))
    
    # Cross terms
    dists_12 = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=-1)
    K_12 = np.exp(-gamma * dists_12)
    
    mmd_sq = np.mean(K_11) + np.mean(K_22) - 2 * np.mean(K_12)
    return max(0.0, float(mmd_sq))

def check_linear_algebra_prerequisites(X: np.ndarray, feature_names: list) -> dict:
    """
    Checks Matrix Rank, Covariance Determinant, Condition Number, and Positive-Definiteness.
    (Bauschke & Combettes, 2011; Golub & Van Loan).
    """
    n_samples, n_features = X.shape
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    cov_matrix = np.cov(X_std, rowvar=False)
    
    # 1. Determinant
    det = np.linalg.det(cov_matrix)
    
    # 2. Rank
    rank = np.linalg.matrix_rank(cov_matrix)
    
    # 3. SVD Condition Number
    U, S, Vt = np.linalg.svd(cov_matrix)
    cond_num = S[0] / S[-1] if S[-1] > 1e-12 else np.inf
    min_eig = np.min(S)
    
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "determinant": float(det),
        "rank": int(rank),
        "full_rank": rank == n_features,
        "condition_number": float(cond_num),
        "min_eigenvalue": float(min_eig),
        "strictly_convex": min_eig > 1e-8
    }

def check_sample_ratio_mismatch(df: pd.DataFrame, group_col: str, expected_ratio: float = 0.5) -> dict:
    """
    Chi-Square Goodness-of-Fit SRM Test (Fabijan et al., Microsoft Research).
    """
    counts = df[group_col].value_counts()
    n_total = len(df)
    n_ctrl = counts.iloc[0] if len(counts) > 0 else 0
    n_treat = counts.iloc[1] if len(counts) > 1 else 0
    
    expected_ctrl = n_total * expected_ratio
    expected_treat = n_total * (1 - expected_ratio)
    
    chi2_stat, p_value = scipy_stats.chisquare(
        f_obs=[n_ctrl, n_treat],
        f_exp=[expected_ctrl, expected_treat]
    )
    
    return {
        "n_ctrl": int(n_ctrl),
        "n_treat": int(n_treat),
        "chi2_stat": float(chi2_stat),
        "p_value": float(p_value),
        "srm_detected": p_value < 0.01
    }

def run_stage0_diagnostics(df: pd.DataFrame, indicator_cols: list, ab_group_col: str = None, ab_metric_col: str = None) -> dict:
    """
    Executes full Stage 0 Diagnostics suite and prints formatted insights.
    """
    print("\n" + "=" * 60)
    print("  STAGE 0 / 5 — Pre-Flight Diagnostics (VALIDATE)")
    print("=" * 60)
    
    # 1. Linear Algebra & Convexity Checks
    print("\n--- 1. Linear Algebra & Convexity Diagnostics ---")
    df_clean = df.dropna(subset=indicator_cols)
    X = df_clean[indicator_cols].values
    la_res = check_linear_algebra_prerequisites(X, indicator_cols)
    
    print(f"  * Feature Matrix Shape: {la_res['n_samples']} samples x {la_res['n_features']} features")
    print(f"  * Covariance Determinant: {la_res['determinant']:.6f}")
    print(f"  * Matrix Rank: {la_res['rank']} / {la_res['n_features']} ({'Full Rank' if la_res['full_rank'] else 'WARNING: Rank Deficient'})")
    print(f"  * SVD Condition Number: {la_res['condition_number']:.2f}")
    print(f"  * Gram Matrix Min Eigenvalue (lambda_min): {la_res['min_eigenvalue']:.6f} ({'Strictly Convex' if la_res['strictly_convex'] else 'Non-Convex / Singular'})")

    # 2. Sample Ratio Mismatch (SRM)
    srm_res = None
    if ab_group_col and ab_group_col in df.columns:
        print("\n--- 2. Sample Ratio Mismatch (SRM) Test ---")
        srm_res = check_sample_ratio_mismatch(df, ab_group_col)
        print(f"  * Allocation Split: Control={srm_res['n_ctrl']}, Treatment={srm_res['n_treat']}")
        print(f"  * Chi-Square p-value: {srm_res['p_value']:.4f}")
        if srm_res['srm_detected']:
            print("  * [WARNING] Sample Ratio Mismatch (SRM) detected! Traffic allocation may be corrupted.")
        else:
            print("  * [OK] No Sample Ratio Mismatch detected. Traffic split is unbiased.")

    # 3. Kernel MMD & Distributional Checks
    mmd_res = None
    if ab_group_col and ab_metric_col and ab_group_col in df.columns and ab_metric_col in df.columns:
        print("\n--- 3. Kernel MMD & Distributional Diagnostics ---")
        df_sub = df.dropna(subset=[ab_group_col, ab_metric_col])
        groups = [val.values.reshape(-1, 1) for _, val in df_sub.groupby(ab_group_col)[ab_metric_col]]
        if len(groups) == 2:
            g1, g2 = groups[0], groups[1]
            mmd_val = compute_rbf_mmd(g1, g2)
            print(f"  • Maximum Mean Discrepancy (MMD² in RKHS): {mmd_val:.6f}")
            
            # Normality check
            _, p_norm1 = scipy_stats.normaltest(g1.ravel())
            _, p_norm2 = scipy_stats.normaltest(g2.ravel())
            is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
            print(f"  • Metric Normality (D'Agostino-Pearson): {'Normal' if is_normal else 'Non-Normal (PyMC / Mann-Whitney recommended)'}")
            
            # Variance Homogeneity (Levene)
            _, p_levene = scipy_stats.levene(g1.ravel(), g2.ravel())
            print(f"  • Homogeneity of Variance (Levene p-val): {p_levene:.4f} ({'Equal Variances' if p_levene > 0.05 else 'Unequal Variances (Welch t-test required)'})")

    passed = la_res['full_rank'] and abs(la_res['determinant']) > 1e-6
    if passed:
        print("\n[Stage 0 PASSED] Pre-flight prerequisites met. Proceeding to Stage 1...\n")
    else:
        print("\n[Stage 0 WARNING] Critical mathematical prerequisites failed (singularity or rank deficiency)!\n")

    return {
        "passed": passed,
        "linear_algebra": la_res,
        "srm": srm_res
    }
