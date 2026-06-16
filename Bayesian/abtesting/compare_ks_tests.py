import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

def demonstrate_goodness_of_fit_superiority():
    print("=" * 60)
    print("DEMONSTRATION 1: KS TEST VS. CHI-SQUARE GOODNESS-OF-FIT")
    print("=" * 60)
    print("Testing if a sample of 100 continuous points follows a Normal distribution.")
    print("We draw samples from a Student's t-distribution with 3 degrees of freedom (which has heavier tails than normal).\n")
    
    # Generate non-normal data (t-distribution with heavy tails)
    data = stats.t.rvs(df=3, size=100)
    mu, sigma = np.mean(data), np.std(data)

    # 1. Kolmogorov-Smirnov Test (Direct CDF Comparison, No Binning)
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(mu, sigma))
    print(f"Kolmogorov-Smirnov Test (No binning needed):")
    print(f"  - p-value: {ks_p:.4f} (Rejects normality if < 0.05: {ks_p < 0.05})\n")

    # 2. Chi-Square Goodness-of-Fit (Requires arbitrary binning)
    print("Chi-Square Goodness-of-Fit (Results depend heavily on arbitrary bin count):")
    for bins in [5, 10, 20]:
        counts, edges = np.histogram(data, bins=bins)
        
        # Expected counts from Normal distribution
        cdf_vals = stats.norm.cdf(edges, loc=mu, scale=sigma)
        expected = (cdf_vals[1:] - cdf_vals[:-1])
        # Normalize expected probabilities to sum to 1 across the bins, 
        # then scale to total observations so sums match exactly.
        expected = expected / np.sum(expected) * np.sum(counts)
        
        # Chi-square requires expected frequency >= 5 in each cell
        # to be mathematically valid, which is often violated with too many bins.
        chi_stat, chi_p = stats.chisquare(counts, f_exp=expected)
        print(f"  - Bins: {bins:2d} | p-value: {chi_p:.4f} (Rejects normality if < 0.05: {chi_p < 0.05})")
    
    print("\nConclusion: The KS test is strictly superior here because it doesn't discard information")
    print("by binning, doesn't suffer from low expected bin frequencies, and returns an exact, unique p-value.")


def demonstrate_ks_vs_ttest():
    print("\n" + "=" * 60)
    print("DEMONSTRATION 2: TWO-SAMPLE KS-TEST VS. T-TEST FOR A/B TESTING")
    print("=" * 60)
    print("Goal: Compare two web-page designs (A vs B).")
    print("Scenario: Users in both groups spend the same average time on site (mean = 10s),")
    print("but Group B has a much wider variance (some leave immediately, others stay very long).\n")

    # Group A: High consistency (Mean = 10, SD = 1)
    group_A = np.random.normal(loc=10.0, scale=1.0, size=500)
    group_A = (group_A - np.mean(group_A)) + 10.0  # Force mean to be exactly 10.0
    
    # Group B: Highly spread out (Mean = 10, SD = 4)
    group_B = np.random.normal(loc=10.0, scale=4.0, size=500)
    group_B = (group_B - np.mean(group_B)) + 10.0  # Force mean to be exactly 10.0

    # 1. T-test (Compares Means only)
    t_stat, t_p = stats.ttest_ind(group_A, group_B)
    print("T-test (Two-sample, equal means comparison):")
    print(f"  - Group A Mean: {np.mean(group_A):.2f} | Group B Mean: {np.mean(group_B):.2f}")
    print(f"  - p-value: {t_p:.4f} (Detects difference in means if < 0.05: {t_p < 0.05})")
    print("    -> T-test concludes Group A and Group B are IDENTICAL because averages are identical.\n")

    # 2. Two-Sample KS Test (Compares overall distribution shape)
    ks_stat, ks_p = stats.ks_2samp(group_A, group_B)
    print("Two-Sample Kolmogorov-Smirnov Test (Distribution shape comparison):")
    print(f"  - p-value: {ks_p:.4e} (Detects difference in distribution if < 0.05: {ks_p < 0.05})")
    print("    -> KS Test correctly concludes the distributions are SIGNIFICANTLY DIFFERENT.\n")
    
    print("Conclusion: The KS test is superior for detecting ANY change in user behavior (e.g., changes in spread).")
    print("However, if your business goal is strictly to increase the average metrics, the T-test is what you need,")
    print("as KS does not tell you *which* group is better (it only tells you they are different).")


def demonstrate_why_ks_cannot_replace_chisq_clicks():
    print("\n" + "=" * 60)
    print("DEMONSTRATION 3: WHY KS TEST CANNOT REPLACE CHI-SQUARE FOR CATEGORICAL A/B CLICKS")
    print("=" * 60)
    print("Goal: Check if Ad B gets significantly more clicks than Ad A (binary conversion data).")
    print("Data: 0 (no click) or 1 (click).\n")

    ad_A = np.random.binomial(n=1, p=0.10, size=1000) # 10% conversion
    ad_B = np.random.binomial(n=1, p=0.14, size=1000) # 14% conversion

    # Contingency table for Chi-Square Test of Independence
    clicks_A = np.sum(ad_A)
    clicks_B = np.sum(ad_B)
    contingency = np.array([
        [clicks_A, 1000 - clicks_A],
        [clicks_B, 1000 - clicks_B]
    ])

    # 1. Chi-Square Test of Independence
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)
    print("Chi-Square Test of Independence (Valid for discrete categories):")
    print(f"  - Ad A Clicks: {clicks_A} | Ad B Clicks: {clicks_B}")
    print(f"  - p-value: {chi2_p:.4f} (Significant if < 0.05: {chi2_p < 0.05})\n")

    # 2. KS Test on Binary data (Mathematically invalid)
    ks_stat, ks_p = stats.ks_2samp(ad_A, ad_B)
    print("Two-Sample KS Test applied to binary data:")
    print(f"  - p-value: {ks_p:.4f}")
    print("    -> DANGER: KS test requires continuous data. Since binary data has only two values (0 and 1),")
    print("       the cumulative distribution functions are step functions with huge ties, making the KS p-values")
    print("       mathematically invalid and highly conservative (unreliable).")


if __name__ == '__main__':
    demonstrate_goodness_of_fit_superiority()
    demonstrate_ks_vs_ttest()
    demonstrate_why_ks_cannot_replace_chisq_clicks()
