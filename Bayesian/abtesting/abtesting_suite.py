import os
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set seed for reproducibility
np.random.seed(42)

# Reference Links (originally in Excercise.py):
# - AB test exercise sheets:
#   https://docs.google.com/spreadsheets/d/1-Kip2_MicfV0ebRZX8N5vhj3b-8RxuQp/edit#gid=1415247164
#   https://docs.google.com/spreadsheets/d/1CpRMgzMfWxmvRASPkfHOd4XBQDNdaW2Q_W1-GitubtU/edit#gid=138966392
# - Z-test Colab Notebook:
#   https://colab.research.google.com/drive/1hu-vJLeUHTQEn8fytWWRMiI2z-mtpIvS?usp=sharing
# - Gaussian Processes:
#   https://docs.pymc.io/Gaussian_Processes.html
#   https://en.wikipedia.org/wiki/Gaussian_process
# - Non-parametric Bayesian reference:
#   https://www.stats.ox.ac.uk/~teh/research/npbayes/Teh2010a.pdf

def generate_simulated_data(output_file='simulated_traffic.csv', num_rows=100000, plot=False):
    """
    Generates simulated user views and conversions based on Log-Normal and Beta distributions.
    Saves the subset (first num_rows rows) to output_file.
    """
    print(f"\nGenerating simulated traffic ({num_rows} rows)...")
    mu = 5
    sigma2 = 1.3
    N = 1000000
    
    # Generate Views (Log-Normal distribution)
    views = np.absolute(np.exp(stats.norm(mu, sigma2).rvs(N)).astype(np.int64) + 1)
    
    if plot:
        p00 = np.percentile(views, 0)
        p99 = np.percentile(views, 99)
        plt.figure()
        ax = sns.histplot(views, bins=np.linspace(p00, p99, 50))
        ax.set_title('Views, P99 = {}'.format(p99))
        ax.set(xlabel = 'Views')
        plt.tight_layout()
        plt.show()
    
    # Generate CTR success rate (Beta distribution)
    success_rate_target = 0.02
    beta_val = 100
    alpha_val = success_rate_target * beta_val / (1 - success_rate_target)
    success_rates = stats.beta(alpha_val, beta_val).rvs(N)
    
    if plot:
        p00 = np.percentile(success_rates, 0)
        p99 = np.percentile(success_rates, 99)
        plt.figure()
        ax = sns.histplot(success_rates, bins=np.linspace(p00, p99, 50))
        ax.set_title('CTR, P99 = {}'.format(p99))
        ax.set(xlabel = 'CTR')
        plt.tight_layout()
        plt.show()
        
    # Simulate actual clicks using binomial distribution
    clicks = np.random.binomial(views, success_rates)
    
    df = pd.DataFrame({
        'views': views,
        'true_ctr': success_rates,
        'clicks': clicks
    })
    
    print(f"Saving {num_rows} rows to {output_file}...")
    df.head(num_rows).to_csv(output_file, index=False)
    print("Data saved successfully.")

def run_chisq_goodness_of_fit(sales_file='DailySale.csv'):
    """
    Evaluates if daily sales data matches a Normal distribution by binning it,
    then running a Chi-Square goodness-of-fit test.
    """
    print("\n" + "=" * 60)
    print("CHI-SQUARE GOODNESS-OF-FIT TEST")
    print("=" * 60)
    try:
        df = pd.read_csv(sales_file)
    except FileNotFoundError:
        print(f"Error: {sales_file} not found.")
        return
        
    hist, edges = np.histogram(
        df['Demand'],
        bins=20,
        range=(0, max(df['Demand'])),
        density=False)

    cumulative = np.cumsum(hist)
    
    Observe = [0 if i == 0 else (cumulative[i] - cumulative[i-1])
                for i in range(len(cumulative)) ]
    
    average = np.mean(df['Demand'])
    std = np.std(df['Demand'])
    
    expCum = stats.norm(average, std).cdf(edges[:-1])
    
    Expected = [expCum[i]*len(df) if i == 0 else (expCum[i] - expCum[i-1])*len(df)
                for i in range(len(expCum))]
    
    # Scale Expected frequencies so that their sum matches the sum of observed frequencies.
    # This prevents SciPy ValueError due to sum mismatch (e.g. because Observe[0] is zeroed).
    Expected = np.array(Expected)
    if np.sum(Expected) > 0:
        Expected = Expected / np.sum(Expected) * np.sum(Observe)
        
    p_value = stats.chisquare(Observe, f_exp=Expected)
    print(f"Sales Data Mean: {average:.2f} | Std Dev: {std:.2f}")
    print(f"Goodness-of-Fit Chi-Square statistic: {p_value.statistic:.4f} | p-value: {p_value.pvalue:.4f}")
    
    # Extra static sample checks from original Chisquare.py
    ob = np.array([0,1,1,3,2,7,12,9,7,8,7,1,0,2,0,0,0])
    ex = np.array([0.15,0.34,0.90,2.01,3.79,6.08,8.28,9.57,9.39,7.82,5.53,3.32,1.69,0.73,0.27,0.08,0.02])
    
    if np.sum(ex) > 0:
        ex = ex / np.sum(ex) * np.sum(ob)
        
    p_value2 = stats.chisquare(ob, f_exp=ex)
    print(f"Static test 1 chisquare p-value: {p_value2.pvalue:.4f}")
    
    _, p1, _, _ = stats.chi2_contingency(np.array([ob, ex]))
    print(f"Static test 1 contingency p-value: {p1:.4f}")
    
    _, p2, _, _ = stats.chi2_contingency(np.array([Observe, Expected]))
    print(f"Sales data observed/expected contingency p-value: {p2:.4f}")

def run_chisq_test_independence(clicks_file='advertisement_clicks.csv', seed=1984):
    """
    Performs Chi-Square Test of Independence and a Bayesian A/B test on binary click data.
    """
    print("\n" + "=" * 60)
    print("CHI-SQUARE TEST OF INDEPENDENCE & BAYESIAN A/B TEST")
    print("=" * 60)
    try:
        df = pd.read_csv(clicks_file)
    except FileNotFoundError:
        print(f"Error: {clicks_file} not found.")
        return
        
    a = df[df['advertisement_id'] == 'A']['action']
    b = df[df['advertisement_id'] == 'B']['action']
    
    A_clk = a.sum()
    A_noclk = a.size - a.sum()
    B_clk = b.sum()
    B_noclk = b.size - b.sum()
    
    T = np.array([[A_clk, A_noclk], [B_clk, B_noclk]])
    print("Contingency Table (Clicks vs No Clicks):")
    print(T)
    
    # Custom Chi-square implementation from ex_chisq.py
    det = T[0,0]*T[1,1] - T[0,1]*T[1,0]
    c2 = float(det) / T[0].sum() * det / T[1].sum() * T.sum() / T[:,0].sum() / T[:,1].sum()
    p = 1 - stats.chi2.cdf(x=c2, df=1)
    print(f"Manual Chi-Square statistic: {c2:.4f} | p-value: {p:.4f}")
    
    # Scipy implementation
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(T, correction=False)
    print(f"SciPy Chi-Square statistic: {chi2_stat:.4f} | p-value: {chi2_p:.4f}")
    
    # Bayesian A/B test
    np.random.seed(seed)
    exposure_a = len(a)
    exposure_b = len(b)
    click_a = A_clk
    click_b = B_clk
    
    # Using Beta distribution with prior: alpha=1, beta=5
    post_a = np.random.beta(1 + click_a, 5 + exposure_a - click_a, 10000)
    post_b = np.random.beta(1 + click_b, 5 + exposure_b - click_b, 10000)
    
    prob_a_better = (post_a > post_b).mean()
    print(f"Bayesian Probability that Ad A is better than Ad B: {prob_a_better:.4f}")

def run_ttest(clicks_file='advertisement_clicks.csv'):
    """
    Performs Student's T-test and Welch's T-test on conversion action data.
    """
    print("\n" + "=" * 60)
    print("T-TEST (STUDENT'S, WELCH'S, AND MANUAL WELCH'S)")
    print("=" * 60)
    try:
        df = pd.read_csv(clicks_file)
    except FileNotFoundError:
        print(f"Error: {clicks_file} not found.")
        return
        
    a = df[df['advertisement_id'] == 'A']['action']
    b = df[df['advertisement_id'] == 'B']['action']
    
    print(f"Group A Mean: {a.mean():.4f}")
    print(f"Group B Mean: {b.mean():.4f}")
    
    # Student's T-test (Equal Variance)
    t_stat, t_p = stats.ttest_ind(a, b)
    print(f"Student's T-test: t-stat = {t_stat:.4f} | p-value = {t_p:.4f}")
    
    # Welch's T-test (Unequal Variance)
    t_welch, p_welch = stats.ttest_ind(a, b, equal_var=False)
    print(f"Welch's T-test:   t-stat = {t_welch:.4f} | p-value = {p_welch:.4f}")
    
    # Manual Welch's T-test implementation
    N1 = len(a)
    s1_sq = a.var()
    N2 = len(b)
    s2_sq = b.var()
    t_manual = (a.mean() - b.mean()) / np.sqrt(s1_sq / N1 + s2_sq / N2)
    
    nu1 = N1 - 1
    nu2 = N2 - 1
    dof_manual = (s1_sq / N1 + s2_sq / N2)**2 / ( (s1_sq**2) / (N1**2 * nu1) + (s2_sq**2) / (N2**2 * nu2) )
    p_manual = (1 - stats.t.cdf(np.abs(t_manual), df=dof_manual)) * 2
    print(f"Manual Welch's:   t-stat = {t_manual:.4f} | p-value = {p_manual:.4f} | dof = {dof_manual:.2f}")

def run_ztests(clicks_file='advertisement_clicks.csv'):
    """
    Runs various Z-tests including one-sample Z-test, two-sample Z-test for proportions,
    and a Z-test comparing Titanic fare means.
    """
    print("\n" + "=" * 60)
    print("Z-TESTS (ONE-SAMPLE, TWO-SAMPLE PROPORTIONS, TITANIC FARES)")
    print("=" * 60)
    
    # 1. One-Sample Z-Test
    print("1. One-Sample Z-Test:")
    np.random.seed(0)
    N = 100
    mu = 0.2
    sigma = 1
    x = np.random.randn(N) * sigma + mu
    mu0 = 0
    mu_hat = x.mean()
    sigma_hat = x.std(ddof=1)
    z = (mu_hat - mu0) / (sigma_hat / np.sqrt(N))
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    print(f"   Sample Mean: {mu_hat:.4f}")
    print(f"   Z-statistic: {z:.4f} | P-value: {p:.4f}")
    print("   Result: " + ("Significant difference from mu0=0" if p < 0.05 else "No significant difference"))
    
    # 2. Two-Sample Z-Test for Proportions
    print("\n2. Two-Sample Z-Test for Proportions (AB Testing clicks):")
    try:
        df = pd.read_csv(clicks_file)
        a = df[df['advertisement_id'] == 'A']['action']
        b = df[df['advertisement_id'] == 'B']['action']
        
        n1 = len(a)
        n2 = len(b)
        p1 = a.mean()
        p2 = b.mean()
        
        p_pooled = (a.sum() + b.sum()) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z_prop = (p1 - p2) / se
        p_prop = 2 * (1 - stats.norm.cdf(np.abs(z_prop)))
        print(f"   Ad A: n={n1}, conv_rate={p1:.4f}")
        print(f"   Ad B: n={n2}, conv_rate={p2:.4f}")
        print(f"   Z-statistic: {z_prop:.4f} | P-value: {p_prop:.4f}")
        print("   Result: " + ("Significant difference" if p_prop < 0.05 else "No significant difference"))
    except FileNotFoundError:
        print(f"   Error: {clicks_file} not found.")
        
    # 3. Titanic Fare Analysis
    print("\n3. Titanic Fare Analysis:")
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    try:
        print(f"   Loading Titanic data from web...")
        df_titanic = pd.read_csv(url)
        x1 = df_titanic[df_titanic['Survived'] == 1]['Fare'].dropna().to_numpy()
        x0 = df_titanic[df_titanic['Survived'] == 0]['Fare'].dropna().to_numpy()
        
        N1 = len(x1)
        N0 = len(x0)
        mu1 = x1.mean()
        mu0 = x0.mean()
        s1 = x1.var(ddof=1)
        s0 = x0.var(ddof=1)
        
        se_diff = np.sqrt(s1 / N1 + s0 / N0)
        z_titanic = (mu1 - mu0) / se_diff
        p_titanic = 2 * (1 - stats.norm.cdf(np.abs(z_titanic)))
        print(f"   Survivors (N={N1}): Mean Fare = {mu1:.2f}")
        print(f"   Non-Survivors (N={N0}): Mean Fare = {mu0:.2f}")
        print(f"   Z-statistic: {z_titanic:.4f} | P-value: {p_titanic:.4e}")
        print("   Result: " + ("Reject Null Hypothesis. Significant difference." if p_titanic < 0.05 else "Fail to reject Null Hypothesis."))
    except Exception as e:
        print(f"   Error loading/processing Titanic data: {e}")

def run_delta_method_test(sim_file='simulated_traffic.csv'):
    """
    Performs standard Z-test and Delta Method Z-test on ratio metric (clicks / views) of clustered data.
    """
    print("\n" + "=" * 60)
    print("DELTA METHOD RATIO VARIANCE TEST ON CLUSTERED TRAFFIC")
    print("=" * 60)
    try:
        df = pd.read_csv(sim_file)
    except FileNotFoundError:
        print(f"Error: {sim_file} not found. Please run --generate-data first.")
        return
        
    total_views = df['views'].sum()
    total_clicks = df['clicks'].sum()
    observed_ctr = total_clicks / total_views if total_views > 0 else 0
    expected_ctr = 0.02
    
    print("Standard (Naive) Ratio Test:")
    print(f"  Total Views:  {total_views:,}")
    print(f"  Total Clicks: {total_clicks:,}")
    print(f"  Observed CTR: {observed_ctr:.4f}")
    print(f"  Expected CTR: {expected_ctr:.4f}")
    
    # Delta Method
    print("\nDelta Method Test:")
    X = df['clicks']
    Y = df['views']
    N = len(df)
    mu_x = X.mean()
    mu_y = Y.mean()
    var_x = X.var()
    var_y = Y.var()
    cov_xy = np.cov(X, Y)[0, 1]
    
    # Delta Method formula for variance of ratio mu_x / mu_y
    var_ratio = (1/N) * (
        (var_x / (mu_y**2)) + 
        (mu_x**2 * var_y / (mu_y**4)) - 
        (2 * mu_x * cov_xy / (mu_y**3))
    )
    se_delta = np.sqrt(var_ratio)
    z_stat = (observed_ctr - expected_ctr) / se_delta
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    
    print(f"  Delta Method SE: {se_delta:.6f}")
    print(f"  Z-statistic:    {z_stat:.4f}")
    print(f"  P-value:        {p_value:.4f}")
    print("  Result: " + ("Reject Null Hypothesis. Significant difference." if p_value < 0.05 else "Fail to reject Null Hypothesis."))

def run_ks_comparisons():
    """
    Demonstrates the Kolmogorov-Smirnov test vs other tests.
    """
    print("\n" + "=" * 60)
    print("KOLMOGOROV-SMIRNOV (KS) TEST COMPARISONS")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Demonstration 1: KS vs Chi-Square Goodness-of-Fit
    print("DEMONSTRATION 1: KS TEST VS. CHI-SQUARE GOODNESS-OF-FIT")
    print("Testing 100 continuous points drawn from t-distribution (df=3, heavy tails) against Normal distribution.\n")
    data = stats.t.rvs(df=3, size=100)
    mu, sigma = np.mean(data), np.std(data)
    
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(mu, sigma))
    print(f"Kolmogorov-Smirnov Test (no binning):")
    print(f"  - p-value: {ks_p:.4f} (Rejects normality if < 0.05: {ks_p < 0.05})\n")
    
    print("Chi-Square Goodness-of-Fit (requires arbitrary binning):")
    for bins in [5, 10, 20]:
        counts, edges = np.histogram(data, bins=bins)
        cdf_vals = stats.norm.cdf(edges, loc=mu, scale=sigma)
        expected = (cdf_vals[1:] - cdf_vals[:-1])
        expected = expected / np.sum(expected) * np.sum(counts)
        chi_stat, chi_p = stats.chisquare(counts, f_exp=expected)
        print(f"  - Bins: {bins:2d} | p-value: {chi_p:.4f} (Rejects normality if < 0.05: {chi_p < 0.05})")
    
    # Demonstration 2: KS vs T-Test for A/B Testing
    print("\nDEMONSTRATION 2: TWO-SAMPLE KS-TEST VS. T-TEST FOR A/B TESTING")
    print("Comparing two groups with identical mean (10s) but different variances (SD 1s vs 4s).\n")
    group_A = np.random.normal(loc=10.0, scale=1.0, size=500)
    group_A = (group_A - np.mean(group_A)) + 10.0
    group_B = np.random.normal(loc=10.0, scale=4.0, size=500)
    group_B = (group_B - np.mean(group_B)) + 10.0
    
    t_stat, t_p = stats.ttest_ind(group_A, group_B)
    print("T-test (compares means only):")
    print(f"  - Group A Mean: {np.mean(group_A):.2f} | Group B Mean: {np.mean(group_B):.2f}")
    print(f"  - p-value: {t_p:.4f} (Rejects identical means if < 0.05: {t_p < 0.05})")
    
    ks_stat, ks_p = stats.ks_2samp(group_A, group_B)
    print("Two-Sample KS Test (compares overall distribution shape):")
    print(f"  - p-value: {ks_p:.4e} (Rejects identical distribution if < 0.05: {ks_p < 0.05})")
    
    # Demonstration 3: KS vs Chi-Square for categorical
    print("\nDEMONSTRATION 3: WHY KS TEST CANNOT REPLACE CHI-SQUARE FOR CATEGORICAL A/B CLICKS")
    print("Testing conversion rate differences (binary data).\n")
    ad_A = np.random.binomial(n=1, p=0.10, size=1000)
    ad_B = np.random.binomial(n=1, p=0.14, size=1000)
    clicks_A = np.sum(ad_A)
    clicks_B = np.sum(ad_B)
    contingency = np.array([
        [clicks_A, 1000 - clicks_A],
        [clicks_B, 1000 - clicks_B]
    ])
    
    chi2_stat, chi2_p, _, _ = stats.chi2_contingency(contingency)
    print("Chi-Square Test of Independence (categorical):")
    print(f"  - Ad A Clicks: {clicks_A} | Ad B Clicks: {clicks_B}")
    print(f"  - p-value: {chi2_p:.4f} (Significant if < 0.05: {chi2_p < 0.05})\n")
    
    ks_stat, ks_p = stats.ks_2samp(ad_A, ad_B)
    print("Two-Sample KS Test applied to binary data (mathematically invalid):")
    print(f"  - p-value: {ks_p:.4f}")
    print("    -> Warning: p-values are conservative and invalid due to ties in discrete data.")

def main():
    parser = argparse.ArgumentParser(description="AB Testing and Statistical Significance Demonstration Suite")
    parser.add_argument("--all", action="store_true", help="Run all statistical tests sequentially")
    parser.add_argument("--generate-data", action="store_true", help="Generate simulated_traffic.csv dataset")
    parser.add_argument("--plot-gen", action="store_true", help="Enable plotting during simulated data generation")
    parser.add_argument("--test", choices=["chisq", "goodness-of-fit", "ttest", "ztest", "delta-method", "ks-comparisons"],
                        help="Run a specific statistical test")
    
    args = parser.parse_args()
    
    if not any([args.all, args.generate_data, args.test]):
        parser.print_help()
        return

    if args.generate_data:
        generate_simulated_data(plot=args.plot_gen)
        
    if args.all:
        run_chisq_goodness_of_fit()
        run_chisq_test_independence()
        run_ttest()
        run_ztests()
        run_delta_method_test()
        run_ks_comparisons()
        
    elif args.test:
        if args.test == "chisq":
            run_chisq_test_independence()
        elif args.test == "goodness-of-fit":
            run_chisq_goodness_of_fit()
        elif args.test == "ttest":
            run_ttest()
        elif args.test == "ztest":
            run_ztests()
        elif args.test == "delta-method":
            run_delta_method_test()
        elif args.test == "ks-comparisons":
            run_ks_comparisons()

if __name__ == "__main__":
    main()
