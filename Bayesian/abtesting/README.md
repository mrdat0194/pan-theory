# AB Testing and Statistical Significance

This folder contains exercises and examples of different statistical tests used in AB testing and data analysis.

## Summary of Statistical Tests

| Test | Best Used For... | Type of Data | Key File |
| :--- | :--- | :--- | :--- |
| **Chi-Square** | Seeing if categories are independent (e.g., did Ad A cause more clicks than Ad B?). | Counts / Categories (Yes/No, Ad A/B) | [ex_chisq.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/ex_chisq.py) |
| **T-Test** | Comparing averages in small groups or when population variance is unknown. | Numbers (Prices, Time spent, Weight) | [ex_ttest.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/ex_ttest.py) |
| **Z-Test** | Comparing averages or proportions in large groups (N > 30). | Numbers or Percentages | [ex_ztest.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/ex_ztest.py) |
| **Delta Method** | Estimating variance for ratios (like clicks/views) on clustered data. | Ratios (Clustered Data) | [ex_simulated_test.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/ex_simulated_test.py) |
| **Kolmogorov-Smirnov (KS)** | Checking if continuous samples match a specific distribution, or if two groups share the same distribution. | Continuous Numbers (e.g., times, scores) | [compare_ks_tests.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/compare_ks_tests.py) |

---

## File Descriptions

### Statistical Examples
*   **[ex_chisq.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/ex_chisq.py)**: Implementation of the Chi-Square test for independence (used for AB testing) and a Bayesian A/B test approximation using Beta posteriors on binary clicks data.
*   **[Chisquare.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/Chisquare.py)**: Implementation of the Chi-Square goodness-of-fit test (used to see if data matches a distribution like the Normal distribution) on sales data.
*   **[ex_ttest.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/ex_ttest.py)**: Implementation of Student's T-test and Welch's T-test for comparing means.
*   **[ex_ztest.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/ex_ztest.py)**: Implementation of Z-tests for means and proportions, including a real-world example using Titanic data.
*   **[ex_simulated_test.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/ex_simulated_test.py)**: Runs statistical tests (standard Z-test vs. Delta Method) on simulated `simulated_traffic.csv` data to demonstrate correct ratio metric variance estimation.
*   **[compare_ks_tests.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/compare_ks_tests.py)**: Comparison demonstrations of the Kolmogorov-Smirnov (KS) test vs. T-test, Chi-Square Goodness-of-Fit, and Chi-Square Independence.

### Utilities and Data
*   **[generate_view.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/generate_view.py)**: A simulation script that generates realistic web traffic ("Views") and conversion rates ("CTR") using Log-Normal and Beta distributions, saving the results to `simulated_traffic.csv` to be used by other scripts.
*   **`advertisement_clicks.csv`**: Sample dataset containing click data for two different advertisements (A and B).
*   **`DailySale.csv`**: Sample dataset used in `Chisquare.py` for sales distribution analysis.
*   **`simulated_traffic.csv`**: Generated dataset containing views, true CTR, and clicks for simulated users.

---

## Chi-Square Comparison: `ex_chisq.py` vs `Chisquare.py`

| Feature | `ex_chisq.py` | `Chisquare.py` |
| :--- | :--- | :--- |
| **Primary Goal** | Compare two groups (A vs B). | Compare data to a theoretical shape. |
| **Dataset** | `advertisement_clicks.csv` | `DailySale.csv` |
| **Test Type** | Chi-Square Test of Independence. | Chi-Square Goodness-of-Fit. |
| **Question Asked** | "Are the ads related to clicks?" | "Does my sales data look like a bell curve?" |

---

## Kolmogorov-Smirnov (KS) Test Comparisons

The KS test is a non-parametric test used for continuous data. It is demonstrated in [compare_ks_tests.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/compare_ks_tests.py) through three comparisons:

### 1. KS vs. Chi-Square Goodness-of-Fit (Testing one distribution against a theoretical shape)
*   **Chi-Square (as in [Chisquare.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/Chisquare.py))**: Requires binning data into arbitrary intervals, which discards information and makes the result sensitive to the number/width of bins. Requires at least 5 expected items per bin.
*   **KS Test**: Compares the empirical cumulative distribution function (CDF) directly to the theoretical CDF. It is exact, requires no binning, and is strictly superior for continuous distributions.

### 2. Two-Sample KS Test vs. T-Test (Comparing two continuous groups in A/B testing)
*   **T-Test (as in [ex_ttest.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/ex_ttest.py))**: Compares group *means* only. If Group A and Group B have the same mean but totally different variance (e.g., Group B has much higher variance), the T-test concludes they are identical.
*   **KS Test**: Compares the overall distribution shape/variance. It will easily detect that the groups have different behaviors.
*   *Note*: Use the T-test if your business goal is strictly to increase the average of a metric; use the KS test to detect *any* change in user behavior.

### 3. KS Test vs. Chi-Square Test of Independence (For categorical/binary conversion metrics)
*   **Chi-Square / Z-Test (as in [ex_chisq.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/ex_chisq.py) / [ex_ztest.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/Bayesian/abtesting/ex_ztest.py))**: Designed for categorical (success/failure) counts.
*   **KS Test**: Invalid for binary data. Since binary data (0 and 1) has massive ties, the empirical CDFs are step functions. Applying the KS test to such data results in mathematically invalid, highly conservative (unreliable) p-values.

