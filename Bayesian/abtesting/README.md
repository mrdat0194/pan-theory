# AB Testing and Statistical Significance

This folder contains exercises and examples of different statistical tests used in AB testing and data analysis.

## Summary of Statistical Tests

| Test | Best Used For... | Type of Data | Key File |
| :--- | :--- | :--- | :--- |
| **Chi-Square** | Seeing if categories are independent (e.g., did Ad A cause more clicks than Ad B?). | Counts / Categories (Yes/No, Ad A/B) | `ex_chisq.py` |
| **T-Test** | Comparing averages in small groups or when population variance is unknown. | Numbers (Prices, Time spent, Weight) | `ex_ttest.py` |
| **Z-Test** | Comparing averages or proportions in large groups (N > 30). | Numbers or Percentages | `ex_ztest.py` |
| **Delta Method** | Estimating variance for ratios (like clicks/views) on clustered data. | Ratios (Clustered Data) | `ex_simulated_test.py` |

---

## File Descriptions

### Statistical Examples
*   **`ex_chisq.py`**: Implementation of the Chi-Square test for independence (used for AB testing).
*   **`Chisquare.py`**: Implementation of the Chi-Square goodness-of-fit test (used to see if data matches a distribution like the Normal distribution).
*   **`ex_ttest.py`**: Implementation of Student's T-test and Welch's T-test for comparing means.
*   **`ex_ztest.py`**: Implementation of Z-tests for means and proportions, including a real-world example using Titanic data.
*   **`ex_simulated_test.py`**: Comparison of a naive Z-test vs. the Delta Method on simulated clumpy data (where standard Z-tests fail).
*   **`verify_ground_truth.py`**: Verification of simulated data's ground truth CTR using T-tests and Bootstrapping.

### Utilities and Data
*   **`generate_view.py`**: A simulation script that generates realistic distributions for web traffic ("Views") and conversion rates ("CTR"). It uses Log-Normal and Beta distributions to model how real-world data typically behaves.
*   **`advertisement_clicks.csv`**: Sample dataset containing click data for two different advertisements (A and B).
*   **`DailySale.csv`**: Sample dataset used in `Chisquare.py` for sales distribution analysis.

---

## Chi-Square Comparison: `ex_chisq.py` vs `Chisquare.py`

| Feature | `ex_chisq.py` | `Chisquare.py` |
| :--- | :--- | :--- |
| **Primary Goal** | Compare two groups (A vs B). | Compare data to a theoretical shape. |
| **Dataset** | `advertisement_clicks.csv` | `DailySale.csv` |
| **Test Type** | Chi-Square Test of Independence. | Chi-Square Goodness-of-Fit. |
| **Question Asked** | "Are the ads related to clicks?" | "Does my sales data look like a bell curve?" |
