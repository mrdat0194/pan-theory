# Latent Profile Analysis (LPA) Python Pipeline

This directory contains a complete, 100% Python-only pipeline for executing Latent Profile Analysis (LPA), performing mean-difference testing (Welch ANOVA + Games-Howell post-hoc), running demographic associations (Chi-Square), and exporting publication-ready APA 7th edition tables and plots.

The pipeline uses the **Titanic dataset** (`adHoc/titanic.csv`) as sample data, focusing on continuous indicators (`Age`, `Fare`, `SibSp`, `Parch`) and categorical demographics (`Sex`, `Pclass`, `Embarked`).

---

## Directory Structure

```
lpa_analysis/
├── config.py              # Configuration file (set column names and chosen K here)
├── main.py                # Pipeline orchestrator
├── requirements.txt       # Python package dependencies
├── step1_prepare_data.py  # Cleans and standardises (z-score) the input dataset
├── step2_run_lpa.py       # Runs Gaussian Mixture Models (K=1 to 6) and assigns profiles
├── step3_test_anova.py    # Welch ANOVA + Games-Howell pairwise tests per indicator
├── step4_test_chi_square.py # Chi-Square + Cramér's V tests per demographic variable
├── step5_plot_profiles.py # Generates a line plot of indicator profiles with 95% CIs
├── step6_apa_tables.py    # Builds a Word document containing APA 7th-style tables
└── outputs/               # Directory containing all generated reports, plots, and tables
```

---

## Installation & Setup

1. **Install Dependencies**:
   Install the required Python packages from the pipeline directory:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Dataset**:
   Ensure that `adHoc/titanic.csv` exists in the parent directory.

---

## Workflow

### 1. Step 1: Run Model Selection (K = 1 to 6)
Run the pipeline once to generate fit statistics across multiple profiles:
```bash
python -X utf8 main.py
```

### 2. Step 2: Determine Best Number of Profiles
Open the generated fit statistics table:
👉 [outputs/lpa_fit_stats.csv](file:///C:/Users/mrdat/PycharmProjects/pan-theory/lpa_analysis/outputs/lpa_fit_stats.csv)

Review the following fit criteria with your co-author:
*   **BIC / aBIC / AIC**: Lower values indicate better fit.
*   **Entropy**: Values closer to `1.0` indicate clearer class separation.
*   **LMR-LRT p-value**: Tells you if a model with $K$ profiles is a statistically significant improvement over a model with $K-1$ profiles.

### 3. Step 3: Choose K and Re-run
1. Open [config.py](file:///C:/Users/mrdat/PycharmProjects/pan-theory/lpa_analysis/config.py).
2. Set `N_PROFILES` to your chosen number of profiles (e.g., `3`).
3. Re-run the script to finalize the statistics, plots, and Word report:
   ```bash
   python -X utf8 main.py
   ```

---

## Outputs Generated

All final results are saved to the `outputs/` folder:

*   **`lpa_fit_stats.csv`**: Contains AIC, BIC, Adjusted BIC, Entropy, and LMR-LRT p-values.
*   **`lpa_profiles.csv`**: Your dataset appended with assigned profiles and posterior probabilities.
*   **`anova_results.csv` & `anova_posthoc.csv`**: Welch's F, eta-squared effect size, and Games-Howell pairwise differences.
*   **`chi_square_results.csv`**: Demographic association test results with Cramér's V.
*   **`profiles_lineplot.png`**: A high-resolution (300 DPI) line plot displaying profile indicator means with 95% confidence intervals.
*   **`apa_tables.docx`**: A formatted Word document featuring 4 APA 7th edition tables ready for publication.
