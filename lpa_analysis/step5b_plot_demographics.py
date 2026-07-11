"""
step5b_plot_demographics.py
───────────────────────────
Generate a publication-quality stacked bar chart displaying the percentage 
distribution of demographic variables (Sex, Pclass, Embarked) across the 
latent profiles.

Saves:
  - outputs/demographics_plot.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import OUTPUT_DIR, DEMOGRAPHIC_COLS

# Colors matching the profile color palette
PALETTE = ["#2E86AB", "#E84855", "#3BB273", "#F18F01"]

def plot_demographics(df: pd.DataFrame) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 3 subplots: Sex, Pclass, Embarked
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=False)
    
    # Map raw value labels for clearer presentation
    pclass_map = {1: "1st Class", 2: "2nd Class", 3: "3rd Class"}
    df_plot = df.copy()
    df_plot["Pclass"] = df_plot["Pclass"].map(pclass_map)
    df_plot["Sex"] = df_plot["Sex"].str.title()
    df_plot["Embarked"] = df_plot["Embarked"].map({"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"})
    
    for i, col in enumerate(DEMOGRAPHIC_COLS):
        ax = axes[i]
        
        # Calculate percentage distribution of demographic categories WITHIN each Profile
        # Pivot table: Index = Profile, Columns = Demographic Category, Values = Count
        ct = pd.crosstab(df_plot["Profile"], df_plot[col], normalize="index") * 100
        
        # Plot stacked bar chart
        ct.plot(kind="bar", stacked=True, ax=ax, color=sns.color_palette("muted", n_colors=len(ct.columns)))
        
        ax.set_title(f"Distribution of {col}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Latent Profile", fontsize=10)
        if i == 0:
            ax.set_ylabel("Percentage (%)", fontsize=10)
        else:
            ax.set_ylabel("")
            
        ax.set_xticklabels([f"Profile {x}" for x in ct.index], rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(title=col, framealpha=0.9, edgecolor="#cccccc")
        
        # Add labels to stacked bars
        for container in ax.containers:
            labels = [f"{val:.1f}%" if val > 5 else "" for val in container.datavalues]
            ax.bar_label(container, labels=labels, label_type="center", fontsize=9, fontweight="bold", color="white")

    plt.suptitle("Demographic Composition of Latent Profiles", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "demographics_plot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Step 5b] Demographic plot saved → {out_path}")

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(OUTPUT_DIR, "lpa_profiles.csv"))
    plot_demographics(df)
