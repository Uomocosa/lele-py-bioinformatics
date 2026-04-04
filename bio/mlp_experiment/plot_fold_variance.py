import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import bio
from loguru import logger


ExperimentConfig = bio.mlp_experiment.Config
def plot_fold_variance(exp_config: ExperimentConfig, save_dir: Path, log_file: Path):
    """
    Generates a bar chart comparing the Coefficient of Determination (R²) across all k-folds.
    
    This graph assesses the stability and reliability of the model:
    - Large discrepancies between bars suggest high variance to the data split.
    - Uniform bars suggest a stable, generalizable model.
    """
    if not log_file.exists():
        return

    df_metrics = pd.read_json(log_file, lines=True)
    if not df_metrics.empty and "fold" in df_metrics.columns and "r2" in df_metrics.columns:
        plt.figure(figsize=(8, 5))
        
        # Bar plot of R² per fold
        ax = sns.barplot(data=df_metrics, x="fold", y="r2", color="cornflowerblue")
        
        valid_r2 = df_metrics["r2"].replace([np.inf, -np.inf], np.nan).dropna()
        mean_r2 = valid_r2.mean() # Add a horizontal line representing the mean R²
        
        # Safely get the minimum, fallback to 0.0 if everything was NaN
        if valid_r2.empty: min_val = 0.0
        else: min_val = min(df_metrics["r2"].min(), mean_r2, 0)
        if min_val < 0: min_val -= 0.1 # Adds a little padding at the bottom
            
        
        plt.axhline(mean_r2, color="red", linestyle="--", label=f"Mean $R^2$ ({mean_r2:.3f})")
        plt.title(f"{exp_config.name} - $R^2$ Variance Across Folds")
        plt.xlabel("Fold")
        plt.ylabel("$R^2$ Score")
        plt.ylim(min_val, 1.0) # Caps top at 1.0
        plt.legend()
        
        img_path = save_dir / f"{exp_config.name}" / "plot_fold_variance.png"
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.close()
        logger.info(f"Saved fold variance plot to: {img_path}")


import pytest
@pytest.mark.todo
def test_():
    pass
