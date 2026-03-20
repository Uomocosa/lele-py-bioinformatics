import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import bio
from bio.__global__ import PSMILES_DICT, SMILES_DICT
from loguru import logger

ExperimentConfig = bio.mlp_experiment.Config
def plot_learning_curves(exp_config: ExperimentConfig, save_dir: Path, log_file: Path):
    """
    Generates learning curves by plotting Training and Validation loss against the number of epochs. 
    
    This graph tracks the model's progress over time:
    - It serves as a diagnostic tool for overfitting and underfitting.
    - Since k-fold cross-validation is used, the lines represent the mean loss across all folds.
    """
    if not log_file.exists(): return

    df_epochs = pd.read_json(log_file, lines=True)
    if not df_epochs.empty:
        plt.figure(figsize=(10, 6))
        
        # Seaborn automatically plots the mean line and a shaded confidence interval across all folds
        sns.lineplot(data=df_epochs, x="epoch", y="train_loss", label="Train Loss", linewidth=2)
        sns.lineplot(data=df_epochs, x="epoch", y="val_loss", label="Validation Loss", linewidth=2)
        
        criterion_name = exp_config.model_config.criterion_fn
        plt.title(f"{exp_config.name} - Learning Curves (Mean across K-Folds)")
        plt.xlabel("Epoch")
        plt.ylabel(f"Loss ({criterion_name.upper()})")
        plt.legend()
        
        img_path = save_dir / f"{exp_config.name}" / "plot_learning_curves.png"
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.close()
        logger.info(f"Saved learning curves plot to: {img_path}")



import pytest
@pytest.mark.todo
def test_():
    pass
