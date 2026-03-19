import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
from experiment.__call__ import Experiment
from bio.__global__ import PSMILES_DICT, SMILES_DICT


def plot_learning_curves(experiment: Experiment, save_dir: Path, log_file: Path):
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
        
        plt.title(f"{experiment.name} - Learning Curves (Mean across K-Folds)")
        plt.xlabel("Epoch")
        plt.ylabel(f"Loss ({experiment.model.criterion.upper()})")
        plt.legend()
        
        img_path = save_dir / f"{experiment.name}" / "plot_learning_curves.png"
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.close()
        logger.info(f"Saved learning curves plot to: {img_path}")




def test_():
    dummy_experiment = Experiment(name="experiment_0")
    save_dir = Path(__file__).parent
    exp_0_dir = save_dir / "experiment_0"
    if not exp_0_dir.exists():
        logger.info("Experiment 0 directory not found. Running __call__.test_().")
        import experiment.__call__.test_ as main_test
        main_test()
    log_file = exp_0_dir / "traing_epochs.jsonl"
    assert log_file.exists(), f"Log file {log_file} not found. Run the experiment first."
    plot_learning_curves(dummy_experiment, save_dir, log_file)
    output_img = exp_0_dir / "plot_learning_curves.png"
    assert output_img.exists(), "Fold variance plot was not created."
    assert output_img.stat().st_size > 0, "Fold variance plot is an empty file."
