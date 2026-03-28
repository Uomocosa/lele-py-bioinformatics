import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from bio.__global__ import REPO_DIR, RESULTS_DIR
from loguru import logger

def test_smiles():
    dir = REPO_DIR / 'SMILES_checkpoints' / '2026_02_07_110058_333737'
    loss_csv = dir / 'loss_data.csv'
    plot_pee_smiles_generator_training_loss(loss_csv)
    
def test_psmiles():
    loss_csv = RESULTS_DIR / 'pee_smiles_generator' / 'loss_data.csv'
    plot_pee_smiles_generator_training_loss(loss_csv)
    
    
    
def plot_pee_smiles_generator_training_loss(loss_csv: Path):
    df = pd.read_csv(loss_csv)
    
    # Calculate a rolling average for a smoother trend line
    df['loss_smoothed'] = df['loss'].rolling(window=100, min_periods=1).mean()
    
    # Create the plot
    plt.plot(df['iteration'], df['loss'], alpha=0.3, color='royalblue', label='Raw Loss')
    plt.plot(df['iteration'], df['loss_smoothed'], color='darkblue', linewidth=2, label='Smoothed Loss (Window=100)')
    
    # Formatting the plot
    plt.title('Training Loss Over Iterations', fontsize=14)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(loss_csv.parent / 'training_loss_plot.png', bbox_inches='tight')
    plt.close()
    logger.info(f"Training loss plot saved to {loss_csv.parent / 'training_loss_plot.png'}")
