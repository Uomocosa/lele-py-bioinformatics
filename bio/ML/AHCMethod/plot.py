import seaborn as sns
import matplotlib.pyplot as mplot
from pathlib import Path
from sklearn.decomposition import PCA
from bio.ML import AHC

def plot(model: AHC, output_path: Path):
    """
    Plots the first two Principal Components and colors by Cluster using Seaborn.
    """
    df = model.df
    # Set a clean seaborn theme
    sns.set_theme(style="whitegrid")
    mplot.figure(figsize=(10, 8))
    
    # We convert the Cluster IDs to strings so Seaborn treats them as distinct 
    # categories rather than a continuous numerical gradient.
    df['Cluster_Label'] = 'Cluster ' + df['Cluster'].astype(str)
    
    # Seaborn does all the heavy lifting here
    sns.scatterplot(
        data=df,
        x='PCA_1',
        y='PCA_2',
        hue='Cluster_Label',
        palette='tab10',  # A great color palette for categorical data
        s=80,             # Marker size
        alpha=0.8,        # Slight transparency
        edgecolor='k'     # Black borders around the dots
    )
    
    mplot.title('2D PCA Projection of Drug-Polymer Clusters', fontsize=14, pad=15)
    mplot.xlabel('Principal Component 1', fontsize=12)
    mplot.ylabel('Principal Component 2', fontsize=12)
    
    # Move the legend outside the plot so it doesn't cover your data
    mplot.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    mplot.tight_layout()
    mplot.savefig(output_path, dpi=300)
    mplot.close()
    print(f"Cluster plot saved to: {output_path}")


import pytest
@pytest.mark.todo
def test_():
    pass
