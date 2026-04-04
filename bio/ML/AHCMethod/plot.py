import seaborn as sns
import matplotlib.pyplot as mplot
from pathlib import Path
from sklearn.decomposition import PCA
import bio
from bio.ML import AHC
from loguru import logger

def plot(model: AHC):
    """
    Plots the first two Principal Components and colors by CLUSTER using Seaborn.
    """
    df = model.df
    # Set a clean seaborn theme
    sns.set_theme(style="whitegrid")
    mplot.figure(figsize=(10, 8))
    
    # We convert the CLUSTER IDs to strings so Seaborn treats them as distinct 
    # categories rather than a continuous numerical gradient.
    df['Cluster_Label'] = 'Cluster ' + df['CLUSTER'].astype(str)
    
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
    save_path = model.options.save_dir / "cluster_plot.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mplot.savefig(save_path, dpi=300)
    mplot.close()
    print(f"Cluster plot saved to: {save_path}")


def test_():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from bio.Dataset import PDCCMethod
    from bio.__global__ import PDCC_CSV
    bio.setup_loguru()
    model_options = AHC.Options(
        n_clusters=5, 
        pca_components=10, 
        scaler=StandardScaler(),
    )
    df = pd.read_csv(PDCC_CSV)
    df = df.head(10)
    df = PDCCMethod.increment_dataset(df)
    df = PDCCMethod.convert_names_to_smiles(df)
    df = PDCCMethod.featurize(df)
    ahc = AHC.cluster(df, model_options)
    plot(ahc)
