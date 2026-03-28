import numpy as np
import matplotlib.pyplot as mplot
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
import bio
from bio.ML import AHC
from loguru import logger

def plot_dendogram(model: AHC):
    """
    Generates and saves a dendrogram using the PCA-transformed data from the AHC model.
    """
    # 1. Ensure the required data exists
    if not hasattr(model, 'X_pca'):
        raise ValueError("X_pca not found in model. Add `model.X_pca = X_pca` to your cluster() function.")
        
    # 2. Extract data and dynamic settings
    X_pca = model.X_pca
    method = model.options.linkage  # Dynamically use 'ward', 'average', etc.
    
    # 3. Calculate linkage using SciPy
    Z = linkage(X_pca, method=method)
    
    # 4. Prepare output path
    save_dir = Path(model.options.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "ahc_dendrogram.png"
    
    # 5. Plot the Dendrogram
    mplot.figure(figsize=(14, 7))
    dendrogram(
        Z, 
        truncate_mode='level', 
        p=5,  # Show only the top 5 levels of the tree to prevent a messy plot
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True
    )
    
    mplot.title(f'Hierarchical Clustering Dendrogram ({method.capitalize()} Linkage)')
    mplot.xlabel('Number of points in node (or index of point if no parenthesis)')
    mplot.ylabel('Distance (Variance merged)')
    mplot.tight_layout()
    mplot.savefig(output_path, dpi=300)
    mplot.close()
    
    logger.info(f"Dendrogram saved to {output_path}. Look for the longest vertical branches to find the natural number of clusters.")
    
    return Z



def test_():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from bio.__global__ import PDCC_CSV
    from bio.Dataset import PDCCMethod
    bio.setup_loguru()
    model_options = AHC.Options(
        n_clusters=5, 
        pca_components=10, 
        scaler=StandardScaler(),
    )
    featurizer_options = PDCCMethod.featurize.Options(
        capping_atoms = ['H'],
        molecule_features_to_calculate = ['logp', 'logd'],
        polymer_features_to_calculate = [],
    )
    df = pd.read_csv(PDCC_CSV)
    df = df.head(10)
    df = PDCCMethod.increment_dataset(df)
    df = PDCCMethod.convert_names_to_smiles(df)
    df = PDCCMethod.featurize(df, options=featurizer_options)
    ahc = AHC.cluster(df, model_options)
    plot_dendogram(ahc)
