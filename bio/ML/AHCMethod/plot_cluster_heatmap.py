import seaborn as sns
import matplotlib.pyplot as mplot
from pathlib import Path
import bio
from bio.ML import AHC
from loguru import logger

def plot_cluster_heatmap(model: AHC):
    """
    Generates and saves a heatmap showing the standardized mean of each feature per cluster.
    """
    # 1. Validation to ensure data exists
    if getattr(model, 'df', None) is None or 'CLUSTER' not in model.df.columns:
        raise ValueError("Cluster column not found. Please run cluster() first.")
    if not hasattr(model, 'pca_feature_names') or model.pca_feature_names is None:
        raise ValueError("PCA features not found. Please run cluster() first.")

    # 2. Extract ONLY the retained numerical features + the CLUSTER column
    features = list(model.pca_feature_names)
    df_eval = model.df[features + ['CLUSTER']].copy()

    # 3. Calculate the mean of each feature per cluster
    cluster_means = df_eval.groupby('CLUSTER').mean()

    # 4. Standardize the means across clusters (Column-wise Z-score)
    # This prevents massive numbers (like Molecular Weight) from hiding small numbers (like LogP)
    cluster_means_scaled = (cluster_means - cluster_means.mean()) / cluster_means.std()
    
    # Failsafe: If a feature has the exact same mean across all clusters, std() is 0, causing NaNs.
    cluster_means_scaled = cluster_means_scaled.fillna(0)

    # 5. Prepare the output directory
    save_dir = Path(model.options.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "cluster_feature_heatmap.png"

    # 6. Plot the Heatmap
    mplot.figure(figsize=(14, 10))  # Slightly taller to fit all feature names on the Y-axis
    
    # We transpose (.T) so Features are rows (Y-axis) and Clusters are columns (X-axis)
    sns.heatmap(
        cluster_means_scaled.T, 
        cmap="coolwarm", 
        center=0, 
        annot=False,  # Set to True if you want the actual numbers printed inside the colored boxes
        cbar_kws={'label': 'Z-Score (Deviation from Global Mean)'}
    )
    
    mplot.title("Cluster Profiles: Standardized Feature Means", fontsize=16, pad=15)
    mplot.xlabel("Cluster ID", fontsize=12)
    mplot.ylabel("Retained Features", fontsize=12)
    
    mplot.tight_layout()
    mplot.savefig(output_path, dpi=300)
    mplot.close()

    logger.info(f"Cluster heatmap saved to {output_path}. Look for deep red (high) or deep blue (low) squares to define your clusters.")


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
    plot_cluster_heatmap(ahc)
