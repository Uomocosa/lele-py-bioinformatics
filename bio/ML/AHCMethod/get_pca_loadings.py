import pandas as pd
from sklearn.decomposition import PCA
import bio
from bio.ML import AHC
from loguru import logger

def get_pca_loadings(model: AHC):
    """
    Extracts and prints the top features driving Principal Components 1 and 2.
    """
    if not hasattr(model, 'pca_object') or not hasattr(model, 'pca_feature_names'):
        raise ValueError("PCA object or feature names not found. Run cluster() first.")
        
    pca = model.pca_object
    features = model.pca_feature_names
    
    # Grab 'n' dynamically from your options dataclass
    n = model.options.pca_components
    
    # pca.components_ contains the eigenvectors (loadings). 
    # Shape is (n_components, n_features)
    loadings_df = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(pca.n_components)], 
        index=features
    )
    loadings_df.to_csv(model.options.save_dir / "pca_loadings_full.csv")
    
    # Find the top 'n' features by absolute magnitude for PC1
    pc1_top_features = loadings_df['PC1'].abs().sort_values(ascending=False).head(n).index
    pc1_loadings = loadings_df.loc[pc1_top_features, 'PC1']
    pc1_loadings.to_csv(model.options.save_dir / "pca_top_drivers_PC1.csv")
    
    # Find the top 'n' features by absolute magnitude for PC2
    pc2_top_features = loadings_df['PC2'].abs().sort_values(ascending=False).head(n).index
    pc2_loadings = loadings_df.loc[pc2_top_features, 'PC2']
    pc2_loadings.to_csv(model.options.save_dir / "pca_top_drivers_PC2.csv")
    
    logger.info(f"--- Top {n} Features Driving PC1 (X-Axis) ---")
    logger.info(pc1_loadings.to_string())
    logger.info(f"\n* Note: High absolute values push points left/right along the x-axis.")
    
    logger.info(f"\n--- Top {n} Features Driving PC2 (Y-Axis) ---")
    logger.info(pc2_loadings.to_string())
    logger.info(f"\n* Note: High absolute values push points up/down along the y-axis.\n")
    logger.info(f"Saved PCA loadings CSVs to {model.options.save_dir}")
    
    return loadings_df


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
    df = get_pca_loadings(ahc)
    print(df)
