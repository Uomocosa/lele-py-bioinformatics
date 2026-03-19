import pandas as pd
from sklearn.decomposition import PCA
from bio.ML import AHC

def get_pca_loadings(model: AHC, n: int = 5):
    """
    Extracts and prints the top features driving Principal Components 1 and 2.
    """
    if not hasattr(model, 'pca_object') or not hasattr(model, 'pca_feature_names'):
        raise ValueError("PCA object or feature names not found. Run cluster() first.")
        
    pca = model.pca_object
    features = model.pca_feature_names
    
    # pca.components_ contains the eigenvectors (loadings). 
    # Shape is (n_components, n_features)
    loadings_df = pd.DataFrame(
        pca.components_.T, 
        columns=[f'PC{i+1}' for i in range(pca.n_components)], 
        index=features
    )
    
    # Find the top features by absolute magnitude for PC1
    pc1_top_features = loadings_df['PC1'].abs().sort_values(ascending=False).head(n).index
    pc1_loadings = loadings_df.loc[pc1_top_features, 'PC1']
    
    # Find the top features by absolute magnitude for PC2
    pc2_top_features = loadings_df['PC2'].abs().sort_values(ascending=False).head(n).index
    pc2_loadings = loadings_df.loc[pc2_top_features, 'PC2']
    
    print(f"--- Top {n} Features Driving PC1 (X-Axis) ---")
    print(pc1_loadings.to_string())
    print(f"\n* Note: High absolute values push points left/right along the x-axis.")
    
    print(f"\n--- Top {n} Features Driving PC2 (Y-Axis) ---")
    print(pc2_loadings.to_string())
    print(f"\n* Note: High absolute values push points up/down along the y-axis.")
    
    return loadings_df
