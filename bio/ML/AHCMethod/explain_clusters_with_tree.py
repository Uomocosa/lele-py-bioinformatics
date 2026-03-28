import matplotlib.pyplot as mplot
from sklearn.tree import DecisionTreeClassifier, plot_tree
from pathlib import Path
import bio
from bio.ML import AHC
from loguru import logger

def explain_clusters_with_tree(model: AHC):
    """
    Trains a shallow decision tree to explain the AHC cluster assignments 
    using the original, interpretable features.
    """
    # 1. Ensure the model has been clustered
    if not hasattr(model, 'pca_feature_names') or model.pca_feature_names is None:
        raise ValueError("PCA features not found. Please run cluster() first.")
        
    # 2. Extract original features and cluster labels directly from the model
    df_features = model.df[model.pca_feature_names]
    
    # Safely grab the cluster column (handling potential casing differences)
    cluster_col = 'CLUSTER' if 'CLUSTER' in model.df.columns else 'Cluster'
    cluster_labels = model.df[cluster_col]
    
    # 3. Train the shallow Surrogate Tree
    # We use model.options.seed to ensure reproducibility
    tree = DecisionTreeClassifier(max_depth=3, random_state=model.options.seed)
    tree.fit(df_features, cluster_labels)
    
    # 4. Prepare the output directory
    save_dir = Path(model.options.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "cluster_decision_tree.png"
    
    # 5. Plot and save
    mplot.figure(figsize=(16, 8))
    plot_tree(
        tree, 
        feature_names=df_features.columns.tolist(), 
        class_names=[f"C{i}" for i in sorted(cluster_labels.unique())],
        filled=True, 
        rounded=True, 
        fontsize=10
    )
    mplot.title("Decision Rules for Cluster Assignments")
    mplot.tight_layout()
    mplot.savefig(output_path, dpi=300)
    mplot.close()
    
    logger.info(f"Tree plot saved to {output_path}. Read it to see the exact thresholds separating your clusters.")
    
    return tree



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
    explain_clusters_with_tree(ahc)
