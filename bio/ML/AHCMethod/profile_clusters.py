import pandas as pd
from pathlib import Path
import bio
from bio.ML import AHC
from loguru import logger

def profile_clusters(model: AHC, top_n: int = 10):
    """
    Identifies the top features that define the differences between clusters
    by ranking them by their scaled variance, saving the results to disk.
    """
    # 1. Validation
    if getattr(model, 'df', None) is None or 'CLUSTER' not in model.df.columns:
        logger.error("Cluster column not found. Please run cluster() first.")
        return
    if not hasattr(model, 'pca_feature_names') or model.pca_feature_names is None:
        logger.error("PCA features not found. Please run cluster() first.")
        return

    # 2. Extract ONLY the retained numerical features + the CLUSTER column
    features = list(model.pca_feature_names)
    df_eval = model.df[features + ['CLUSTER']].copy()

    # 3. Calculate the raw mean of each feature per cluster
    cluster_means = df_eval.groupby('CLUSTER').mean()

    # 4. Standardize the means before calculating variance
    cluster_means_scaled = (cluster_means - cluster_means.mean()) / cluster_means.std()
    cluster_means_scaled = cluster_means_scaled.fillna(0)

    # 5. Find the features that vary the most across the clusters
    variance_across_clusters = cluster_means_scaled.var().sort_values(ascending=False)
    top_features = variance_across_clusters.head(top_n).index

    # Create the DataFrame to export
    report_df = cluster_means[top_features].T.round(3)

    # 6. Prepare output path
    save_dir = Path(model.options.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / f"cluster_profile_top_{top_n}.txt"

    # 7. Format and write the text file (Human readable)
    report_str = (
        f"{'='*70}\n"
        f" TOP {top_n} DEFINING FEATURES FOR CLUSTERS (Ranked by Scaled Variance)\n"
        f"{'='*70}\n\n"
        f"{report_df.to_string()}\n\n"
        f"{'='*70}\n"
    )
    
    with open(txt_path, "w") as f: f.write(report_str)
    logger.info(f"Saved cluster profiles (TXT) to {save_dir}")
    return report_df



def test_():
    import pandas as pd
    import bio
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
    profile_clusters(ahc)
