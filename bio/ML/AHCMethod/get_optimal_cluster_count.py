import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import bio
from bio.ML import AHC
from loguru import logger

def get_optimal_cluster_count(model: AHC) -> int:
    """
    Automatically calculates the optimal number of clusters using the Silhouette Score.
    Returns the optimal integer k without generating plots or files.
    """
    if not hasattr(model, 'X_pca'):
        raise ValueError("X_pca not found. Add `model.X_pca = X_pca` to your cluster() function.")

    X = model.X_pca
    linkage_method = model.options.linkage
    
    # Failsafe: You can't have more clusters than data points (minus 1)
    max_k = len(X) - 1
    
    k_values = range(2, max_k + 1)
    scores = []

    # 1. Test every possible cluster count
    for k in k_values:
        temp_model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        labels = temp_model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores.append(score)

    # 2. Identify the winner
    best_index = np.argmax(scores)
    
    # Ensure it returns a standard Python int (np.argmax returns a numpy type)
    best_k = int(k_values[best_index]) 
    best_score = scores[best_index]

    logger.info(f"Evaluated k=2 to {max_k} (number of datapoints). Optimal clusters found: {best_k} (Silhouette Score: {best_score:.3f})")

    return best_k



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
    n = get_optimal_cluster_count(ahc)
    print(n)
