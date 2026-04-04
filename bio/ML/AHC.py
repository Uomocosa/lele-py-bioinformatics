import sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg') # tryin to solve a pytest error caused by this file
import matplotlib.pyplot as mplot
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Any
import bio
from bio.Dataset import PDCCMethod
from bio.ML import AHCMethod
from loguru import logger
from bio.ML.__global__ import HELPER_DIR

@dataclass
class Options:
    save_dir: Path = HELPER_DIR / "AHC"
    n_clusters: int = 5
    metric: str = 'euclidean'
    linkage: str = 'ward'
    variance_threshold: float = 0.0
    pca_components: int = 15
    scaler: Optional[StandardScaler] = None
    seed: int = 42


@dataclass
class AHC:
    """
    Agglomerative Hierarchical Clustering (AHC) using sklearn's AgglomerativeClustering.
    """
    df: pd.DataFrame
    options: Options = field(default_factory=lambda: Options())
    pca_feature_names: Optional[list[str]] = None
    pca_object: Optional[PCA] = None
    X_pca: Optional[np.ndarray] = None
    
    @staticmethod
    def cluster(df: pd.DataFrame, options=Options()): return AHCMethod.cluster(df, options)
    
    def get_optimal_cluster_count(self) -> int: return AHCMethod.get_optimal_cluster_count(self)
    def plot(self): return AHCMethod.plot(self)
    def get_pca_loadings(self): return AHCMethod.get_pca_loadings(self)
    def explain_clusters_with_tree(self): return AHCMethod.explain_clusters_with_tree(self)
    def plot_boxplot_by_capacity(self): return AHCMethod.plot_boxplot_by_capacity(self)
    def plot_cluster_heatmap(self): return AHCMethod.plot_cluster_heatmap(self)
    def plot_dendogram(self): return AHCMethod.plot_dendogram(self)
    def profile_clusters(self): return AHCMethod.profile_clusters(self)


import pytest
@pytest.mark.above10s
def test_ahc_cluster():
    # pixi run pytest -rFP -q -s bio\ML\AHC.py::test_ahc_cluster -o "addopts="
    import bio
    from bio.Dataset import PDCCMethod
    from bio.__global__ import PDCC_CSV
    bio.setup_loguru()
    model_options = Options(
        pca_components=10, 
        scaler=StandardScaler(),
    )
    df = pd.read_csv(PDCC_CSV)
    df = df.head(25)
    df = PDCCMethod.increment_dataset(df)
    df = PDCCMethod.convert_names_to_smiles(df)
    df = PDCCMethod.featurize(df)
    ahc_model = AHC.cluster(df.copy(), model_options)
    n = ahc_model.get_optimal_cluster_count()
    model_options.n_clusters = n
    ahc_model = AHC.cluster(df.copy(), model_options)
    ahc_model.plot()
    ahc_model.get_pca_loadings()
    ahc_model.explain_clusters_with_tree()
    ahc_model.plot_boxplot_by_capacity()
    ahc_model.plot_cluster_heatmap()
    ahc_model.plot_dendogram()
    ahc_model.profile_clusters()
