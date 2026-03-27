import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as mplot
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

@dataclass
class Options:
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
    
    def cluster(self): return AHCMethod.cluster(self)
    def plot(self, output_path: Path): return AHCMethod.plot(self, output_path)
    def get_pca_loadings(self, n: int = 5): return AHCMethod.get_pca_loadings(self, n)


import pytest
@pytest.fixture
def pytest_logger_setup():
    logger.remove()
    logger.add(
        sys.stderr,
        format = bio.__global__.LOGURU_SIMPLE_FORMAT,
        filter = {
            "bio.ML.MLPMethod.train_model": "WARNING",
        },
        level = "INFO"
    )


def run_and_save_ahc(model: AHC, output_name: str, print_pca: bool = True):
    """Handles the repetitive task of clustering, logging, saving CSVs, and plotting."""
    model.cluster()
    
    print(f"\n--- {output_name.upper()} CLUSTER COUNTS ---")
    print(model.df['Cluster'].value_counts())
    
    model.df.to_csv(HELPER_DIR / f"{output_name}.csv", index=False)
    model.plot(output_path=HELPER_DIR / f"{output_name}.png")
    
    if print_pca:
        pca_loadings = model.get_pca_loadings()
        print(f"\n--- {output_name.upper()} PCA LOADINGS ---")
        print(pca_loadings)
        

@pytest.mark.skip(reason="useless")
# pixi run pytest -rFP -q -s bio\ML\AHC.py::test_ahc_cluster -o "addopts="
def test_ahc_cluster():
    import bio
    from bio.Dataset import PDCC, PDCCMethod
    from bio.__global__ import CONVERTED_PDCC_CSV
    from bio.ML.__global__ import HELPER_DIR
    df = pd.read_csv(CONVERTED_PDCC_CSV)
    df = PDCCMethod.featurize(
        df,
        options = PDCCMethod.featurize.Options(capping_atoms_dict={'H': 1})
    )
    model = AHC(df, options=Options(n_clusters=5, scaler=StandardScaler()))
    run_and_save_ahc(model, "clustered_output")


@pytest.mark.skip(reason="useless")
def test_one_hot_clustering():
    # pixi run pytest -q -s bio\ML\AHC.py::test_one_hot_clustering
    import bio
    from bio.Dataset import PDCC, PDCCMethod
    from bio.__global__ import CONVERTED_PDCC_CSV
    from bio.ML.__global__ import HELPER_DIR
    df = pd.read_csv(CONVERTED_PDCC_CSV)
    df = pd.get_dummies(df, columns=['DRUG', 'POLYMER_USED'], dtype=int)
    model = AHC(df, options=Options(n_clusters=5, pca_components=10, scaler=StandardScaler()))
    run_and_save_ahc(model, "one_hot_clustered_output")


def cluster_by_features(
    name: str, 
    model_options: Options, 
    featurizer_options: PDCCMethod.featurize.Options
):
    import bio
    from bio.Dataset import PDCC, PDCCMethod
    from bio.__global__ import CONVERTED_PDCC_CSV, SMILES_DICT, PSMILES_DICT
    from bio.ML.__global__ import HELPER_DIR
    df = pd.read_csv(CONVERTED_PDCC_CSV)
    REV_SMILES = {v: k for k, v in SMILES_DICT.items()}
    REV_PSMILES = {v: k for k, v in PSMILES_DICT.items()}
    df_features = PDCCMethod.featurize(df, featurizer_options)
    df_labels = df.loc[df_features.index].copy()
    overlap = [c for c in df_labels.columns if c in df_features.columns]
    df_features = df_features.drop(columns=overlap)
    df_features = df_features.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
    group_keys = [df_labels['POLYMER_USED'], df_labels['DRUG']]
    system_features = df_features.groupby(group_keys).mean().reset_index(drop=True)
    system_labels = df_labels.groupby(['POLYMER_USED', 'DRUG']).mean(numeric_only=True).reset_index()
    model = AHC(system_features, model_options)
    model.cluster()
    system_labels['CLUSTER'] = model.df['Cluster']
    cols = ['CLUSTER'] + [c for c in system_labels.columns if c != 'CLUSTER']
    system_labels = system_labels[cols]
    system_labels['POLYMER_USED'] = system_labels['POLYMER_USED'].apply(lambda x: REV_PSMILES.get(x, x))
    system_labels['DRUG'] = system_labels['DRUG'].apply(lambda x: REV_SMILES.get(x, x))
        
    print("\n--- MEANINGFUL CLUSTER ASSIGNMENTS ---")
    for cluster_id in sorted(system_labels['CLUSTER'].unique()):
        print(f"\nCLUSTER #{cluster_id}:")
        members = system_labels[system_labels['CLUSTER'] == cluster_id][['POLYMER_USED', 'DRUG', 'CAPACITY']]
        print(members.sort_values(by='CAPACITY', ascending=False).to_string(index=False))
        
    system_labels.to_csv(HELPER_DIR / f"{name}.csv", index=False)
    # matplotlib.use('Agg')
    model.plot(output_path=HELPER_DIR / f"{name}.png")


def test_simple_feature_clustering(pytest_logger_setup):
    # pixi run pytest -rFP -q -s bio\ML\AHC.py::test_simple_feature_clustering -o "addopts="
    name = "simple_feature_clustering"
    model_options = Options(
        n_clusters=5, 
        pca_components=10, 
        scaler=StandardScaler(),
    )
    featurizer_options = PDCCMethod.featurize.Options(
        capping_atoms = ['H'],
        molecule_features_to_calculate = ['logp', 'logd'],
        polymer_features_to_calculate = [],
    )
    cluster_by_features(name, model_options, featurizer_options)


def test_full_feature_clustering_n3_pca15(pytest_logger_setup):
    # pixi run pytest -rFP -q -s bio\ML\AHC.py::test_full_feature_clustering_n3_pca15 -o "addopts="
    name = "full_feature_clustering_n3_pca15"
    model_options = Options(
        n_clusters=3, 
        pca_components=15, 
        scaler=StandardScaler(),
    )
    featurizer_options = PDCCMethod.featurize.Options(
        capping_atoms = ['H'],
        # molecule_features_to_calculate = ['logp', 'logd'],
        # polymer_features_to_calculate = [],
    )
    cluster_by_features(name, model_options, featurizer_options)


def test_full_feature_clustering_n5_pca25(pytest_logger_setup):
    # pixi run pytest -q -s bio\ML\AHC.py::test_full_feature_clustering_n5_pca25
    name = "full_feature_clustering_n5_pca25"
    model_options = Options(
        n_clusters=5, 
        pca_components=25, 
        scaler=StandardScaler(),
    )
    featurizer_options = PDCCMethod.featurize.Options(
        capping_atoms = ['H'],
        # molecule_features_to_calculate = ['logp', 'logd'],
        # polymer_features_to_calculate = [],
    )
    cluster_by_features(name, model_options, featurizer_options)
