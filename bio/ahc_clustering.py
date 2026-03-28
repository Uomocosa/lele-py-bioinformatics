import tyro
import pandas as pd
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path
import bio
from bio.ML import AHC
from bio.Dataset import PDCCMethod
from bio.__global__ import PDCC_CSV
from bio.ML.__global__ import HELPER_DIR as ML_HELPER_DIR
from bio.__global__ import RESULTS_DIR

IncrementDatasetOptions = PDCCMethod.increment_dataset.Options
FeaturizerOptions = PDCCMethod.featurize.Options

@dataclass
class AHCConfig(AHC.Options):
    csv_file: str = PDCC_CSV
    save_dir: Path = RESULTS_DIR / "ahc_clustering"
    use_optimal_cluster_count: bool = False
    max_size: Optional[int] = None
    incerement_dataset_options: IncrementDatasetOptions = field(default_factory=lambda: IncrementDatasetOptions())
    featurizer_options: FeaturizerOptions = field(default_factory=lambda: FeaturizerOptions())

def test_():
    bio.setup_loguru()
    config = AHCConfig()
    config.save_dir = ML_HELPER_DIR / "AHC"
    config.max_size = 25
    run_with_config(config)

def main():
    bio.setup_loguru()
    config = tyro.cli(AHCConfig)
    run_with_config(config)
    
def run_with_config(config: AHCConfig):
    # This automatically maps all matching keys to the AHC.Options constructor
    model_options = AHC.Options(**{
        k: v for k, v in asdict(config).items() 
        if k in AHC.Options.__dataclass_fields__
    })
    
    df = pd.read_csv(PDCC_CSV)
    if config.max_size: df = df.head(config.max_size)
    df = PDCCMethod.increment_dataset(df, options=config.incerement_dataset_options)
    df = PDCCMethod.convert_names_to_smiles(df)
    df = PDCCMethod.featurize(df, options=config.featurizer_options)
    
    ahc_model = AHC.cluster(df.copy(), model_options)
    
    if config.use_optimal_cluster_count:
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
