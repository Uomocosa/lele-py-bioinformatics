import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as mplot
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import lele, bio
from loguru import logger

@dataclass
class Options:
    cmap: str = 'coolwarm'
    center: float = 0.0
    figsize: tuple = (15, 15)
    linewidths: float = 0.5
    method: str = 'ward'
    core_feature: str = 'CAPACITY'
    n_highest_correlated_features: int = 20
    dpi: int = 300
    drop_axis: int = 0 # 0 = drop rows, 1 = drop columns

@dataclass
class Clustermap:
    """
    Generates a hierarchically-clustered heatmap of the correlation matrix 
    using Seaborn's clustermap.
    >>> The Heatmap (The Colors): 
        It takes your correlation matrix and colors the grid based 
        on the values. In your coolwarm setup, strong positive 
        correlations (close to 1.0) show up as deep red, strong negative 
        correlations (close to -1.0) show up as deep blue, and zero 
        correlations are neutral/white.
    >>> The Hierarchical Clustering (The Dendrograms): 
        This is the magic part. Unlike a standard heatmap that just 
        lists columns in the order you gave them, a clustermap runs 
        Agglomerative Hierarchical Clustering on the rows and columns. 
        It reorders the variables so that features behaving similarly 
        (highly correlated with each other) are grouped together.
    >>> The Trees (Dendrograms): 
        The little branch-like lines on the top and side of the visual 
        are called dendrograms. They show you the "lineage" of the 
        clusters. If two features merge really early (close to the 
        edge of the map), it means they are highly similar.
    """
    df: pd.DataFrame
    options: Options = field(default_factory=lambda: Options())
    
    def cluster(self, output_path: Optional[Path] = None):
        df_numeric = self.df.select_dtypes(include=['float64', 'int64', 'float32', 'int32'])
        if self.options.drop_axis == 0: # drop rows
            df_numeric = df_numeric.dropna()
        elif self.options.drop_axis == 1: # drop rows
            df_numeric = df_numeric.dropna(axis=1) # Drops failed calculation columns, saves rows
        else:
            raise ValueError("drop_axis must be 0 (rows) or 1 (columns)")
        
        df_numeric = df_numeric.loc[:, df_numeric.std() > 0] # Prevents the crash
        if not self.options.core_feature:
            logger.warning(f"Core feature not defined") 
        elif self.options.core_feature not in df_numeric.columns:
            logger.warning(f"Core feature '{self.options.core_feature}' not found in dataframe. Plotting all remaining features.")    
        else:
            full_corr = df_numeric.corr()
            orig_corrs = full_corr[self.options.core_feature]
            # Sort the INDEX by the absolute values descending
            top_cols = orig_corrs.abs().sort_values(ascending=False).head(self.options.n_highest_correlated_features + 1).index.tolist()
            top_corrs = orig_corrs[top_cols]
            logger.debug(f"Top {self.options.n_highest_correlated_features} features correlated with {self.options.core_feature} (sorted by magnitude):\n{top_corrs}")
            corr_dict = top_corrs.to_dict()
            logger.info(f"Calculated top {self.options.n_highest_correlated_features} features correlated with {self.options.core_feature}: \n{top_corrs}")
            logger.bind(
                log_type="correlation_data", 
                top_correlations=corr_dict
            )
            df_numeric = df_numeric[top_cols]
        corr_matrix = df_numeric.corr()
        cluster_grid = sns.clustermap(
            corr_matrix, 
            cmap=self.options.cmap, 
            center=self.options.center, 
            figsize=self.options.figsize, 
            linewidths=self.options.linewidths, 
            method=self.options.method
        )
        if output_path:
            mplot.savefig(output_path, dpi=self.options.dpi, bbox_inches="tight")
            mplot.close() # Close to free up memory
        return cluster_grid



import pytest
@pytest.mark.above10s
def test_without_fingerprints_clustermap():
    """
    Hierarchical Clustering on calculated chemical features only.
    Drops the one-hot encoded SMILES strings to avoid visual bloat 
    and meaningless correlations, allowing the physics to group naturally.
    """
    NAME = "without_fingerprints_clustermap"
    import pandas as pd
    from bio.Dataset import PDCC, PDCCMethod
    from bio.__global__ import CONVERTED_PDCC_CSV
    from bio.ML.__global__ import HELPER_DIR
    logger.remove()
    logger.add(
        sys.stderr,
        format = bio.__global__.LOGURU_SIMPLE_FORMAT,
        filter = {
            "bio.ML.MLPMethod.train_model": "WARNING",
        },
        level = "INFO"
    )
    logger.add(
        lele.Loguru.CleanJSONLSink(HELPER_DIR / f"{NAME}.jsonl"),
        filter=lambda record: record["extra"].get("log_type") == "correlation_data",
        level="TRACE",
    )
    df = pd.read_csv(CONVERTED_PDCC_CSV)
    df_features = PDCCMethod.featurize(
        df,
        options = PDCCMethod.featurize.Options(
            capping_atoms = ['H'],
            molecule_features_to_calculate = ['logp', 'logd', 'homo_lumo_eV'],
            polymer_features_to_calculate = ['logp', 'logd', 'homo_lumo_eV'],
        )
    )
    df_features['NORMALIZED_CAPACITY'] = df_features['CAPACITY'] / df_features['CONCENTRATION']
    df_features['NORMALIZED_CAPACITY'] = df_features['NORMALIZED_CAPACITY'].replace([np.inf, -np.inf], np.nan)
    cmap_model = Clustermap(
        df_features, 
        options = Options(
            core_feature='NORMALIZED_CAPACITY',
        )
    )
    cmap_model.cluster(output_path=HELPER_DIR / f"{NAME}.png")
