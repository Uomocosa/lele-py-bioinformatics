import seaborn as sns
import matplotlib.pyplot as mplot
from pathlib import Path
import bio
from bio.ML import AHC
from loguru import logger
import warnings

def plot_boxplot_by_capacity(model: AHC):
    """
    Generates and saves a boxplot showing the distribution of Drug Loading Capacity across clusters.
    """
    # 1. Ensure the model has been clustered and has the required data
    if getattr(model, 'df', None) is None or 'CLUSTER' not in model.df.columns:
        raise ValueError("Cluster column not found. Please run cluster() first.")
        
    if 'CAPACITY' not in model.df.columns:
        # Note: If your target variable is named differently (e.g., 'capacity'), update this check!
        raise KeyError("'CAPACITY' column is missing from the model's DataFrame.")

    # 2. Prepare the output directory
    save_dir = Path(model.options.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "capacity_by_cluster.png"

    # 3. Plot the Boxplot
    mplot.figure(figsize=(10, 6))
    
    # Using hue='CLUSTER' and legend=False prevents deprecation warnings in newer seaborn versions
    with warnings.catch_warnings():
        # IMPORTANT! Maybe by doing 'pixi add seaborn>=0.13.0' it will be solved, still I dont want to do that.
        warnings.simplefilter("ignore", category=PendingDeprecationWarning)
        sns.boxplot(
            data=model.df, 
            x='CLUSTER', 
            y='CAPACITY', 
            palette='tab10',
            hue='CLUSTER',
            legend=False
        )
        
    # 4. Add formatting
    mplot.title('Drug Loading Capacity Distribution by Cluster Group', fontsize=14, pad=15)
    mplot.xlabel('Cluster ID', fontsize=12)
    mplot.ylabel('Capacity', fontsize=12)
    mplot.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 5. Save and close
    mplot.tight_layout()
    mplot.savefig(output_path, dpi=300)
    mplot.close()
    
    logger.info(f"Capacity boxplot saved to {output_path}. Check this to verify if clusters correlate with performance.")
    
    
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
    plot_boxplot_by_capacity(ahc)
