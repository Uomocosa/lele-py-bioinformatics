import sys, yaml, json
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tyro
from rdkit import Chem
from typing import Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable, Tuple, Annotated
from polymetrix.featurizers.chemical_featurizer import *
import lele, bio
from bio.ML import MLP, MLPMethod
from bio.Dataset import PDCC, PDCCMethod
from bio.__global__ import CONVERTED_PDCC_CSV
from loguru import logger

ALL_POSSIBLE_FEATURES = [
    NumHBondDonors, NumHBondAcceptors, NumRotatableBonds, NumRings,
    NumNonAromaticRings, NumAromaticRings, NumAtoms, TopologicalSurfaceArea,
    FractionBicyclicRings, NumAliphaticHeterocycles, SlogPVSA1, BalabanJIndex,
    MolecularWeight, Sp3CarbonCountFeaturizer, Sp2CarbonCountFeaturizer,
    MaxEStateIndex, SmrVSA5, FpDensityMorgan1, HalogenCounts, BondCounts,
    BridgingRingsCount, MaxRingSize, HeteroatomCount, HeteroatomDensity,
]
FEATURE_MAP = {cls.__name__: cls for cls in ALL_POSSIBLE_FEATURES}

@dataclass
class DatasetConfig:
    csv_file: Path = CONVERTED_PDCC_CSV
    train_validation_test_pecentages: Tuple[float, float, float] = (0.6, 0.2, 0.2)
    max_size: Optional[int] = None

    def to_pdcc_config(self, seed: int) -> PDCC.Config:
        """Translates to the internal PDCC.Config, injecting the seed."""
        return PDCC.Config(
            csv_file=self.csv_file,
            max_size=self.max_size,
            seed=seed  # Injected from Experiment level
        )
            
@dataclass
class FeaturizerConfig:
    capping_atoms: list = field(default_factory=lambda: [
        'H', 'C', 'O'
    ])
    fingerprint_radius: int = 2
    fingerprint_n_bits: int = 256
    protonate_precision: float = 1.0
    molecule_features_to_calculate: list = field(default_factory=lambda: [
        'logp', 'logd', 'homo_lumo_eV', 'fingerprint'
    ])
    polymer_features_to_calculate: list = field(default_factory=lambda: [
        'logp',  'logd',  'homo_lumo_eV', 'fingerprint',
    ])
    molecule_polymetrix_features: list = field(default_factory=lambda: ["ALL"]) 
    polymer_polymetrix_features: list = field(default_factory=lambda: ["ALL"])
    sidechain_polymetrix_features: list = field(default_factory=lambda: ["ALL"])
    backbone_polymetrix_features: list = field(default_factory=lambda: ["ALL"])
    
    def to_featurizer_config(self) -> PDCCMethod.featurize.Options:
        """Translates to the internal featurize.Options."""
        pt = Chem.GetPeriodicTable()
        capping_atoms_dict = {
            symbol: pt.GetAtomicNumber(symbol) 
            for symbol in self.capping_atoms
        }
        logger.info(f"capping_atoms_dict: {capping_atoms_dict}")
        
        def _parse_polymetrix(features: list) -> list:
            if "ALL" in features: return ALL_POSSIBLE_FEATURES
            if "all" in features: return ALL_POSSIBLE_FEATURES
            return [FEATURE_MAP[f] for f in features if f in FEATURE_MAP]
                    
        return PDCCMethod.featurize.Options(
            capping_atoms_dict=capping_atoms_dict,
            fingerprint_radius=self.fingerprint_radius,
            fingerprint_n_bits=self.fingerprint_n_bits,
            protonate_precision=self.protonate_precision,
            molecule_features_to_calculate=self.molecule_features_to_calculate,
            polymer_features_to_calculate=self.polymer_features_to_calculate,
            molecule_polymetrix_features=_parse_polymetrix(self.molecule_polymetrix_features),
            polymer_polymetrix_features=_parse_polymetrix(self.polymer_polymetrix_features),
            sidechain_polymetrix_features=_parse_polymetrix(self.sidechain_polymetrix_features),
            backbone_polymetrix_features=_parse_polymetrix(self.backbone_polymetrix_features),
        )
            
CRITERION_MAP = {
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
}

@dataclass
class ModelConfig():
    k_fold: int = 5 # Note! if you set it to -1 -> implies LOOCV
    hidden_dims: list = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.2
    weight_decay: float = 1e-4
    criterion: str = "mse"
    learning_rate: float = 1e-3
    epochs: int = 1000
    early_stop_patience: int = 100
    batch_size: int = 16
    num_workers: int = 0   
    
    def to_mlp_config(
        self, 
        seed: int, 
        save_dir: Optional[Path] = None,
    ) -> MLP.Config:
        """Translates to the internal MLP.Config, mapping strings and injecting fields."""
        config = MLP.Config(
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            weight_decay=self.weight_decay,
            criterion=CRITERION_MAP.get(self.criterion.lower(), nn.MSELoss()),
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            early_stop_patience=self.early_stop_patience,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            seed=seed,                   # Injected
            best_model_save_dir=save_dir # Injected
        )
        config.k_fold = self.k_fold
        return config

@dataclass
class Experiment():
    name: str = "experiment_0"
    yaml_base_config: Optional[Path] = None
    dataset: DatasetConfig = field(default_factory=lambda: DatasetConfig())
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    features: FeaturizerConfig = field(default_factory=lambda: FeaturizerConfig())
    seed: int = 42

def __call__():
    # pixi run experiment --help
    """Entry point for handling YAML and CLI overriding."""
    cli_args = tyro.cli(Experiment)
    default_experiment = Experiment()
    if cli_args.yaml_base_config and cli_args.yaml_base_config.exists():
        with open(cli_args.yaml_base_config, "r") as f:
            yaml_dict = yaml.safe_load(f)
        with open(cli_args.yaml_base_config, "r") as f:
            default_experiment = tyro.extras.from_yaml(Experiment, f)
    final_args = tyro.cli(Experiment, default=default_experiment)
    
    run(final_args)
    
def run(experiment: Experiment):
    save_dir = lele.P(__file__).parent
    setup_loguru(experiment, save_dir)
    yaml.SafeDumper.add_multi_representer(
        Path, 
        lambda dumper, data: dumper.represent_str(str(data))
    )
    formatted_config = yaml.safe_dump(
        asdict(experiment), 
        default_flow_style=False, 
        sort_keys=False
    )
    logger.info(f"Running Experiment '{experiment.name}' with config:\n{formatted_config}")
        
    bio.ML.set_seed(experiment.seed)
    
    pdcc_config = experiment.dataset.to_pdcc_config(seed=experiment.seed)
    mlp_config = experiment.model.to_mlp_config(seed=experiment.seed, save_dir=None)
    featurizer_options = experiment.features.to_featurizer_config()
    
    dataset = bio.Dataset.PDCC(pdcc_config)
    featurize_fn = lambda df: PDCCMethod.featurize(df, featurizer_options)
    dataset.featurize = featurize_fn
    torch_dataset = dataset.to_torch_dataset()
    trn, val, tst = pdcc_config.train_validation_test_pecentages
    splitted_dataset = bio.Dataset.split_dataset(
        dataset = torch_dataset,
        train_percentage = trn,
        validation_percentage = val,
        test_percentage = tst,
        seed = experiment.seed,
    )
    
    model = MLP(
        splitted_dataset = splitted_dataset, 
        featurize = featurize_fn,
        config = mlp_config,
        scaler = None,
    )
    
    MLPMethod.k_fold_cross_validation(model, k=mlp_config.k_fold)
    
    config_save_path = save_dir / f"{experiment.name}" / "config.yaml"
    with open(config_save_path, "w") as f:
        # asdict converts the entire Experiment dataclass tree into a dictionary
        yaml.safe_dump(asdict(experiment), f, default_flow_style=False, sort_keys=False)

    create_graphs_from_logs(experiment, save_dir)



class CleanJSONLSink:
    def __init__(self, filepath: Path):
        # Open in 'w' mode to start fresh every time you run the experiment
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.file = filepath.open("w")
        
    def __call__(self, message):
        # Extract ONLY the bound variables and write as JSON
        json.dump(message.record["extra"], self.file)
        self.file.write("\n")
        self.file.flush()
        
        
def get_log_files(experiment: Experiment, save_dir: Path):
    return {
        "traing_epochs": save_dir / f"{experiment.name}" / "traing_epochs.jsonl",
        "fold_predictions": save_dir / f"{experiment.name}" / "fold_predictions.jsonl",
        "fold_metrics": save_dir / f"{experiment.name}" / "fold_metrics.jsonl",
        "aggregate": save_dir / f"{experiment.name}" / "aggregated_results.jsonl",
    }

def setup_loguru(experiment: Experiment, save_dir: Path):
    log = get_log_files(experiment, save_dir)
    logger.remove()
    logger.add(
        sys.stderr,
        format = bio.__global__.LOGURU_SIMPLE_FORMAT,
        filter = {
            "bio.ML.MLPMethod.train_model": "WARNING",
            # "": "INFO",
        },
        level = "INFO"
    )
    logger.add(
        CleanJSONLSink(log["traing_epochs"]),
        filter=lambda record: record["extra"].get("log_type") == "epoch_trace",
        level="TRACE",
    )    
    logger.add(
        CleanJSONLSink(log["fold_predictions"]),
        filter=lambda record: record["extra"].get("log_type") == "prediction_trace",
        level="TRACE",
    )
    logger.add(
        CleanJSONLSink(log["fold_metrics"]),
        filter=lambda record: record["extra"].get("log_type") == "fold_metric_trace",
        level="TRACE",
    )
    logger.add(
        CleanJSONLSink(log["aggregate"]),
        filter=lambda record: record["extra"].get("log_type") == "aggregate_metrics",
        level="TRACE",
    )
    return logger
    
    
def create_graphs_from_logs(experiment: Experiment, save_dir: Path):
    log = get_log_files(experiment, save_dir)
    sns.set_theme(style="whitegrid", palette="muted")
    if log["traing_epochs"].exists():
        """
        Generates learning curves by plotting Training and Validation loss against the number of epochs. 
        
        This graph tracks the model's progress over time:
        - It serves as a diagnostic tool for **overfitting** (where training loss continues to drop 
          but validation loss rises) and **underfitting** (where neither loss reaches a low value).
        - Since k-fold cross-validation is used, the lines represent the mean loss across all folds, 
          providing a more robust view of the model's convergence than a single run.
        """
        df_epochs = pd.read_json(log["traing_epochs"], lines=True)
        if not df_epochs.empty:
            plt.figure(figsize=(10, 6))
            
            # If your epochs log has a 'fold' column, Seaborn will automatically 
            # plot the mean line and a shaded confidence interval across all folds!
            sns.lineplot(data=df_epochs, x="epoch", y="train_loss", label="Train Loss", linewidth=2)
            sns.lineplot(data=df_epochs, x="epoch", y="val_loss", label="Validation Loss", linewidth=2)
            
            plt.title(f"{experiment.name} - Learning Curves (Mean across K-Folds)")
            plt.xlabel("Epoch")
            plt.ylabel(f"Loss ({experiment.model.criterion.upper()})")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_dir / f"{experiment.name}" / "plot_learning_curves.png", dpi=300)
            plt.close()
            logger.info("Saved learning curves plot.")

    # --- 2. PARITY PLOT (Actual vs Predicted) ---
    if log["fold_predictions"].exists():
        """
        Generates a Parity Plot (also known as a predicted-vs-actual plot).
        
        This graph visualizes the accuracy and precision of the regression model:
        - The **Red Dashed Line** represents a perfect model where predicted values exactly equal actual values ($y=x$).
        - **Closer clusters** to this line indicate higher accuracy.
        - Coloring the points by **fold** allows for the detection of "outlier folds," helping 
          determine if the model's performance is consistent across different subsets of data.
        - Dispersion (spread) away from the line indicates the error variance.
        """
        df_preds = pd.read_json(log["fold_predictions"], lines=True)
        if not df_preds.empty:
            plt.figure(figsize=(8, 8))
            
            # Scatter plot, colored by fold to see if one fold behaved weirdly
            hue_arg = "fold" if "fold" in df_preds.columns else None
            sns.scatterplot(data=df_preds, x="actual", y="predicted", hue=hue_arg, alpha=0.7, edgecolor=None)
            
            # Draw the perfect prediction y = x line
            min_val = min(df_preds["actual"].min(), df_preds["predicted"].min())
            max_val = max(df_preds["actual"].max(), df_preds["predicted"].max())
            plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Prediction")
            
            plt.title(f"{experiment.name} - Parity Plot")
            plt.xlabel("Actual Value")
            plt.ylabel("Predicted Value")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_dir / f"{experiment.name}" / "plot_parity.png", dpi=300)
            plt.close()
            logger.info("Saved parity plot.")

    # --- 3. FOLD METRICS SUMMARY ---
    if log["fold_metrics"].exists():
        """
        Generates a bar chart comparing the Coefficient of Determination ($R^2$) across all k-folds.
        
        This graph assesses the **stability and reliability** of the model:
        - The **Bar Height** shows the $R^2$ for each specific fold, indicating how much variance 
          in the target variable was captured by that specific model instance.
        - The **Mean $R^2$ Line** provides a high-level summary of expected performance.
        - Large discrepancies between bars suggest that the model is sensitive to the 
          particular data split (high variance), while uniform bars suggest a stable, generalizable model.
        """
        df_metrics = pd.read_json(log["fold_metrics"], lines=True)
        if not df_metrics.empty and "fold" in df_metrics.columns and "r2" in df_metrics.columns:
            plt.figure(figsize=(8, 5))
            
            # Bar plot of R² per fold
            ax = sns.barplot(data=df_metrics, x="fold", y="r2", color="cornflowerblue")
            
            # Add a horizontal line representing the mean R²
            mean_r2 = df_metrics["r2"].mean()
            plt.axhline(mean_r2, color="red", linestyle="--", label=f"Mean $R^2$ ({mean_r2:.3f})")
            
            plt.title(f"{experiment.name} - $R^2$ Variance Across Folds")
            plt.xlabel("Fold")
            plt.ylabel("$R^2$ Score")
            plt.ylim(0, 1) # Assuming R2 is between 0 and 1, adjust if needed
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(save_dir / f"{experiment.name}" / "plot_fold_variance.png", dpi=300)
            plt.close()
            logger.info("Saved fold variance plot.")
    
def test_():
    default_experiment = Experiment()
    default_experiment.dataset.max_size = 10
    default_experiment.model.epochs = 10
    run(default_experiment)
