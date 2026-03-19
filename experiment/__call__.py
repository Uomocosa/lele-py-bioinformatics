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
import experiment
from bio.ML import MLP, MLPMethod
from bio.Dataset import PDCC, PDCCMethod
from bio.__global__ import CONVERTED_PDCC_CSV
from loguru import logger

print(dir(lele.Loguru))
from lele.Loguru import CleanJSONLSink

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
    # pixi run example --help
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
    
def run(exp: Experiment):
    save_dir = lele.P(__file__).parent
    setup_loguru(exp, save_dir)
    yaml.SafeDumper.add_multi_representer(
        Path, 
        lambda dumper, data: dumper.represent_str(str(data))
    )
    formatted_config = yaml.safe_dump(
        asdict(exp), 
        default_flow_style=False, 
        sort_keys=False
    )
    logger.info(f"Running Experiment '{exp.name}' with config:\n{formatted_config}")
        
    bio.ML.set_seed(exp.seed)
    
    pdcc_config = exp.dataset.to_pdcc_config(seed=exp.seed)
    mlp_config = exp.model.to_mlp_config(seed=exp.seed, save_dir=None)
    featurizer_options = exp.features.to_featurizer_config()
    
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
        seed = exp.seed,
    )
    
    model = MLP(
        splitted_dataset = splitted_dataset, 
        featurize = featurize_fn,
        config = mlp_config,
        scaler = None,
    )
    
    MLPMethod.k_fold_cross_validation(model, k=mlp_config.k_fold)
    
    config_save_path = save_dir / f"{exp.name}" / "config.yaml"
    with open(config_save_path, "w") as f:
        # asdict converts the entire Experiment dataclass tree into a dictionary
        yaml.safe_dump(asdict(exp), f, default_flow_style=False, sort_keys=False)

    create_graphs_from_logs(exp, save_dir)



def setup_loguru(exp: Experiment, save_dir: Path):
    log = experiment.get_log_files(exp, save_dir)
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
        lele.Loguru.CleanJSONLSink(log["traing_epochs"]),
        filter=lambda record: record["extra"].get("log_type") == "epoch_trace",
        level="TRACE",
    )    
    logger.add(
        lele.Loguru.CleanJSONLSink(log["fold_predictions"]),
        filter=lambda record: record["extra"].get("log_type") == "prediction_trace",
        level="TRACE",
    )
    logger.add(
        lele.Loguru.CleanJSONLSink(log["fold_metrics"]),
        filter=lambda record: record["extra"].get("log_type") == "fold_metric_trace",
        level="TRACE",
    )
    logger.add(
        lele.Loguru.CleanJSONLSink(log["aggregate"]),
        filter=lambda record: record["extra"].get("log_type") == "aggregate_metrics",
        level="TRACE",
    )
    return logger
    
    
def create_graphs_from_logs(exp: Experiment, save_dir: Path):
    """Orchestrates the creation of all example visualizations."""
    log = experiment.get_log_files(exp, save_dir)
    sns.set_theme(style="whitegrid", palette="muted")
    experiment.plot_learning_curves(exp, save_dir, log["traing_epochs"])
    experiment.plot_parity(exp, save_dir, log["fold_predictions"])
    experiment.plot_fold_variance(exp, save_dir, log["fold_metrics"])
    
        
def test_():
    default_experiment = Experiment()
    default_experiment.dataset.max_size = 10
    default_experiment.model.epochs = 10
    run(default_experiment)
