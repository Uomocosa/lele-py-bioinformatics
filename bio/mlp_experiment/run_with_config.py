import sys, types
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import pandas as pd
import seaborn as sns
from dataclasses import dataclass, field, asdict
import copy, warnings
from pathlib import Path
from typing import Optional, Callable, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import lele, bio
from bio.ML import MLP, MLPMethod
from bio.Dataset import PDCC, PDCCMethod
from bio.__global__ import PDCC_CSV, RESULTS_DIR
from loguru import logger

ExperimentConfig = bio.mlp_experiment.Config

from bio.__global__ import CACHE_MEMORY
@CACHE_MEMORY.cache
def run_with_config(exp_config: ExperimentConfig):
    setup_loguru(exp_config)
    
    if exp_config.dataset_config.seed != exp_config.seed: 
        logger.warning(f"Dataset seed {exp_config.dataset_config.seed} does not match exp_config seed {exp_config.seed}, setting seed to {exp_config.seed}")
        exp_config.dataset_config.seed = exp_config.seed
    if exp_config.model_config.seed != exp_config.seed: 
        logger.warning(f"ModelConfig seed {exp_config.model_config.seed} does not match exp_config seed {exp_config.seed}, setting seed to {exp_config.seed}")
        exp_config.model_config.seed = exp_config.seed
    
    
    yaml.SafeDumper.add_multi_representer(
        Path, 
        lambda dumper, data: dumper.represent_str(str(data))
    )
    formatted_config = yaml.safe_dump(
        asdict(exp_config), 
        default_flow_style=False, 
        sort_keys=False
    )
    config_save_path = exp_config.save_dir / f"{exp_config.name}" / "exp_config.yaml"
    config_save_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    with open(config_save_path, "w") as f:
        f.write(formatted_config)
    logger.info(f"Running Experiment '{exp_config.name}' with exp_config:\n{formatted_config}")
        
    x_scaler_fn = exp_config.get_x_scaler_fn()
    y_scaler_fn = exp_config.get_y_scaler_fn()
    forward_fn = exp_config.get_forward_fn()
        
    
        
    bio.ML.set_seed(exp_config.seed)
    
    psmiles_dict = bio.integrate_paper_scraper.resolve_dict_sources(
        exp_config.dataset_config.psmiles_dicts, "psmiles"
    )
    smiles_dict = bio.integrate_paper_scraper.resolve_dict_sources(
        exp_config.dataset_config.smiles_dicts, "smiles"
    )

    featurize_fn = lambda df: PDCCMethod.featurize(df, options=exp_config.featurizer_options)
    dataset = bio.Dataset.PDCC(config = exp_config.dataset_config)
    dataset.increment_dataset(options=exp_config.incerement_dataset_options)
    dataset.convert_names_to_smiles(
        PDCCMethod.convert_names_to_smiles.Options(
            psmiles_dict=psmiles_dict,
            smiles_dict=smiles_dict,
        )
    )
    dataset.featurize_fn = featurize_fn
    
    torch_dataset = dataset.to_torch_dataset()
    x_sample, y_sample = torch_dataset[0]
    trn, val, tst = exp_config.dataset_config.train_validation_test_pecentages
    splitted_dataset = bio.Dataset.split_dataset(
        dataset = torch_dataset,
        train_percentage = trn,
        validation_percentage = val,
        test_percentage = tst,
        seed = exp_config.seed,
    )
    
    x_scaler = None
    y_scaler = None
    if x_scaler_fn:
        x_scaler = splitted_dataset.scale(
            feature_col_indexes = range(torch_dataset.num_features),
            feature_attribute = "X",
            scaler_fn = x_scaler_fn,
        )
    if y_scaler_fn:
        y_scaler = splitted_dataset.scale(
            feature_col_indexes = range(len(y_sample.shape)),
            feature_attribute = "y",
            scaler_fn = y_scaler_fn,
        )
    
    model = MLP(
        splitted_dataset = splitted_dataset, 
        featurize_fn = dataset.featurize_fn,
        x_scaler = x_scaler,
        y_scaler = y_scaler,
        config = exp_config.model_config,
    )
    model.forward = types.MethodType(forward_fn, model)
    
    MLPMethod.k_fold_cross_validation(model, k=exp_config.k_fold, cv_method=exp_config.cv_method)
            
    log = bio.mlp_experiment.get_log_files(exp_config, exp_config.save_dir)
    sns.set_theme(style="whitegrid", palette="muted")
    bio.mlp_experiment.plot_learning_curves(exp_config, exp_config.save_dir, log["traing_epochs"])
    bio.mlp_experiment.plot_parity(exp_config, exp_config.save_dir, log["fold_predictions"])
    bio.mlp_experiment.plot_fold_variance(exp_config, exp_config.save_dir, log["fold_metrics"])
        
        
def setup_loguru(exp_config: ExperimentConfig):
    log = bio.mlp_experiment.get_log_files(exp_config, exp_config.save_dir)
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
    
    
def test_k_fold():
    from bio.mlp_experiment.__global__ import HELPER_DIR
    ExperimentConfig = bio.mlp_experiment.Config
    DatasetConfig = bio.mlp_experiment.Config.DatasetConfig
    ModelConfig = bio.mlp_experiment.Config.ModelConfig
    FeaturizerOptions = bio.mlp_experiment.Config.FeaturizerOptions
    experiment_config = ExperimentConfig(
        name="experiment_test_k_fold_cross_validation",
        k_fold = 3,
        save_dir = HELPER_DIR,
        dataset_config = DatasetConfig(
            max_size = 30,
        ),
    )
    run_with_config(experiment_config)


def test_loocv():
    from bio.mlp_experiment.__global__ import HELPER_DIR
    ExperimentConfig = bio.mlp_experiment.Config
    DatasetConfig = bio.mlp_experiment.Config.DatasetConfig
    ModelConfig = bio.mlp_experiment.Config.ModelConfig
    FeaturizerOptions = bio.mlp_experiment.Config.FeaturizerOptions
    experiment_config = ExperimentConfig(
        name="experiment_test_leave_one_out_cross_validation",
        k_fold = -1,
        save_dir = HELPER_DIR,
        dataset_config = DatasetConfig(
            max_size = 10,
        ),
    )
    run_with_config(experiment_config)
