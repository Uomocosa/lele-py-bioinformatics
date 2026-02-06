import lele, bio
from mingpt.model import GPT
from mingpt.trainer import Trainer
import time
import torch
from loguru import logger
import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)

def get_trainer_config_from_model_config_and_dataset(
    config: bio.MinGPT.ModelConfig, 
    dataset: torch.utils.data.Dataset
):
    trainer_config = Trainer.get_default_config()
    trainer_config.learning_rate = config.learning_rate
    trainer_config.batch_size = config.batch_size
    iters_per_epoch = (len(dataset) + config.batch_size - 1) // config.batch_size
    trainer_config.max_iters = config.epochs * iters_per_epoch
    trainer_config.num_workers = config.num_workers
    return trainer_config

def test_():
    from bio.MinGPT.__global__ import MIN_GPT_CONFIG_FILE
    config = bio.MinGPT.ModelConfig()
    unlabeled_smiles = bio.Dataset.UnlabeledSmiles.load(
        csv_file=config.dataset.csv_file,
        train_validation_test_pecentages=config.dataset.train_validation_test_pecentages, 
        max_dataset_size=config.dataset.max_size
    )
    dataset = unlabeled_smiles.to_torch_dataset(block_size=config.block_size)
    model = bio.MinGPT.get_model_from_config_and_dataset(config, dataset)
    trainer_config = get_trainer_config_from_model_config_and_dataset(config, dataset)
    print(trainer_config)
