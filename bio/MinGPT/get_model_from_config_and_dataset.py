import lele, bio

from mingpt.model import GPT
import time
import torch
from loguru import logger
import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)

def get_model_from_config_and_dataset(
    config: bio.MinGPT.ModelConfig, 
    dataset: torch.utils.data.Dataset
):
    a = time.perf_counter()
    model_config = GPT.get_default_config()
    model_config.model_type = config.model_type
    model_config.block_size = config.block_size
    model_config.vocab_size = len(dataset.tokenizer)
    model_config.device = bio.ML.get_torch_device()
    model = GPT(model_config)
    if config.starting_state_dict: 
        model.load_state_dict(torch.load(
            lele.P(STARTING_STATE_DICT), 
            map_location=model_config.device
        ))
        model.to(torch.device(model_config.device))
    b = time.perf_counter()
    logger.debug(f"model:\n{model}")
    logger.debug(f"model loaded in {(b-a):.2f} s")
    return model
    
    
def test_():
    from bio.MinGPT.__global__ import MIN_GPT_CONFIG_FILE
    config = bio.MinGPT.ModelConfig()
    unlabeled_smiles = bio.Dataset.UnlabeledSmiles.load(
        csv_file=config.dataset.csv_file,
        train_validation_test_pecentages=config.dataset.train_validation_test_pecentages, 
        max_dataset_size=config.dataset.max_size
    )
    dataset = unlabeled_smiles.to_torch_dataset(block_size=config.block_size)
    model = get_model_from_config_and_dataset(config, dataset)
