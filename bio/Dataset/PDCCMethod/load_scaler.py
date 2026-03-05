import pickle
import bio
from bio.Dataset import PDCC
from loguru import logger


def load_scaler(dataset: PDCC):
    if dataset._scaler_path is None:
        logger.error("_scaler_path not defined")
        return
        
    if not dataset._scaler_path.exists():
        logger.error(f"Scaler file not found at {dataset._scaler_path}")
        return
        
    with open(dataset._scaler_path, "rb") as f:
        dataset.scaler = pickle.load(f)
        
    logger.info(f"Scaler successfully loaded from {dataset._scaler_path}")
    assert dataset.scaler is not None, "Scaler must have been loaded"
    return dataset.scaler

import pytest
@pytest.mark.todo
def test_():
    pass
