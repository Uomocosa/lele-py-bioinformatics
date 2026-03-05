import pickle
import bio
from bio.Dataset import PDCC
from loguru import logger


def save_scaler(dataset: PDCC):
    if dataset._scaler_path is None: 
        logger.error("_scaler_path not defined")
        return
        
    if dataset.scaler is None:
        logger.warning("No fitted scaler to save (dataset.scaler is None).")
        return

    dataset._scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset._scaler_path, "wb") as f:
        pickle.dump(dataset.scaler, f)
        
    logger.info(f"Fitted scaler saved locally to {dataset._scaler_path}")


import pytest
@pytest.mark.todo
def test_():
    pass
