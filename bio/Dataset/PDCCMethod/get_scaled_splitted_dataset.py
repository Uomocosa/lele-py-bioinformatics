import torch
import pickle
import bio
from bio.Dataset import PDCC
from loguru import logger


def get_scaled_splitted_dataset(dataset: PDCC) -> bio.Dataset.Splitted:
    torch_dataset = PDCC.to_torch_dataset(config)
    trn, val, tst = config.train_validation_test_pecentages
    splitted = bio.Dataset.split_dataset(
        dataset = torch_dataset,
        train_percentage = trn,
        validation_percentage = val,
        test_percentage = tst,
        seed = config.seed,
    )
    if config._scaler_fn is None: return splitted
    
    X_np = torch_dataset.X.numpy()
    train_indices = splitted.train.indices
    train_X_np = X_np[train_indices]
    config.scaler = config._scaler_fn.fit(train_X_np)
    logger.info(f"config.scaler: {config.scaler}")
    config.save_scaler()
    X_scaled = config.scaler.transform(X_np)
    torch_dataset.X = torch.tensor(X_scaled, dtype=torch.float32)
    logger.info("Scaler fitted on train data and applied to all dataset splits.")
    splitted = bio.Dataset.split_dataset(
        dataset = torch_dataset,
        train_percentage = trn,
        validation_percentage = val,
        test_percentage = tst,
        seed = config.seed,
    )
    return splitted


def test_():
    config = PDCC.Config()
    splitted = get_splititted_dataset_from_config(config)
    assert len(splitted.train) > 0, "Train split should not be empty."
    assert len(splitted.validation) > 0, "Validation split should not be empty."
    assert len(splitted.test) > 0, "Test split should not be empty."
