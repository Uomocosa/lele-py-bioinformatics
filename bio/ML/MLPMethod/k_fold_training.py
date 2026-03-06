import numpy as np
from bio.ML import MLP, MLPMethod
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Subset
import copy
import torch
from loguru import logger


def k_fold_training(model: MLP, k: int = 5):
    """
    Evaluates the MLP using K-Fold Cross-Validation.
    If we set k = 1 -> We get the Leave-One-Out Cross-Validation (LOOCV) equivalent.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=model.config.seed)
    full_dataset = model.data.original
    n_samples = len(full_dataset)
    untrained_weights = copy.deepcopy(model.state_dict())
    all_targets = []
    all_predictions = []
    logger.info(f"K-Fold Training for {n_samples} samples (k={k})...")
    
    for fold, (train_index, val_index) in enumerate(kf.split(range(n_samples))):
        logger.info(f"--- K-Fold {fold + 1}/{n_samples} ---")
        model.load_state_dict(copy.deepcopy(untrained_weights))
        model.data.train = Subset(full_dataset, train_index.tolist())
        model.data.validation = Subset(full_dataset, val_index.tolist())
        MLPMethod.train_model(model)
        model.eval()
        with torch.no_grad():
            device = next(model.parameters()).device
            x_val, y_val = full_dataset[val_index[0]]
            x_val = x_val.unsqueeze(0).to(device) # Add batch dimension
            prediction = model(x_val).cpu().item()
            target = y_val.item() if isinstance(y_val, torch.Tensor) else y_val
            all_predictions.append(prediction)
            all_targets.append(target)
            logger.info(f"Fold {fold + 1} Result | Target: {target:.4f} | Predicted: {prediction:.4f}")
    
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    logger.success("=== LOOCV Evaluation Results ===")
    logger.info(f"Aggregate MSE  : {mse:.4f}")
    logger.info(f"Aggregate RMSE : {rmse:.4f}")
    logger.info(f"Aggregate MAE  : {mae:.4f}")
    logger.info(f"Aggregate R²   : {r2:.4f}")
    return mse, rmse, mae, r2


import pytest
@pytest.mark.todo
def test_():
    pass
