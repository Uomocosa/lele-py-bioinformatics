import numpy as np
from bio.ML import MLP, MLPMethod
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Subset
import copy
import torch
from loguru import logger


def k_fold_cross_validation(model: MLP, k: int = 5):
    full_dataset = model.data.original
    n_samples = len(full_dataset)
    if k == -1: 
        k = n_samples
        kf = LeaveOneOut()
        logger.info(f"Running LOOCV for {n_samples} samples...")
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=model.config.seed)
    
    untrained_weights = copy.deepcopy(model.state_dict())
    
    all_targets = []
    all_predictions = []
    
    logger.info(f"K-Fold Training for {n_samples} samples (k={k})...")
    for fold, (train_index, val_index) in enumerate(kf.split(range(n_samples))):
        current_fold = fold + 1
        logger.info(f"--- K-Fold {current_fold}/{k} ---")
        model.load_state_dict(copy.deepcopy(untrained_weights))
        model.data.train = Subset(full_dataset, train_index.tolist())
        model.data.validation = Subset(full_dataset, val_index.tolist())
        
        MLPMethod.train_model(model)
        
        model.eval()
        fold_predictions = []
        fold_targets = []
        
        with torch.no_grad():
            device = next(model.parameters()).device
            for idx in val_index:
                x_val, y_val = full_dataset[idx]
                x_val = x_val.unsqueeze(0).to(device)
                prediction = model(x_val).cpu().item()
                target = y_val.item() if isinstance(y_val, torch.Tensor) else y_val
                fold_predictions.append(prediction)
                fold_targets.append(target)
                logger.bind(
                    log_type="prediction_trace",
                    fold=current_fold,
                    sample_idx=int(idx),
                    actual=float(target),
                    predicted=float(prediction)
                ).trace("eval_prediction")

        all_predictions.extend(fold_predictions)
        all_targets.extend(fold_targets)
        
        fold_mse = mean_squared_error(fold_targets, fold_predictions)
        fold_rmse = np.sqrt(fold_mse)
        fold_mae = mean_absolute_error(fold_targets, fold_predictions)
        fold_r2 = r2_score(fold_targets, fold_predictions) if len(fold_targets) > 1 else np.nan
        logger.bind(
            log_type="fold_metric_trace",
            fold=current_fold,
            mse=float(fold_mse),
            rmse=float(fold_rmse),
            mae=float(fold_mae),
            r2=float(fold_r2)
        ).trace("fold_metrics")
        logger.info(f"Fold {fold + 1} Result | MSE: {fold_mse:.4f} | R²: {fold_r2:.4f}")

    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    eval_type = "LOOCV" if k == n_samples else f"{k}-Fold"
    logger.bind(
        log_type="aggregate_metrics",
        eval_method=eval_type,
        mse=float(mse),
        rmse=float(rmse),
        mae=float(mae),
        r2=float(r2)
    ).success(
        f"=== {eval_type} Results === | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}"
    )
    return mse, rmse, mae, r2

import pytest
@pytest.mark.todo
def test_():
    pass
