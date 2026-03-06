
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from bio.ML.MLP import MLP
from bio.utils.logger import logger


def check_model_accuracy(model: MLP) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_ds = model.data.test
    if not test_ds or len(test_ds) == 0:
        logger.warning("Test dataset is empty or not found.")
        return

    logger.info(f"Evaluating on Test Dataset (Size: {len(test_ds)})")
    test_loader = DataLoader(test_ds, batch_size=model.config.batch_size, shuffle=False)
    
    all_preds = []
    all_targets = []

    # Disable gradient calculation for faster, memory-efficient inference
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            
            all_preds.append(predictions)
            all_targets.append(batch_y)

    # Combine all batches into single 1D tensors
    preds_tensor = torch.cat(all_preds).squeeze()
    targets_tensor = torch.cat(all_targets).squeeze()

    # Calculate Regression Metrics
    mse = F.mse_loss(preds_tensor, targets_tensor).item()
    mae = F.l1_loss(preds_tensor, targets_tensor).item()
    rmse = mse ** 0.5
    
    # Calculate R-squared (Coefficient of Determination)
    target_mean = torch.mean(targets_tensor)
    ss_tot = torch.sum((targets_tensor - target_mean) ** 2)
    ss_res = torch.sum((targets_tensor - preds_tensor) ** 2)
    r2 = (1 - (ss_res / ss_tot)).item()

    # Log the results
    logger.info("=== Test Set Evaluation Results ===")
    logger.info(f"MSE  (Mean Squared Error)      : {mse:.4f}")
    logger.info(f"RMSE (Root Mean Squared Error) : {rmse:.4f}")
    logger.info(f"MAE  (Mean Absolute Error)     : {mae:.4f}")
    logger.info(f"R² Score                       : {r2:.4f}")
    
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


import pytest
@pytest.mark.todo
def test_():
    pass
