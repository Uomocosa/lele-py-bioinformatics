from bio.ML import MLP
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import copy
from loguru import logger


def train_model(model: MLP):
    torch.manual_seed(model.config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    logger.info(f"Training on device: {device}")
    model.to(device)
    train_ds = model.data.train
    val_ds = model.data.validation
    logger.info(f"Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}")

    train_loader = DataLoader(
        train_ds, 
        batch_size=model.config.batch_size, 
        shuffle=True, 
        pin_memory=use_cuda,
        num_workers=model.config.num_workers if use_cuda else 0
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=model.config.batch_size, 
        shuffle=False, 
        pin_memory=use_cuda,
        num_workers=model.config.num_workers if use_cuda else 0
    )
    optimizer = optim.Adam(model.parameters(), lr=model.config.learning_rate)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_weights = None

    for epoch in range(model.config.epochs):
        model.train() 
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            predictions = model(batch_x)
            loss = model.config.criterion(predictions, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
        
        train_loss /= len(train_ds)

        model.eval() 
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = model.config.criterion(predictions, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        
        val_loss /= len(val_ds)
        
        is_best = val_loss < best_val_loss
        if (epoch + 1) % 10 == 0 or epoch == 0 or is_best:
            log_msg = f"Epoch {epoch+1:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}"
            if is_best: log_msg = f"<green>{log_msg}</green>"
            logger.opt(colors=True).info(log_msg)
        
        if model.config.early_stop_patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= model.config.early_stop_patience:
                    logger.warning(f"EARLY STOPPING: Validation loss hasn't improved for {model.config.early_stop_patience} epochs.")
                    break 

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        logger.info(f"Restored best model weights with Val MSE: {best_val_loss:.4f}")

    logger.success("Training complete!")
    return model


import pytest
@pytest.mark.todo
def test_():
    pass
