import numpy as np
from bio.ML import MLP, MLPMethod
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Subset
import copy
import torch
import bio
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
    
    eval_type = "LOOCV" if k == n_samples else f"{k}-Fold"
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
                scaled_prediction = model(x_val).cpu().item()
                scaled_target = y_val.item() if isinstance(y_val, torch.Tensor) else y_val
                if getattr(model, 'y_scaler', None) is not None:
                    prediction = model.y_scaler.inverse_transform([[scaled_prediction]])[0][0]
                    target = model.y_scaler.inverse_transform([[scaled_target]])[0][0]
                else:
                    prediction = scaled_prediction
                    target = scaled_target
                fold_predictions.append(prediction)
                fold_targets.append(target)
                poly_smiles = full_dataset.metadata.iloc[idx]['POLYMER_USED']
                drug_smiles = full_dataset.metadata.iloc[idx]['DRUG']
                logger.bind(
                    log_type="prediction_trace",
                    fold=current_fold,
                    sample_idx=int(idx),
                    drug=str(drug_smiles),
                    polymer=str(poly_smiles),
                    actual=float(target),
                    predicted=float(prediction)
                ).trace("eval_prediction")

        all_predictions.extend(fold_predictions)
        all_targets.extend(fold_targets)
        
        fold_mse = mean_squared_error(fold_targets, fold_predictions)
        fold_rmse = np.sqrt(fold_mse)
        fold_mae = mean_absolute_error(fold_targets, fold_predictions)
        fold_r2 = r2_score(fold_targets, fold_predictions) if len(fold_targets) > 1 else np.nan
        if eval_type == "LOOCV": log_msg = f"Fold {fold + 1} Result | MSE: {fold_mse:.4f}"
        else: log_msg = f"Fold {fold + 1} Result | MSE: {fold_mse:.4f} | R²: {fold_r2:.4f}"
        logger.bind(
            log_type="fold_metric_trace",
            fold=current_fold,
            mse=float(fold_mse),
            rmse=float(fold_rmse),
            mae=float(fold_mae),
            r2=float(fold_r2)
        ).trace("fold_metrics")
        logger.info(log_msg)

    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    if eval_type == "LOOCV": log_msg = f"=== LOOCV Result === | MSE: {mse:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}"
    else: log_msg = f"=== {eval_type} Result === | MSE: {mse:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}"
    logger.bind(
        log_type="aggregate_metrics",
        eval_method=eval_type,
        mse=float(mse),
        rmse=float(rmse),
        mae=float(mae),
        r2=float(r2)
    ).success(log_msg)
    return mse, rmse, mae, r2



def test_():
    import types
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import torch.nn.functional as F
    from bio.__global__ import PDCC_CSV
    from bio.ML import MLP, MLPMethod
    from bio.Dataset import PDCC, PDCCMethod
    seed = 42
    bio.ML.set_seed(seed)
    dataset_config = PDCC.Config(
        csv_file=PDCC_CSV,
        max_size=10,
        seed=seed,
    )
    model_config = MLP.Config(
        seed=seed,
    )
    x_scaler_fn = StandardScaler()
    y_scaler_fn = MinMaxScaler(feature_range=(0, 1))
    def forward_softplus_fn(mlp, x):
        x = mlp.model(x)
        return F.softplus(mlp.output(x))
    forward_fn = forward_softplus_fn
    k_fold = -1
    
    dataset = bio.Dataset.PDCC(config = dataset_config)
    dataset.increment_dataset()
    dataset.convert_names_to_smiles()
    torch_dataset = dataset.to_torch_dataset()
    x_sample, y_sample = torch_dataset[0]
    trn, val, tst = dataset_config.train_validation_test_pecentages
    splitted_dataset = bio.Dataset.split_dataset(
        dataset = torch_dataset,
        train_percentage = trn,
        validation_percentage = val,
        test_percentage = tst,
        seed = seed,
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
        config = model_config,
    )
    model.forward = types.MethodType(forward_fn, model)
    
    MLPMethod.k_fold_cross_validation(model, k=k_fold)
