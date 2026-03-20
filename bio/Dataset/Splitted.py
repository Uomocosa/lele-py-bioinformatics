import torch
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger

@dataclass
class Splitted:
    original: torch.utils.data.Dataset
    train: torch.utils.data.dataset.Subset
    validation: torch.utils.data.dataset.Subset
    test: torch.utils.data.dataset.Subset
    
    def scale(
        self, 
        feature_col_indexes: list[int],
        feature_attribute: str = "X",  
        scaler_fn = StandardScaler(),
    ) -> StandardScaler:
        """
        Scales specific feature columns in val and test datasets 
        based on the fit from the train dataset.
        """
        dataset = self.train.dataset
        logger.debug(f"dataset:\n{dataset}")
        if not hasattr(dataset, feature_attribute):
            raise AttributeError(
                f"Cannot scale: Dataset is missing the '{feature_attribute}' attribute. "
                f"Please provide the correct feature_attribute string."
            )
        feature_tensor = getattr(dataset, feature_attribute)
        cols = torch.tensor(feature_col_indexes, dtype=torch.long)
        train_rows = torch.tensor(self.train.indices, dtype=torch.long).unsqueeze(1)
        train_features = feature_tensor[train_rows, cols].numpy()       
        scaler = scaler_fn.fit(train_features)
        scaler.fit(train_features)
        for split in [self.train, self.validation, self.test]:
            if len(split) == 0: continue
            rows = torch.tensor(split.indices, dtype=torch.long).unsqueeze(1)
            features_to_scale = feature_tensor[rows, cols].numpy()
            scaled_features = scaler.transform(features_to_scale)
            feature_tensor[rows, cols] = torch.tensor(
                scaled_features, 
                dtype=feature_tensor.dtype
            )

        return scaler        
    
    
def from_datasets(
    train_dataset: torch.utils.data.Dataset,
    validation_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
) -> Splitted:
    train_len = len(train_dataset)
    val_len = len(validation_dataset)
    test_len = len(test_dataset)
    full_dataset = torch.utils.data.ConcatDataset([
        train_dataset, 
        validation_dataset, 
        test_dataset,
    ])
    train_indices = list(range(0, train_len))
    val_indices = list(range(train_len, train_len + val_len))
    test_indices = list(range(train_len + val_len, train_len + val_len + test_len))
    return Splitted(
        original=full_dataset,
        train=torch.utils.data.dataset.Subset(full_dataset, train_indices),
        validation=torch.utils.data.dataset.Subset(full_dataset, val_indices),
        test=torch.utils.data.dataset.Subset(full_dataset, test_indices)
    )


import pytest
@pytest.mark.todo
def test_():
    pass
