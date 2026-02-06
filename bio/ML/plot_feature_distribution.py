import torch
from typing import List, Optional
import matplotlib.pyplot as mplot
import lele, bio
from bio import Dataset
from loguru import logger

"""
Dataset.Splitted has this form:
    train: torch.utils.data.Dataset
    validation: torch.utils.data.Dataset
    test: torch.utils.data.Dataset
"""

def plot_feature_distribution(dataset: Dataset.Splitted, feature_names: List[str]):
    nrows = len(feature_names)
    ncols = 3 # train, val, test
    fig, axes = mplot.subplots(nrows=nrows, ncols=ncols, figsize=(15, 3.5 * nrows))
    fig.suptitle('', fontsize=25, y=0.98)    
    if len(feature_names) == 1: axes = axes.reshape(1, -1)
    colors = {'train': '#1f77b4', 'val': '#ff7f0e', 'test': '#2ca02c'}
    common_args = dict(bins=20, alpha=0.5, density=True, histtype='stepfilled')
    features = get_features_dict(dataset, feature_names)
    for i, (feature_name, values) in enumerate(features.items()):
        for j, (key, data) in enumerate(values.items()):
            assert key in ["train", "val", "test"]
            color = colors[key]
            ax = axes[i, j]
            ax.hist(data, **common_args, label=key.title(), color=color, edgecolor=color)
            if i == 0: ax.set_title(key.title(), fontsize=14, fontweight='bold')
            if j == 0: ax.set_ylabel(f"{feature_name}\nDensity", fontsize=12, fontweight='bold')
            if i == nrows - 1: ax.set_xlabel('Value')
    return ax
    
def get_features_dict(dataset: Dataset.Splitted, feature_names: List[str]):
    features = dict()
    train_features = torch.stack([x[1] for x in dataset.train]).numpy()
    val_features   = torch.stack([x[1] for x in dataset.validation]).numpy()
    test_features  = torch.stack([x[1] for x in dataset.test]).numpy()
    for i, feature in enumerate(feature_names):
        features[feature] = dict()
        train_values = train_features[:, i]
        val_values   = val_features[:, i]
        test_values  = test_features[:, i]
        features[feature]["train"] = train_values
        features[feature]["val"] = val_values
        features[feature]["test"] = test_values
    return features

def test_numbers():
    from bio.Dataset.__global__ import ZINC_ZSCORE_CSV
    from bio.ML.__global__ import HELPER_DIR
    import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)
    p_train, p_val, p_test = (0.7, 0.15, 0.15)
    labled_smiles = bio.Dataset.LabeledSmiles.load(ZINC_ZSCORE_CSV, max_dataset_size=100)
    entire_dataset = labled_smiles.to_torch_dataset(block_size=8)
    print(f"Example torch.Dataset: {entire_dataset[0]}")
    dataset = bio.Dataset.split_dataset(entire_dataset, p_train, p_val, p_test, seed=42)
    print(f"Example torch.Dataset (train): {dataset.train[0]}")
    features = get_features_dict(dataset, labled_smiles.feature_names)
    for feature, values in features.items():
        train_values = values["train"]
        val_values   = values["val"]
        test_values  = values["test"]
        print(f"MEAN {feature}:")
        print(f"\ttrain: {train_values.mean():.3f}")
        print(f"\tval: {val_values.mean():.3f}")
        print(f"\ttest: {test_values.mean():.3f}")
        print(f"STANDARD DEVIATION {feature}:")
        print(f"\ttrain: {train_values.std():.3f}")
        print(f"\tval: {val_values.std():.3f}")
        print(f"\ttest: {test_values.std():.3f}")
        print()
    
    
import pytest
@pytest.mark.verbose
def test_plot_100():
    _testing_function(max_dataset_size=100)
    
import pytest
@pytest.mark.verbose
@pytest.mark.above10s
def test_plot_all():
    _testing_function(max_dataset_size=None)

def _testing_function(max_dataset_size: Optional[int]):
    from bio.Dataset.__global__ import ZINC_ZSCORE_CSV
    from bio.ML.__global__ import HELPER_DIR
    import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)
    p_train, p_val, p_test = (0.7, 0.15, 0.15)
    labled_smiles = bio.Dataset.LabeledSmiles.load(ZINC_ZSCORE_CSV, max_dataset_size=max_dataset_size)
    entire_dataset = labled_smiles.to_torch_dataset(block_size=8)
    print(f"Example torch.Dataset: {entire_dataset[0]}")
    dataset = bio.Dataset.split_dataset(entire_dataset, p_train, p_val, p_test, seed=42)
    print(f"Example torch.Dataset (train): {dataset.train[0]}")
    ax = plot_feature_distribution(dataset, feature_names=labled_smiles.feature_names)
    # ax.legend()
    mplot.tight_layout()
    if not max_dataset_size: name = "fulldataset"
    else: name = str(max_dataset_size)
    mplot.savefig(HELPER_DIR/('distribution_check_'+name+'.png'))
    mplot.show()
