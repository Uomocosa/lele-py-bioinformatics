import pandas as pd
import torch
import bio
from bio.Dataset import PDCCMethod
from loguru import logger


def handle_input_to_predict(
    polymer_psmile: str, 
    drug_smile: str, 
    concentration: float,
    featurize = PDCCMethod.featurize_v1,
) -> torch.Tensor:
    """
    Transforms raw P-SMILES, SMILES and concentration into a featurized tensor 
    matching the training distribution.
    """
    df = pd.DataFrame([{
        'POLYMER_USED': polymer_psmile,
        'DRUG': drug_smile,
        'CONCENTRATION': concentration
    }])
    df = featurize(df)
    return torch.tensor(df.values, dtype=torch.float32).squeeze(0)


import pytest
@pytest.mark.todo
def test_():
    pass
