import pandas as pd
import bio
from bio.Dataset import PDCC, PDCCMethod
from bio.__global__ import PDCC_DATASET, PSMILES_DICT, SMILES_DICT, INTERPOLATED_CSV
from loguru import logger

def handle_dataframe(
    df: pd.DataFrame, 
    options: PDCC.Options = PDCC.Options(),
    featurize = PDCCMethod.featurize_v1,
) -> pd.DataFrame:
    n = options.n_points_to_interpolate_for_each_pair
    assert PDCC_DATASET.exists()
    df = pd.read_csv(PDCC_DATASET)
    df = bio.interpolate_combined_datasets(df, n)
    logger.debug(df)

    missing_polymers = set(df['POLYMER_USED']) - set(PSMILES_DICT.keys())
    missing_drugs = set(df['DRUG']) - set(SMILES_DICT.keys())
    
    if missing_polymers:
        logger.warning(f"Polymers missing from PSMILES_DICT: {missing_polymers}")
    if missing_drugs:
        logger.warning(f"Drugs missing from SMILES_DICT: {missing_drugs}")

    df['POLYMER_USED'] = df['POLYMER_USED'].map(PSMILES_DICT)
    df['DRUG'] = df['DRUG'].map(SMILES_DICT)
    
    df = featurize(df)

    logger.debug(df)
    df.to_csv(INTERPOLATED_CSV, index=False)
    return df


import pytest
@pytest.mark.todo
def test_():
    pass
