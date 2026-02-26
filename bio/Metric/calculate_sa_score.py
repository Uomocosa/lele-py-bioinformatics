import pandas as pd
from rdkit import Chem

import bio
from loguru import logger


def calculate_sa_score(
    df: pd.DataFrame, 
    column_name: str, 
) -> pd.DataFrame:
    stats_df = df[column_name].apply(compute_min_max_sa_score)
    df_out = pd.concat([df, stats_df], axis=1)
    return df_out

def compute_min_max_sa_score(smiles_str: str) -> pd.Series:
    min_lable = f'min_sa_score'
    max_lable = f'max_sa_score'
    null_result = pd.Series({min_lable: None, max_lable: None})
    smiles_str = str(smiles_str)
    valid_smiles_list = bio.Bioinformatics.transform_into_smiles(smiles_str)
    if not valid_smiles_list: return null_result
    mols = [Chem.MolFromSmiles(smile) for smile in valid_smiles_list]
    scores = [bio.PolyGen.calculate_sa_score(mol) for mol in mols]
    scores = [sa for sa in scores if sa is not None]
    if not scores: return null_result
    logger.debug(f'scores: {scores}')
    return pd.Series({min_lable: min(scores), max_lable: max(scores)})

import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "calculate_sa_score_generated.csv"
    df = pd.read_csv(dataset_csv).head(10) # Start small, xTB is ~1000x slower than RDKit
    df = calculate_sa_score(df, column_name="PSMILES")
    print(df)
    df.to_csv(csv_file, index=False)
