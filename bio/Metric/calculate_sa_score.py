import pandas as pd
from rdkit import Chem

import bio
from bio.Bioinformatics.transform_into_smiles import DEFAULT_CAPPING_ATOMS
from loguru import logger


def calculate_sa_score(
    df: pd.DataFrame, 
    column_name: str, 
    capping_atoms_dict: dict = DEFAULT_CAPPING_ATOMS,
    starting_lable: str = '',
) -> pd.DataFrame:
    df_out = df.copy()
    fn = lambda s: single_calculation(
        s,
        capping_atoms_dict,
        starting_lable,
    )
    stats_df = df_out[column_name].apply(fn)
    df_out = pd.concat([df, stats_df], axis=1)
    return df_out


def single_calculation(
    smiles_str: str, 
    capping_atoms_dict: dict,
    starting_lable: str,
) -> pd.Series:
    smiles_str = str(smiles_str)
    lable = f'{starting_lable}_sa_score'
    if not starting_lable: lable = lable.removeprefix("_")
    null_result = pd.Series({})
    if not smiles_str: return null_result
    
    valid_smiles_dict = bio.Bioinformatics.transform_into_smiles(smiles_str, capping_atoms_dict)
    if not valid_smiles_dict: return null_result
    
    df = dict()
    for atom, smile in valid_smiles_dict.items():
        mol = Chem.MolFromSmiles(smile)
        if mol is None: continue
        key = f"{lable}_{atom}"
        key = key.removesuffix('_')
        sa_score = bio.PolyGen.calculate_sa_score(mol)
        if sa_score is None: continue
        df[key] = sa_score
        
    logger.debug(f"sa scores: {df}")
    return pd.Series(df)


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
