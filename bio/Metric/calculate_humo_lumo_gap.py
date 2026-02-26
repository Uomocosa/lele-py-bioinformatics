import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from dimorphite_dl import protonate_smiles
import bio
from loguru import logger


"""
Based on:
    https://tblite.readthedocs.io/en/latest/tutorial/python/singlepoint.html#homo-lumo-gap

# TODO !!!
"""
def calculate_humo_lumo_gap(
    df: pd.DataFrame, 
    column_name: str, 
    target_smile: str,
) -> pd.DataFrame:
    RDLogger.DisableLog('rdApp.*') # disable RDKit warnings
    df_out = df.copy()    
    stats_df = df_out[column_name].apply(lambda s: compute_min_max_humo_lumo_gap(s, target_smile))
    df_out = pd.concat([df_out, stats_df], axis=1)
    return df_out


def compute_min_max_humo_lumo_gap(smiles_str: str, target_smile: str) -> pd.Series:
    """
    Protonates a single SMILES string across a pH range and 
    calculates the min/max logD.
    """
    smiles_str = str(smiles_str)
    min_lable = f'min_logd_pH_{ph_min}-{ph_max}'
    max_lable = f'max_logd_pH_{ph_min}-{ph_max}'
    null_result = pd.Series({min_lable: None, max_lable: None})
    if pd.isna(smiles_str): return null_result
    valid_smiles_list = bio.Bioinformatics.transform_into_smiles(smiles_str)
    if not valid_smiles_list: return null_result
    protonated_mols = protonate_smiles(valid_smiles_list, ph_min=ph_min, ph_max=ph_max, precision=precision)
    protonated_mols = [Chem.MolFromSmiles(p_mol) for p_mol in protonated_mols]
    protonated_mols = [p_mol for p_mol in protonated_mols if p_mol]
    logd_values = [Descriptors.MolLogP(p_mol) for p_mol in protonated_mols]
    logger.debug(f"logd_values: {logd_values}")
    if not logd_values: return null_result
    return pd.Series({min_lable: min(logd_values), max_lable: max(logd_values)})


import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "calculate_humo_lumo_energy_generated.csv"
    df = pd.read_csv(dataset_csv)
    df = calculate_humo_lumo_energy(
        df.head(100), 
        column_name="PSMILES", 
        ph_min = 6.8,
        ph_max = 7.9,
        precision = 0.5,
    )
    print(df)
    df.to_csv(csv_file, index=False)
