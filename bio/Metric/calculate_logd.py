import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from dimorphite_dl import protonate_smiles
import bio
from bio.Bioinformatics.transform_into_smiles import DEFAULT_CAPPING_ATOMS
from loguru import logger


"""
Calculates the minimum and maximum logD for a given pH range and appends
it to the DataFrame.
"""
def calculate_logd(
    df: pd.DataFrame, 
    column_name: str, 
    ph_min: float = 7.0, 
    ph_max: float = 7.4, 
    precision: float = 1.0,
    capping_atoms_dict: dict = DEFAULT_CAPPING_ATOMS,
    starting_lable: str = '',
) -> pd.DataFrame:
    RDLogger.DisableLog('rdApp.*') # disable RDKit warnings
    df_out = df.copy()
    if starting_lable: starting_lable = starting_lable + "_"
    if ph_min == ph_max: starting_lable = f'{starting_lable}logd_pH_{ph_max}'
    else: starting_lable = f'{starting_lable}logd_pH_{ph_min}-{ph_max}'
    fn = lambda s: compute_most_probable_logd(
        s, 
        ph_min, 
        ph_max, 
        precision, 
        capping_atoms_dict,
        starting_lable,
    )
    stats_df = df_out[column_name].apply(fn)
    df_out = pd.concat([df_out, stats_df], axis=1)
    return df_out


def compute_most_probable_logd(
    smiles_str: str, 
    ph_min: float, 
    ph_max: float, 
    precision: float,
    capping_atoms_dict: dict,
    starting_lable: str,
) -> pd.Series:
    """
    Protonates a single SMILES string across a pH range and 
    calculates the min/max logD.
    """
    smiles_str = str(smiles_str)
    null_result = pd.Series({})
    if pd.isna(smiles_str): return null_result
    lable = starting_lable
    
    valid_smiles_dict = bio.Bioinformatics.transform_into_smiles(smiles_str, capping_atoms_dict)
    if not valid_smiles_dict: return null_result
    # protonated_mols = protonate_smiles(valid_smiles_dict, ph_min=ph_min, ph_max=ph_max, precision=precision)
    df = dict()
    for atom, smile in valid_smiles_dict.items():
        protonated_mols = protonate_smiles(
            smile,
            ph_min=ph_min, 
            ph_max=ph_max, 
            precision=precision,
        )
        logger.debug(f"atom: {atom}")
        logger.debug(f"smile: {smile}")
        logger.debug(f"protonated_mols: {protonated_mols}")
        # Convert the most dominant protonated SMILES to an RDKit Mol
        # Dimorphite usually returns the most probable states first
        mol = Chem.MolFromSmiles(protonated_mols[0])
        if not mol: continue
        key = f"{lable}_{atom}"
        key = key.removesuffix('_')
        df[key] = Descriptors.MolLogP(mol)
    logger.debug(f"logd values: {df}")
    return pd.Series(df)


def compute_min_max_logd(
    smiles_str: str, 
    ph_min: float, 
    ph_max: float, 
    precision: float,
    capping_atoms_dict: dict,
    starting_lable: str,
) -> pd.Series:
    logger.warn("DEPRECATED! Use compute_most_probable_logd instead!")
    """
    DEPRECATED!
    Protonates a single SMILES string across a pH range and 
    calculates the min/max logD.
    """
    smiles_str = str(smiles_str)
    null_result = pd.Series({})
    if pd.isna(smiles_str): return null_result
    lable = starting_lable
    
    valid_smiles_dict = bio.Bioinformatics.transform_into_smiles(smiles_str, capping_atoms_dict)
    if not valid_smiles_dict: return null_result
    # protonated_mols = protonate_smiles(valid_smiles_dict, ph_min=ph_min, ph_max=ph_max, precision=precision)
    df = dict()
    for atom, smile in valid_smiles_dict.items():
        protonated_mols = protonate_smiles(
            smile,
            ph_min=ph_min, 
            ph_max=ph_max, 
            precision=precision,
        )
        logger.debug(f"atom: {atom}")
        logger.debug(f"smile: {smile}")
        logger.debug(f"protonated_mols: {protonated_mols}")
        mols = [Chem.MolFromSmiles(p_mol) for p_mol in protonated_mols]
        logd_values = [Descriptors.MolLogP(mol) for mol in mols]
        if not mols: continue
        key = f"{lable}_{atom}"
        key = key.removesuffix('_')
        
        df[f"min_{key}"] = min(logd_values)
        df[f"max_{key}"] = max(logd_values)
    logger.debug(f"logd values: {df}")
    return pd.Series(df)


import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "calculate_logd_generated.csv"
    df = pd.read_csv(dataset_csv)
    df = calculate_logd(
        df.head(100), 
        column_name="PSMILES", 
        ph_min = 6.8,
        ph_max = 7.9,
        precision = 0.5,
    )
    print(df)
    df.to_csv(csv_file, index=False)

import pytest
@pytest.mark.above10s
def test_no_ph_range():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "calculate_logd_no_ph_range_generated.csv"
    df = pd.read_csv(dataset_csv)
    df = calculate_logd(
        df.head(1000), 
        column_name="PSMILES", 
        ph_min = 7.0,
        ph_max = 7.0,
        precision = 1.0,
    )
    print(df)
    df.to_csv(csv_file, index=False)
