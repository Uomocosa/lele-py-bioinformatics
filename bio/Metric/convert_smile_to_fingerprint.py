import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from bio.Bioinformatics.transform_into_smiles import DEFAULT_CAPPING_ATOMS
import bio
from loguru import logger


def convert_smile_to_fingerprint(
    df: pd.DataFrame, 
    column_name: str,
    capping_atoms_dict: dict = DEFAULT_CAPPING_ATOMS,
    radius: int = 2,
    nBits: int = 2048,
    starting_lable: str = '',
) -> pd.DataFrame:
    RDLogger.DisableLog('rdApp.*') # disable warnings
    df_out = df.copy() 
    fn = lambda s: single_calculation(
        s, 
        capping_atoms_dict,
        radius=radius,
        nBits=nBits,
        starting_lable=starting_lable,
    )
    stats_df = df_out[column_name].apply(fn)
    df_out = pd.concat([df_out, stats_df], axis=1)
    return df_out
    

def single_calculation(
    smiles_str: str, 
    capping_atoms_dict: dict, 
    radius: int,
    nBits: int,
    starting_lable: str,
):
    smiles_str = str(smiles_str)
    null_result = pd.Series({})
    if not smiles_str: return null_result
    lable = f'{starting_lable}_fingerprint'
    if not starting_lable: lable = lable.removeprefix("_")
        
    valid_smiles_dict = bio.Bioinformatics.transform_into_smiles(smiles_str, capping_atoms_dict)
    if not valid_smiles_dict: return null_result
    
    df = dict()
    for atom, smile in valid_smiles_dict.items():
        mol = Chem.MolFromSmiles(smile)
        if not mol: continue
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        key = f"{lable}_{atom}".removesuffix('_')
        df[key] = fingerprint.ToBitString()
    logger.trace(f"fingerprints: {df}")
    return pd.Series(df)


import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "convert_smile_to_fingerprint_generated.csv"
    df = pd.read_csv(dataset_csv)
    df = convert_smile_to_fingerprint(df.head(100), column_name="PSMILES")
    print(df)
    df.to_csv(csv_file, index=False)
