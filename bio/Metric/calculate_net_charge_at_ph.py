import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from dimorphite_dl import protonate_smiles
import bio
from bio.Bioinformatics.transform_into_smiles import DEFAULT_CAPPING_ATOMS
from loguru import logger

def calculate_net_charge_at_ph(
    df: pd.DataFrame, 
    column_name: str, 
    capping_atoms_dict: dict = DEFAULT_CAPPING_ATOMS,
    ph_col: str = 'WATER_PH', 
    precision: float = 1.0,
    starting_label: str = ''
) -> pd.DataFrame:
    """
    Calculates the RDKit formal charge of a molecule at a specific pH.
    Charge can assume values like: ..., -1, 0, 1, 2, ... (not sure if there is a max or min)
    """
    null_result = pd.Series({})
    label = f"{starting_label}charge_at_{ph_col}" if starting_label else f"charge_at_{ph_col}"
    
    def get_charge(row):
        smiles_str = str(row[column_name])
        valid_smiles_dict = bio.Bioinformatics.transform_into_smiles(smiles_str, capping_atoms_dict)
        if not valid_smiles_dict: return null_result
        target_ph = row[ph_col]
        if pd.isna(smiles_str) or pd.isna(target_ph): return null_result
        
        df_dict = dict()
        for atom, smile in valid_smiles_dict.items():
            protonated_mols = protonate_smiles(
                smile,
                ph_min=target_ph, 
                ph_max=target_ph, 
                precision=precision,
            )
            logger.debug(f"atom: {atom}")
            logger.debug(f"smile: {smile}")
            logger.debug(f"protonated_mols: {protonated_mols}")
            # Convert the most dominant protonated SMILES to an RDKit Mol
            # Dimorphite usually returns the most probable states first
            mol = Chem.MolFromSmiles(protonated_mols[0])
            if not mol: continue
            key = f"{label}_{atom}"
            key = key.removesuffix('_')
            df_dict[key] = rdmolops.GetFormalCharge(mol)
        logger.debug(f"net_charge values: {df_dict}")
        return pd.Series(df_dict)
        
    # Apply the calculation
    df_out = df.copy()
    charge_df = df_out.apply(get_charge, axis=1)
    df_out = pd.concat([df_out, charge_df], axis=1)
    return df_out


import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import CONVERTED_PDCC_CSV
    from bio.Metric.__global__ import HELPER_DIR
    csv_file = HELPER_DIR / "calculate_net_charge_at_ph_generated.csv"
    df = pd.read_csv(CONVERTED_PDCC_CSV)
    df = df.head(100)
    df = calculate_net_charge_at_ph(
        df, 
        column_name="POLYMER_USED", 
    )
    df = calculate_net_charge_at_ph(
        df, 
        column_name="DRUG", 
    )
    print(df)
    df.to_csv(csv_file, index=False)
