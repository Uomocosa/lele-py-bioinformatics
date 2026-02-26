import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import bio
from loguru import logger

def calculate_logp(df_smiles_and_psmiles: pd.DataFrame, column_name: str) -> pd.DataFrame:
    RDLogger.DisableLog('rdApp.*') # disable warnings
    df = df_smiles_and_psmiles.copy()    
    stats_df = df[column_name].apply(compute_min_max_logp)
    df = pd.concat([df, stats_df], axis=1)
    return df


"""
NOTE! Made using AI.
"""
def compute_min_max_logp(smiles_str):
        # Default return if something fails
        smiles_str = str(smiles_str)
        null_result = pd.Series({'min_logp': None, 'max_logp': None})
        if pd.isna(smiles_str): return null_result
            
        # 1. Get the list of all valid capped SMILES
        valid_smiles_list = bio.Bioinformatics.transform_into_smiles(smiles_str)
        
        if not valid_smiles_list: return null_result
            
        # 2. Calculate logp for EVERY valid variant
        logp_values = []
        for chosen_smiles in valid_smiles_list:
            mol = Chem.MolFromSmiles(chosen_smiles)
            if mol is not None: 
                try:
                    val = Descriptors.MolLogP(mol)
                    logp_values.append(val)
                except Exception:
                    pass
                    
        # 3. Find the minimum and maximum values
        if logp_values:
            return pd.Series({
                'min_logp': min(logp_values), 
                'max_logp': max(logp_values)
            })
        else:
            return null_result


import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "calculate_logp_generated.csv"
    df = pd.read_csv(dataset_csv)
    df = calculate_logp(df.head(100), column_name="PSMILES")
    print(df)
    df.to_csv(csv_file, index=False)
