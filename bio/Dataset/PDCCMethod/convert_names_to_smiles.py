import pandas as pd
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
import bio
from loguru import logger
from bio.__global__ import CONVERTED_PDCC_CSV, PSMILES_DICT, SMILES_DICT

@dataclass
class Options:
    psmiles_dict: dict = field(default_factory=lambda: PSMILES_DICT)
    smiles_dict: dict = field(default_factory=lambda: SMILES_DICT)

from bio.__global__ import CACHE_MEMORY
@CACHE_MEMORY.cache
def convert_names_to_smiles(df: pd.DataFrame, options=Options()):
    psmiles_dict_lower = {str(k).lower(): v for k, v in options.psmiles_dict.items()}
    smiles_dict_lower = {str(k).lower(): v for k, v in options.smiles_dict.items()}
    
    poly_lower = df['POLYMER_USED'].astype(str).str.lower()
    drug_lower = df['DRUG'].astype(str).str.lower()
    
    missing_polymers = {p for p in poly_lower if is_missing_or_empty(p, psmiles_dict_lower)}
    missing_drugs = {d for d in drug_lower if is_missing_or_empty(d, smiles_dict_lower)}
    
    if missing_polymers: logger.warning(f"Polymers missing from PSMILES_DICT!:\n{missing_polymers}")
    if missing_drugs: logger.warning(f"Molecules missing from SMILES_DICT!:\n{missing_drugs}")

    df['POLYMER_USED'] = poly_lower.map(psmiles_dict_lower)
    df['DRUG'] = drug_lower.map(smiles_dict_lower)
    
    # Convert empty strings (""), spaces ("   "), and literal "nan" strings to true np.nan
    df['POLYMER_USED'] = df['POLYMER_USED'].replace(r'^\s*$', np.nan, regex=True).replace(['nan', 'NaN', 'None'], np.nan)
    df['DRUG'] = df['DRUG'].replace(r'^\s*$', np.nan, regex=True).replace(['nan', 'NaN', 'None'], np.nan)
    df = df.dropna()
    return df
    
    
    
def is_missing_or_empty(item, mapping_dict):
    # 1. Ignore items that are already NaN, empty, or literal "nan" in the dataset
    if pd.isna(item) or str(item).strip().lower() in ["", "nan", "none", "null"]:
        return False 
    
    # 2. If the valid item is completely missing from the dictionary
    if item not in mapping_dict:
        return True
        
    # 3. If the item is in the dictionary, but the mapped value is invalid/empty
    val = mapping_dict[item]
    if pd.isna(val) or str(val).strip().lower() in ["", "nan", "none", "null"]:
        return True
        
    return False

    

def test_():
    from bio.__global__ import PDCC_CSV
    from bio.Dataset import PDCCMethod
    df = pd.read_csv(PDCC_CSV)
    df = PDCCMethod.increment_dataset(
        df, 
        PDCCMethod.increment_dataset.Options(
            method="interpolate_then_add_origins", 
            n_points=10
        )
    )
    df = convert_names_to_smiles(df)
    print(f"df.head(): {df.head()}")
