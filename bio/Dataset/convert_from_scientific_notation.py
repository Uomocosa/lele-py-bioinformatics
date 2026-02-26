import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
import bio
from loguru import logger

def convert_from_scientific_notation(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df[column_name] = df[column_name].apply(format_float)
    return df
    
def format_float(val):
    if pd.isna(val): return ""
    # Format to 10 decimal places and strip unnecessary trailing zeros/dots
    return f"{val:.10f}".rstrip('0').rstrip('.')


import pytest
@pytest.mark.above10s
def test_generated():
    from bio.Dataset.__global__ import HELPER_DIR
    dataset_csv = HELPER_DIR / "polymer_drug_concentration_capacity.csv"
    csv_file = HELPER_DIR / "converted.csv"
    assert dataset_csv.exists()
    df = pd.read_csv(dataset_csv)
    df = convert_from_scientific_notation(df, column_name="CAPACITY")
    print(df)
    df.to_csv(csv_file, index=False)
