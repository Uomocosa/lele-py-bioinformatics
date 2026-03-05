import pandas as pd
import bio

def featurize_v1(df: pd.DataFrame) -> pd.DataFrame:
    polymer_metrix = bio.Metric.featurize_psmiles(df[['POLYMER_USED']], "POLYMER_USED")
    molecule_metrix = bio.Metric.featurize_smiles(df[['DRUG']], "DRUG")
    polymer_metrix = polymer_metrix.drop(columns=['POLYMER_USED'])
    molecule_metrix = molecule_metrix.drop(columns=['DRUG'])
    polymer_metrix = polymer_metrix.add_prefix('poly_')
    molecule_metrix = molecule_metrix.add_prefix('drug_')
    df = pd.concat([
        polymer_metrix, 
        molecule_metrix,
        df.drop(columns=['POLYMER_USED', 'DRUG']),
    ], axis=1)
    df = df.dropna()
    return df


import pytest
@pytest.mark.todo
def test_():
    pass
