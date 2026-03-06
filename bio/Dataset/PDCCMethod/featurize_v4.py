from dataclasses import dataclass, field
import torch.nn as nn
from typing import Optional
import pandas as pd
from rdkit import Chem
import bio
from bio.Bioinformatics.transform_into_smiles import DEFAULT_CAPPING_ATOMS


@dataclass
class Options:
    capping_atoms_dict: dict = field(default_factory=lambda: DEFAULT_CAPPING_ATOMS)
    fingerprint_radius: int = 2
    fingerprint_n_bits: int = 2048

def featurize_v4(
    df: pd.DataFrame,
    options: Options = Options()
) -> pd.DataFrame:
    polymer_features = df.copy()
    molecule_features = df.copy()
    polymer_features = bio.Metric.convert_smile_to_fingerprint(
        polymer_features, 
        column_name = 'DRUG',
        capping_atoms_dict = options.capping_atoms_dict,
        radius = options.fingerprint_radius,
        nBits = options.fingerprint_n_bits,
    )
    molecule_features = bio.Metric.convert_smile_to_fingerprint(
        molecule_features, 
        column_name = 'POLYMER_USED',
        capping_atoms_dict = options.capping_atoms_dict,
        radius = options.fingerprint_radius,
        nBits = options.fingerprint_n_bits,
    )
    cols_to_drop = df.columns
    polymer_features = polymer_features.drop(columns=cols_to_drop, errors='ignore')
    molecule_features = molecule_features.drop(columns=cols_to_drop, errors='ignore')
    polymer_features = expand_fingerprints(polymer_features)
    molecule_features = expand_fingerprints(molecule_features)
    polymer_features = polymer_features.add_prefix('poly_')
    molecule_features = molecule_features.add_prefix('drug_')
    df = pd.concat([
        polymer_features, 
        molecule_features,
        df.drop(columns=['POLYMER_USED', 'DRUG']),
    ], axis=1)
    df = df.dropna()
    return df


def expand_fingerprints(df):
    fp_cols = [col for col in df.columns if col.startswith('fingerprint')]
    for col in fp_cols:
        bit_series = df[col].apply(lambda x: [int(bit) for bit in list(x)] if pd.notna(x) else [])
        expanded_df = pd.DataFrame(bit_series.tolist(), index=df.index)
        expanded_df.columns = [f"{col}_bit_{i}" for i in range(expanded_df.shape[1])]
        df = pd.concat([df, expanded_df], axis=1).drop(columns=[col])
    return df

        
def test_():
    from bio.__global__ import PDCC_DATASET, PSMILES_DICT, SMILES_DICT
    from bio.Dataset.__global__ import HELPER_DIR
    df = pd.read_csv(PDCC_DATASET)
    df['POLYMER_USED'] = df['POLYMER_USED'].map(PSMILES_DICT)
    df['DRUG'] = df['DRUG'].map(SMILES_DICT)
    df = featurize_v4(df)
    print(df.head(10))
    # Cannot save it as a csv since it contains binary data
    df.to_csv(HELPER_DIR / "featurize_v4.csv", index=False)
