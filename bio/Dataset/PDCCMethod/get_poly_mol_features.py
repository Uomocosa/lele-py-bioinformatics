from dataclasses import dataclass, field
import torch.nn as nn
from typing import Optional
import pandas as pd
from rdkit import Chem
import bio
from bio.Bioinformatics.transform_into_smiles import DEFAULT_CAPPING_ATOMS
from loguru import logger

@dataclass
class Options:
    capping_atoms_dict: dict = field(default_factory=lambda: DEFAULT_CAPPING_ATOMS)
    fingerprint_radius: int = 2
    fingerprint_n_bits: int = 2048
    protonate_precision: float = 1.0
    molecule_features_to_calculate: list = field(default_factory=lambda: [
        'logp', 'logd', 'homo_lumo_eV', 'net_charge', 'fingerprint',
    ])
    polymer_features_to_calculate: list = field(default_factory=lambda: [
        'logp', 'logd', 'homo_lumo_eV', 'net_charge', 'fingerprint',
    ])


def get_poly_mol_features(
    df: pd.DataFrame,
    options: Options = Options()
) -> pd.DataFrame:
    molecule_features = df.copy()
    polymer_features = df.copy()
    
    if 'logp' in options.molecule_features_to_calculate:
        molecule_features = bio.Metric.calculate_logp(
            molecule_features, 
            'DRUG', 
            options.capping_atoms_dict
        )
    if 'logp' in options.polymer_features_to_calculate:
        polymer_features = bio.Metric.calculate_logp(
            polymer_features, 
            'POLYMER_USED', 
            options.capping_atoms_dict
        )
        
    if 'logd' in options.molecule_features_to_calculate:
        molecule_features = calculate_logd(
            molecule_features, 
            'DRUG', 
            options
        )
    if 'logd' in options.polymer_features_to_calculate:
        polymer_features = calculate_logd(
            polymer_features, 
            'POLYMER_USED', 
            options
        )
        
    if 'homo_lumo_eV' in options.molecule_features_to_calculate:
        molecule_features = bio.Metric.calculate_homo_lumo_energies(
            molecule_features, 
            'DRUG', 
            options.capping_atoms_dict
        )
    if 'homo_lumo_eV' in options.polymer_features_to_calculate:
        polymer_features = bio.Metric.calculate_homo_lumo_energies(
            polymer_features, 
            'POLYMER_USED', 
            options.capping_atoms_dict
        )
        
    if 'net_charge' in options.molecule_features_to_calculate:
        molecule_features = bio.Metric.calculate_net_charge_at_ph(
            molecule_features, 
            'DRUG', 
            options.capping_atoms_dict
        )
    if 'net_charge' in options.polymer_features_to_calculate:
        polymer_features = bio.Metric.calculate_net_charge_at_ph(
            polymer_features, 
            'POLYMER_USED', 
            options.capping_atoms_dict
        )
        
    if 'fingerprint' in options.molecule_features_to_calculate:
        molecule_features = bio.Metric.convert_smile_to_fingerprint(
            molecule_features, 
            column_name = 'DRUG',
            capping_atoms_dict = options.capping_atoms_dict,
            radius = options.fingerprint_radius,
            nBits = options.fingerprint_n_bits,
        )

    if 'fingerprint' in options.polymer_features_to_calculate:
        polymer_features = bio.Metric.convert_smile_to_fingerprint(
            polymer_features, 
            column_name = 'POLYMER_USED',
            capping_atoms_dict = options.capping_atoms_dict,
            radius = options.fingerprint_radius,
            nBits = options.fingerprint_n_bits,
        )
    polymer_features = expand_fingerprints(polymer_features)
    molecule_features = expand_fingerprints(molecule_features)
        
    cols_to_drop = df.columns
    polymer_features = polymer_features.drop(columns=cols_to_drop, errors='ignore')
    molecule_features = molecule_features.drop(columns=cols_to_drop, errors='ignore')
    return polymer_features, molecule_features


def calculate_logd(df, column_name, options):
    def compute_logd_safely(row):
        water_ph = row['WATER_PH']
        if pd.isna(water_ph): 
            logger.warning(f"Found NaN WATER_PH for POLYMER_USED: {row['POLYMER_USED']}, DRUG: {row['DRUG']}.\nComplete row: {row}")
            return pd.Series({})
            
        return bio.Metric.calculate_logd.compute_most_probable_logd(
            smiles_str = row[column_name], 
            ph_min = water_ph, 
            ph_max = water_ph, 
            precision = options.protonate_precision, 
            capping_atoms_dict = options.capping_atoms_dict,
            starting_lable = 'logd_at_WATER_PH',
        )
    logd_df = df.apply(compute_logd_safely, axis=1)
    df = pd.concat([df, logd_df], axis=1)
    return df


def expand_fingerprints(df):
    fp_cols = [col for col in df.columns if col.startswith('fingerprint')]
    for col in fp_cols:
        bit_series = df[col].apply(lambda x: [int(bit) for bit in list(x)] if pd.notna(x) else [])
        expanded_df = pd.DataFrame(bit_series.tolist(), index=df.index)
        expanded_df.columns = [f'{col}_bit_{i}' for i in range(expanded_df.shape[1])]
        df = pd.concat([df, expanded_df], axis=1).drop(columns=[col])
    return df


# import pytest
# @pytest.mark.skip(reason="no reason") # LEAVE THIS COMMENETED!
def test_():
    from bio.__global__ import PDCC_DATASET, PSMILES_DICT, SMILES_DICT
    from bio.Dataset.__global__ import HELPER_DIR
    df = pd.read_csv(PDCC_DATASET)
    df['POLYMER_USED'] = df['POLYMER_USED'].map(PSMILES_DICT)
    df['DRUG'] = df['DRUG'].map(SMILES_DICT)
    poly_features, mol_features = get_poly_mol_features(df.head(5))
    logger.debug(poly_features)
    logger.debug(mol_features)


def test_only_logd():
    import sys
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    from bio.__global__ import PDCC_DATASET, PSMILES_DICT, SMILES_DICT
    from bio.Dataset.__global__ import HELPER_DIR
    df = pd.read_csv(PDCC_DATASET)
    df['POLYMER_USED'] = df['POLYMER_USED'].map(PSMILES_DICT)
    df['DRUG'] = df['DRUG'].map(SMILES_DICT)
    poly_features, mol_features = get_poly_mol_features(
        df,
        options = Options(
            molecule_features_to_calculate = ['logd'],
            polymer_features_to_calculate = ['logd'],
        )
    )
    print(poly_features)
    print(mol_features)
