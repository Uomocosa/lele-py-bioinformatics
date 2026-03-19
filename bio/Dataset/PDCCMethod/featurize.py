from dataclasses import dataclass, field
import torch.nn as nn
from typing import Optional
import pandas as pd
from rdkit import Chem
from polymetrix.featurizers.chemical_featurizer import *
import bio
from bio.Dataset import PDCCMethod
from loguru import logger


ALL_FEATURES = [
    NumHBondDonors, NumHBondAcceptors, NumRotatableBonds, NumRings,
    NumNonAromaticRings, NumAromaticRings, NumAtoms, TopologicalSurfaceArea,
    FractionBicyclicRings, NumAliphaticHeterocycles, SlogPVSA1, BalabanJIndex,
    MolecularWeight, Sp3CarbonCountFeaturizer, Sp2CarbonCountFeaturizer,
    MaxEStateIndex, SmrVSA5, FpDensityMorgan1, HalogenCounts, BondCounts,
    BridgingRingsCount, MaxRingSize, HeteroatomCount, HeteroatomDensity,
]
POLYMER_FEATURES = ALL_FEATURES
MOLECULE_FEATURES = ALL_FEATURES
SIDECHAIN_FEATURES = ALL_FEATURES
BACKBONE_FEATURES = ALL_FEATURES

DEFAULT_CAPPING_ATOMS = {
    'H': 1,
    # 'C': 6,
    # 'O': 8
}

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
        'logp', 'logd', 'homo_lumo_eV', 'net_charge', 'fingerprint'
    ])
    molecule_polymetrix_features: list = field(default_factory=lambda: MOLECULE_FEATURES)
    polymer_polymetrix_features: list = field(default_factory=lambda: POLYMER_FEATURES)
    sidechain_polymetrix_features: list = field(default_factory=lambda: SIDECHAIN_FEATURES)
    backbone_polymetrix_features: list = field(default_factory=lambda: BACKBONE_FEATURES)


def featurize(
    df: pd.DataFrame,
    options: Options = Options()
) -> pd.DataFrame:
    polymetrix_options = PDCCMethod.get_poly_mol_features_polymetrix.Options(
        molecule_features = options.molecule_polymetrix_features,
        polymer_features = options.polymer_polymetrix_features,
        sidechain_features = options.sidechain_polymetrix_features,
        backbone_features = options.backbone_polymetrix_features,
    )
    polymer_features_1, molecule_features_1 = PDCCMethod.get_poly_mol_features(df, options)
    polymer_features_2, molecule_features_2 = PDCCMethod.get_poly_mol_features_polymetrix(df, polymetrix_options)
    polymer_features_1 = polymer_features_1.add_prefix('poly_')
    polymer_features_2 = polymer_features_2.add_prefix('poly_')
    molecule_features_1 = molecule_features_1.add_prefix('drug_')
    molecule_features_2 = molecule_features_2.add_prefix('drug_')
    df = pd.concat([
        df.drop(columns=['POLYMER_USED', 'DRUG']),
        polymer_features_1, 
        polymer_features_2, 
        molecule_features_1,
        molecule_features_2,
    ], axis=1)
    df = df.dropna()
    return df

# import pytest
# @pytest.mark.skip(reason="no reason") # LEAVE THIS COMMENETED!
def test_():
    from bio.__global__ import PDCC_DATASET, PSMILES_DICT, SMILES_DICT
    from bio.Dataset.__global__ import HELPER_DIR
    df = pd.read_csv(PDCC_DATASET)
    df['POLYMER_USED'] = df['POLYMER_USED'].map(PSMILES_DICT)
    df['DRUG'] = df['DRUG'].map(SMILES_DICT)
    df = featurize(df.head(10))
    print(df)
    df.to_csv(HELPER_DIR / 'featurize_FINAL.csv', index=False)


def test_only_logd():
    import sys
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    from bio.__global__ import PDCC_DATASET, PSMILES_DICT, SMILES_DICT
    from bio.Dataset.__global__ import HELPER_DIR
    df = pd.read_csv(PDCC_DATASET)
    df['POLYMER_USED'] = df['POLYMER_USED'].map(PSMILES_DICT)
    df['DRUG'] = df['DRUG'].map(SMILES_DICT)
    df = featurize(
        df,
        options = Options(
            molecule_features_to_calculate = ['logd'],
            polymer_features_to_calculate = ['logd'],
            molecule_polymetrix_features = [],
            polymer_polymetrix_features = [],
            sidechain_polymetrix_features = [],
            backbone_polymetrix_features = [],
        )
    )
    print(df)
