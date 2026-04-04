from dataclasses import dataclass, field
import torch.nn as nn
from typing import Optional
import pandas as pd
from rdkit import Chem
import bio
from bio.Bioinformatics.transform_into_smiles import DEFAULT_CAPPING_ATOMS
from loguru import logger

from polymetrix.featurizers.polymer import Polymer
from polymetrix.featurizers.sidechain_backbone_featurizer import (
    SideChainFeaturizer,
    BackBoneFeaturizer,
    FullPolymerFeaturizer
)
from polymetrix.featurizers.multiple_featurizer import MultipleFeaturizer
from polymetrix.featurizers.chemical_featurizer import *
from polymetrix.featurizers.molecule import Molecule, FullMolecularFeaturizer


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


@dataclass
class Options:
    molecule_features: list = field(default_factory=lambda: MOLECULE_FEATURES)
    polymer_features: list = field(default_factory=lambda: POLYMER_FEATURES)
    sidechain_features: list = field(default_factory=lambda: SIDECHAIN_FEATURES)
    backbone_features: list = field(default_factory=lambda: BACKBONE_FEATURES)


def get_poly_mol_features_polymetrix(
    df: pd.DataFrame,
    options: Options = Options()
) -> pd.DataFrame:
    molecule_multi_featurize = MultipleFeaturizer([
        FullMolecularFeaturizer(f()) for f in options.polymer_features
    ])
    polymer_multi_featurizer = MultipleFeaturizer([
        FullPolymerFeaturizer(f()) for f in options.polymer_features
    ])
    sidechain_multi_featurizer = MultipleFeaturizer([
        SideChainFeaturizer(f()) for f in options.sidechain_features
    ])
    backbone_multi_featurizer = MultipleFeaturizer([
        BackBoneFeaturizer(f()) for f in options.polymer_features
    ])
    
    polymer_features = bio.Metric.featurize_psmiles(
        df[['POLYMER_USED']], "POLYMER_USED",
        polymer_multi_featurizer,
        sidechain_multi_featurizer,
        backbone_multi_featurizer,
    )
    molecule_features = bio.Metric.featurize_smiles(
        df[['DRUG']], "DRUG", 
        molecule_multi_featurize,
    )
    cols_to_drop = df.columns
    polymer_features = polymer_features.drop(columns=cols_to_drop, errors='ignore')
    molecule_features = molecule_features.drop(columns=cols_to_drop, errors='ignore')
    return polymer_features, molecule_features


def test_():
    from bio.__global__ import PDCC_DATASET, PSMILES_DICT, SMILES_DICT
    from bio.Dataset.__global__ import HELPER_DIR
    df = pd.read_csv(PDCC_DATASET)
    df['POLYMER_USED'] = df['POLYMER_USED'].map(PSMILES_DICT)
    df['DRUG'] = df['DRUG'].map(SMILES_DICT)
    poly_features, mol_features = get_poly_mol_features_polymetrix(df.head(10))
    logger.debug(poly_features)
    logger.debug(mol_features)
    

def test_empty():
    from bio.__global__ import PDCC_DATASET, PSMILES_DICT, SMILES_DICT
    from bio.Dataset.__global__ import HELPER_DIR
    df = pd.read_csv(PDCC_DATASET)
    df['POLYMER_USED'] = df['POLYMER_USED'].map(PSMILES_DICT)
    df['DRUG'] = df['DRUG'].map(SMILES_DICT)
    poly_features, mol_features = get_poly_mol_features_polymetrix(
        df.head(10), 
        Options(
            molecule_features = [],
            polymer_features = [],
            sidechain_features = [],
            backbone_features = [],
        ),
    )
    logger.debug(poly_features)
    logger.debug(mol_features)
