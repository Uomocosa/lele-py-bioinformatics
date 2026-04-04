from dataclasses import dataclass, field
import torch.nn as nn
from typing import Optional
import pandas as pd
from rdkit import Chem
from polymetrix.featurizers.chemical_featurizer import *
import bio
from bio.Dataset import PDCCMethod
from loguru import logger

ALL_POLYMETRIX_FEATURES = [
    NumHBondDonors, NumHBondAcceptors, NumRotatableBonds, NumRings,
    NumNonAromaticRings, NumAromaticRings, NumAtoms, TopologicalSurfaceArea,
    FractionBicyclicRings, NumAliphaticHeterocycles, SlogPVSA1, BalabanJIndex,
    MolecularWeight, Sp3CarbonCountFeaturizer, Sp2CarbonCountFeaturizer,
    MaxEStateIndex, SmrVSA5, FpDensityMorgan1, HalogenCounts, BondCounts,
    BridgingRingsCount, MaxRingSize, HeteroatomCount, HeteroatomDensity,
]
FEATURE_MAP = {cls.__name__: cls for cls in ALL_POLYMETRIX_FEATURES}



@dataclass
class Options:
    capping_atoms: list = field(default_factory=lambda: [
        'H',
        # 'C',
        # 'O',
    ])
    fingerprint_radius: int = 2
    fingerprint_n_bits: int = 256
    protonate_precision: float = 1.0
    molecule_features_to_calculate: list = field(default_factory=lambda: [
        'logp', 'logd', 'homo_lumo_eV', 'net_charge', 'fingerprint',
    ])
    polymer_features_to_calculate: list = field(default_factory=lambda: [
        'logp', 'logd', 'homo_lumo_eV', 'net_charge', 'fingerprint'
    ])
    molecule_polymetrix_features: list = field(default_factory=lambda: ["ALL"]) 
    polymer_polymetrix_features: list = field(default_factory=lambda: ["ALL"])
    sidechain_polymetrix_features: list = field(default_factory=lambda: ["ALL"])
    backbone_polymetrix_features: list = field(default_factory=lambda: ["ALL"])

    @property
    def capping_atoms_dict(self) -> dict:
        return _parse_capping_atoms_dict(self.capping_atoms)

    @property
    def parsed_molecule_polymetrix_features(self) -> list:
        return _parse_polymetrix(self.molecule_polymetrix_features)

    @property
    def parsed_polymer_polymetrix_features(self) -> list:
        return _parse_polymetrix(self.polymer_polymetrix_features)

    @property
    def parsed_sidechain_polymetrix_features(self) -> list:
        return _parse_polymetrix(self.sidechain_polymetrix_features)

    @property
    def parsed_backbone_polymetrix_features(self) -> list:
        return _parse_polymetrix(self.backbone_polymetrix_features)
        
        
def _parse_capping_atoms_dict(capping_atoms: list) -> dict:
    pt = Chem.GetPeriodicTable()
    capping_atoms_dict = {
        symbol: pt.GetAtomicNumber(symbol) 
        for symbol in capping_atoms
    }
    logger.debug(f"capping_atoms_dict: {capping_atoms_dict}")
    return capping_atoms_dict
    
def _parse_polymetrix(features: list) -> list:
    if "ALL" in features: return ALL_POLYMETRIX_FEATURES
    if "all" in features: return ALL_POLYMETRIX_FEATURES
    return [FEATURE_MAP[f] for f in features if f in FEATURE_MAP]
        



from bio.__global__ import CACHE_MEMORY
@CACHE_MEMORY.cache
def featurize(
    df: pd.DataFrame,
    options: Options = Options()
) -> pd.DataFrame:
    polymetrix_options = PDCCMethod.get_poly_mol_features_polymetrix.Options(
        molecule_features = options.parsed_molecule_polymetrix_features,
        polymer_features = options.parsed_polymer_polymetrix_features,
        sidechain_features = options.parsed_sidechain_polymetrix_features,
        backbone_features = options.parsed_backbone_polymetrix_features,
    )
    
    polymer_features_1, molecule_features_1 = PDCCMethod.get_poly_mol_features(df, options)
    polymer_features_2, molecule_features_2 = PDCCMethod.get_poly_mol_features_polymetrix(df, polymetrix_options)
    
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




def test_complete_featurizer():
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
    df = PDCCMethod.convert_names_to_smiles(df)
    df = featurize(df.head(10))
    print(df)


def test_only_logd():
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
    df = PDCCMethod.convert_names_to_smiles(df)
    df = featurize(
        df.head(10),
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
