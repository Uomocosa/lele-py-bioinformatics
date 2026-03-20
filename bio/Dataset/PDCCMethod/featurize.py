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

    def parse(self):
        self.capping_atoms_dict = _parse_capping_atoms_dict(self.capping_atoms)
        self.molecule_polymetrix_features = _parse_polymetrix(self.molecule_polymetrix_features)
        self.polymer_polymetrix_features = _parse_polymetrix(self.polymer_polymetrix_features)
        self.sidechain_polymetrix_features = _parse_polymetrix(self.sidechain_polymetrix_features)
        self.backbone_polymetrix_features = _parse_polymetrix(self.backbone_polymetrix_features)
        return self

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
    parsed_options = options.parse()
    polymetrix_options = PDCCMethod.get_poly_mol_features_polymetrix.Options(
        molecule_features = parsed_options.molecule_polymetrix_features,
        polymer_features = parsed_options.polymer_polymetrix_features,
        sidechain_features = parsed_options.sidechain_polymetrix_features,
        backbone_features = parsed_options.backbone_polymetrix_features,
    )
    polymer_features_1, molecule_features_1 = PDCCMethod.get_poly_mol_features(df, parsed_options)
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
