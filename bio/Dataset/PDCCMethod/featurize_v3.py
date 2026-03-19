from dataclasses import dataclass, field
import torch.nn as nn
from typing import Optional
import pandas as pd
from rdkit import Chem
import bio
from bio.Bioinformatics.transform_into_smiles import DEFAULT_CAPPING_ATOMS

from polymetrix.featurizers.polymer import Polymer
from polymetrix.featurizers.sidechain_backbone_featurizer import (
    SideChainFeaturizer,
    BackBoneFeaturizer,
    FullPolymerFeaturizer
)
from polymetrix.featurizers.multiple_featurizer import MultipleFeaturizer
from polymetrix.featurizers.chemical_featurizer import *
from polymetrix.featurizers.molecule import Molecule, FullMolecularFeaturizer


POLYMER_FEATURIZERS = [TopologicalSurfaceArea]
MOLECULE_FEATURIZERS = []
COMMON_FEATURIZERS = [
    NumHBondDonors, NumHBondAcceptors, NumNonAromaticRings, NumAromaticRings,
    
    # ALL OF THEM (pick from them)
    # NumHBondDonors, NumHBondAcceptors, NumRotatableBonds, NumRings,
    # NumNonAromaticRings, NumAromaticRings, NumAtoms, TopologicalSurfaceArea,
    # FractionBicyclicRings, NumAliphaticHeterocycles, SlogPVSA1, BalabanJIndex,
    # MolecularWeight, Sp3CarbonCountFeaturizer, Sp2CarbonCountFeaturizer,
    # MaxEStateIndex, SmrVSA5, FpDensityMorgan1, HalogenCounts, BondCounts,
    # BridgingRingsCount, MaxRingSize, HeteroatomCount, HeteroatomDensity,
]

SIDECHAIN_MULTI_FEATURIZER = MultipleFeaturizer([
    SideChainFeaturizer(NumHBondDonors()),
    SideChainFeaturizer(NumHBondAcceptors())
])

BACKBONE_MULTI_FEATURIZER = MultipleFeaturizer([
    BackBoneFeaturizer(NumAtoms()), 
    BackBoneFeaturizer(Sp2CarbonCountFeaturizer()), 
    BackBoneFeaturizer(Sp3CarbonCountFeaturizer())
])

POLYMER_MULTI_FEATURIZER = MultipleFeaturizer([
    FullPolymerFeaturizer(f()) for f in COMMON_FEATURIZERS + POLYMER_FEATURIZERS
])
MOLECULE_MULTI_FEATURIZER = MultipleFeaturizer([
    FullMolecularFeaturizer(f()) for f in COMMON_FEATURIZERS + MOLECULE_FEATURIZERS
])

@dataclass
class Options:
    # train_data: Optional[pd.DataFrame] = None
    capping_atoms_dict: dict = field(default_factory=lambda: DEFAULT_CAPPING_ATOMS) 
    protonate_precision: float = 1.0
    molecule_multi_featurizer = MOLECULE_MULTI_FEATURIZER
    polymer_multi_featurizer = POLYMER_MULTI_FEATURIZER
    sidechain_multi_featurizer = SIDECHAIN_MULTI_FEATURIZER
    backbone_multi_featurizer = BACKBONE_MULTI_FEATURIZER


def featurize_v3(
    df: pd.DataFrame,
    options: Options = Options()
) -> pd.DataFrame:
    polymer_features = bio.Metric.featurize_psmiles(
        df[['POLYMER_USED']], "POLYMER_USED",
        options.polymer_multi_featurizer,
        options.sidechain_multi_featurizer,
        options.backbone_multi_featurizer,
    )
    molecule_features = bio.Metric.featurize_smiles(
        df[['DRUG']], "DRUG", 
        options.molecule_multi_featurizer,
    )
    molecule_features = bio.Metric.calculate_logp(molecule_features, "DRUG", options.capping_atoms_dict)
    fn = lambda row: bio.Metric.calculate_logd.compute_most_probable_logd(
        row['DRUG'], 
        ph_min = row['WATER_PH'], 
        ph_max = row['WATER_PH'], 
        precision = options.protonate_precision, 
        capping_atoms_dict = options.capping_atoms_dict,
        starting_lable = 'DRUG_logd_at_WATER_PH',
    )
    logd_df = df.apply(fn, axis=1)
    molecule_features = pd.concat([molecule_features, logd_df], axis=1)    
    molecule_features = bio.Metric.calculate_homo_lumo_energies(molecule_features, "DRUG", options.capping_atoms_dict)
    polymer_features = bio.Metric.calculate_homo_lumo_energies(polymer_features, "POLYMER_USED", options.capping_atoms_dict)
    
    polymer_features = polymer_features.drop(columns=['POLYMER_USED'])
    molecule_features = molecule_features.drop(columns=['DRUG'])
    polymer_features = polymer_features.add_prefix('poly_')
    molecule_features = molecule_features.add_prefix('drug_')
    df = pd.concat([
        polymer_features, 
        molecule_features,
        df.drop(columns=['POLYMER_USED', 'DRUG']),
    ], axis=1)
    df = df.dropna()
    return df


def test_():
    from bio.__global__ import PDCC_DATASET, PSMILES_DICT, SMILES_DICT
    from bio.Dataset.__global__ import HELPER_DIR
    df = pd.read_csv(PDCC_DATASET)
    df['POLYMER_USED'] = df['POLYMER_USED'].map(PSMILES_DICT)
    df['DRUG'] = df['DRUG'].map(SMILES_DICT)
    df = featurize_v3(df)
    print(df.head(10))
    df.to_csv(HELPER_DIR / "featurize_v3.csv", index=False)
