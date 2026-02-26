import pandas as pd
from rdkit import Chem
from loguru import logger

import logging
from typing import List, Optional
import pandas as pd

from polymetrix.featurizers.polymer import Polymer
from polymetrix.featurizers.molecule import Molecule
from polymetrix.featurizers.sidechain_backbone_featurizer import (
    SideChainFeaturizer,
    BackBoneFeaturizer,
)
from polymetrix.featurizers.chemical_featurizer import (
    NumHBondDonors,
    NumHBondAcceptors,
    NumRotatableBonds,
    NumRings,
    NumNonAromaticRings,
    NumAromaticRings,
    NumAtoms,
    TopologicalSurfaceArea,
    FractionBicyclicRings,
    NumAliphaticHeterocycles,
    SlogPVSA1,
    BalabanJIndex,
    MolecularWeight,
    Sp3CarbonCountFeaturizer,
    Sp2CarbonCountFeaturizer,
    MaxEStateIndex,
    SmrVSA5,
    FpDensityMorgan1,
    HalogenCounts,
    BondCounts,
    BridgingRingsCount,
    MaxRingSize,
    HeteroatomCount,
    HeteroatomDensity,
)
from polymetrix.featurizers.molecule import FullMolecularFeaturizer
from polymetrix.featurizers.sidechain_backbone_featurizer import FullPolymerFeaturizer
from polymetrix.featurizers.multiple_featurizer import MultipleFeaturizer
import bio

POLYMER_FEATURIZERS = [
    FullPolymerFeaturizer(NumHBondDonors()),
    FullPolymerFeaturizer(NumHBondAcceptors()),
    FullPolymerFeaturizer(NumRotatableBonds()),
    FullPolymerFeaturizer(NumRings()),
    FullPolymerFeaturizer(NumNonAromaticRings()),
    FullPolymerFeaturizer(NumAromaticRings()),
    FullPolymerFeaturizer(NumAtoms()),
    FullPolymerFeaturizer(TopologicalSurfaceArea()),
    FullPolymerFeaturizer(FractionBicyclicRings()),
    FullPolymerFeaturizer(NumAliphaticHeterocycles()),
    FullPolymerFeaturizer(SlogPVSA1()),
    FullPolymerFeaturizer(BalabanJIndex()),
    FullPolymerFeaturizer(MolecularWeight()),
    FullPolymerFeaturizer(Sp3CarbonCountFeaturizer()),
    FullPolymerFeaturizer(Sp2CarbonCountFeaturizer()),
    FullPolymerFeaturizer(MaxEStateIndex()),
    FullPolymerFeaturizer(SmrVSA5()),
    FullPolymerFeaturizer(FpDensityMorgan1()),
    FullPolymerFeaturizer(HalogenCounts()),
    FullPolymerFeaturizer(BondCounts()),
    FullPolymerFeaturizer(BridgingRingsCount()),
    FullPolymerFeaturizer(MaxRingSize()),
    FullPolymerFeaturizer(HeteroatomCount()),
    FullPolymerFeaturizer(HeteroatomDensity()),
]
POLYMER_MULTI_FEATURIZER = MultipleFeaturizer(POLYMER_FEATURIZERS)

MOLECULE_FEATURIZERS = [
    FullMolecularFeaturizer(NumHBondDonors()),
    FullMolecularFeaturizer(NumHBondAcceptors()),
    FullMolecularFeaturizer(NumRotatableBonds()),
    FullMolecularFeaturizer(NumRings()),
    FullMolecularFeaturizer(NumNonAromaticRings()),
    FullMolecularFeaturizer(NumAromaticRings()),
    FullMolecularFeaturizer(NumAtoms()),
    FullMolecularFeaturizer(TopologicalSurfaceArea()),
    FullMolecularFeaturizer(FractionBicyclicRings()),
    FullMolecularFeaturizer(NumAliphaticHeterocycles()),
    FullMolecularFeaturizer(SlogPVSA1()),
    FullMolecularFeaturizer(BalabanJIndex()),
    FullMolecularFeaturizer(MolecularWeight()),
    FullMolecularFeaturizer(Sp3CarbonCountFeaturizer()),
    FullMolecularFeaturizer(Sp2CarbonCountFeaturizer()),
    FullMolecularFeaturizer(MaxEStateIndex()),
    FullMolecularFeaturizer(SmrVSA5()),
    FullMolecularFeaturizer(FpDensityMorgan1()),
    FullMolecularFeaturizer(HalogenCounts()),
    FullMolecularFeaturizer(BondCounts()),
    FullMolecularFeaturizer(BridgingRingsCount()),
    FullMolecularFeaturizer(MaxRingSize()),
    FullMolecularFeaturizer(HeteroatomCount()),
    FullMolecularFeaturizer(HeteroatomDensity()),
]
MOLECULE_MULTI_FEATURIZER = MultipleFeaturizer(MOLECULE_FEATURIZERS)

SIDECHAIN_FEATURIZERS = [ # Sidechain H-Bond Dynamics
    SideChainFeaturizer(NumHBondDonors()),
    SideChainFeaturizer(NumHBondAcceptors())
]
SIDECHAIN_MULTI_FEATURIZER = MultipleFeaturizer(SIDECHAIN_FEATURIZERS)

BACKBONE_FEATURIZERS = [ # Backbone Connectivity and Hybridization
    BackBoneFeaturizer(NumAtoms()), 
    BackBoneFeaturizer(Sp2CarbonCountFeaturizer()), 
    BackBoneFeaturizer(Sp3CarbonCountFeaturizer())
]
BACKBONE_MULTI_FEATURIZER = MultipleFeaturizer(BACKBONE_FEATURIZERS)


"""
For Polymetrix featurizes take a look at this examples/links:
- https://lamalab-org.github.io/PolyMetriX/use_featurizers/
- NOTE! If needed there are also: 
    - SIDECHAIN_MULTI_FEATURIZER and BACKBONE_MULTI_FEATURIZER
    - Comparators to Compare Polymer and Molecule Features
"""
def add_polymetrix_to_df(df_smiles_and_psmiles: pd.DataFrame, column_name: str):
    df = df_smiles_and_psmiles.copy()
    is_psmiles = df[column_name].apply(lambda x: bio.Bioinformatics.is_psmiles_string_valid(str(x)))
    is_smiles = ~is_psmiles & df[column_name].apply(lambda x: bio.Bioinformatics.Smile(str(x)).is_valid)
    smiles = df.loc[is_smiles, column_name]
    psmiles = df.loc[is_psmiles, column_name]
    assert len(smiles) + len(psmiles) <= len(df), "A SMILES should not be considered a valid P-SMILES and vice-versa"
    
    if not smiles.empty:
        molecules = smiles.apply(Chem.MolFromSmiles)
        df.loc[is_smiles, 'SA_score'] = molecules.apply(bio.PolyGen.calculate_sa_score)
        df.loc[is_smiles, 'diversity'] = bio.Metric.calculate_mean_diversity(smiles.to_list())
        molecules = smiles.apply(Molecule.from_smiles)
        feat_values = molecules.apply(MOLECULE_MULTI_FEATURIZER.featurize)
        labels = MOLECULE_MULTI_FEATURIZER.feature_labels()
        molecule_feats = pd.DataFrame(feat_values.tolist(), index=smiles.index, columns=labels)
        df = pd.concat([df, molecule_feats], axis=1)
    if not psmiles.empty:
        polymers = psmiles.apply(Polymer.from_psmiles)
        
        polymer_feat_values = polymers.apply(POLYMER_MULTI_FEATURIZER.featurize)
        polymer_labels = POLYMER_MULTI_FEATURIZER.feature_labels()
        polymer_feats = pd.DataFrame(polymer_feat_values.tolist(), index=psmiles.index, columns=polymer_labels)
        
        sc_feat_values = polymers.apply(SIDECHAIN_MULTI_FEATURIZER.featurize)
        sc_labels = SIDECHAIN_MULTI_FEATURIZER.feature_labels()
        sc_feats = pd.DataFrame(sc_feat_values.tolist(), index=psmiles.index, columns=sc_labels)
        
        bb_feat_values = polymers.apply(BACKBONE_MULTI_FEATURIZER.featurize)
        bb_labels = BACKBONE_MULTI_FEATURIZER.feature_labels()
        bb_feats = pd.DataFrame(bb_feat_values.tolist(), index=psmiles.index, columns=bb_labels)
        
        df = pd.concat([df, polymer_feats, sc_feats, bb_feats], axis=1)
    return df


# import pytest
# @pytest.mark.above10s
# def test_psmiles():
#     from bio.__global__ import DATASETS_DIR
#     from bio.Metric.__global__ import HELPER_DIR
#     csv_file = HELPER_DIR / "add_polymetrix_to_df_psmiles.csv"
#     dataset_csv = DATASETS_DIR / "PI1M" / "PI1M.csv"
#     df = pd.read_csv(dataset_csv)
#     df = add_polymetrix_to_df(df.head(100), column_name="PSMILES")
#     print(df)
#     df.to_csv(csv_file, index=False)


# import pytest
# @pytest.mark.above10s
# def test_combined():
#     from bio.__global__ import DATASETS_DIR
#     from bio.Metric.__global__ import HELPER_DIR
#     csv_file = HELPER_DIR / "add_polymetrix_to_df_combined.csv"
#     dataset_csv = DATASETS_DIR / "PI1M+ZINC_base" / "combined.csv"
#     df = pd.read_csv(dataset_csv)
#     df = add_polymetrix_to_df(df.head(100), column_name="PSMILES")
#     print(df)
#     df.to_csv(csv_file, index=False)


import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "add_polymetrix_to_df_generated.csv"
    df = pd.read_csv(dataset_csv)
    df = add_polymetrix_to_df(df.head(100), column_name="PSMILES")
    print(df)
    df.to_csv(csv_file, index=False)
