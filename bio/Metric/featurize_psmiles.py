import pandas as pd
from rdkit import Chem
import bio

from polymetrix.featurizers.polymer import Polymer
from polymetrix.featurizers.sidechain_backbone_featurizer import (
    SideChainFeaturizer,
    BackBoneFeaturizer,
    FullPolymerFeaturizer
)
from polymetrix.featurizers.multiple_featurizer import MultipleFeaturizer
from polymetrix.featurizers.chemical_featurizer import (
    NumHBondDonors, NumHBondAcceptors, NumRotatableBonds, NumRings,
    NumNonAromaticRings, NumAromaticRings, NumAtoms, TopologicalSurfaceArea,
    FractionBicyclicRings, NumAliphaticHeterocycles, SlogPVSA1, BalabanJIndex,
    MolecularWeight, Sp3CarbonCountFeaturizer, Sp2CarbonCountFeaturizer,
    MaxEStateIndex, SmrVSA5, FpDensityMorgan1, HalogenCounts, BondCounts,
    BridgingRingsCount, MaxRingSize, HeteroatomCount, HeteroatomDensity,
)

# 1. Global Polymer Descriptors
POLYMER_FEATURIZERS = [
    FullPolymerFeaturizer(f()) for f in [
        NumHBondDonors, NumHBondAcceptors, NumRotatableBonds, NumRings,
        NumNonAromaticRings, NumAromaticRings, NumAtoms, TopologicalSurfaceArea,
        FractionBicyclicRings, NumAliphaticHeterocycles, SlogPVSA1, BalabanJIndex,
        MolecularWeight, Sp3CarbonCountFeaturizer, Sp2CarbonCountFeaturizer,
        MaxEStateIndex, SmrVSA5, FpDensityMorgan1, HalogenCounts, BondCounts,
        BridgingRingsCount, MaxRingSize, HeteroatomCount, HeteroatomDensity,
    ]
]
POLYMER_MULTI_FEATURIZER = MultipleFeaturizer(POLYMER_FEATURIZERS)

# 2. Sidechain-specific Descriptors (H-Bond Dynamics)
SIDECHAIN_MULTI_FEATURIZER = MultipleFeaturizer([
    SideChainFeaturizer(NumHBondDonors()),
    SideChainFeaturizer(NumHBondAcceptors())
])

# 3. Backbone-specific Descriptors (Connectivity/Hybridization)
BACKBONE_MULTI_FEATURIZER = MultipleFeaturizer([
    BackBoneFeaturizer(NumAtoms()), 
    BackBoneFeaturizer(Sp2CarbonCountFeaturizer()), 
    BackBoneFeaturizer(Sp3CarbonCountFeaturizer())
])

def featurize_psmiles(
    df: pd.DataFrame, column_name: str,
    polymer_multi_featurizer = POLYMER_MULTI_FEATURIZER,
    sidechain_multi_featurizer = SIDECHAIN_MULTI_FEATURIZER,
    backbone_multi_featurizer = BACKBONE_MULTI_FEATURIZER,
) -> pd.DataFrame:
    """
    Processes only valid P-SMILES strings, extracting global polymer features
    as well as isolated sidechain and backbone descriptors.
    """
    df = df.copy()
    
    # Identify valid P-SMILES entries
    is_psmiles = df[column_name].apply(
        lambda x: bio.Bioinformatics.is_psmiles_string_valid(str(x)) if pd.notnull(x) else False
    )
    
    psmiles_subset = df.loc[is_psmiles, column_name]
    
    if not psmiles_subset.empty:
        # Convert strings to Polymetrix Polymer objects
        polymers = psmiles_subset.apply(Polymer.from_psmiles)
        
        # --- Global Polymer Features ---
        poly_feat_values = polymers.apply(polymer_multi_featurizer.featurize)
        poly_labels = polymer_multi_featurizer.feature_labels()
        df_poly = pd.DataFrame(poly_feat_values.tolist(), index=psmiles_subset.index, columns=poly_labels)
        
        # --- Sidechain Features ---
        sc_feat_values = polymers.apply(sidechain_multi_featurizer.featurize)
        sc_labels = sidechain_multi_featurizer.feature_labels()
        df_sc = pd.DataFrame(sc_feat_values.tolist(), index=psmiles_subset.index, columns=sc_labels)
        
        # --- Backbone Features ---
        bb_feat_values = polymers.apply(backbone_multi_featurizer.featurize)
        bb_labels = backbone_multi_featurizer.feature_labels()
        df_bb = pd.DataFrame(bb_feat_values.tolist(), index=psmiles_subset.index, columns=bb_labels)
        
        # Concatenate all new feature columns
        df = pd.concat([df, df_poly, df_sc, df_bb], axis=1)
        
    return df
    
import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "featurize_psmiles_generated.csv"
    df = pd.read_csv(dataset_csv)
    df = featurize_psmiles(df.head(100), column_name="PSMILES")
    print(df)
    df.to_csv(csv_file, index=False)
