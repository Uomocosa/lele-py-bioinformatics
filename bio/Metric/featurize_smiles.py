import pandas as pd
from rdkit import Chem
import bio

from polymetrix.featurizers.molecule import Molecule, FullMolecularFeaturizer
from polymetrix.featurizers.multiple_featurizer import MultipleFeaturizer
from polymetrix.featurizers.chemical_featurizer import *

# Simplified list containing only Molecular Featurizers
MOLECULE_FEATURIZERS = [
    FullMolecularFeaturizer(f()) for f in [
        NumHBondDonors, NumHBondAcceptors, NumRotatableBonds, NumRings,
        NumNonAromaticRings, NumAromaticRings, NumAtoms, TopologicalSurfaceArea,
        FractionBicyclicRings, NumAliphaticHeterocycles, SlogPVSA1, BalabanJIndex,
        MolecularWeight, Sp3CarbonCountFeaturizer, Sp2CarbonCountFeaturizer,
        MaxEStateIndex, SmrVSA5, FpDensityMorgan1, HalogenCounts, BondCounts,
        BridgingRingsCount, MaxRingSize, HeteroatomCount, HeteroatomDensity,
    ]
]
MOLECULE_MULTI_FEATURIZER = MultipleFeaturizer(MOLECULE_FEATURIZERS)

def featurize_smiles(
    df: pd.DataFrame, 
    column_name: str,
    molecule_multi_featurizer: MultipleFeaturizer = MOLECULE_MULTI_FEATURIZER,
) -> pd.DataFrame:
    """
    Processes only valid SMILES strings, calculating SA scores, 
    diversity, and molecular descriptors.
    """
    if not molecule_multi_featurizer.featurizers: return df
    
    df = df.copy()
    
    # Validate SMILES (Using the provided bio.Bioinformatics utility)
    is_smiles = df[column_name].apply(
        lambda x: bio.Bioinformatics.Smile(str(x)).is_valid if pd.notnull(x) else False
    )
    
    smiles_subset = df.loc[is_smiles, column_name]
    
    if smiles_subset.empty: return df
    
    pm_molecules = smiles_subset.apply(Molecule.from_smiles)
    feat_values = pm_molecules.apply(molecule_multi_featurizer.featurize)
    labels = molecule_multi_featurizer.feature_labels()
    molecule_feats = pd.DataFrame(feat_values.tolist(), index=smiles_subset.index, columns=labels)
    df = pd.concat([df, molecule_feats], axis=1)
    return df


def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "featurize_smiles_generated.csv"
    df = pd.read_csv(dataset_csv)
    df = featurize_smiles(df.head(100), column_name="PSMILES")
    print(df)
    df.to_csv(csv_file, index=False)


def test_empty():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "featurize_smiles_generated.csv"
    df = pd.read_csv(dataset_csv)
    df = featurize_smiles(
        df.head(100), 
        column_name="PSMILES", 
        molecule_multi_featurizer=MultipleFeaturizer([]),
    )
    print(df)
    df.to_csv(csv_file, index=False)
