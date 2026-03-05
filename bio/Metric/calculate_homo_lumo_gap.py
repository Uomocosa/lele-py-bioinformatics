import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import qcelemental as qcel
import tblite.interface as tb
from ase import Atoms
from ase.optimize import BFGS
from tblite.ase import TBLite
from typing import Optional

import bio
from bio.Bioinformatics.transform_into_smiles import DEFAULT_CAPPING_ATOMS
from loguru import logger


def calculate_homo_lumo_gap(
    df: pd.DataFrame, 
    column_name: str, 
    capping_atoms_dict: dict = DEFAULT_CAPPING_ATOMS
) -> pd.DataFrame:
    """
    Calculates the HOMO-LUMO gap for each SMILES in the dataframe.
    """
    stats_df = df[column_name].apply(lambda s: compute_min_max_gap(s, capping_atoms_dict))
    df_out = pd.concat([df, stats_df], axis=1)
    return df_out


def compute_min_max_gap(smiles_str: str, capping_atoms_dict: dict) -> pd.Series:
    """
    Performs 3D optimization and GFN2-xTB calculation to find the HOMO-LUMO gap.
    """
    lable = f'homo_lumo_gap'
    # null_result = pd.Series({lable: None})
    null_result = pd.Series({})
    smiles_str = str(smiles_str)
    valid_smiles_dict = bio.Bioinformatics.transform_into_smiles(smiles_str, capping_atoms_dict)
    if not valid_smiles_dict: return null_result
    mols = {atom: Chem.MolFromSmiles(smile) for atom, smile in valid_smiles_dict.items()}
    energies_dict = {atom: bio.Metric.calculate_homo_lumo_energies.from_mol(mol) for atom, mol in mols.items()}
    energies_dict = {atom: energies for atom, energies in energies_dict.items() if energies is not None}
    if not energies_dict: return null_result
    df = dict()
    for atom, energies in energies_dict.items():
        homo_eV, lumo_eV = energies
        key = f"{lable}_{atom}"
        key = key.removesuffix('_')
        df[key] = lumo_eV - homo_eV
    logger.debug(f'gaps: {df}')
    return pd.Series(df)
    

import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "calculate_homo_lumo_gap_generated.csv"
    df = pd.read_csv(dataset_csv).head(10) # Start small, xTB is ~1000x slower than RDKit
    df = calculate_homo_lumo_gap(df, column_name="PSMILES")
    print(df)
    df.to_csv(csv_file, index=False)

    
# def from_mol(mol: Chem.Mol) -> Optional[float]:
#     """
#     based on :
#         - https://tblite.readthedocs.io/en/latest/tutorial/python/singlepoint.html#homo-lumo-gap
#     """
#     mol_3D = bio.Bioinformatics.transform_into_3D_geometry(mol)
#     if mol_3D is None: return None
#     atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol_3D.GetAtoms()])
#     coords_angstrom = mol_3D.GetConformer().GetPositions()
#     logger.debug(f"atomic_numbers: {atomic_numbers}")
#     logger.debug(f"coords_angstrom:\n{coords_angstrom}")
    
#     angstrom_to_bohr = qcel.constants.conversion_factor("angstrom", "bohr")
#     geometry_bohr = coords_angstrom * angstrom_to_bohr
    
#     xtb = tb.Calculator("GFN2-xTB", atomic_numbers, geometry_bohr)
#     xtb.set("verbosity", 0) # Disable energy table output
#     results = xtb.singlepoint()
    
#     logger.debug(f"Energy: {results['energy']} Hartree")
    
#     orbital_energies = results["orbital-energies"]
#     orbital_occupations = results["orbital-occupations"]
    
#     lumo_index = np.argmax(orbital_occupations)
#     homo_index = lumo_index - 1
#     gap_ev = (orbital_energies[lumo_index] - orbital_energies[homo_index]) * qcel.constants.conversion_factor("hartree", "eV")
#     logger.debug(f"HOMO-LUMO Gap: {gap_ev:.4f} eV")
#     return gap_ev
