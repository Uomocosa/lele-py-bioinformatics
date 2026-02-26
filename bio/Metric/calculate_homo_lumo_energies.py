import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import qcelemental as qcel
import tblite.interface as tb
from ase import Atoms
from ase.optimize import BFGS
from tblite.ase import TBLite
from typing import Optional, Tuple

import bio
from loguru import logger


def calculate_homo_lumo_energies(
    df: pd.DataFrame, 
    column_name: str, 
) -> pd.DataFrame:
    """
    Calculates the HOMO-LUMO gap for each SMILES in the dataframe.
    """
    stats_df = df[column_name].apply(compute_min_max_gap)
    df_out = pd.concat([df, stats_df], axis=1)
    return df_out


def compute_min_max_gap(smiles_str: str) -> pd.Series:
    """
    Performs 3D optimization and GFN2-xTB calculation to find the HOMO-LUMO gap.
    """
    lable = "homo_lumo_energies"
    null_result = pd.Series({lable: None})
    smiles_str = str(smiles_str)
    valid_smiles_list = bio.Bioinformatics.transform_into_smiles(smiles_str)
    if not valid_smiles_list: return null_result
    mols = [Chem.MolFromSmiles(smile) for smile in valid_smiles_list]
    homo_lumo_energies = [from_mol(mol) for mol in mols]
    homo_lumo_energies = [eV for eV in homo_lumo_energies if eV is not None]
    if not homo_lumo_energies: return null_result
    logger.debug(f'homo_lumo_energies: {homo_lumo_energies}')
    return pd.Series({lable: homo_lumo_energies})
    
    
def from_mol(mol: Chem.Mol) -> Optional[Tuple[float, float]]:
    """
    based on :
        - https://tblite.readthedocs.io/en/latest/tutorial/python/singlepoint.html#homo-lumo-gap
    """
    mol_3D = bio.Bioinformatics.transform_into_3D_geometry(mol)
    if mol_3D is None: return None
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol_3D.GetAtoms()])
    coords_angstrom = mol_3D.GetConformer().GetPositions()
    logger.debug(f"atomic_numbers: {atomic_numbers}")
    logger.debug(f"coords_angstrom:\n{coords_angstrom}")
    
    angstrom_to_bohr = qcel.constants.conversion_factor("angstrom", "bohr")
    geometry_bohr = coords_angstrom * angstrom_to_bohr
    
    xtb = tb.Calculator("GFN2-xTB", atomic_numbers, geometry_bohr)
    xtb.set("verbosity", 0) # Disable energy table output
    results = xtb.singlepoint()
    
    logger.debug(f"Energy: {results['energy']} Hartree")
    
    orbital_energies = results["orbital-energies"]
    orbital_occupations = results["orbital-occupations"]
    
    lumo_index = np.argmax(orbital_occupations)
    homo_index = lumo_index - 1
    homo_energy = orbital_energies[homo_index] * qcel.constants.conversion_factor("hartree", "eV")
    lumo_energy = orbital_energies[lumo_index] * qcel.constants.conversion_factor("hartree", "eV")
    logger.debug(f"homo_energy: {homo_energy:.4f} eV")
    logger.debug(f"lumo_energy: {lumo_energy:.4f} eV")
    return homo_energy, lumo_energy
    
    
import pytest
@pytest.mark.above10s
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "calculate_homo_lumo_energies_generated.csv"
    df = pd.read_csv(dataset_csv).head(10) # Start small, xTB is ~1000x slower than RDKit
    df = calculate_homo_lumo_energies(df, column_name="PSMILES")
    print(df)
    df.to_csv(csv_file, index=False)
