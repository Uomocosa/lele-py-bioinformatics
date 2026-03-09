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
from bio.Bioinformatics.transform_into_smiles import DEFAULT_CAPPING_ATOMS
from loguru import logger


def calculate_homo_lumo_energies(
    df: pd.DataFrame, 
    column_name: str, 
    capping_atoms_dict: dict = DEFAULT_CAPPING_ATOMS,
    starting_lable: str = '',
) -> pd.DataFrame:
    """
    Calculates the HOMO-LUMO gap for each SMILES in the dataframe.
    """
    fn = lambda s: single_calculation(
        s, 
        capping_atoms_dict,
        starting_lable,
    )
    stats_df = df[column_name].apply(fn)
    df_out = pd.concat([df, stats_df], axis=1)
    return df_out


def single_calculation(
    smiles_str: str, 
    capping_atoms_dict: dict, 
    starting_lable: str,
) -> pd.Series:
    """
    Performs 3D optimization and GFN2-xTB calculation to find the HOMO-LUMO gap.
    """
    homo_lable = f"{starting_lable}_homo_eV"
    lumo_lable = f"{starting_lable}_lumo_eV"
    if not starting_lable: homo_lable = homo_lable.removeprefix("_")
    if not starting_lable: lumo_lable = lumo_lable.removeprefix("_")
    null_result = pd.Series({})
    smiles_str = str(smiles_str)
    valid_smiles_dict = bio.Bioinformatics.transform_into_smiles(smiles_str, capping_atoms_dict)
    logger.debug(f"valid_smiles_dict: {valid_smiles_dict}")
    if not valid_smiles_dict: return null_result
    
    mols = {atom: Chem.MolFromSmiles(smile) for atom, smile in valid_smiles_dict.items()}
    homo_lumo_energies = dict()
    for atom, mol in mols.items():
        if mol is None:
            continue
        try:
            homo_lumo_energies[atom] = from_mol(mol)
        except Exception as e:
            if not atom: logger.warning(f"xTB calculation failed: {e}")
            else: logger.warning(f"xTB calculation failed for atom '{atom}': {e}")
            homo_lumo_energies[atom] = None
    homo_lumo_energies = {atom: eVs for atom, eVs in homo_lumo_energies.items() if eVs is not None}
    
    df = dict()
    for atom, energies in homo_lumo_energies.items():
        homo_eV, lumo_eV = energies
        homo_key = f"{homo_lable}_{atom}"
        lumo_key = f"{lumo_lable}_{atom}"
        homo_key = homo_key.removesuffix('_')
        lumo_key = lumo_key.removesuffix('_')
        df[homo_key] = homo_eV
        df[lumo_key] = lumo_eV
    if not homo_lumo_energies: return null_result
    logger.debug(f'homo_lumo_energies: {df}')
    return pd.Series(df)
    
    
def from_mol(mol: Chem.Mol) -> Optional[Tuple[float, float]]:
    """
    based on :
        - https://tblite.readthedocs.io/en/latest/tutorial/python/singlepoint.html#homo-lumo-gap
    """
    mol_3D = bio.Bioinformatics.transform_into_3D_geometry(mol)
    
    if mol_3D is None:
        logger.warning("Skipping QM calculation: Could not generate 3D geometry.")
        return None
        
    atomic_numbers = np.array([atom.GetAtomicNum() for atom in mol_3D.GetAtoms()])
    logger.trace(f"atomic_numbers: {atomic_numbers}")
    
    try:
        coords_angstrom = mol_3D.GetConformer().GetPositions()
    except ValueError:
        logger.warning("Molecule has no 3D coordinates generated. Skipping.")
        return None
    angstrom_to_bohr = qcel.constants.conversion_factor("angstrom", "bohr")
    geometry_bohr = coords_angstrom * angstrom_to_bohr
    logger.trace(f"geometry_bohr: {geometry_bohr}")
    
    xtb = tb.Calculator("GFN2-xTB", atomic_numbers, geometry_bohr)
    xtb.set("verbosity", 0) # Disable energy table output
    try:
        results = xtb.singlepoint()
    except Exception as e:
        logger.error(f"xTB singlepoint failed (likely SCF non-convergence or clashing atoms): {e}")
        return None 
    logger.trace(f"Energy: {results['energy']} Hartree")    
    
    orbital_energies = results["orbital-energies"]
    orbital_occupations = results["orbital-occupations"]   

    # lumo_index = np.argmax(orbital_occupations) # This is taken from the official docs!
    lumo_index = np.argmin(orbital_occupations) # GEMINI AI: Use argmin instead of argmax to find the first 0.0 occupation
    homo_index = lumo_index - 1
        
    homo_index = lumo_index - 1
    homo_energy = orbital_energies[homo_index] * qcel.constants.conversion_factor("hartree", "eV")
    lumo_energy = orbital_energies[lumo_index] * qcel.constants.conversion_factor("hartree", "eV")
    logger.trace(f"homo_energy: {homo_energy:.4f} eV")
    logger.trace(f"lumo_energy: {lumo_energy:.4f} eV")
    return homo_energy, lumo_energy


import pytest
# @pytest.mark.skip("REMOVE THIS LINE")
def test_():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "calculate_homo_lumo_energies_generated.csv"
    df = pd.read_csv(dataset_csv).head(3) 
    df = calculate_homo_lumo_energies(df, column_name="PSMILES", capping_atoms_dict={"H": 1})
    print(df)
    df.to_csv(csv_file, index=False)
  
   
    
import pytest
@pytest.mark.above10s
# @pytest.mark.skip("REMOVE THIS LINE")
def test_generated():
    from bio.__global__ import BIOINFORMATICS_DIR
    from bio.Metric.__global__ import HELPER_DIR
    dataset_csv = BIOINFORMATICS_DIR / "COMBINED_checkpoints" / "2026_02_07_202304_051020" / "generate_mnt128_t100000000" / "2026_02_10_093248_774466" / "generated_smiles.csv"
    csv_file = HELPER_DIR / "calculate_homo_lumo_energies_generated.csv"
    df = pd.read_csv(dataset_csv).head(10) # Start small, xTB is ~1000x slower than RDKit
    df = calculate_homo_lumo_energies(df, column_name="PSMILES")
    print(df)
    df.to_csv(csv_file, index=False)


    
import pytest
@pytest.mark.above10s    
@pytest.mark.skip("changes logger")
def test_all():
    import sys
    import lele
    from bio.__global__ import CONVERTED_PDCC_CSV, LOGURU_SIMPLE_FORMAT
    from bio.Metric.__global__ import HELPER_DIR
    logger.remove()
    logger.add(
        sys.stderr,
        format = LOGURU_SIMPLE_FORMAT,
        level = "INFO"
    )
    csv_file = HELPER_DIR / "calculate_homo_lumo_energies_polymer_used.csv"
    df = pd.read_csv(CONVERTED_PDCC_CSV) 
    df = calculate_homo_lumo_energies(df, column_name="POLYMER_USED", capping_atoms_dict={"H": 1})
    print(df)
    df.to_csv(csv_file, index=False)
    csv_file = HELPER_DIR / "calculate_homo_lumo_energies_drugs.csv"
    df = pd.read_csv(CONVERTED_PDCC_CSV) 
    df = calculate_homo_lumo_energies(df, column_name="DRUG", capping_atoms_dict={"H": 1})
    print(df)
    df.to_csv(csv_file, index=False)
