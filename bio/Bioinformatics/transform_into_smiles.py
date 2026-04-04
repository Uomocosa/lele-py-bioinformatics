from loguru import logger
from rdkit import Chem
from rdkit import RDLogger

from rdkit import Chem
from rdkit import RDLogger

# Mapping atomic numbers to their periodic symbols
DEFAULT_CAPPING_ATOMS = {
    "H": 1,
    "C": 6,
    "O": 8
}

def transform_into_smiles(
    psmiles_str: str,
    capping_atoms_dict: dict = DEFAULT_CAPPING_ATOMS
) -> dict:
    """
    NOTE! Made using AI.
    TODO! Change the return type instead of a list it should be a dict {atom: smile}
    
    Takes a P-SMILES string containing dummy atoms (*).
    Returns a dict {atom_symbol: smile} of valid standard SMILES.
    """
    RDLogger.DisableLog('rdApp.*') 
    if not isinstance(psmiles_str, str): 
        return {}
    
    mol = Chem.MolFromSmiles(psmiles_str)
    if mol is None:  return {}
        
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    
    # CASE 1: Standard SMILES (No dummy atoms)
    if not dummy_indices:
        try:
            Chem.SanitizeMol(mol)
            return {"": Chem.MolToSmiles(mol)}
        except Exception: 
            return {}
            
    # CASE 2: P-SMILES (Has dummy atoms)
    results = {}


    for symbol, atomic_num in capping_atoms_dict.items():
        mol_copy = Chem.Mol(mol)
        
        # Replace all dummy atoms with the target element
        for idx in dummy_indices:
            atom = mol_copy.GetAtomWithIdx(idx)
            atom.SetAtomicNum(atomic_num)
            # Ensure isotope/formal charge info from dummy is cleared
            atom.SetIsotope(0) 
            
        try:
            # Check chemical validity (valency, aromaticity, etc.)
            Chem.SanitizeMol(mol_copy)
            
            # For H-capping, we usually want to remove explicit [H] 
            # for a "clean" standard SMILES string.
            clean_mol = Chem.RemoveHs(mol_copy)
            clean_smiles = Chem.MolToSmiles(clean_mol)
            
            results[symbol] = clean_smiles
        except Exception:
            # Skip if the capping element violates valency rules
            continue 
    return results
    
    
import pytest
@pytest.mark.parametrize("psmiles, expected_keys", [
    ("*/C=C/c1cc(CCCCCC)c(*)s1", ["H", "C", "O"]),
    ("*Oc1ccccc1C(=O)NCCCSCCSCSc1cc(*)s1", []), # Invalid SMILES (unclosed ring/paren)
    ("c1ccccc1", [""]), # No dummies
])
def test_transform(psmiles, expected_keys):
    output = transform_into_smiles(psmiles)
    print(f"\nInput: {psmiles}.\nOutput:\n\t{output}\n\n")
    assert isinstance(output, dict)
    assert sorted(list(output.keys())) == sorted(expected_keys)
