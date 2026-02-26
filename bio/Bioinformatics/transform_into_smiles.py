from loguru import logger
from rdkit import Chem
from rdkit import RDLogger

"""
NOTE! Made using AI.

Takes a P-SMILES or SMILES string.
Returns a list of valid standard SMILES by replacing dummy atoms (*).
If the base molecule is invalid or no caps work, returns an empty list.
"""
def transform_into_smiles(smiles_str: str) -> list:
    RDLogger.DisableLog('rdApp.*') # disable warnings
    if not isinstance(smiles_str, str): return []
    
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is None: return [] 
        
    # Find all dummy atoms (*)
    dummy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    
    # CASE 1: Standard SMILES (No dummy atoms)
    if not dummy_indices:
        try:
            Chem.SanitizeMol(mol)
            return [Chem.MolToSmiles(mol)]
        except Exception: 
            return []
            
    # CASE 2: P-SMILES (Has dummy atoms)
    valid_smiles = set() # Use a set to prevent duplicate SMILES strings
    
    # Try replacing all dummy atoms uniformly with H, C, and O
    CAPPING_ATOMIC_NUMS=(1, 6, 8)
    for atomic_num in CAPPING_ATOMIC_NUMS:
        mol_copy = Chem.Mol(mol) # Create a fresh copy for each attempt
        
        # Apply the current cap to all dummy atoms
        for idx in dummy_indices:
            atom = mol_copy.GetAtomWithIdx(idx)
            atom.SetAtomicNum(atomic_num)
            
        try:
            # The strict chemistry test
            Chem.SanitizeMol(mol_copy)
            
            # Clean up the molecule (removes explicit Hydrogens we may have just added)
            # so the resulting SMILES looks standard (e.g., 'C' instead of '[H]C')
            mol_copy = Chem.RemoveHs(mol_copy) 
            
            clean_smiles = Chem.MolToSmiles(mol_copy)
            valid_smiles.add(clean_smiles)
        except Exception:
            # If replacing with this atom breaks valency (e.g., H on a double bond), we just skip it
            pass 
            
    return list(valid_smiles)

import pytest
@pytest.mark.parametrize("input, expected", [
    ["*/C=C/c1cc(CCCCCC)c(*)s1", ['C/C=C/c1cc(CCCCCC)c(C)s1', 'CCCCCCc1cc(/C=C/O)sc1O', '[H]/C=C/c1cc(CCCCCC)cs1']],
    ["*Oc1ccccc1C(=O)NCCCSCCSCSc1cc(*)s1", []],  # non-closed parentesis
])
def test_(input, expected):
    output = sorted(transform_into_smiles(input))
    assert output == expected, f"\n\tExpected: >>> {expected}\n\tGot: >>> {output}"
