from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from typing import Tuple

"""
FUNCTION SUMMARY:
Calculates the count of bridgehead atoms and spiro atoms in a molecule.
- Bridgehead Atoms: Atoms shared by three or more rings in a polycyclic system.
- Spiro Atoms: A single atom that serves as the only common point between two rings.
These descriptors are often used to estimate molecular complexity (e.g., Fraction Csp3).
"""
def num_bridgeheads_and_spiro(mol) -> Tuple[int, int]:
    if mol is None: return 0, 0
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return n_bridgehead, n_spiro


def test_bridgeheads_and_spiro():
    # Spiro example: Spiro[4.4]nonane
    spiro_mol = Chem.MolFromSmiles('C1CCC2(C1)CCCC2') 
    # Bridgehead example: Adamantane
    bridge_mol = Chem.MolFromSmiles('C1C2CC3CC1CC(C2)C3') 
    
    b1, s1 = num_bridgeheads_and_spiro(spiro_mol)
    b2, s2 = num_bridgeheads_and_spiro(bridge_mol)
    
    print(f"Spiro Molecule -> Bridgeheads: {b1}, Spiro: {s1}")
    print(f"Bridgehead Molecule -> Bridgeheads: {b2}, Spiro: {s2}")
    
    assert s1 == 1 and b1 == 0
    assert b2 == 4 and s2 == 0
