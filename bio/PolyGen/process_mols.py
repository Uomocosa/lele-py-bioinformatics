from rdkit import Chem
from typing import List
import bio

"""
NOTE! Refactored using GEMINI (AI)
NOTE! Seems kinda useless
"""
def process_mols(mols: List[Chem.Mol]) -> None:
    print('smiles\tName\tsa_score')
    for i, mol in enumerate(mols):
        if mol is None: continue
        s = bio.PolyGen.calculate_sa_score(mol)
        smiles = Chem.MolToSmiles(mol)
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else "NameNotGiven"
        print(f"{smiles}\t{name}\t{s:.3f}")


import pytest
@pytest.mark.parametrize("smiles,expected", [
    (["CCO"], None),
    (["c1ccccc1"], None)
])
def test_process_mols(smiles, expected):
    print()
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    result = process_mols(mols)
    assert result == expected, f"Expected {expected}, but got {result}"
