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


def test_():
    smiles = ["CCO", "c1ccccc1"]
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    print(process_mols(mols))
