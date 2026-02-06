from bio.Bioinformatics import Smile
from typing import Optional
from rdkit import Chem
import logging; logging.getLogger("deepchem").setLevel(logging.ERROR)

def canonicalize_smiles(smile: Smile) -> Optional[str]:
    mol = Chem.MolFromSmiles(smile)
    if mol is not None:
        smile_str = Chem.MolToSmiles(mol, isomericSmiles=True)
        return smile_str
    else:
        return None


def test_str():
    import lele
    smiles_str = "COC"
    print(f"smiles_str: {smiles_str}")
    print(f"canonicalize_smiles(smiles_str): {canonicalize_smiles(smiles_str)}")


def test_smile():
    import lele
    smiles_str = Smile("COC")
    print(f"smiles_str: {smiles_str}")
    print(f"canonicalize_smiles(smiles_str): {canonicalize_smiles(smiles_str)}")
