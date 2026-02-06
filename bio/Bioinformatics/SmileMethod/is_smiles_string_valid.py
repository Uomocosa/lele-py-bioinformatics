from bio.Bioinformatics import Smile
from rdkit import Chem, rdBase
from loguru import logger

def is_smiles_string_valid(smile: Smile):
    rdBase.DisableLog('rdApp.error') # disable warnings
    mol = Chem.MolFromSmiles(smile)
    logger.debug(f"SMILES string {smile} valid?: {mol is not None}")
    if mol is None: return False
    else: return True


import pytest
@pytest.mark.parametrize("input, expected", [
    [Smile("CCO"), True],
    [Smile("C(CO"), False],  # non-closed parentesis
    [Smile("C[Invalid]C"), False], # incorrect atoms
])
def test_(input, expected):
    assert is_smiles_string_valid(input) == expected
