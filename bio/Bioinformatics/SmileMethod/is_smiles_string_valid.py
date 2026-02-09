from loguru import logger
from polymetrix.featurizers.molecule import Molecule
from rdkit import RDLogger

def is_smiles_string_valid(smile: str) -> bool:
    RDLogger.DisableLog('rdApp.*') # disable warnings
    try: Molecule.from_smiles(smile)
    except (AssertionError, ValueError): return False
    return True


import pytest
@pytest.mark.parametrize("input, expected", [
    ["CCO", True],
    ["C(CO", False],  # non-closed parentesis
    ["C[Invalid]C", False], # incorrect atoms
])
def test_(input, expected):
    assert is_smiles_string_valid(input) == expected
