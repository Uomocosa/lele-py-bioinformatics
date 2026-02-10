from loguru import logger
from polymetrix.featurizers.polymer import Polymer
from rdkit import RDLogger

def is_psmiles_string_valid(psmile: str) -> bool:
    if psmile.count('*') != 2: return False
    RDLogger.DisableLog('rdApp.*')
    try: Polymer.from_psmiles(psmile)
    except (AssertionError, ValueError): return False
    return True
    
import pytest
@pytest.mark.parametrize("input, output", [
    ("*CCCC1CCCC(*)c2ccc2Ncc1", True),
    ("*C(=O)O*", True),
    ("*C(=O)O(*)", True),
    ("*CCCC1CCCC*c2ccc2Ncc1", True),
    ("C1=CC=C", False), # Invalid p-smiles (no stars), but valid molecule
    ("*C1=CC=C", False), # Invalid p-smiles (1 star)
    ("*CCCCCC", False), # Invalid p-smiles (1 star)
    ("InvalidSMILESString", False)
])
def test_(input, output):
    assert is_psmiles_string_valid(input) == output
