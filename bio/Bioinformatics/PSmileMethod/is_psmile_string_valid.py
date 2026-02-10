"""
We test 1x, 2x, 3x, and 5x lengths to ensure ring closures and 
syntax work when polymerized.
TODO: This might not be the best way to validate p-smiles.
TODO: 1, 2, 3 and 5 are MAGIC NUMBERS.
"""

from loguru import logger
from polymetrix.featurizers.polymer import Polymer
from rdkit import RDLogger
# import lele, bio
# PSmile = bio.Bioinformatics.PSmile.PSmile

def is_psmile_string_valid(psmile: str) -> bool:
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
    assert is_psmile_string_valid(input) == output
