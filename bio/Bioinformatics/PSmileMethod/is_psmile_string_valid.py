"""
We test 1x, 2x, 3x, and 5x lengths to ensure ring closures and 
syntax work when polymerized.
TODO: This might not be the best way to validate p-smiles.
TODO: 1, 2, 3 and 5 are MAGIC NUMBERS.
"""

from loguru import logger
import lele, bio
PSmile = bio.Bioinformatics.PSmile.PSmile

def is_psmile_string_valid(psmile: str) -> bool:
    psmile = PSmile.from_str(psmile)
    if not psmile: return False
    valid_checks = map(lambda n: psmile.to_smile(n).is_valid, [1, 2, 3, 5])
    valid_checks = list(valid_checks)
    logger.debug(f"valid_checks: {valid_checks}")
    logger.debug(f"any(valid_checks): {any(valid_checks)}")
    return any(valid_checks)
    
import pytest
@pytest.mark.parametrize("input, output", [
    ("*CCCC1CCCC(*)c2ccc2Ncc1", True),
    ("*C(=O)O*", True),
    ("*C(=O)O(*)", False),
    ("*CCCC1CCCC*c2ccc2Ncc1", False),
    ("C1=CC=C", False), # Invalid p-smiles (no stars), but valid molecule
    ("InvalidSMILESString", False)
])
def test_(input, output):
    assert is_psmile_string_valid(input) == output
