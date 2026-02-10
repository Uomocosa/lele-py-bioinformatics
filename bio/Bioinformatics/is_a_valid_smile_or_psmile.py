import bio
from loguru import logger

def is_a_valid_smile_or_psmile(string: str) -> bool:
    if bio.Bioinformatics.PSmileMethod.is_psmile_string_valid(string):
        logger.debug(f"P-SMILE string '{string}' is valid.")
        return True
    if "*" in string: 
        return False
    if bio.Bioinformatics.SmileMethod.is_smiles_string_valid(string):
        logger.debug(f"SMILE string '{string}' is valid.")
        return True
    return False


import pytest
@pytest.mark.parametrize("input, expected", [
    ("*CCCCCC", False)
])
def test_(input, expected):
    assert is_a_valid_smile_or_psmile(input) == expected
