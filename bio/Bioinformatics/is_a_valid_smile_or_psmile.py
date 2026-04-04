import bio
from loguru import logger

def is_a_valid_smile_or_psmile(string: str) -> bool:
    if bio.Bioinformatics.is_psmiles_string_valid(string):
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
    ("CCOc1cc(oc1)/C=C\\2/N(C(=O)N(C2=O)C)CC(=O)N", True), # TODO: really??? this is valid??? 
    ("*CCCCCC", False)
])
def test_(input, expected):
    assert is_a_valid_smile_or_psmile(input) == expected
